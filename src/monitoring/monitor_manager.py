"""
Central manager for all monitoring components with enhanced capabilities
"""

from typing import Dict, List, Optional, Any, Union, TypeVar, overload, cast
from dataclasses import dataclass, field
import structlog
from pathlib import Path
import asyncio
import redis.asyncio as aioredis
from prometheus_client import start_http_server
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import warnings

# Optional dependencies with fallbacks
try:
    import sentry_sdk
    from sentry_sdk.integrations.redis import RedisIntegration
    from sentry_sdk.integrations.logging import LoggingIntegration
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False
    warnings.warn(
        "Sentry SDK not available - error reporting will be disabled",
        ImportWarning
    )

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False
    warnings.warn(
        "OpenTelemetry not available - distributed tracing will be disabled",
        ImportWarning
    )

# Local imports
from ..utils.unified_metrics import UnifiedMetricsSystem, MetricConfig
from .specialized.trade_monitor import TradeMonitor, TradeMetrics
from .specialized.system_monitor import SystemMonitor, ResourceType, ResourceThreshold
from .core.metrics_manager import metrics_manager
from .visualization.learning_insights import LearningInsightsVisualizer

logger = structlog.get_logger(__name__)

T = TypeVar('T')

@dataclass
class MonitoringConfig:
    """Configuration for monitoring system"""
    metrics_interval: int = 60  # seconds
    log_level: str = "INFO"
    enable_metrics: bool = True
    enable_tracing: bool = True
    enable_profiling: bool = True
    trade_cache_ttl: int = 3600  # 1 hour
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    sentry_dsn: Optional[str] = None
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'min_profit': -0.1,
        'max_gas_price': 100.0,
        'max_execution_time': 5.0,
        'min_success_rate': 95.0
    })
    resource_thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'cpu': {'warning': 80.0, 'critical': 90.0},
        'memory': {'warning': 80.0, 'critical': 90.0},
        'disk': {'warning': 85.0, 'critical': 95.0},
        'network': {'warning': 5.0, 'critical': 10.0}
    })
    ml_config: Dict[str, Any] = field(default_factory=lambda: {
        'profit_prediction': {
            'window_size': 24,
            'train_size': 0.8,
            'feature_importance_threshold': 0.05
        },
        'anomaly_detection': {
            'contamination': 0.1,
            'n_estimators': 100,
            'max_features': 0.8
        }
    })
    
    @overload
    def get(self, key: str) -> Any: ...
    
    @overload
    def get(self, key: str, default: T) -> Union[Any, T]: ...
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with fallback"""
        return getattr(self, key, default)
    
    def get_resource_thresholds(self) -> Dict[ResourceType, ResourceThreshold]:
        """Get resource thresholds in correct format"""
        thresholds: Dict[ResourceType, ResourceThreshold] = {}
        for resource_str, values in self.resource_thresholds.items():
            try:
                resource_type = ResourceType(resource_str)
                thresholds[resource_type] = ResourceThreshold(
                    warning=values['warning'],
                    critical=values['critical']
                )
            except (ValueError, KeyError) as e:
                logger.error(
                    "Invalid resource threshold configuration",
                    resource=resource_str,
                    error=str(e)
                )
        return thresholds

class MonitorManager:
    """Central manager for all monitoring components"""
    
    def __init__(
        self,
        config: Union[Dict[str, Any], MonitoringConfig],
        storage_path: str = "data/monitoring",
        prometheus_port: int = 8000,
        cache_enabled: bool = True
    ):
        """Initialize monitoring manager"""
        self.config = config if isinstance(config, MonitoringConfig) else MonitoringConfig(**config)
        self.storage_path = Path(storage_path)
        self.prometheus_port = prometheus_port
        self.cache_enabled = cache_enabled
        
        # Create storage directories
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize unified metrics
        self.metrics = UnifiedMetricsSystem(
            enable_prometheus=self.config.enable_metrics
        )
        
        # Initialize monitoring components
        self.trade_monitor = TradeMonitor(
            storage_path=str(self.storage_path / "trades"),
            max_memory_entries=10000,
            flush_interval=100,
            enable_prometheus=self.config.enable_metrics,
            performance_thresholds=self.config.alert_thresholds,
            ml_config=self.config.ml_config,
            metrics_system=self.metrics
        )
        
        self.system_monitor = SystemMonitor(
            storage_path=str(self.storage_path / "system"),
            alert_webhook_url=self.config.get('alert_webhook_url'),
            collection_interval=self.config.metrics_interval,
            enable_prometheus=self.config.enable_metrics,
            resource_thresholds=self._convert_resource_thresholds(),
            metrics_system=self.metrics
        )
        
        # Initialize visualization if available
        self.learning_viz = LearningInsightsVisualizer(self.trade_monitor)
        
        # Initialize Redis cache if enabled
        self.cache: Optional[aioredis.Redis] = None
        if cache_enabled:
            self._setup_cache()
        
        # Setup monitoring infrastructure
        self._setup_logging()
        self._setup_tracing()
        
        # Background tasks
        self._tasks: List[asyncio.Task] = []
        self._running = False

    def _convert_resource_thresholds(self) -> Dict[ResourceType, ResourceThreshold]:
        """Convert resource thresholds from config format to monitor format"""
        thresholds = {}
        for resource_name, values in self.config.resource_thresholds.items():
            try:
                resource_type = ResourceType(resource_name)
                thresholds[resource_type] = ResourceThreshold(
                    warning=values['warning'],
                    error=values['warning'] + (values['critical'] - values['warning']) * 0.5,
                    critical=values['critical']
                )
            except (ValueError, KeyError) as e:
                logger.error(
                    "Invalid resource threshold configuration",
                    resource=resource_name,
                    error=str(e)
                )
        return thresholds

    def _setup_metrics_export(self):
        """Setup metrics export"""
        if not self.config.enable_metrics:
            return
            
        try:
            # Start Prometheus server
            start_http_server(self.prometheus_port)
            logger.info("Metrics export initialized", prometheus_port=self.prometheus_port)
        except Exception as e:
            logger.error("Error setting up metrics export", error=str(e))
    
    def _setup_logging(self):
        """Setup logging and error reporting"""
        try:
            # Configure logging
            logging.basicConfig(
                level=self.config.log_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Initialize Sentry if configured and available
            if SENTRY_AVAILABLE and self.config.sentry_dsn:
                sentry_sdk.init(
                    dsn=self.config.sentry_dsn,
                    traces_sample_rate=0.1,
                    integrations=[
                        RedisIntegration(),
                        LoggingIntegration(
                            level=None,
                            event_level=logging.ERROR
                        )
                    ]
                )
                logger.info("Sentry error reporting initialized")
            
        except Exception as e:
            logger.error("Error setting up logging", error=str(e))
    
    def _setup_tracing(self):
        """Setup distributed tracing"""
        if not self.config.enable_tracing or not TRACING_AVAILABLE:
            return
            
        try:
            # Set up tracer
            trace.set_tracer_provider(TracerProvider())
            tracer = trace.get_tracer_provider()
            
            # Configure OTLP exporter
            otlp_exporter = OTLPSpanExporter()
            span_processor = BatchSpanProcessor(otlp_exporter)
            tracer.add_span_processor(span_processor)
            
            logger.info("Distributed tracing initialized")
            
        except Exception as e:
            logger.error("Error setting up tracing", error=str(e))
    
    def _setup_cache(self):
        """Setup Redis cache"""
        if not self.cache_enabled:
            return
            
        try:
            self.cache = aioredis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db
            )
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.error("Failed to connect to Redis cache", error=str(e))
            self.cache_enabled = False
    
    async def start(self):
        """Start all monitoring components"""
        if self._running:
            return
            
        self._running = True
        try:
            # Start system monitoring
            await self.system_monitor.start_monitoring()
            
            # Start trade monitoring tasks
            await self.trade_monitor.start_monitoring()
            
            logger.info("Started all monitoring components")
            
        except Exception as e:
            logger.error("Error starting monitoring components", error=str(e))
            raise

    async def stop(self):
        """Stop all monitoring components"""
        if not self._running:
            return
            
        self._running = False
        try:
            # Stop system monitoring
            await self.system_monitor.stop_monitoring()
            
            # Stop trade monitoring
            await self.trade_monitor.stop_monitoring()
            
            # Cancel all tasks
            for task in self._tasks:
                task.cancel()
            
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()
            
            logger.info("Stopped all monitoring components")
            
        except Exception as e:
            logger.error("Error stopping monitoring components", error=str(e))
            raise
    
    async def _monitor_trades(self):
        """Monitor trading activity"""
        while self._running:
            try:
                # Get recent trades
                trades = self.trade_monitor.get_trade_history(
                    start_time=datetime.now() - pd.Timedelta(minutes=5)
                )
                
                if not trades.empty:
                    # Update cache if enabled
                    if self.cache_enabled and self.cache:
                        await self._cache_trade_data(trades)
                    
                    # Check for anomalies
                    anomalies = self.trade_monitor.detect_anomalies()
                    if anomalies:
                        logger.warning(
                            "Detected trade anomalies",
                            count=len(anomalies['anomaly_indices'])
                        )
                
            except Exception as e:
                logger.error("Error monitoring trades", error=str(e))
            
            await asyncio.sleep(60)  # Check every minute
    
    async def _analyze_performance(self):
        """Analyze system and trading performance"""
        while self._running:
            try:
                # Analyze trade performance
                trade_metrics = self.trade_monitor.analyze_performance(
                    timeframe='1h'
                )
                
                if trade_metrics:
                    # Generate visualizations if available
                    if self.learning_viz:
                        self.learning_viz.visualize_performance(trade_metrics)
                    
                    # Cache performance data if enabled
                    if self.cache_enabled and self.cache:
                        await self._cache_performance_data(trade_metrics)
                
            except Exception as e:
                logger.error("Error analyzing performance", error=str(e))
            
            await asyncio.sleep(300)  # Analyze every 5 minutes
    
    async def _cache_trade_data(self, trades: pd.DataFrame):
        """Cache recent trade data
        
        Args:
            trades: DataFrame with trade data
        """
        if not self.cache:
            return
            
        try:
            for _, trade in trades.iterrows():
                key = f"trade:{trade['timestamp'].isoformat()}"
                await self.cache.setex(
                    key,
                    self.config.trade_cache_ttl,
                    trade.to_json()
                )
        except Exception as e:
            logger.error("Error caching trade data", error=str(e))
    
    async def _cache_performance_data(self, metrics: Dict[str, Any]):
        """Cache performance metrics
        
        Args:
            metrics: Performance metrics dictionary
        """
        if not self.cache:
            return
            
        try:
            key = f"performance:{datetime.now().isoformat()}"
            await self.cache.setex(
                key,
                self.config.trade_cache_ttl,
                str(metrics)
            )
        except Exception as e:
            logger.error("Error caching performance data", error=str(e))
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics
        
        Returns:
            Dictionary with system metrics
        """
        metrics = {}
        
        for resource_type in self.system_monitor._resource_history:
            history = self.system_monitor._resource_history[resource_type]
            if history:
                metrics[resource_type.value] = history[-1]['metrics']
        
        return metrics
    
    def get_trade_metrics(self) -> Dict[str, Any]:
        """Get current trade metrics
        
        Returns:
            Dictionary with trade metrics
        """
        return self.trade_monitor.analyze_performance(timeframe='5m')
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts
        
        Returns:
            List of active alerts
        """
        alerts = []
        
        # System alerts
        for alert in self.system_monitor._alert_history:
            alerts.append({
                'timestamp': alert.timestamp.isoformat(),
                'type': 'system',
                'resource': alert.resource_type.value,
                'severity': alert.severity.value,
                'message': alert.message
            })
        
        # Trade alerts
        for alert in self.trade_monitor._recent_alerts:
            alerts.append({
                'timestamp': alert.timestamp.isoformat(),
                'type': 'trade',
                'metric': alert.metric,
                'value': alert.value,
                'threshold': alert.threshold,
                'severity': alert.severity,
                'message': alert.message
            })
        
        return alerts
    
    async def record_trade(
        self,
        strategy: str,
        token_pair: str,
        dex: str,
        profit: float,
        gas_price: float,
        execution_time: float,
        success: bool,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a trade with all monitoring components
        
        Args:
            strategy: Trading strategy name
            token_pair: Token pair traded
            dex: DEX used
            profit: Trade profit
            gas_price: Gas price used
            execution_time: Execution time
            success: Whether trade was successful
            additional_data: Additional metadata
        """
        try:
            # Create trade metrics
            metrics = TradeMetrics(
                timestamp=datetime.now(),
                strategy=strategy,
                token_pair=token_pair,
                dex=dex,
                profit=profit,
                gas_price=gas_price,
                execution_time=execution_time,
                success=success,
                additional_data=additional_data or {}
            )
            
            # Record with trade monitor
            await self.trade_monitor.record_trade(metrics)
            
            # Cache recent trade data if enabled
            if self.cache_enabled and self.cache:
                await self._cache_trade_data(pd.DataFrame([{
                    'timestamp': metrics.timestamp,
                    'strategy': metrics.strategy,
                    'token_pair': metrics.token_pair,
                    'dex': metrics.dex,
                    'profit': metrics.profit,
                    'gas_price': metrics.gas_price,
                    'execution_time': metrics.execution_time,
                    'success': metrics.success,
                    'additional_data': metrics.additional_data
                }]))
            
            logger.info(
                "Trade recorded",
                strategy=strategy,
                token_pair=token_pair,
                profit=profit,
                success=success
            )
            
        except Exception as e:
            logger.error("Error recording trade", error=str(e))
    
    def predict_trade_profit(
        self,
        strategy: str,
        token_pair: str,
        market_conditions: Dict[str, float]
    ) -> Dict[str, Any]:
        """Predict profit for a potential trade
        
        Args:
            strategy: Trading strategy
            token_pair: Token pair
            market_conditions: Current market conditions
            
        Returns:
            Dictionary with prediction results
        """
        prediction = self.trade_monitor.predict_profit(
            strategy=strategy,
            token_pair=token_pair,
            market_conditions=market_conditions
        )
        
        return {
            'predicted_profit': prediction.predicted_value,
            'confidence': prediction.confidence,
            'important_features': prediction.features_importance,
            'model_metrics': prediction.model_metrics
        }
    
    def export_metrics(self, export_path: Optional[str] = None) -> None:
        """Export all metrics to file
        
        Args:
            export_path: Optional path to export metrics to
        """
        try:
            if not export_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                export_path = str(self.storage_path / f"metrics_export_{timestamp}")
            
            # Export system metrics
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'system_metrics': self.get_system_metrics(),
                'trade_metrics': self.get_trade_metrics(),
                'alerts': self.get_alerts()
            }
            
            Path(f"{export_path}.json").write_text(
                str(metrics_data)
            )
            
            logger.info(f"Exported metrics to {export_path}")
            
        except Exception as e:
            logger.error("Error exporting metrics", error=str(e))
    
    async def get_trade_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        strategy: Optional[str] = None,
        token_pair: Optional[str] = None,
        include_memory: bool = True
    ) -> pd.DataFrame:
        """Get historical trade data
        
        Args:
            start_time: Start time for history
            end_time: End time for history
            strategy: Filter by strategy
            token_pair: Filter by token pair
            include_memory: Whether to include in-memory trades
            
        Returns:
            DataFrame with historical trade data
        """
        return self.trade_monitor.get_trade_history(
            start_time=start_time,
            end_time=end_time,
            strategy=strategy,
            token_pair=token_pair,
            include_memory=include_memory
        ) 