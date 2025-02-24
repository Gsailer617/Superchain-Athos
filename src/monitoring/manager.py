"""
Unified monitoring manager that coordinates all monitoring components
"""

from typing import Dict, Any, Optional, List
import structlog
from pathlib import Path
import asyncio
import redis
from prometheus_client import start_http_server
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .specialized.trade_monitor import TradeMonitor, TradeMetrics
from .specialized.system_monitor import SystemMonitor, ResourceType
from .core.base_monitor import ResourceThreshold
from visualization.learning_insights import LearningInsightsVisualizer

logger = structlog.get_logger(__name__)

class MonitorManager:
    """Central manager for all monitoring components with enhanced learning capabilities"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        storage_path: str = "data/monitoring",
        prometheus_port: int = 8000,
        cache_enabled: bool = True
    ):
        """Initialize monitoring manager
        
        Args:
            config: Configuration dictionary
            storage_path: Base path for storing monitoring data
            prometheus_port: Port for Prometheus metrics server
            cache_enabled: Whether to enable Redis caching
        """
        self.config = config
        self.storage_path = Path(storage_path)
        self.prometheus_port = prometheus_port
        self.cache_enabled = cache_enabled
        
        # Create storage directories
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize monitoring components
        self.trade_monitor = TradeMonitor(
            storage_path=str(self.storage_path / "trades"),
            max_memory_entries=config.get('max_memory_entries', 10000),
            flush_interval=config.get('flush_interval', 100),
            enable_prometheus=True
        )
        
        self.system_monitor = SystemMonitor(
            storage_path=str(self.storage_path / "system"),
            max_memory_entries=config.get('max_memory_entries', 10000),
            flush_interval=config.get('flush_interval', 100),
            enable_prometheus=True,
            collection_interval=config.get('collection_interval', 10)
        )
        
        # Setup system monitoring thresholds
        self._setup_monitoring_thresholds()
        
        # Initialize visualization
        self.learning_viz = LearningInsightsVisualizer(
            history_manager=self.trade_monitor  # type: ignore
        )
        
        # Initialize Redis cache if enabled
        self.cache: Optional[redis.Redis] = None
        if cache_enabled:
            self._setup_cache()
        
        # Start Prometheus server
        try:
            start_http_server(prometheus_port)
            logger.info(f"Started Prometheus server on port {prometheus_port}")
        except Exception as e:
            logger.error("Failed to start Prometheus server", error=str(e))
    
    def _setup_monitoring_thresholds(self):
        """Setup default monitoring thresholds"""
        # CPU thresholds
        self.system_monitor.set_threshold(
            ResourceType.CPU,
            warning=80.0,
            critical=90.0,
            check_interval=30
        )
        
        # Memory thresholds
        self.system_monitor.set_threshold(
            ResourceType.MEMORY,
            warning=80.0,
            critical=90.0,
            check_interval=30
        )
        
        # Disk thresholds
        self.system_monitor.set_threshold(
            ResourceType.DISK,
            warning=85.0,
            critical=95.0,
            check_interval=300
        )
        
        # Network thresholds
        self.system_monitor.set_threshold(
            ResourceType.NETWORK,
            warning=5.0,  # 5% error rate
            critical=10.0,  # 10% error rate
            check_interval=60
        )
    
    def _setup_cache(self):
        """Setup Redis cache"""
        try:
            self.cache = redis.Redis(
                host=self.config.get('redis_host', 'localhost'),
                port=self.config.get('redis_port', 6379),
                db=self.config.get('redis_db', 0)
            )
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.error("Failed to connect to Redis cache", error=str(e))
    
    async def start(self):
        """Start all monitoring components"""
        try:
            # Start trade monitoring
            await self.trade_monitor.start_monitoring()
            
            # Start system monitoring
            await self.system_monitor.start_monitoring()
            
            logger.info("Started all monitoring components")
            
        except Exception as e:
            logger.error("Error starting monitoring components", error=str(e))
            raise
    
    async def stop(self):
        """Stop all monitoring components"""
        try:
            # Stop trade monitoring
            await self.trade_monitor.stop_monitoring()
            
            # Stop system monitoring
            await self.system_monitor.stop_monitoring()
            
            logger.info("Stopped all monitoring components")
            
        except Exception as e:
            logger.error("Error stopping monitoring components", error=str(e))
            raise
    
    def record_trade(
        self,
        strategy: str,
        token_pair: str,
        dex: str,
        profit: float,
        gas_price: float,
        execution_time: float,
        success: bool,
        additional_data: Optional[Dict[str, Any]] = None
    ):
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
            self.trade_monitor.record_trade(metrics)
            
            # Cache recent trade data if enabled
            if self.cache:
                self._cache_trade_data(metrics)
            
            logger.info(
                "Trade recorded",
                strategy=strategy,
                token_pair=token_pair,
                profit=profit,
                success=success
            )
            
        except Exception as e:
            logger.error("Error recording trade", error=str(e))
    
    def _cache_trade_data(self, metrics: TradeMetrics):
        """Cache trade data in Redis
        
        Args:
            metrics: Trade metrics to cache
        """
        try:
            # Create cache key
            timestamp = metrics.timestamp.strftime('%Y%m%d_%H%M%S')
            cache_key = f"trade:{timestamp}"
            
            # Cache trade data
            trade_data = {
                'timestamp': metrics.timestamp.isoformat(),
                'strategy': metrics.strategy,
                'token_pair': metrics.token_pair,
                'dex': metrics.dex,
                'profit': metrics.profit,
                'gas_price': metrics.gas_price,
                'execution_time': metrics.execution_time,
                'success': metrics.success
            }
            
            self.cache.setex(
                cache_key,
                timedelta(hours=24),  # Cache for 24 hours
                str(trade_data)
            )
            
        except Exception as e:
            logger.error("Error caching trade data", error=str(e))
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics
        
        Returns:
            Dictionary containing system metrics
        """
        return {
            'system_health': self.system_monitor.get_system_health(),
            'trade_performance': self.trade_monitor.get_performance_summary()
        }
    
    def get_trade_analytics(
        self,
        timeframe: str = '24h',
        strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get trade analytics
        
        Args:
            timeframe: Timeframe for analysis
            strategy: Optional strategy to analyze
            
        Returns:
            Dictionary containing trade analytics
        """
        return self.trade_monitor.analyze_performance(
            timeframe=timeframe,
            strategy=strategy
        )
    
    def get_learning_features(
        self,
        lookback_period: str = '7d'
    ) -> pd.DataFrame:
        """Get features for AI learning
        
        Args:
            lookback_period: Period to look back for features
            
        Returns:
            DataFrame with features for AI learning
        """
        return self.trade_monitor.get_learning_features(
            lookback_period=lookback_period
        )
    
    def export_metrics(self, export_path: Optional[str] = None) -> None:
        """Export all metrics to file
        
        Args:
            export_path: Optional path to export metrics to
        """
        try:
            if not export_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                export_path = str(self.storage_path / f"metrics_export_{timestamp}")
            
            # Export trade metrics
            self.trade_monitor.export_metrics(
                f"{export_path}_trades.json"
            )
            
            # Export system metrics
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'system_health': self.system_monitor.get_system_health(),
                'trade_analytics': self.get_trade_analytics(),
                'learning_features': self.get_learning_features().to_dict()
            }
            
            Path(f"{export_path}_system.json").write_text(
                str(metrics_data)
            )
            
            logger.info(f"Exported metrics to {export_path}")
            
        except Exception as e:
            logger.error("Error exporting metrics", error=str(e)) 