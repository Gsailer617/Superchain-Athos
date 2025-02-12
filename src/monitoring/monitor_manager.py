from typing import Dict, Any, Optional, List, Tuple
import structlog
from pathlib import Path
import asyncio
from datetime import datetime
import psutil
import redis
from prometheus_client import start_http_server
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.aiohttp import AioHttpClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
import sentry_sdk
from sentry_sdk.integrations.redis import RedisIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from .performance_monitor import PerformanceMonitor
from src.history.trade_history import TradeHistoryManager, TradeMetrics
from visualization.learning_insights import LearningInsightsVisualizer
import logging
import time
from dataclasses import dataclass
import prometheus_client as prom
from prometheus_client import Counter, Gauge, Histogram, Summary
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
import aiohttp
import asyncio
from functools import wraps
import socket
import json

logger = structlog.get_logger(__name__)

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    prometheus_port: int = 9090
    grafana_port: int = 3000
    jaeger_host: str = "localhost"
    jaeger_port: int = 6831
    log_level: str = "INFO"
    metrics_interval: int = 15  # seconds
    enable_tracing: bool = True
    enable_metrics: bool = True
    enable_logging: bool = True

class MonitorManager:
    """Central manager for all monitoring components with enhanced learning capabilities"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        storage_path: str = "data/monitoring",
        prometheus_port: int = 8000,
        cache_enabled: bool = True
    ):
        """Initialize monitoring manager with enhanced learning capabilities
        
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
        self.metrics = MetricsManager()
        
        # Initialize components
        self._setup_monitoring()
        
        # Initialize tracing
        self._setup_tracing()
        
        # Initialize error reporting
        self._setup_error_reporting()
        
        # Initialize components with tighter integration
        self.performance_monitor = PerformanceMonitor(
            port=prometheus_port,
            metrics_logger=self._handle_performance_metrics
        )
        
        self.trade_history = TradeHistoryManager(
            storage_path=str(self.storage_path / "trade_history"),
            max_memory_entries=config.get('max_memory_entries', 10000),
            flush_interval=config.get('flush_interval', 100)
        )
        
        self.learning_viz = LearningInsightsVisualizer(self.trade_history)
        
        # Initialize anomaly detection
        self.anomaly_detector = None
        self._initialize_anomaly_detection()
        
        # Initialize Redis cache if enabled
        self.cache = None
        if cache_enabled:
            try:
                self.cache = redis.Redis(
                    host=config.get('redis_host', 'localhost'),
                    port=config.get('redis_port', 6379),
                    db=config.get('redis_db', 0)
                )
                # Instrument Redis for tracing
                RedisInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())
                logger.info("Redis cache initialized")
            except Exception as e:
                logger.error("Failed to initialize Redis cache", error=str(e))
        
        # Background tasks
        self._tasks = []
        self._running = False
        
        # Learning state
        self.learning_state = {
            'strategy_performance': {},
            'anomaly_scores': [],
            'feature_importance': {},
            'optimization_suggestions': []
        }
        
        # Instrument HTTP client for tracing
        AioHttpClientInstrumentor().instrument()

    def _setup_tracing(self):
        """Setup OpenTelemetry tracing"""
        try:
            # Create and set tracer provider
            trace.set_tracer_provider(TracerProvider())
            
            # Configure Jaeger exporter
            jaeger_exporter = JaegerExporter(
                agent_host_name=self.config.get('jaeger', {}).get('host', 'localhost'),
                agent_port=self.config.get('jaeger', {}).get('port', 6831)
            )
            
            # Add span processor
            trace.get_tracer_provider().add_span_processor(
                BatchSpanProcessor(jaeger_exporter)
            )
            
            logger.info("Tracing initialized with Jaeger")
            
        except Exception as e:
            logger.error(f"Error setting up tracing: {str(e)}")

    def _setup_error_reporting(self):
        """Setup Sentry error reporting"""
        try:
            sentry_dsn = self.config.get('sentry', {}).get('dsn')
            if sentry_dsn:
                sentry_sdk.init(
                    dsn=sentry_dsn,
                    traces_sample_rate=0.1,
                    integrations=[
                        RedisIntegration(),
                        LoggingIntegration(
                            level=None,  # Capture all logs
                            event_level=logging.ERROR  # Send errors to Sentry
                        )
                    ]
                )
                logger.info("Sentry error reporting initialized")
            
        except Exception as e:
            logger.error(f"Error setting up error reporting: {str(e)}")

    def _initialize_anomaly_detection(self):
        """Initialize anomaly detection model"""
        try:
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42
            )
        except Exception as e:
            logger.error("Failed to initialize anomaly detection", error=str(e))

    async def _handle_performance_metrics(self, metrics: Dict[str, Any]):
        """Handle performance metrics with enhanced tracing"""
        try:
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span("handle_performance_metrics") as span:
                span.set_attribute("metrics_count", len(metrics))
                
                # Update learning state
                if metrics.get('strategy') in self.learning_state['strategy_performance']:
                    self.learning_state['strategy_performance'][metrics['strategy']].append({
                        'timestamp': metrics['timestamp'],
                        'profit': metrics.get('profit', 0),
                        'gas_cost': metrics.get('gas_cost', 0),
                        'execution_time': metrics.get('execution_time', 0)
                    })
                else:
                    self.learning_state['strategy_performance'][metrics['strategy']] = [{
                        'timestamp': metrics['timestamp'],
                        'profit': metrics.get('profit', 0),
                        'gas_cost': metrics.get('gas_cost', 0),
                        'execution_time': metrics.get('execution_time', 0)
                    }]
                
                # Detect anomalies if enough data
                with tracer.start_span("detect_anomalies") as anomaly_span:
                    await self._detect_anomalies(metrics)
                    
                # Generate optimization suggestions
                with tracer.start_span("generate_optimizations") as opt_span:
                    await self._generate_optimization_suggestions()
                
        except Exception as e:
            logger.error("Error handling performance metrics", error=str(e))
            sentry_sdk.capture_exception(e)

    async def _detect_anomalies(self, metrics: Dict[str, Any]):
        """Detect anomalies in performance metrics
        
        Args:
            metrics: Performance metrics dictionary
        """
        try:
            if len(self.learning_state['strategy_performance']) < 100:
                return
            
            # Prepare data for anomaly detection
            recent_data = []
            for strategy in self.learning_state['strategy_performance'].values():
                recent_data.extend(strategy[-100:])  # Last 100 points per strategy
            
            # Convert to numpy array
            X = np.array([[
                d['profit'],
                d['gas_cost'],
                d['execution_time']
            ] for d in recent_data])
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Detect anomalies
            scores = self.anomaly_detector.fit_predict(X_scaled)
            self.learning_state['anomaly_scores'] = scores.tolist()
            
            # Log anomalies
            if -1 in scores:  # Anomaly detected
                logger.warning(
                    "Anomaly detected in performance metrics",
                    anomaly_count=sum(scores == -1)
                )
                
        except Exception as e:
            logger.error("Error detecting anomalies", error=str(e))

    async def _generate_optimization_suggestions(self):
        """Generate optimization suggestions based on learning state"""
        try:
            suggestions = []
            
            # Analyze strategy performance
            for strategy, metrics in self.learning_state['strategy_performance'].items():
                if len(metrics) < 10:
                    continue
                
                recent_metrics = metrics[-10:]  # Last 10 trades
                avg_profit = sum(m['profit'] for m in recent_metrics) / len(recent_metrics)
                avg_gas = sum(m['gas_cost'] for m in recent_metrics) / len(recent_metrics)
                avg_time = sum(m['execution_time'] for m in recent_metrics) / len(recent_metrics)
                
                # Generate suggestions based on metrics
                if avg_profit < 0:
                    suggestions.append(f"Strategy '{strategy}' showing negative profit. Consider adjusting parameters.")
                if avg_gas > self.config.get('gas_threshold', 100):
                    suggestions.append(f"High gas costs in strategy '{strategy}'. Consider implementing gas optimization.")
                if avg_time > self.config.get('execution_threshold', 2):
                    suggestions.append(f"Slow execution in strategy '{strategy}'. Consider optimizing execution path.")
            
            self.learning_state['optimization_suggestions'] = suggestions
            
            if suggestions:
                logger.info("Generated optimization suggestions", suggestions=suggestions)
                
        except Exception as e:
            logger.error("Error generating optimization suggestions", error=str(e))

    async def get_learning_insights(self) -> Dict[str, Any]:
        """Get current learning insights with tracing"""
        try:
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span("get_learning_insights") as span:
                insights = {
                    'strategy_performance': self.learning_state['strategy_performance'],
                    'anomaly_scores': self.learning_state['anomaly_scores'],
                    'optimization_suggestions': self.learning_state['optimization_suggestions'],
                    'feature_importance': self.learning_state['feature_importance']
                }
                span.set_attribute("insights_count", len(insights))
                return insights
                
        except Exception as e:
            logger.error("Error getting learning insights", error=str(e))
            sentry_sdk.capture_exception(e)
            return {}

    def update_feature_importance(self, features: Dict[str, float]):
        """Update feature importance scores
        
        Args:
            features: Dictionary of feature names and their importance scores
        """
        try:
            self.learning_state['feature_importance'] = features
            logger.info("Updated feature importance scores", features=features)
        except Exception as e:
            logger.error("Error updating feature importance", error=str(e))

    async def start(self):
        """Start all monitoring components"""
        try:
            self._running = True
            
            # Start background tasks
            self._tasks = [
                asyncio.create_task(self._monitor_system_resources()),
                asyncio.create_task(self._periodic_analysis()),
                asyncio.create_task(self._cache_cleanup())
            ]
            
            logger.info("Monitor manager started")
            
        except Exception as e:
            logger.error("Error starting monitor manager", error=str(e))
            raise
    
    async def stop(self):
        """Stop all monitoring components"""
        try:
            self._running = False
            
            # Cancel background tasks
            for task in self._tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self._tasks, return_exceptions=True)
            
            # Stop components
            self.performance_monitor.stop()
            
            # Flush trade history
            self.trade_history.flush_to_disk()
            
            # Close Redis connection
            if self.cache:
                await self.cache.close()
            
            logger.info("Monitor manager stopped")
            
        except Exception as e:
            logger.error("Error stopping monitor manager", error=str(e))
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
            
            # Record with performance monitor
            self.performance_monitor.record_transaction(
                success=success,
                gas_price=gas_price,
                execution_time=execution_time,
                profit=profit,
                strategy=strategy,
                dex=dex,
                token_pair=token_pair
            )
            
            # Record in trade history
            self.trade_history.record_trade(metrics)
            
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
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics
        
        Returns:
            Dictionary of system metrics
        """
        try:
            process = psutil.Process()
            
            return {
                'cpu_percent': process.cpu_percent(),
                'memory_percent': process.memory_percent(),
                'memory_mb': process.memory_info().rss / (1024 * 1024),
                'threads': process.num_threads(),
                'open_files': len(process.open_files()),
                'connections': len(process.connections())
            }
            
        except Exception as e:
            logger.error("Error getting system metrics", error=str(e))
            return {}
    
    async def _monitor_system_resources(self, interval: int = 60):
        """Monitor system resources periodically
        
        Args:
            interval: Monitoring interval in seconds
        """
        while self._running:
            try:
                metrics = self.get_system_metrics()
                
                # Record with performance monitor
                self.performance_monitor.record_resource_usage(
                    component='arbitrage_agent',
                    memory_mb=metrics.get('memory_mb', 0),
                    cpu_percent=metrics.get('cpu_percent', 0)
                )
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error monitoring system resources", error=str(e))
                await asyncio.sleep(interval)
    
    async def _periodic_analysis(self, interval: int = 300):
        """Run periodic analysis of trading performance
        
        Args:
            interval: Analysis interval in seconds
        """
        while self._running:
            try:
                # Analyze recent performance
                metrics = self.trade_history.analyze_performance(timeframe='1h')
                
                # Cache analysis results
                if self.cache and metrics:
                    await self.cache.setex(
                        'recent_performance',
                        interval * 2,  # Cache for 2x interval
                        str(metrics)
                    )
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error running periodic analysis", error=str(e))
                await asyncio.sleep(interval)
    
    async def _cache_cleanup(self, interval: int = 3600):
        """Clean up expired cache entries
        
        Args:
            interval: Cleanup interval in seconds
        """
        while self._running and self.cache:
            try:
                # Implement cache cleanup logic
                # This is a placeholder for actual cache cleanup implementation
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error cleaning cache", error=str(e))
                await asyncio.sleep(interval)
    
    def _cache_trade_data(self, metrics: TradeMetrics):
        """Cache recent trade data
        
        Args:
            metrics: Trade metrics to cache
        """
        if not self.cache:
            return
            
        try:
            # Cache trade data with expiration
            key = f"trade:{metrics.timestamp.isoformat()}"
            value = {
                'strategy': metrics.strategy,
                'token_pair': metrics.token_pair,
                'dex': metrics.dex,
                'profit': metrics.profit,
                'success': metrics.success
            }
            
            self.cache.setex(
                key,
                self.config.get('trade_cache_ttl', 3600),  # 1 hour default
                str(value)
            )
            
        except Exception as e:
            logger.error("Error caching trade data", error=str(e))

    async def record_telegram_event(self, event_type: str, data: Dict[str, Any]):
        """Record Telegram bot events"""
        try:
            # Record event metrics
            self.metrics.increment('telegram_events_total', {'type': event_type})
            
            if event_type == 'command':
                self.metrics.increment('telegram_commands_total', {'command': data['command']})
            elif event_type == 'trade':
                self.metrics.observe('telegram_trade_profit', data['profit'])
                self.metrics.observe('telegram_trade_volume', data['volume'])
            elif event_type == 'error':
                self.metrics.increment('telegram_errors_total', {'error': data['error']})
            
            # Store event data
            await self._store_event_data('telegram', event_type, data)
            
        except Exception as e:
            logger.error(f"Error recording Telegram event: {str(e)}")
    
    async def record_visualization_event(self, event_type: str, data: Dict[str, Any]):
        """Record visualization system events"""
        try:
            # Record event metrics
            self.metrics.increment('visualization_events_total', {'type': event_type})
            
            if event_type == 'chart_update':
                self.metrics.observe('chart_update_duration_seconds', data['duration'])
            elif event_type == 'user_interaction':
                self.metrics.increment('user_interactions_total', {'action': data['action']})
            elif event_type == 'error':
                self.metrics.increment('visualization_errors_total', {'error': data['error']})
            
            # Store event data
            await self._store_event_data('visualization', event_type, data)
            
        except Exception as e:
            logger.error(f"Error recording visualization event: {str(e)}")
    
    async def get_telegram_metrics(self) -> Dict[str, Any]:
        """Get Telegram bot metrics"""
        try:
            metrics = {
                'commands': {
                    'total': self.metrics.get_counter('telegram_commands_total'),
                    'by_type': self.metrics.get_counter_by_label('telegram_commands_total', 'command')
                },
                'trades': {
                    'profit': self.metrics.get_gauge('telegram_trade_profit'),
                    'volume': self.metrics.get_gauge('telegram_trade_volume')
                },
                'errors': {
                    'total': self.metrics.get_counter('telegram_errors_total'),
                    'by_type': self.metrics.get_counter_by_label('telegram_errors_total', 'error')
                }
            }
            return metrics
        except Exception as e:
            logger.error(f"Error getting Telegram metrics: {str(e)}")
            return {}
    
    async def get_visualization_metrics(self) -> Dict[str, Any]:
        """Get visualization system metrics"""
        try:
            metrics = {
                'chart_updates': {
                    'duration': self.metrics.get_histogram('chart_update_duration_seconds'),
                    'count': self.metrics.get_counter('visualization_events_total')
                },
                'user_interactions': {
                    'total': self.metrics.get_counter('user_interactions_total'),
                    'by_action': self.metrics.get_counter_by_label('user_interactions_total', 'action')
                },
                'errors': {
                    'total': self.metrics.get_counter('visualization_errors_total'),
                    'by_type': self.metrics.get_counter_by_label('visualization_errors_total', 'error')
                }
            }
            return metrics
        except Exception as e:
            logger.error(f"Error getting visualization metrics: {str(e)}")
            return {}
    
    async def _store_event_data(self, source: str, event_type: str, data: Dict[str, Any]):
        """Store event data for analysis"""
        if not self.cache_enabled:
            return
            
        try:
            # Create event document
            event = {
                'timestamp': time.time(),
                'source': source,
                'type': event_type,
                'data': data
            }
            
            # Store in appropriate collection
            collection = f"{source}_events"
            await self._store_document(collection, event)
            
        except Exception as e:
            logger.error(f"Error storing event data: {str(e)}")

class MonitoringManager:
    """Centralized monitoring and observability manager"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self._setup_monitoring()
    
    def _setup_monitoring(self):
        """Initialize monitoring components"""
        if self.config.enable_metrics:
            self._setup_metrics()
        
        if self.config.enable_tracing:
            self._setup_tracing()
        
        if self.config.enable_logging:
            self._setup_logging()
        
        # Initialize system metrics
        self._setup_system_metrics()
        
        # Start metrics collection
        self._start_metrics_collection()
    
    def _setup_metrics(self):
        """Setup Prometheus metrics"""
        # Performance metrics
        self.execution_time = Histogram(
            'execution_time_seconds',
            'Time spent executing operations',
            ['operation']
        )
        
        self.operation_counter = Counter(
            'operations_total',
            'Number of operations performed',
            ['operation', 'status']
        )
        
        self.error_counter = Counter(
            'errors_total',
            'Number of errors encountered',
            ['type', 'component']
        )
        
        # Business metrics
        self.active_trades = Gauge(
            'active_trades',
            'Number of active trades'
        )
        
        self.trade_volume = Counter(
            'trade_volume_total',
            'Total trading volume',
            ['token']
        )
        
        self.profit_loss = Gauge(
            'profit_loss',
            'Current profit/loss'
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage'
        )
        
        self.memory_usage = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes'
        )
        
        self.network_io = Counter(
            'network_io_bytes',
            'Network I/O in bytes',
            ['direction']
        )
    
    def _setup_tracing(self):
        """Setup distributed tracing"""
        # Configure Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name=self.config.jaeger_host,
            agent_port=self.config.jaeger_port
        )
        
        # Set up trace provider
        provider = TracerProvider()
        processor = BatchSpanProcessor(jaeger_exporter)
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        
        # Get tracer
        self.tracer = trace.get_tracer(__name__)
        
        # Instrument libraries
        AioHttpClientInstrumentor().instrument()
        RedisInstrumentor().instrument()
        SQLAlchemyInstrumentor().instrument()
    
    def _setup_logging(self):
        """Setup centralized logging"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _setup_system_metrics(self):
        """Setup system resource monitoring"""
        self.system_metrics = {
            'cpu_percent': Gauge(
                'system_cpu_percent',
                'System CPU usage percentage'
            ),
            'memory_percent': Gauge(
                'system_memory_percent',
                'System memory usage percentage'
            ),
            'disk_usage': Gauge(
                'system_disk_usage_percent',
                'System disk usage percentage',
                ['mount_point']
            ),
            'network_io_counters': Counter(
                'system_network_io_bytes',
                'System network I/O in bytes',
                ['interface', 'direction']
            )
        }
    
    def _start_metrics_collection(self):
        """Start periodic metrics collection"""
        async def collect_metrics():
            while True:
                self._collect_system_metrics()
                await asyncio.sleep(self.config.metrics_interval)
        
        asyncio.create_task(collect_metrics())
    
    def _collect_system_metrics(self):
        """Collect system resource metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_metrics['cpu_percent'].set(cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.system_metrics['memory_percent'].set(memory.percent)
            
            # Disk metrics
            for partition in psutil.disk_partitions():
                usage = psutil.disk_usage(partition.mountpoint)
                self.system_metrics['disk_usage'].labels(
                    mount_point=partition.mountpoint
                ).set(usage.percent)
            
            # Network metrics
            net_io = psutil.net_io_counters()
            for interface, stats in psutil.net_if_stats().items():
                if stats.isup:
                    self.system_metrics['network_io_counters'].labels(
                        interface=interface,
                        direction='bytes_sent'
                    ).inc(net_io.bytes_sent)
                    self.system_metrics['network_io_counters'].labels(
                        interface=interface,
                        direction='bytes_recv'
                    ).inc(net_io.bytes_recv)
        
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
    
    def trace_function(self, name: Optional[str] = None):
        """Decorator for function tracing"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                operation_name = name or func.__name__
                
                with self.tracer.start_as_current_span(operation_name) as span:
                    start_time = time.time()
                    try:
                        result = await func(*args, **kwargs)
                        duration = time.time() - start_time
                        
                        # Record metrics
                        self.execution_time.labels(
                            operation=operation_name
                        ).observe(duration)
                        
                        self.operation_counter.labels(
                            operation=operation_name,
                            status='success'
                        ).inc()
                        
                        # Add span attributes
                        span.set_attribute('duration', duration)
                        span.set_status(Status(StatusCode.OK))
                        
                        return result
                        
                    except Exception as e:
                        # Record error metrics
                        self.error_counter.labels(
                            type=type(e).__name__,
                            component=operation_name
                        ).inc()
                        
                        # Add error to span
                        span.set_status(
                            Status(StatusCode.ERROR, str(e))
                        )
                        span.record_exception(e)
                        
                        raise
            
            return wrapper
        return decorator
    
    def monitor_operation(self, name: str):
        """Context manager for operation monitoring"""
        return OperationMonitor(self, name)
    
    async def record_trade(
        self,
        token: str,
        amount: float,
        profit: float
    ):
        """Record trade metrics"""
        self.active_trades.inc()
        self.trade_volume.labels(token=token).inc(amount)
        self.profit_loss.inc(profit)
        
        with self.tracer.start_span('trade_execution') as span:
            span.set_attributes({
                'token': token,
                'amount': amount,
                'profit': profit
            })
    
    async def export_metrics(self) -> Dict[str, Any]:
        """Export current metrics"""
        metrics = {}
        
        # Collect from Prometheus
        for metric in prom.REGISTRY.collect():
            for sample in metric.samples:
                metrics[sample.name] = {
                    'value': sample.value,
                    'labels': sample.labels
                }
        
        return metrics
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': {
                p.mountpoint: psutil.disk_usage(p.mountpoint).percent
                for p in psutil.disk_partitions()
            },
            'network_connections': len(psutil.net_connections()),
            'open_files': len(psutil.Process().open_files())
        }
    
    def start_prometheus_server(self):
        """Start Prometheus metrics server"""
        prom.start_http_server(self.config.prometheus_port)
    
    async def setup_grafana_dashboard(self):
        """Setup Grafana dashboard"""
        dashboard_config = self._load_dashboard_config()
        
        async with aiohttp.ClientSession() as session:
            # Create data source
            await self._create_datasource(session)
            
            # Create dashboard
            await self._create_dashboard(session, dashboard_config)
    
    def _load_dashboard_config(self) -> Dict[str, Any]:
        """Load Grafana dashboard configuration"""
        dashboard_path = Path(__file__).parent / 'dashboards/main.json'
        with open(dashboard_path) as f:
            return json.load(f)
    
    async def _create_datasource(self, session: aiohttp.ClientSession):
        """Create Prometheus data source in Grafana"""
        datasource = {
            'name': 'Prometheus',
            'type': 'prometheus',
            'url': f'http://localhost:{self.config.prometheus_port}',
            'access': 'proxy',
            'isDefault': True
        }
        
        async with session.post(
            f'http://localhost:{self.config.grafana_port}/api/datasources',
            json=datasource,
            headers={'Content-Type': 'application/json'}
        ) as response:
            if response.status not in (200, 409):  # 409 means already exists
                raise Exception(
                    f"Failed to create datasource: {await response.text()}"
                )
    
    async def _create_dashboard(
        self,
        session: aiohttp.ClientSession,
        dashboard_config: Dict[str, Any]
    ):
        """Create Grafana dashboard"""
        async with session.post(
            f'http://localhost:{self.config.grafana_port}/api/dashboards/db',
            json={'dashboard': dashboard_config, 'overwrite': True},
            headers={'Content-Type': 'application/json'}
        ) as response:
            if response.status != 200:
                raise Exception(
                    f"Failed to create dashboard: {await response.text()}"
                )

class OperationMonitor:
    """Context manager for operation monitoring"""
    
    def __init__(self, manager: MonitoringManager, operation_name: str):
        self.manager = manager
        self.operation_name = operation_name
        self.start_time = None
        self.span = None
    
    async def __aenter__(self):
        self.start_time = time.time()
        self.span = self.manager.tracer.start_span(self.operation_name)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        # Record metrics
        self.manager.execution_time.labels(
            operation=self.operation_name
        ).observe(duration)
        
        if exc_type is None:
            self.manager.operation_counter.labels(
                operation=self.operation_name,
                status='success'
            ).inc()
            self.span.set_status(Status(StatusCode.OK))
        else:
            self.manager.error_counter.labels(
                type=exc_type.__name__,
                component=self.operation_name
            ).inc()
            self.span.set_status(
                Status(StatusCode.ERROR, str(exc_val))
            )
            self.span.record_exception(exc_val)
        
        self.span.end() 