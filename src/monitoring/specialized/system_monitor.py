"""
Enhanced system monitoring that consolidates all resource monitoring functionality.
"""

from typing import Dict, List, Optional, Any, Set, Tuple, cast
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import structlog
import psutil
import asyncio
import numpy as np
from pathlib import Path
import aiohttp
from ..core.base_monitor import BaseMonitor
from ..core.metrics_manager import metrics_manager, MetricType, MetricConfig
from scipy import stats
import numpy.typing as npt
from ...utils.unified_metrics import UnifiedMetricsSystem

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
    TREND_ANALYSIS_ENABLED = True
except ImportError:
    logger.warning("statsmodels not installed - trend analysis disabled")
    ExponentialSmoothing = None
    seasonal_decompose = None
    TREND_ANALYSIS_ENABLED = False

logger = structlog.get_logger(__name__)

class ResourceType(str, Enum):
    """Resource types for monitoring"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    IO = "io"
    GPU = "gpu"  # For future expansion

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ResourceThreshold:
    """Resource threshold configuration"""
    warning: float
    error: float
    critical: float
    duration: int = 60  # Duration in seconds
    cooldown: int = 300  # Cooldown period between alerts

@dataclass
class ResourceAlert:
    """Alert for resource issues"""
    timestamp: datetime
    resource_type: ResourceType
    severity: AlertSeverity
    value: float
    threshold: float
    message: str
    metadata: Dict[str, Any]

class SystemMonitor(BaseMonitor):
    """Enhanced system monitoring system"""
    
    def __init__(
        self,
        storage_path: str = "data/monitoring/system",
        alert_webhook_url: Optional[str] = None,
        collection_interval: int = 10,
        enable_prometheus: bool = True,
        resource_thresholds: Optional[Dict[ResourceType, ResourceThreshold]] = None,
        metrics_system: Optional[UnifiedMetricsSystem] = None
    ):
        """Initialize system monitor
        
        Args:
            storage_path: Path for storing monitoring data
            alert_webhook_url: Optional webhook URL for alerts
            collection_interval: Metric collection interval in seconds
            enable_prometheus: Whether to enable Prometheus metrics
            resource_thresholds: Custom resource thresholds
            metrics_system: Optional unified metrics system
        """
        super().__init__(
            storage_path=storage_path,
            enable_prometheus=enable_prometheus
        )
        
        self.alert_webhook_url = alert_webhook_url
        self.collection_interval = collection_interval
        
        # Resource monitoring state
        self._thresholds = resource_thresholds or {
            ResourceType.CPU: ResourceThreshold(
                warning=70.0,
                error=85.0,
                critical=95.0
            ),
            ResourceType.MEMORY: ResourceThreshold(
                warning=75.0,
                error=85.0,
                critical=95.0
            ),
            ResourceType.DISK: ResourceThreshold(
                warning=80.0,
                error=90.0,
                critical=95.0
            )
        }
        
        # Historical data storage
        self._resource_history: Dict[ResourceType, List[Dict[str, Any]]] = {
            rt: [] for rt in ResourceType
        }
        
        # Alert tracking
        self._active_alerts: Set[str] = set()
        self._alert_history: List[ResourceAlert] = []
        self._alert_cooldowns: Dict[str, datetime] = {}
        
        # Initialize metrics
        self._setup_resource_metrics()
        
        # Start monitoring tasks
        self._monitoring_tasks: List[asyncio.Task] = []
        self._is_running = False
        
        # Risk thresholds
        self.risk_thresholds = {
            'price_impact': 0.02,  # 2% price impact
            'slippage': 0.01,      # 1% slippage
            'liquidity': 100000,   # $100k minimum liquidity
            'volume': 50000,       # $50k minimum 24h volume
            'volatility': 0.8,     # 80% volatility threshold
            'gas_price': 200,      # 200 gwei
            'network_load': 0.8,   # 80% network load
            'error_rate': 0.05     # 5% error rate
        }
        
        # Initialize trend metrics
        self.trend_strength = metrics_manager.gauge(
            'metric_trend_strength',
            'Strength of trend in metric (positive or negative)',
            ['metric_name']
        )
        
        self.seasonality_strength = metrics_manager.gauge(
            'metric_seasonality_strength', 
            'Strength of seasonal patterns in metric',
            ['metric_name']
        )
        
        self.forecast_value = metrics_manager.gauge(
            'metric_forecast_value',
            'Forecasted value for metric',
            ['metric_name', 'horizon']
        )
        
        self.forecast_error = metrics_manager.gauge(
            'metric_forecast_error',
            'Mean absolute percentage error of forecasts',
            ['metric_name']
        )
        
        # Use provided metrics system or create new one
        self.metrics = metrics_system or UnifiedMetricsSystem(
            enable_prometheus=enable_prometheus
        )
        
        # Initialize monitoring
        self._initialize_monitoring()
        
        # Background task
        self._task: Optional[asyncio.Task] = None
    
    def _setup_resource_metrics(self):
        """Setup resource-specific metrics"""
        # Resource usage metrics
        self._metrics['resource_usage'] = metrics_manager.create_metric(
            MetricConfig(
                name="system_resource_usage",
                description="Resource usage percentage",
                type=MetricType.GAUGE,
                labels=['resource_type', 'component']
            )
        )
        
        # Resource saturation metrics
        self._metrics['resource_saturation'] = metrics_manager.create_metric(
            MetricConfig(
                name="system_resource_saturation",
                description="Resource saturation level",
                type=MetricType.GAUGE,
                labels=['resource_type']
            )
        )
        
        # Alert metrics
        self._metrics['alerts'] = metrics_manager.create_metric(
            MetricConfig(
                name="system_alerts_total",
                description="Number of alerts generated",
                type=MetricType.COUNTER,
                labels=['resource_type', 'severity']
            )
        )
        
        # IO metrics
        self._metrics['io_operations'] = metrics_manager.create_metric(
            MetricConfig(
                name="system_io_operations",
                description="IO operations count",
                type=MetricType.COUNTER,
                labels=['operation_type', 'device']
            )
        )
    
    def _initialize_monitoring(self):
        """Initialize system monitoring"""
        # Set default thresholds if not provided
        if not self._thresholds:
            self._thresholds = {
                ResourceType.CPU: ResourceThreshold(warning=80.0, error=90.0, critical=95.0),
                ResourceType.MEMORY: ResourceThreshold(warning=80.0, error=90.0, critical=95.0),
                ResourceType.DISK: ResourceThreshold(warning=85.0, error=95.0, critical=95.0),
                ResourceType.NETWORK: ResourceThreshold(warning=5.0, critical=10.0)
            }
            
    async def start_monitoring(self):
        """Start system monitoring"""
        if self._is_running:
            return
        
        self._is_running = True
        self._monitoring_tasks = [
            asyncio.create_task(self._collect_metrics()),
            asyncio.create_task(self._analyze_trends()),
            asyncio.create_task(self._cleanup_old_data())
        ]
        
        logger.info("Started system monitoring")
    
    async def stop_monitoring(self):
        """Stop system monitoring"""
        self._is_running = False
        for task in self._monitoring_tasks:
            task.cancel()
        
        await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
        logger.info("Stopped system monitoring")
    
    async def _collect_metrics(self):
        """Collect system metrics periodically"""
        while self._is_running:
            try:
                # Collect CPU metrics
                cpu_metrics = await self._collect_cpu_metrics()
                await self._process_metrics(ResourceType.CPU, cpu_metrics)
                
                # Collect memory metrics
                memory_metrics = await self._collect_memory_metrics()
                await self._process_metrics(ResourceType.MEMORY, memory_metrics)
                
                # Collect disk metrics
                disk_metrics = await self._collect_disk_metrics()
                await self._process_metrics(ResourceType.DISK, disk_metrics)
                
                # Collect network metrics
                network_metrics = await self._collect_network_metrics()
                await self._process_metrics(ResourceType.NETWORK, network_metrics)
                
                # Collect IO metrics
                io_metrics = await self._collect_io_metrics()
                await self._process_metrics(ResourceType.IO, io_metrics)
                
            except Exception as e:
                logger.error("Error collecting system metrics", error=str(e))
            
            await asyncio.sleep(self.collection_interval)
    
    async def _collect_cpu_metrics(self) -> Dict[str, float]:
        """Collect CPU metrics"""
        cpu_times = psutil.cpu_times_percent()
        cpu_stats = psutil.cpu_stats()
        cpu_freq = psutil.cpu_freq()
        
        metrics = {
            'usage': psutil.cpu_percent(),
            'user': cpu_times.user,
            'system': cpu_times.system,
            'idle': cpu_times.idle,
            'iowait': getattr(cpu_times, 'iowait', 0),
            'interrupts': cpu_stats.interrupts,
            'ctx_switches': cpu_stats.ctx_switches,
            'frequency': cpu_freq.current if cpu_freq else 0
        }
        
        # Per-CPU metrics
        per_cpu = psutil.cpu_percent(percpu=True)
        for i, usage in enumerate(per_cpu):
            metrics[f'cpu{i}'] = usage
        
        return metrics
    
    async def _collect_memory_metrics(self) -> Dict[str, float]:
        """Collect memory metrics"""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'total': mem.total,
            'available': mem.available,
            'used': mem.used,
            'free': mem.free,
            'usage': mem.percent,
            'swap_total': swap.total,
            'swap_used': swap.used,
            'swap_free': swap.free,
            'swap_usage': swap.percent
        }
    
    async def _collect_disk_metrics(self) -> Dict[str, float]:
        """Collect disk metrics"""
        metrics = {}
        
        # Disk usage
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                metrics[f'{partition.mountpoint}_total'] = usage.total
                metrics[f'{partition.mountpoint}_used'] = usage.used
                metrics[f'{partition.mountpoint}_free'] = usage.free
                metrics[f'{partition.mountpoint}_usage'] = usage.percent
            except Exception:
                continue
        
        # Disk IO
        io_counters = psutil.disk_io_counters()
        if io_counters:
            metrics.update({
                'read_bytes': io_counters.read_bytes,
                'write_bytes': io_counters.write_bytes,
                'read_time': io_counters.read_time,
                'write_time': io_counters.write_time,
                'read_count': io_counters.read_count,
                'write_count': io_counters.write_count
            })
        
        return metrics
    
    async def _collect_network_metrics(self) -> Dict[str, float]:
        """Collect network metrics"""
        metrics = {}
        
        # Network IO
        net_io = psutil.net_io_counters()
        metrics.update({
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv,
            'errin': net_io.errin,
            'errout': net_io.errout,
            'dropin': net_io.dropin,
            'dropout': net_io.dropout
        })
        
        # Per-interface metrics
        net_if_stats = psutil.net_if_stats()
        for interface, stats in net_if_stats.items():
            metrics[f'{interface}_speed'] = stats.speed
            metrics[f'{interface}_mtu'] = stats.mtu
            metrics[f'{interface}_up'] = int(stats.isup)
        
        return metrics
    
    async def _collect_io_metrics(self) -> Dict[str, float]:
        """Collect IO metrics"""
        metrics = {}
        
        try:
            # System-wide IO counters
            io_counters = psutil.disk_io_counters()
            if io_counters:
                metrics.update({
                    'system_read_bytes': float(io_counters.read_bytes),
                    'system_write_bytes': float(io_counters.write_bytes),
                    'system_read_count': float(io_counters.read_count),
                    'system_write_count': float(io_counters.write_count)
                })
            
            # Process-specific metrics
            process = psutil.Process()
            with process.oneshot():
                metrics.update({
                    'proc_cpu_percent': float(process.cpu_percent()),
                    'proc_memory_percent': float(process.memory_percent()),
                    'proc_num_threads': float(process.num_threads()),
                    'proc_num_fds': float(process.num_fds())
                })
                
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired) as e:
            logger.warning("Could not collect process metrics", error=str(e))
        
        return metrics
    
    async def _process_metrics(self, resource_type: ResourceType, metrics: Dict[str, float]):
        """Process collected metrics"""
        try:
            # Record metrics
            for metric_name, value in metrics.items():
                self._metrics['resource_usage'].labels(
                    resource_type=resource_type.value,
                    component=metric_name
                ).set(value)
                
            # Store historical data
            self._resource_history[resource_type].append({
                'timestamp': datetime.now(),
                'metrics': metrics
            })
            if len(self._resource_history[resource_type]) > 1000:  # Keep last 1000 points
                self._resource_history[resource_type].pop(0)
            
            # Check thresholds
            await self._evaluate_thresholds(resource_type, metrics['usage'], metrics)
            
            # Detect anomalies
            anomaly_score = await self._detect_anomalies(resource_type, metrics['usage'])
            if anomaly_score is not None:
                self._metrics['resource_saturation'].labels(
                    resource_type=resource_type.value
                ).set(anomaly_score)
                
        except Exception as e:
            logger.error("Error processing metrics", error=str(e))

    async def _evaluate_thresholds(self,
                                resource_type: ResourceType,
                                value: float,
                                metadata: Optional[Dict[str, Any]] = None):
        """Evaluate resource usage against thresholds"""
        if resource_type not in self._thresholds:
            return
            
        threshold = self._thresholds[resource_type]
        
        # Check cooldown period
        alert_key = f"{resource_type.value}"
        if alert_key in self._alert_cooldowns:
            if datetime.now() - self._alert_cooldowns[alert_key] < timedelta(seconds=threshold.cooldown):
                return
        
        if value >= threshold.critical:
            await self._send_alert(
                resource_type,
                value,
                threshold.critical,
                AlertSeverity.CRITICAL,
                metadata or {}
            )
        elif value >= threshold.error:
            await self._send_alert(
                resource_type,
                value,
                threshold.error,
                AlertSeverity.ERROR,
                metadata or {}
            )
        elif value >= threshold.warning:
            await self._send_alert(
                resource_type,
                value,
                threshold.warning,
                AlertSeverity.WARNING,
                metadata or {}
            )

    async def _send_alert(self,
                        resource_type: ResourceType,
                        value: float,
                        threshold: float,
                        severity: AlertSeverity,
                        metadata: Dict[str, Any]):
        """Generate and send a resource alert"""
        alert = ResourceAlert(
            timestamp=datetime.now(),
            resource_type=resource_type,
            severity=severity,
            value=value,
            threshold=threshold,
            message=f"{resource_type.value} usage at {value:.1f}% (threshold: {threshold:.1f}%)",
            metadata=metadata
        )
        
        # Track alert in metrics
        self._metrics['alerts'].labels(
            resource_type=resource_type.value,
            severity=severity.value
        ).inc()
        
        # Store alert history
        self._alert_history.append(alert)
        if len(self._alert_history) > 1000:  # Keep last 1000 alerts
            self._alert_history.pop(0)
        
        # Update cooldown
        alert_key = f"{resource_type.value}"
        self._alert_cooldowns[alert_key] = datetime.now()
        
        # Send alert if webhook configured
        if self.alert_webhook_url:
            try:
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        self.alert_webhook_url,
                        json={
                            'timestamp': alert.timestamp.isoformat(),
                            'resource_type': alert.resource_type.value,
                            'severity': alert.severity.value,
                            'value': alert.value,
                            'threshold': alert.threshold,
                            'message': alert.message,
                            'metadata': alert.metadata
                        }
                    )
            except Exception as e:
                logger.error("Failed to send alert", error=str(e))
    
    async def _analyze_trends(self):
        """Analyze resource usage trends"""
        while self._is_running:
            try:
                for resource_type in ResourceType:
                    history = self._resource_history[resource_type]
                    if len(history) < 2:
                        continue
                    
                    # Calculate trend using last hour of data
                    recent = history[-360:]  # Last hour (assuming 10s interval)
                    values = [entry['metrics'].get('usage', 0) for entry in recent]
                    
                    trend = np.polyfit(range(len(values)), values, 1)[0]
                    
                    # Update trend metric
                    self._metrics['resource_usage'].labels(
                        resource_type=resource_type.value,
                        component='trend'
                    ).set(trend)
                    
            except Exception as e:
                logger.error("Error analyzing trends", error=str(e))
            
            await asyncio.sleep(60)  # Analyze trends every minute
    
    async def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        while self._is_running:
            try:
                now = datetime.now()
                
                # Keep last 24 hours of data
                cutoff = now - timedelta(hours=24)
                
                for resource_type in ResourceType:
                    self._resource_history[resource_type] = [
                        entry for entry in self._resource_history[resource_type]
                        if entry['timestamp'] > cutoff
                    ]
                
                # Clear old alerts
                self._alert_history = [
                    alert for alert in self._alert_history
                    if alert.timestamp > cutoff
                ]
                
                # Clear expired active alerts
                for alert_key in list(self._active_alerts):
                    resource_type = alert_key.split('_')[0]
                    if resource_type in self._thresholds:
                        threshold = self._thresholds[ResourceType(resource_type)]
                        if (now - self._alert_history[-1].timestamp).total_seconds() > threshold.cooldown:
                            self._active_alerts.remove(alert_key)
                
            except Exception as e:
                logger.error("Error cleaning up old data", error=str(e))
            
            await asyncio.sleep(300)  # Clean up every 5 minutes 

    def configure_threshold(self, 
                          resource_type: ResourceType,
                          threshold: ResourceThreshold):
        """Configure resource thresholds"""
        self._thresholds[resource_type] = threshold

    async def _detect_anomalies(self, resource_type: ResourceType, value: float) -> Optional[float]:
        """Detect anomalies in resource usage using statistical methods"""
        history = self._resource_history[resource_type]
        if len(history) < 30:  # Need enough data points
            return None
            
        values = [entry['metrics'].get('usage', 0) for entry in history[-30:]]
        mean = float(np.mean(values))
        std = float(np.std(values))
        z_score = (value - mean) / std if std > 0 else 0.0
        
        return float(abs(z_score)) if abs(z_score) > 3 else None  # 3 sigma rule

    async def analyze_trend(
        self,
        metric_name: str,
        values: List[float],
        timestamps: List[datetime]
    ) -> Dict[str, float]:
        """
        Analyze trend components using time series decomposition.
        """
        if not TREND_ANALYSIS_ENABLED:
            self.logger.warning("Trend analysis disabled - statsmodels not installed")
            return {}
            
        if len(values) < 2:
            return {}
            
        try:
            # Convert to numpy array and ensure float type
            values_array = np.array(values, dtype=np.float64)
            
            # Perform seasonal decomposition
            result = seasonal_decompose(
                values_array,
                period=min(len(values) // 2, 24),  # Assume daily seasonality if enough data
                extrapolate_trend='freq'
            )
            
            # Calculate trend strength
            trend_strength = float(1 - np.var(result.resid) / np.var(values_array - result.seasonal))
            self.trend_strength.labels(metric_name=metric_name).set(trend_strength)
            
            # Calculate seasonality strength
            seasonality_strength = float(1 - np.var(result.resid) / np.var(values_array - result.trend))
            self.seasonality_strength.labels(metric_name=metric_name).set(seasonality_strength)
            
            return {
                'trend_strength': trend_strength,
                'seasonality_strength': seasonality_strength,
                'trend_direction': float(np.mean(np.diff(result.trend))),
                'last_trend_value': float(result.trend[-1]),
                'last_seasonal_value': float(result.seasonal[-1])
            }
            
        except Exception as e:
            self.logger.error("Error analyzing trend",
                            metric_name=metric_name,
                            error=str(e))
            return {}

    async def forecast_metric(
        self,
        metric_name: str,
        values: List[float],
        horizon: int = 24
    ) -> Dict[str, List[float]]:
        """
        Generate forecasts using Holt-Winters method.
        """
        if len(values) < horizon:
            return {}
            
        try:
            # Fit model
            model = ExponentialSmoothing(
                values,
                seasonal_periods=min(len(values) // 2, 24),
                trend='add',
                seasonal='add'
            ).fit()
            
            # Generate forecast
            forecast = model.forecast(horizon)
            
            # Calculate forecast error
            mape = np.mean(np.abs(model.resid / values)) * 100
            self.forecast_error.labels(metric_name=metric_name).set(mape)
            
            # Record forecasts
            for h, value in enumerate(forecast):
                self.forecast_value.labels(
                    metric_name=metric_name,
                    horizon=str(h+1)
                ).set(value)
            
            return {
                'forecast': forecast.tolist(),
                'mape': mape,
                'lower_bound': (forecast - 2 * model.resid.std()).tolist(),
                'upper_bound': (forecast + 2 * model.resid.std()).tolist()
            }
            
        except Exception as e:
            self.logger.error("Error generating forecast",
                            metric_name=metric_name,
                            error=str(e))
            return {}

    async def check_risk_metrics(self):
        """Check all risk metrics and generate alerts if needed"""
        try:
            current_state = self.get_current_state()
            
            # Check price impacts
            for token_pair, state in current_state.items():
                price_impact = state.get('price_impact', 0)
                if price_impact > self.risk_thresholds['price_impact']:
                    await self.alert_manager.create_alert(
                        'price_impact',
                        'high',
                        f"High price impact detected for {token_pair}",
                        {
                            'token_pair': token_pair,
                            'impact': price_impact,
                            'threshold': self.risk_thresholds['price_impact']
                        }
                    )
                    
            # Check liquidity levels
            for token_pair, state in current_state.items():
                liquidity = state.get('liquidity', 0)
                if liquidity < self.risk_thresholds['liquidity']:
                    await self.alert_manager.create_alert(
                        'liquidity',
                        'high',
                        f"Low liquidity detected for {token_pair}",
                        {
                            'token_pair': token_pair,
                            'liquidity': liquidity,
                            'threshold': self.risk_thresholds['liquidity']
                        }
                    )
                    
            # Check network conditions
            for chain, state in current_state.items():
                gas_price = state.get('gas_price', 0)
                if gas_price > self.risk_thresholds['gas_price']:
                    await self.alert_manager.create_alert(
                        'gas_price',
                        'medium',
                        f"High gas price on {chain}",
                        {
                            'chain': chain,
                            'gas_price': gas_price,
                            'threshold': self.risk_thresholds['gas_price']
                        }
                    )
                
                network_load = state.get('network_load', 0)
                if network_load > self.risk_thresholds['network_load']:
                    await self.alert_manager.create_alert(
                        'network_load',
                        'medium',
                        f"High network load on {chain}",
                        {
                            'chain': chain,
                            'load': network_load,
                            'threshold': self.risk_thresholds['network_load']
                        }
                    )
                    
        except Exception as e:
            self.logger.error(f"Error checking risk metrics: {str(e)}")

    async def analyze_correlations(
        self,
        metrics: Dict[str, List[float]]
    ) -> Dict[Tuple[str, str], float]:
        """
        Analyze correlations between different metrics.
        """
        correlations = {}
        
        try:
            metric_names = list(metrics.keys())
            
            for i in range(len(metric_names)):
                for j in range(i + 1, len(metric_names)):
                    name1, name2 = metric_names[i], metric_names[j]
                    values1, values2 = metrics[name1], metrics[name2]
                    
                    if len(values1) == len(values2):
                        correlation = stats.pearsonr(values1, values2)[0]
                        correlations[(name1, name2)] = correlation
            
            return correlations
            
        except Exception as e:
            self.logger.error("Error analyzing correlations", error=str(e))
            return {}

    def get_current_state(self) -> Dict[str, Any]:
        """Get current monitoring state"""
        return self._state