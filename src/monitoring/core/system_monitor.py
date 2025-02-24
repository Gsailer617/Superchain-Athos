"""
Enhanced system monitoring that consolidates all resource monitoring functionality.
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import structlog
import psutil
import asyncio
import numpy as np
from pathlib import Path
import aiohttp
from .base_monitor import BaseMonitor
from .metrics_manager import metrics_manager, MetricType, MetricConfig

logger = structlog.get_logger(__name__)

class ResourceType(Enum):
    """Types of resources to monitor"""
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
        resource_thresholds: Optional[Dict[ResourceType, ResourceThreshold]] = None
    ):
        """Initialize system monitor
        
        Args:
            storage_path: Path for storing monitoring data
            alert_webhook_url: Optional webhook URL for alerts
            collection_interval: Metric collection interval in seconds
            enable_prometheus: Whether to enable Prometheus metrics
            resource_thresholds: Custom resource thresholds
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
        
        # Active alerts tracking
        self._active_alerts: Set[str] = set()
        self._alert_history: List[ResourceAlert] = []
        
        # Initialize metrics
        self._setup_resource_metrics()
        
        # Start monitoring tasks
        self._monitoring_tasks: List[asyncio.Task] = []
        self._is_running = False
    
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
                self._update_resource_metrics(ResourceType.CPU, cpu_metrics)
                
                # Collect memory metrics
                memory_metrics = await self._collect_memory_metrics()
                self._update_resource_metrics(ResourceType.MEMORY, memory_metrics)
                
                # Collect disk metrics
                disk_metrics = await self._collect_disk_metrics()
                self._update_resource_metrics(ResourceType.DISK, disk_metrics)
                
                # Collect network metrics
                network_metrics = await self._collect_network_metrics()
                self._update_resource_metrics(ResourceType.NETWORK, network_metrics)
                
                # Collect IO metrics
                io_metrics = await self._collect_io_metrics()
                self._update_resource_metrics(ResourceType.IO, io_metrics)
                
                # Check thresholds
                await self._check_thresholds()
                
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
            # Process IO counters
            proc_io = psutil.Process().io_counters()
            metrics.update({
                'proc_read_bytes': proc_io.read_bytes,
                'proc_write_bytes': proc_io.write_bytes,
                'proc_read_count': proc_io.read_count,
                'proc_write_count': proc_io.write_count
            })
        except Exception:
            pass
        
        return metrics
    
    def _update_resource_metrics(
        self,
        resource_type: ResourceType,
        metrics: Dict[str, float]
    ):
        """Update resource metrics
        
        Args:
            resource_type: Type of resource
            metrics: Resource metrics
        """
        # Store historical data
        self._resource_history[resource_type].append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })
        
        # Update Prometheus metrics
        for name, value in metrics.items():
            self._metrics['resource_usage'].labels(
                resource_type=resource_type.value,
                component=name
            ).set(value)
        
        # Calculate and update saturation
        if 'usage' in metrics:
            self._metrics['resource_saturation'].labels(
                resource_type=resource_type.value
            ).set(metrics['usage'])
    
    async def _check_thresholds(self):
        """Check resource metrics against thresholds"""
        for resource_type, threshold in self._thresholds.items():
            history = self._resource_history[resource_type]
            if not history:
                continue
            
            current_metrics = history[-1]['metrics']
            usage = current_metrics.get('usage', 0)
            
            # Check critical threshold
            if usage >= threshold.critical:
                await self._generate_alert(
                    resource_type,
                    usage,
                    threshold.critical,
                    AlertSeverity.CRITICAL,
                    current_metrics
                )
            # Check error threshold
            elif usage >= threshold.error:
                await self._generate_alert(
                    resource_type,
                    usage,
                    threshold.error,
                    AlertSeverity.ERROR,
                    current_metrics
                )
            # Check warning threshold
            elif usage >= threshold.warning:
                await self._generate_alert(
                    resource_type,
                    usage,
                    threshold.warning,
                    AlertSeverity.WARNING,
                    current_metrics
                )
    
    async def _generate_alert(
        self,
        resource_type: ResourceType,
        value: float,
        threshold: float,
        severity: AlertSeverity,
        metadata: Dict[str, Any]
    ):
        """Generate and send a resource alert
        
        Args:
            resource_type: Type of resource
            value: Current value
            threshold: Threshold value
            severity: Alert severity
            metadata: Additional alert metadata
        """
        # Create alert
        alert = ResourceAlert(
            timestamp=datetime.now(),
            resource_type=resource_type,
            severity=severity,
            value=value,
            threshold=threshold,
            message=f"{resource_type.value} usage at {value:.1f}% (threshold: {threshold:.1f}%)",
            metadata=metadata
        )
        
        # Update metrics
        self._metrics['alerts'].labels(
            resource_type=resource_type.value,
            severity=severity.value
        ).inc()
        
        # Check alert cooldown
        alert_key = f"{resource_type.value}_{severity.value}"
        if alert_key in self._active_alerts:
            return
        
        self._active_alerts.add(alert_key)
        self._alert_history.append(alert)
        
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