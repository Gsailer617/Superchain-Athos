"""
Performance Monitoring Module

This module provides comprehensive performance monitoring:
- System resource monitoring
- Application performance metrics
- Bottleneck detection
- Anomaly detection
- Alert generation
- Performance reporting
"""

import asyncio
import psutil
import structlog
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from ..utils.metrics import metrics_manager, MetricConfig, MetricType, track_metric
import numpy as np
from prometheus_client import Gauge, Counter, Histogram
import aiohttp
import json

logger = structlog.get_logger(__name__)

class ResourceType(Enum):
    """Types of resources to monitor"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    IO = "io"

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

@dataclass
class Alert:
    """Alert data structure"""
    timestamp: datetime
    resource_type: ResourceType
    severity: AlertSeverity
    value: float
    threshold: float
    message: str

class PerformanceMonitor:
    """Performance monitoring system"""
    
    def __init__(self,
                 alert_webhook_url: Optional[str] = None,
                 collection_interval: int = 10):
        self.alert_webhook_url = alert_webhook_url
        self.collection_interval = collection_interval
        self._thresholds: Dict[ResourceType, ResourceThreshold] = {}
        self._historical_data: Dict[ResourceType, List[float]] = {
            rt: [] for rt in ResourceType
        }
        self._active_alerts: Set[str] = set()
        self._setup_metrics()

    def _setup_metrics(self):
        """Setup performance metrics"""
        # Resource usage metrics
        self._resource_usage = metrics_manager.create_metric(MetricConfig(
            name="resource_usage",
            description="Resource usage percentage",
            type=MetricType.GAUGE,
            labels=['resource_type']
        ))
        
        # Resource saturation metrics
        self._resource_saturation = metrics_manager.create_metric(MetricConfig(
            name="resource_saturation",
            description="Resource saturation level",
            type=MetricType.GAUGE,
            labels=['resource_type']
        ))
        
        # Alert metrics
        self._alerts = metrics_manager.create_metric(MetricConfig(
            name="alerts_total",
            description="Number of alerts generated",
            type=MetricType.COUNTER,
            labels=['resource_type', 'severity']
        ))
        
        # Performance metrics
        self._performance_metrics = metrics_manager.create_metric(MetricConfig(
            name="performance_metrics",
            description="Various performance metrics",
            type=MetricType.HISTOGRAM,
            labels=['metric_type'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        ))

    def configure_threshold(self, 
                          resource_type: ResourceType,
                          threshold: ResourceThreshold):
        """Configure resource thresholds"""
        self._thresholds[resource_type] = threshold

    async def _collect_cpu_metrics(self) -> Dict[str, float]:
        """Collect CPU metrics"""
        cpu_times = psutil.cpu_times_percent()
        cpu_stats = psutil.cpu_stats()
        
        return {
            'usage': psutil.cpu_percent(),
            'user': cpu_times.user,
            'system': cpu_times.system,
            'idle': cpu_times.idle,
            'interrupts': cpu_stats.interrupts,
            'ctx_switches': cpu_stats.ctx_switches
        }

    async def _collect_memory_metrics(self) -> Dict[str, float]:
        """Collect memory metrics"""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'usage': mem.percent,
            'available': mem.available / mem.total * 100,
            'swap_usage': swap.percent,
            'swap_free': swap.free / swap.total * 100
        }

    async def _collect_disk_metrics(self) -> Dict[str, float]:
        """Collect disk metrics"""
        disk = psutil.disk_usage('/')
        io_counters = psutil.disk_io_counters()
        
        return {
            'usage': disk.percent,
            'read_bytes': io_counters.read_bytes,
            'write_bytes': io_counters.write_bytes,
            'read_time': io_counters.read_time,
            'write_time': io_counters.write_time
        }

    async def _collect_network_metrics(self) -> Dict[str, float]:
        """Collect network metrics"""
        net = psutil.net_io_counters()
        
        return {
            'bytes_sent': net.bytes_sent,
            'bytes_recv': net.bytes_recv,
            'packets_sent': net.packets_sent,
            'packets_recv': net.packets_recv,
            'errin': net.errin,
            'errout': net.errout
        }

    async def _detect_anomalies(self, 
                              resource_type: ResourceType,
                              current_value: float) -> Optional[float]:
        """Detect anomalies using statistical methods"""
        data = self._historical_data[resource_type]
        if len(data) < 30:  # Need enough data points
            return None
            
        mean = np.mean(data)
        std = np.std(data)
        z_score = (current_value - mean) / std if std > 0 else 0
        
        return abs(z_score) if abs(z_score) > 3 else None  # 3 sigma rule

    async def _generate_alert(self,
                            resource_type: ResourceType,
                            value: float,
                            threshold: float,
                            severity: AlertSeverity):
        """Generate and send an alert"""
        alert = Alert(
            timestamp=datetime.now(),
            resource_type=resource_type,
            severity=severity,
            value=value,
            threshold=threshold,
            message=f"{resource_type.value} usage at {value:.1f}% (threshold: {threshold:.1f}%)"
        )
        
        # Track alert in metrics
        self._alerts.labels(
            resource_type=resource_type.value,
            severity=severity.value
        ).inc()
        
        # Avoid alert spam
        alert_key = f"{resource_type.value}_{severity.value}"
        if alert_key in self._active_alerts:
            return
            
        self._active_alerts.add(alert_key)
        
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
                            'message': alert.message
                        }
                    )
            except Exception as e:
                logger.error(f"Failed to send alert: {str(e)}")

    async def _check_thresholds(self,
                              resource_type: ResourceType,
                              value: float):
        """Check resource usage against thresholds"""
        if resource_type not in self._thresholds:
            return
            
        threshold = self._thresholds[resource_type]
        
        if value >= threshold.critical:
            await self._generate_alert(
                resource_type,
                value,
                threshold.critical,
                AlertSeverity.CRITICAL
            )
        elif value >= threshold.error:
            await self._generate_alert(
                resource_type,
                value,
                threshold.error,
                AlertSeverity.ERROR
            )
        elif value >= threshold.warning:
            await self._generate_alert(
                resource_type,
                value,
                threshold.warning,
                AlertSeverity.WARNING
            )
        else:
            # Clear active alerts for this resource
            alert_keys = [
                key for key in self._active_alerts
                if key.startswith(resource_type.value)
            ]
            for key in alert_keys:
                self._active_alerts.remove(key)

    @track_metric("collect_metrics")
    async def collect_metrics(self):
        """Collect all performance metrics"""
        # Collect resource metrics
        cpu_metrics = await self._collect_cpu_metrics()
        memory_metrics = await self._collect_memory_metrics()
        disk_metrics = await self._collect_disk_metrics()
        network_metrics = await self._collect_network_metrics()
        
        # Update resource usage metrics
        self._resource_usage.labels(
            resource_type=ResourceType.CPU.value
        ).set(cpu_metrics['usage'])
        self._resource_usage.labels(
            resource_type=ResourceType.MEMORY.value
        ).set(memory_metrics['usage'])
        self._resource_usage.labels(
            resource_type=ResourceType.DISK.value
        ).set(disk_metrics['usage'])
        
        # Update historical data
        for rt, value in [
            (ResourceType.CPU, cpu_metrics['usage']),
            (ResourceType.MEMORY, memory_metrics['usage']),
            (ResourceType.DISK, disk_metrics['usage'])
        ]:
            self._historical_data[rt].append(value)
            if len(self._historical_data[rt]) > 1000:  # Keep last 1000 points
                self._historical_data[rt].pop(0)
            
            # Check for anomalies
            anomaly_score = await self._detect_anomalies(rt, value)
            if anomaly_score:
                self._performance_metrics.labels(
                    metric_type=f"{rt.value}_anomaly"
                ).observe(anomaly_score)
            
            # Check thresholds
            await self._check_thresholds(rt, value)
        
        # Update saturation metrics
        self._resource_saturation.labels(
            resource_type=ResourceType.CPU.value
        ).set(100 - cpu_metrics['idle'])
        self._resource_saturation.labels(
            resource_type=ResourceType.MEMORY.value
        ).set(100 - memory_metrics['available'])
        self._resource_saturation.labels(
            resource_type=ResourceType.DISK.value
        ).set(disk_metrics['usage'])

    async def start_monitoring(self):
        """Start the monitoring loop"""
        while True:
            try:
                await self.collect_metrics()
            except Exception as e:
                logger.error(f"Error collecting metrics: {str(e)}")
            
            await asyncio.sleep(self.collection_interval)

# Global performance monitor instance
performance_monitor = PerformanceMonitor() 