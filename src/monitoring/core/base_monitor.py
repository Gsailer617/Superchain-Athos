"""
Base monitoring system that consolidates core functionality from all monitoring components
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import structlog
from prometheus_client import Counter, Gauge, Histogram
import psutil
import json
import asyncio
from pathlib import Path
import numpy as np
import pandas as pd
from functools import wraps

logger = structlog.get_logger(__name__)

@dataclass
class MetricConfig:
    """Configuration for metrics"""
    name: str
    description: str
    type: str
    labels: Optional[List[str]] = None
    buckets: Optional[List[float]] = None

@dataclass
class ResourceThreshold:
    """Resource threshold configuration"""
    warning: float
    critical: float
    check_interval: int = 60

class BaseMonitor:
    """Base monitoring system that consolidates core functionality"""
    
    def __init__(
        self,
        storage_path: str = "data/monitoring",
        max_memory_entries: int = 10000,
        flush_interval: int = 100,
        enable_prometheus: bool = True
    ):
        """Initialize base monitoring system
        
        Args:
            storage_path: Path for storing monitoring data
            max_memory_entries: Maximum entries to keep in memory
            flush_interval: Interval for flushing to disk
            enable_prometheus: Whether to enable Prometheus metrics
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_entries = max_memory_entries
        self.flush_interval = flush_interval
        self.enable_prometheus = enable_prometheus
        
        # Core metrics storage
        self._metrics_history: List[Dict[str, Any]] = []
        self._resource_metrics: Dict[str, List[float]] = {
            'cpu': [], 'memory': [], 'disk': [], 'network': []
        }
        
        # Performance tracking
        self._performance_data = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_execution_time': 0.0,
            'operation_counts': {},
            'error_counts': {}
        }
        
        # Initialize Prometheus metrics if enabled
        if enable_prometheus:
            self._setup_prometheus_metrics()
            
        # Anomaly detection state
        self._anomaly_baselines = {}
        self._anomaly_thresholds = {}
        
        # Start monitoring tasks
        self._monitoring_tasks = []
        self._is_running = False
        
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        self._metrics = {
            # Resource metrics
            'resource_usage': Gauge(
                'resource_usage_percent',
                'Resource usage percentage',
                ['resource_type']
            ),
            'resource_saturation': Gauge(
                'resource_saturation_level',
                'Resource saturation level',
                ['resource_type']
            ),
            
            # Operation metrics
            'operation_duration': Histogram(
                'operation_duration_seconds',
                'Operation duration in seconds',
                ['operation_type'],
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            ),
            'operation_count': Counter(
                'operation_count_total',
                'Total operation count',
                ['operation_type', 'status']
            ),
            
            # Error metrics
            'error_count': Counter(
                'error_count_total',
                'Total error count',
                ['error_type']
            ),
            
            # Performance metrics
            'performance_score': Gauge(
                'performance_score',
                'Overall performance score',
                ['component']
            )
        }
    
    async def start_monitoring(self):
        """Start all monitoring tasks"""
        if self._is_running:
            return
            
        self._is_running = True
        self._monitoring_tasks = [
            asyncio.create_task(self._collect_resource_metrics()),
            asyncio.create_task(self._analyze_performance()),
            asyncio.create_task(self._detect_anomalies())
        ]
        
        logger.info("Started monitoring tasks")
    
    async def stop_monitoring(self):
        """Stop all monitoring tasks"""
        self._is_running = False
        for task in self._monitoring_tasks:
            task.cancel()
        await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
        logger.info("Stopped monitoring tasks")
    
    def track_operation(self, operation_type: str):
        """Decorator for tracking operation metrics"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = datetime.now()
                try:
                    result = await func(*args, **kwargs)
                    duration = (datetime.now() - start_time).total_seconds()
                    
                    self._record_operation(
                        operation_type=operation_type,
                        duration=duration,
                        success=True
                    )
                    
                    return result
                except Exception as e:
                    duration = (datetime.now() - start_time).total_seconds()
                    self._record_operation(
                        operation_type=operation_type,
                        duration=duration,
                        success=False,
                        error=str(e)
                    )
                    raise
                    
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = datetime.now()
                try:
                    result = func(*args, **kwargs)
                    duration = (datetime.now() - start_time).total_seconds()
                    
                    self._record_operation(
                        operation_type=operation_type,
                        duration=duration,
                        success=True
                    )
                    
                    return result
                except Exception as e:
                    duration = (datetime.now() - start_time).total_seconds()
                    self._record_operation(
                        operation_type=operation_type,
                        duration=duration,
                        success=False,
                        error=str(e)
                    )
                    raise
                    
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def _record_operation(
        self,
        operation_type: str,
        duration: float,
        success: bool,
        error: Optional[str] = None
    ):
        """Record operation metrics"""
        self._performance_data['total_operations'] += 1
        if success:
            self._performance_data['successful_operations'] += 1
        else:
            self._performance_data['failed_operations'] += 1
            
        self._performance_data['total_execution_time'] += duration
        
        # Update operation counts
        if operation_type not in self._performance_data['operation_counts']:
            self._performance_data['operation_counts'][operation_type] = {
                'total': 0, 'success': 0, 'failed': 0
            }
        self._performance_data['operation_counts'][operation_type]['total'] += 1
        if success:
            self._performance_data['operation_counts'][operation_type]['success'] += 1
        else:
            self._performance_data['operation_counts'][operation_type]['failed'] += 1
            
        # Update error counts if applicable
        if error:
            if error not in self._performance_data['error_counts']:
                self._performance_data['error_counts'][error] = 0
            self._performance_data['error_counts'][error] += 1
            
        # Update Prometheus metrics if enabled
        if self.enable_prometheus:
            self._metrics['operation_duration'].labels(
                operation_type=operation_type
            ).observe(duration)
            
            self._metrics['operation_count'].labels(
                operation_type=operation_type,
                status='success' if success else 'failed'
            ).inc()
            
            if error:
                self._metrics['error_count'].labels(
                    error_type=error
                ).inc()
    
    async def _collect_resource_metrics(self):
        """Collect system resource metrics"""
        while self._is_running:
            try:
                # Collect CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                self._resource_metrics['cpu'].append(cpu_percent)
                
                # Collect memory metrics
                memory = psutil.virtual_memory()
                self._resource_metrics['memory'].append(memory.percent)
                
                # Collect disk metrics
                disk = psutil.disk_usage('/')
                self._resource_metrics['disk'].append(disk.percent)
                
                # Update Prometheus metrics if enabled
                if self.enable_prometheus:
                    self._metrics['resource_usage'].labels(
                        resource_type='cpu'
                    ).set(cpu_percent)
                    self._metrics['resource_usage'].labels(
                        resource_type='memory'
                    ).set(memory.percent)
                    self._metrics['resource_usage'].labels(
                        resource_type='disk'
                    ).set(disk.percent)
                
                # Trim history if needed
                for resource_type in self._resource_metrics:
                    if len(self._resource_metrics[resource_type]) > self.max_memory_entries:
                        self._resource_metrics[resource_type] = (
                            self._resource_metrics[resource_type][-self.max_memory_entries:]
                        )
                
                await asyncio.sleep(60)  # Collect every minute
                
            except Exception as e:
                logger.error("Error collecting resource metrics", error=str(e))
                await asyncio.sleep(60)
    
    async def _analyze_performance(self):
        """Analyze system performance"""
        while self._is_running:
            try:
                # Calculate performance metrics
                total_ops = self._performance_data['total_operations']
                if total_ops > 0:
                    success_rate = (
                        self._performance_data['successful_operations'] / total_ops
                    ) * 100
                    avg_execution_time = (
                        self._performance_data['total_execution_time'] / total_ops
                    )
                    
                    # Update Prometheus metrics if enabled
                    if self.enable_prometheus:
                        self._metrics['performance_score'].labels(
                            component='success_rate'
                        ).set(success_rate)
                        self._metrics['performance_score'].labels(
                            component='avg_execution_time'
                        ).set(avg_execution_time)
                
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
            except Exception as e:
                logger.error("Error analyzing performance", error=str(e))
                await asyncio.sleep(300)
    
    async def _detect_anomalies(self):
        """Detect system anomalies"""
        while self._is_running:
            try:
                for resource_type, metrics in self._resource_metrics.items():
                    if len(metrics) < 30:  # Need enough data points
                        continue
                        
                    # Calculate baseline if not exists
                    if resource_type not in self._anomaly_baselines:
                        self._anomaly_baselines[resource_type] = np.mean(metrics)
                        self._anomaly_thresholds[resource_type] = np.std(metrics) * 2
                    
                    # Check for anomalies
                    current_value = metrics[-1]
                    baseline = self._anomaly_baselines[resource_type]
                    threshold = self._anomaly_thresholds[resource_type]
                    
                    if abs(current_value - baseline) > threshold:
                        logger.warning(
                            "Anomaly detected",
                            resource_type=resource_type,
                            value=current_value,
                            baseline=baseline,
                            threshold=threshold
                        )
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error("Error detecting anomalies", error=str(e))
                await asyncio.sleep(300)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary
        
        Returns:
            Dictionary containing performance metrics
        """
        total_ops = self._performance_data['total_operations']
        
        return {
            'total_operations': total_ops,
            'success_rate': (
                (self._performance_data['successful_operations'] / total_ops * 100)
                if total_ops > 0 else 0
            ),
            'average_execution_time': (
                self._performance_data['total_execution_time'] / total_ops
                if total_ops > 0 else 0
            ),
            'operation_counts': self._performance_data['operation_counts'],
            'error_counts': self._performance_data['error_counts'],
            'resource_usage': {
                resource_type: np.mean(metrics[-10:])  # Average of last 10 measurements
                for resource_type, metrics in self._resource_metrics.items()
                if metrics
            }
        }
    
    def export_metrics(self, export_path: Optional[str] = None) -> None:
        """Export metrics to file
        
        Args:
            export_path: Optional path to export metrics to. If not provided,
                       uses storage_path/metrics_{timestamp}.json
        """
        try:
            if not export_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                export_path = str(self.storage_path / f"metrics_{timestamp}.json")
            
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'performance_data': self._performance_data,
                'resource_metrics': {
                    k: v[-100:] for k, v in self._resource_metrics.items()
                },
                'summary': self.get_performance_summary()
            }
            
            with open(export_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
            logger.info(f"Metrics exported to {export_path}")
            
        except Exception as e:
            logger.error("Error exporting metrics", error=str(e)) 