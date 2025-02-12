"""
Performance Monitoring Module

This module provides comprehensive performance monitoring capabilities including:
- CPU and memory usage tracking
- Render time monitoring
- Error tracking and reporting
- Optimization suggestions
- Performance metrics collection
- Resource threshold monitoring
"""

import psutil
import time
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
import structlog
from prometheus_client import Counter, Gauge, Histogram
from functools import wraps
import threading
from concurrent.futures import ThreadPoolExecutor
import socket
import aiohttp
import json
from pathlib import Path
from datetime import datetime, timedelta

logger = structlog.get_logger(__name__)

# Performance metrics
METRICS = {
    'cpu_usage': Gauge('cpu_usage_percent', 'CPU usage percentage'),
    'memory_usage': Gauge('memory_usage_bytes', 'Memory usage in bytes'),
    'render_time': Histogram(
        'render_time_seconds',
        'Time taken for rendering operations',
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
    ),
    'error_count': Counter('error_count_total', 'Total number of errors', ['type']),
    'io_wait': Gauge('io_wait_percent', 'IO wait percentage'),
    'network_latency': Histogram(
        'network_latency_seconds',
        'Network operation latency',
        buckets=[0.01, 0.05, 0.1, 0.5, 1.0]
    ),
    'chart_render_time': Histogram(
        'chart_render_time_seconds',
        'Time taken for chart rendering',
        ['chart_type'],
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
    ),
    'websocket_latency': Histogram(
        'websocket_latency_seconds',
        'WebSocket message latency',
        buckets=[0.001, 0.01, 0.1, 0.5, 1.0]
    ),
    'data_update_frequency': Counter(
        'data_update_total',
        'Number of data updates',
        ['component']
    ),
    'chart_interaction_count': Counter(
        'chart_interaction_total',
        'Number of chart interactions',
        ['action']
    )
}

@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring"""
    cpu_threshold: float = 90.0  # CPU usage threshold (%)
    memory_threshold: float = 90.0  # Memory usage threshold (%)
    render_time_threshold: float = 5.0  # Chart render time threshold (seconds)
    monitoring_interval: float = 0.1  # Monitoring interval (seconds)
    enable_suggestions: bool = True  # Enable performance suggestions
    enable_metrics: bool = True  # Enable Prometheus metrics
    log_to_file: bool = False
    log_file_path: str = "performance.log"
    network_check_urls: Optional[List[str]] = None  # URLs to check for network latency
    test_mode: bool = False  # Enable test mode for integration testing

class PerformanceMonitor:
    """
    Performance monitoring system with resource tracking and optimization suggestions.
    
    Features:
    - Real-time CPU and memory monitoring
    - Render time tracking
    - Error tracking and reporting
    - Automatic optimization suggestions
    - Resource threshold monitoring
    - Background monitoring task
    """
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._is_running = False
        self._lock = threading.Lock()
        self._performance_history: List[Dict[str, Any]] = []
        self._suggestions: List[str] = []  # Store suggestions
        
        # Initialize metrics
        if self.config.enable_metrics:
            self._setup_metrics()
    
    def _setup_metrics(self) -> None:
        """Initialize metric tracking"""
        from prometheus_client import CollectorRegistry
        self.registry = CollectorRegistry()
        
        self.metrics = {
            'cpu_usage': Gauge('cpu_usage_percent', 'CPU usage percentage', registry=self.registry),
            'memory_usage': Gauge('memory_usage_bytes', 'Memory usage in bytes', registry=self.registry),
            'render_time': Histogram(
                'render_time_seconds',
                'Time taken for rendering operations',
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
                registry=self.registry
            ),
            'error_count': Counter('error_count_total', 'Total number of errors', ['type'], registry=self.registry),
            'io_wait': Gauge('io_wait_percent', 'IO wait percentage', registry=self.registry),
            'network_latency': Histogram(
                'network_latency_seconds',
                'Network operation latency',
                buckets=[0.01, 0.05, 0.1, 0.5, 1.0],
                registry=self.registry
            ),
            'chart_render_time': Histogram(
                'chart_render_time_seconds',
                'Time taken for chart rendering',
                ['chart_type'],
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
                registry=self.registry
            ),
            'websocket_latency': Histogram(
                'websocket_latency_seconds',
                'WebSocket message latency',
                buckets=[0.001, 0.01, 0.1, 0.5, 1.0],
                registry=self.registry
            ),
            'data_update_frequency': Counter(
                'data_update_total',
                'Number of data updates',
                ['component'],
                registry=self.registry
            ),
            'chart_interaction_count': Counter(
                'chart_interaction_total',
                'Number of chart interactions',
                ['action'],
                registry=self.registry
            )
        }
    
    async def start_monitoring(self) -> None:
        """Start the performance monitoring loop"""
        if self._monitoring_task is not None:
            return
        
        self._is_running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop the performance monitoring loop"""
        self._is_running = False
        if self._monitoring_task is not None:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
            logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop that collects and analyzes performance metrics"""
        while self._is_running:
            try:
                # Collect metrics
                metrics = await self._collect_metrics()
                
                # Update Prometheus metrics
                if self.config.enable_metrics:
                    self._update_prometheus_metrics(metrics)
                
                # Store history
                with self._lock:
                    self._performance_history.append(metrics)
                    if len(self._performance_history) > 100:
                        self._performance_history.pop(0)
                
                # Check thresholds and generate suggestions
                if self.config.enable_suggestions:
                    await self._check_thresholds(metrics)
                
                await asyncio.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
                if self.config.enable_metrics:
                    METRICS['error_count'].labels(type='monitoring').inc()
                await asyncio.sleep(self.config.monitoring_interval)
    
    async def _collect_metrics(self) -> Dict[str, Union[float, str]]:
        """Collect current performance metrics
        
        Returns:
            Dictionary containing metrics with float values and timestamp as string
        """
        try:
            # Run CPU-bound operations in thread pool
            loop = asyncio.get_event_loop()
            cpu_usage = await loop.run_in_executor(
                self._executor,
                lambda: psutil.cpu_percent(interval=0.1)
            )
            memory = await loop.run_in_executor(
                self._executor,
                lambda: psutil.Process().memory_info()
            )
            
            # Get base metrics
            metrics = {
                'cpu_usage': float(cpu_usage),
                'memory_usage': float(memory.rss),
                'memory_percent': float(memory.rss / psutil.virtual_memory().total * 100),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add IO metrics if available
            try:
                io_counters = psutil.disk_io_counters()
                if io_counters:
                    metrics['io_wait'] = float(io_counters.read_time + io_counters.write_time)
            except Exception:
                metrics['io_wait'] = 0.0
            
            # Add visualization-specific metrics
            if hasattr(self, 'metrics'):
                chart_samples = self.metrics['chart_render_time'].collect()
                interaction_samples = self.metrics['chart_interaction_count'].collect()
                update_samples = self.metrics['data_update_frequency'].collect()
                
                metrics.update({
                    'chart_count': float(len(chart_samples)),
                    'interaction_rate': float(sum(s.value for s in interaction_samples) / 60),
                    'update_frequency': float(sum(s.value for s in update_samples) / 60)
                })
            
            return metrics
            
        except Exception as e:
            logger.error("Error collecting metrics", error=str(e))
            return {
                'cpu_usage': 0.0,
                'memory_usage': 0.0,
                'memory_percent': 0.0,
                'io_wait': 0.0,
                'timestamp': datetime.now().isoformat()
            }
    
    def _update_prometheus_metrics(self, metrics: Dict[str, Union[float, str]]) -> None:
        """Update Prometheus metrics with current values
        
        Args:
            metrics: Dictionary containing metrics with float values and timestamp as string
        """
        try:
            if not hasattr(self, 'metrics'):
                return
                
            # Update numeric metrics only
            numeric_metrics = {
                k: v for k, v in metrics.items() 
                if isinstance(v, (int, float)) and k in self.metrics
            }
            
            for name, value in numeric_metrics.items():
                if name in self.metrics:
                    if isinstance(self.metrics[name], (Gauge, Counter)):
                        self.metrics[name].set(value)
                    elif isinstance(self.metrics[name], Histogram):
                        self.metrics[name].observe(value)
                        
        except Exception as e:
            logger.error("Error updating Prometheus metrics", error=str(e))
            if hasattr(self, 'metrics') and 'error_count' in self.metrics:
                self.metrics['error_count'].labels(type='prometheus').inc()
    
    async def _check_thresholds(self, metrics: Dict[str, float]) -> None:
        """Check resource thresholds and generate optimization suggestions"""
        try:
            suggestions = []
            
            # CPU threshold check
            if metrics.get('cpu_usage', 0) > self.config.cpu_threshold:
                suggestions.append(
                    "High CPU usage detected. Consider reducing update frequency or "
                    "optimizing heavy computations."
                )
            
            # Memory threshold check
            if metrics.get('memory_percent', 0) > self.config.memory_threshold:
                suggestions.append(
                    "High memory usage detected. Consider implementing data cleanup "
                    "or reducing cache sizes."
                )
            
            # IO wait check
            if metrics.get('io_wait', 0) > 1000:  # 1 second
                suggestions.append(
                    "High I/O wait times detected. Consider optimizing disk operations "
                    "or using caching."
                )
            
            # Log suggestions
            if suggestions and self.config.enable_suggestions:
                for suggestion in suggestions:
                    logger.warning(suggestion)
                    
        except Exception as e:
            logger.error("Error checking thresholds", error=str(e))
            if self.config.enable_metrics:
                METRICS['error_count'].labels(type='threshold').inc()
    
    def monitor_performance(self, operation_name: str = "unknown"):
        """Decorator for monitoring function performance"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    if self.config.enable_metrics:
                        METRICS['render_time'].observe(duration)
                    
                    if duration > self.config.render_time_threshold:
                        logger.warning(
                            f"Slow operation detected",
                            operation=operation_name,
                            duration=duration
                        )
                    
                    return result
                    
                except Exception as e:
                    if self.config.enable_metrics:
                        METRICS['error_count'].labels(type='operation').inc()
                    logger.error(
                        f"Error in monitored operation",
                        operation=operation_name,
                        error=str(e)
                    )
                    raise
            
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    if self.config.enable_metrics:
                        METRICS['render_time'].observe(duration)
                    
                    if duration > self.config.render_time_threshold:
                        logger.warning(
                            f"Slow operation detected",
                            operation=operation_name,
                            duration=duration
                        )
                    
                    return result
                    
                except Exception as e:
                    if self.config.enable_metrics:
                        METRICS['error_count'].labels(type='operation').inc()
                    logger.error(
                        f"Error in monitored operation",
                        operation=operation_name,
                        error=str(e)
                    )
                    raise
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of current performance metrics"""
        with self._lock:
            if not self._performance_history:
                return {}
            
            recent_metrics = self._performance_history[-1]
            avg_metrics = {
                key: sum(h[key] for h in self._performance_history) / len(self._performance_history)
                for key in recent_metrics
                if key != 'timestamp'
            }
            
            return {
                'current': recent_metrics,
                'averages': avg_metrics,
                'suggestions': self._generate_optimization_suggestions(avg_metrics)
            }
    
    def _generate_optimization_suggestions(
        self,
        metrics: Dict[str, float]
    ) -> List[str]:
        """Generate optimization suggestions based on performance metrics"""
        suggestions = []
        
        if metrics.get('cpu_usage', 0) > self.config.cpu_threshold:
            suggestions.append({
                'type': 'cpu',
                'severity': 'high',
                'message': (
                    "High average CPU usage. Consider implementing caching, "
                    "reducing update frequency, or optimizing computations."
                )
            })
        
        if metrics.get('memory_percent', 0) > self.config.memory_threshold:
            suggestions.append({
                'type': 'memory',
                'severity': 'high',
                'message': (
                    "High average memory usage. Consider implementing cleanup "
                    "routines, reducing cache sizes, or optimizing data structures."
                )
            })
        
        return suggestions

    async def check_network_health(self) -> Dict[str, float]:
        """Check network health by measuring latency to specified endpoints"""
        results = {}
        if not self.config.network_check_urls:
            return results

        async with aiohttp.ClientSession() as session:
            for url in self.config.network_check_urls:
                try:
                    start_time = time.time()
                    async with session.get(url) as response:
                        latency = time.time() - start_time
                        results[url] = latency
                        METRICS['network_latency'].observe(latency)
                except Exception as e:
                    logger.error(f"Network check failed for {url}", error=str(e))
                    results[url] = -1
        return results

    def track_chart_render(self, chart_type: str, duration: float) -> None:
        """Track chart rendering time"""
        if self.config.enable_metrics:
            METRICS['chart_render_time'].labels(chart_type=chart_type).observe(duration)

    def track_chart_interaction(self, action: str) -> None:
        """Track chart interactions"""
        if self.config.enable_metrics:
            METRICS['chart_interaction_count'].labels(action=action).inc()

    def track_data_update(self, component: str) -> None:
        """Track data updates by component"""
        if self.config.enable_metrics:
            METRICS['data_update_frequency'].labels(component=component).inc()

    async def run_health_check(self) -> Dict[str, Any]:
        """Run a comprehensive health check"""
        try:
            # Get current metrics
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.Process().memory_percent()
            
            # Calculate average render time
            render_times = [
                h.get('render_time', 0.0) 
                for h in self._performance_history[-100:]
            ]
            avg_render_time = sum(render_times) / len(render_times) if render_times else 0
            
            # Determine system health
            is_healthy = (
                cpu_percent < self.config.cpu_threshold and
                memory_percent < self.config.memory_threshold and
                avg_render_time < self.config.render_time_threshold
            )
            
            # Generate suggestions if enabled
            suggestions = []
            if self.config.enable_suggestions:
                if cpu_percent > self.config.cpu_threshold:
                    suggestions.append(
                        "Consider reducing update frequency or optimizing computations"
                    )
                if memory_percent > self.config.memory_threshold:
                    suggestions.append(
                        "Consider implementing data cleanup or reducing cache size"
                    )
                if avg_render_time > self.config.render_time_threshold:
                    suggestions.append(
                        "Consider optimizing chart rendering or reducing data points"
                    )
            
            return {
                'status': 'healthy' if is_healthy else 'degraded',
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'avg_render_time': avg_render_time,
                'suggestions': suggestions,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Error running health check", error=str(e))
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def get_performance_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get performance history within the specified time range"""
        if not start_time:
            start_time = datetime.now() - timedelta(hours=1)
        if not end_time:
            end_time = datetime.now()
            
        return [
            entry for entry in self._performance_history
            if start_time <= datetime.fromisoformat(entry['timestamp']) <= end_time
        ]

    async def export_metrics(self, export_path: str) -> None:
        """Export metrics to file for analysis"""
        try:
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': self._performance_history,
                'summary': self.get_performance_summary()
            }
            
            Path(export_path).parent.mkdir(parents=True, exist_ok=True)
            with open(export_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
            logger.info(f"Metrics exported to {export_path}")
        except Exception as e:
            logger.error("Error exporting metrics", error=str(e))

# Convenience function to create a monitor instance with default config
def create_monitor(
    cpu_threshold: float = 90.0,
    memory_threshold: float = 90.0,
    render_time_threshold: float = 5.0,
    monitoring_interval: float = 0.1,
    enable_suggestions: bool = True,
    enable_metrics: bool = True
) -> PerformanceMonitor:
    """Create a performance monitor instance with custom configuration"""
    config = PerformanceConfig(
        cpu_threshold=cpu_threshold,
        memory_threshold=memory_threshold,
        render_time_threshold=render_time_threshold,
        monitoring_interval=monitoring_interval,
        enable_suggestions=enable_suggestions,
        enable_metrics=enable_metrics
    )
    return PerformanceMonitor(config)

# Add test utilities
def create_test_monitor() -> PerformanceMonitor:
    """Create a monitor instance for testing"""
    config = PerformanceConfig(
        cpu_threshold=90.0,
        memory_threshold=90.0,
        render_time_threshold=5.0,
        monitoring_interval=0.1,
        enable_suggestions=True,
        enable_metrics=True,
        test_mode=True,
        network_check_urls=['http://localhost:8050']  # Dash default port
    )
    return PerformanceMonitor(config)

async def run_integration_test(
    monitor: PerformanceMonitor,
    test_duration: float = 10.0
) -> Tuple[bool, str]:
    """Run integration test on the monitor"""
    try:
        # Start monitoring
        await monitor.start_monitoring()
        
        # Simulate some load
        start_time = time.time()
        while time.time() - start_time < test_duration:
            # Simulate chart renders
            monitor.track_chart_render('test_chart', 0.5)
            monitor.track_chart_interaction('zoom')
            monitor.track_data_update('test_component')
            await asyncio.sleep(0.1)
        
        # Run health check
        health_check = await monitor.run_health_check()
        
        # Stop monitoring
        await monitor.stop_monitoring()
        
        # Verify results
        success = all([
            len(monitor._performance_history) > 0,
            health_check['status'] in ['healthy', 'degraded'],
            isinstance(health_check['suggestions'], list)
        ])
        
        return success, "Integration test completed successfully" if success else "Test failed"
        
    except Exception as e:
        return False, f"Integration test failed: {str(e)}" 