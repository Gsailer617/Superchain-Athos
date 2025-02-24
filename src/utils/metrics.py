"""
Metrics Module

This module provides centralized Prometheus metrics setup and reporting functionality.
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    CollectorRegistry,
    REGISTRY,
    Sample
)
from prometheus_client.metrics import MetricWrapperBase
from typing import Dict, Union, Optional, cast, Any, List, Iterable
import logging
from contextlib import contextmanager
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MetricValue = Union[int, float]
MetricDict = Dict[str, MetricValue]
Metric = Union[Counter, Gauge, Histogram]

@dataclass
class MetricConfig:
    """Configuration for a metric"""
    name: str
    description: str
    type: str = "gauge"  # gauge, counter, histogram
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None

class MetricsManager:
    """
    Centralized metrics management with type-safe metric access
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize metrics manager with optional custom registry
        
        Args:
            registry: Optional custom Prometheus registry
        """
        self._metrics: Dict[str, Metric] = {}
        self._registry = registry or REGISTRY
        self._initialize_metrics()
        
    def _get_metric_value(self, metric: MetricWrapperBase) -> float:
        """Safely get current value from a metric.
        
        Args:
            metric: Prometheus metric wrapper
            
        Returns:
            Current metric value as float
        """
        try:
            if isinstance(metric, (Counter, Gauge)):
                return float(metric._value.get())
            elif isinstance(metric, Histogram):
                samples = metric.collect()[0].samples
                return sum(s.value for s in samples if s.name.endswith('_sum'))
            return 0.0
        except Exception:
            return 0.0
            
    def _get_histogram_stats(
        self,
        histogram: Histogram
    ) -> Dict[str, float]:
        """Get statistics from a histogram.
        
        Args:
            histogram: Prometheus histogram
            
        Returns:
            Dict with sum, count and average
        """
        try:
            samples = histogram.collect()[0].samples
            total = sum(s.value for s in samples if s.name.endswith('_sum'))
            count = sum(s.value for s in samples if s.name.endswith('_count'))
            return {
                'sum': total,
                'count': count,
                'average': total / count if count > 0 else 0.0
            }
        except Exception:
            return {'sum': 0.0, 'count': 0.0, 'average': 0.0}

    @contextmanager
    def timer(self, metric: Histogram):
        """Context manager for timing operations.
        
        Args:
            metric: Histogram to record duration in
            
        Example:
            with metrics.timer(metrics.api_latency):
                # Timed operation
        """
        start = time.monotonic()
        try:
            yield
        finally:
            duration = time.monotonic() - start
            metric.observe(duration)
        
    def _initialize_metrics(self):
        """Initialize default metrics"""
        # Protocol metrics
        self._add_metric(MetricConfig(
            name="protocol_tvl",
            description="Protocol TVL in USD",
            type="gauge",
            labels=["protocol"]
        ))
        self._add_metric(MetricConfig(
            name="protocol_volume_24h",
            description="Protocol 24h volume in USD",
            type="gauge",
            labels=["protocol"]
        ))
        self._add_metric(MetricConfig(
            name="protocol_fees_24h",
            description="Protocol 24h fees in USD",
            type="gauge",
            labels=["protocol"]
        ))
        self._add_metric(MetricConfig(
            name="protocol_revenue_24h",
            description="Protocol 24h revenue in USD",
            type="gauge",
            labels=["protocol"]
        ))
        
        # Token metrics
        self._add_metric(MetricConfig(
            name="token_price",
            description="Token price in USD",
            type="gauge",
            labels=["token"]
        ))
        self._add_metric(MetricConfig(
            name="token_volume_24h",
            description="Token 24h volume in USD",
            type="gauge",
            labels=["token"]
        ))
        self._add_metric(MetricConfig(
            name="token_liquidity",
            description="Token liquidity in USD",
            type="gauge",
            labels=["token"]
        ))
        
        # Chain metrics
        self._add_metric(MetricConfig(
            name="chain_tvl",
            description="Chain TVL in USD",
            type="gauge",
            labels=["chain"]
        ))
        self._add_metric(MetricConfig(
            name="chain_volume_24h",
            description="Chain 24h volume in USD",
            type="gauge",
            labels=["chain"]
        ))
        self._add_metric(MetricConfig(
            name="chain_fees_24h",
            description="Chain 24h fees in USD",
            type="gauge",
            labels=["chain"]
        ))
        
        # Error metrics
        self._add_metric(MetricConfig(
            name="api_errors_total",
            description="Total API errors",
            type="counter",
            labels=["integration", "error"]
        ))
    
    def _add_metric(self, config: MetricConfig):
        """Add a new metric"""
        name = f"defillama_{config.name}"
        
        if config.type == "gauge":
            metric: Metric = Gauge(
                name,
                config.description,
                config.labels,
                registry=self._registry
            )
        elif config.type == "counter":
            metric = Counter(
                name,
                config.description,
                config.labels,
                registry=self._registry
            )
        elif config.type == "histogram":
            metric = Histogram(
                name,
                config.description,
                config.labels,
                buckets=config.buckets or Histogram.DEFAULT_BUCKETS,
                registry=self._registry
            )
        else:
            raise ValueError(f"Unknown metric type: {config.type}")
            
        self._metrics[config.name] = metric
    
    def observe(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Observe a metric value
        
        Args:
            name: Metric name
            value: Metric value
            labels: Optional metric labels
        """
        try:
            metric = self._metrics.get(name)
            if metric:
                if labels:
                    if isinstance(metric, Gauge):
                        metric.labels(**labels).set(value)
                    elif isinstance(metric, Histogram):
                        metric.labels(**labels).observe(value)
                else:
                    if isinstance(metric, Gauge):
                        metric.set(value)
                    elif isinstance(metric, Histogram):
                        metric.observe(value)
        except Exception as e:
            logger.error(f"Error observing metric: {str(e)}")
            
    def inc_counter(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None
    ):
        """Increment a counter metric
        
        Args:
            name: Counter name
            labels: Optional metric labels
        """
        try:
            metric = self._metrics.get(name)
            if metric and isinstance(metric, Counter):
                if labels:
                    metric.labels(**labels).inc()
                else:
                    metric.inc()
        except Exception as e:
            logger.error(f"Error incrementing counter: {str(e)}")
            
    def get_metric(self, name: str) -> Optional[Metric]:
        """Get a metric by name
        
        Args:
            name: Metric name
            
        Returns:
            Metric object if found, None otherwise
        """
        return self._metrics.get(name)
        
    def record_api_error(
        self,
        integration: str,
        error: str
    ):
        """Record an API error
        
        Args:
            integration: Integration identifier
            error: Error message
        """
        self.inc_counter(
            "api_errors_total",
            {
                "integration": integration,
                "error": error
            }
        )
    
    def clear(self):
        """Clear all metrics"""
        try:
            metrics = list(self._metrics.values())
            for metric in metrics:
                try:
                    self._registry.unregister(metric)
                except Exception:
                    pass  # Ignore unregister errors
            self._metrics.clear()
            self._initialize_metrics()
        except Exception as e:
            logger.error(f"Error clearing metrics: {str(e)}")
            
    def observe_protocol_metrics(
        self,
        protocol: str,
        metrics: Dict[str, float]
    ):
        """Observe protocol metrics
        
        Args:
            protocol: Protocol identifier
            metrics: Dictionary of metric values
        """
        try:
            for name, value in metrics.items():
                metric = self._metrics.get(f"protocol_{name}")
                if metric and isinstance(metric, Gauge):
                    metric.labels(protocol=protocol).set(value)
        except Exception as e:
            logger.error(f"Error observing protocol metrics: {str(e)}")
    
    def observe_token_metrics(
        self,
        token: str,
        metrics: Dict[str, float]
    ):
        """Observe token metrics
        
        Args:
            token: Token identifier
            metrics: Dictionary of metric values
        """
        try:
            for name, value in metrics.items():
                metric = self._metrics.get(f"token_{name}")
                if metric and isinstance(metric, Gauge):
                    metric.labels(token=token).set(value)
        except Exception as e:
            logger.error(f"Error observing token metrics: {str(e)}")
    
    def observe_chain_metrics(
        self,
        chain: str,
        metrics: Dict[str, float]
    ):
        """Observe chain metrics
        
        Args:
            chain: Chain identifier
            metrics: Dictionary of metric values
        """
        try:
            for name, value in metrics.items():
                metric = self._metrics.get(f"chain_{name}")
                if metric and isinstance(metric, Gauge):
                    metric.labels(chain=chain).set(value)
        except Exception as e:
            logger.error(f"Error observing chain metrics: {str(e)}")
            
    def __getattr__(self, name: str) -> Any:
        """Handle attribute access for observe_* methods"""
        if name.startswith('observe_'):
            return getattr(self, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'") 