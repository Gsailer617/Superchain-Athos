"""
Centralized metrics management system for all monitoring components.
Provides unified metric collection, storage, and Prometheus integration.
"""

from typing import Dict, List, Optional, Any, Union, Tuple, Set, cast
from dataclasses import dataclass
from enum import Enum
import structlog
from prometheus_client import Counter, Gauge, Histogram, Summary
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import time
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

logger = structlog.get_logger(__name__)

@dataclass
class AggregatedValue:
    """Aggregated metric value"""
    timestamp: float
    mean: float
    min: float
    max: float
    count: int

class MetricType(Enum):
    """Types of metrics supported"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class MetricConfig:
    """Configuration for a metric"""
    name: str
    description: str
    type: MetricType
    labels: Optional[List[str]] = None
    buckets: Optional[List[float]] = None
    quantiles: Optional[List[float]] = None
    optimization_config: Optional['MetricOptimizationConfig'] = None

@dataclass
class MetricOptimizationConfig:
    """Configuration for metric optimization"""
    max_cardinality: int = 1000
    aggregation_interval: int = 300  # 5 minutes
    retention_hours: int = 24
    enable_trend_analysis: bool = True
    trend_window: int = 24  # hours
    seasonality_window: int = 24  # periods

class MetricsManager:
    """Centralized metrics management system"""
    
    def __init__(self, optimization_config: Optional[MetricOptimizationConfig] = None):
        """Initialize metrics manager"""
        self._metrics: Dict[str, Union[Counter, Gauge, Histogram, Summary]] = {}
        self._metric_configs: Dict[str, MetricConfig] = {}
        self._metric_values: Dict[str, List[float]] = {}
        
        # Optimization configuration
        self._optimization_config = optimization_config or MetricOptimizationConfig()
        
        # Storage for raw and aggregated metrics
        self._raw_metrics: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self._aggregated_metrics: Dict[str, List[Tuple[float, AggregatedValue]]] = defaultdict(list)
        self._label_sets: Dict[str, Set[Tuple]] = defaultdict(set)
        
        # Default buckets for histograms
        self._default_buckets = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        
        # Default quantiles for summaries
        self._default_quantiles = [0.5, 0.75, 0.9, 0.95, 0.99]
        
        # Initialize system metrics
        self._setup_system_metrics()
    
    def _setup_system_metrics(self):
        """Setup internal system metrics"""
        # Cardinality metrics
        self.cardinality = cast(Gauge, self.create_metric(MetricConfig(
            name="metric_cardinality",
            description="Number of unique label combinations",
            type=MetricType.GAUGE,
            labels=['metric_name']
        )))
        
        # Storage metrics
        self.storage_size = cast(Gauge, self.create_metric(MetricConfig(
            name="metric_storage_bytes",
            description="Estimated storage size of metrics",
            type=MetricType.GAUGE,
            labels=['metric_name', 'type']
        )))
        
        # Trend metrics
        self.trend_strength = cast(Gauge, self.create_metric(MetricConfig(
            name="metric_trend_strength",
            description="Strength of trend in metric",
            type=MetricType.GAUGE,
            labels=['metric_name']
        )))
        
        self.seasonality_strength = cast(Gauge, self.create_metric(MetricConfig(
            name="metric_seasonality_strength",
            description="Strength of seasonal patterns",
            type=MetricType.GAUGE,
            labels=['metric_name']
        )))
        
        # Forecast metrics
        self.forecast_error = cast(Gauge, self.create_metric(MetricConfig(
            name="metric_forecast_error",
            description="Forecast error percentage",
            type=MetricType.GAUGE,
            labels=['metric_name']
        )))
    
    def configure_optimization(self, config: MetricOptimizationConfig) -> None:
        """Configure optimization settings for all metrics
        
        Args:
            config: Optimization configuration
        """
        self._optimization_config = config
        for metric_config in self._metric_configs.values():
            metric_config.optimization_config = config
    
    def create_metric(
        self,
        config: MetricConfig
    ) -> Union[Counter, Gauge, Histogram, Summary]:
        """Create a new metric
        
        Args:
            config: Metric configuration
            
        Returns:
            Created metric object
        """
        if config.name in self._metrics:
            return self._metrics[config.name]
        
        metric: Union[Counter, Gauge, Histogram, Summary]
        
        if config.type == MetricType.COUNTER:
            metric = Counter(
                config.name,
                config.description,
                labelnames=config.labels or []
            )
        elif config.type == MetricType.GAUGE:
            metric = Gauge(
                config.name,
                config.description,
                labelnames=config.labels or []
            )
        elif config.type == MetricType.HISTOGRAM:
            metric = Histogram(
                config.name,
                config.description,
                labelnames=config.labels or [],
                buckets=config.buckets or self._default_buckets
            )
        else:  # Summary
            metric = Summary(
                config.name,
                config.description,
                labelnames=config.labels or []
            )
        
        self._metrics[config.name] = metric
        self._metric_configs[config.name] = config
        self._metric_values[config.name] = []
        
        # Set optimization config
        config.optimization_config = self._optimization_config
        
        return metric
    
    def _update_metric_value(
        self,
        metric: Union[Counter, Gauge, Histogram, Summary],
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Update a metric value based on its type"""
        if isinstance(metric, Counter):
            if labels:
                metric.labels(**labels).inc(value)
            else:
                metric.inc(value)
        elif isinstance(metric, Gauge):
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)
        else:  # Histogram or Summary
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)
    
    async def record_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> bool:
        """Record a metric value with optimization
        
        Args:
            name: Metric name
            value: Metric value
            labels: Optional metric labels
            
        Returns:
            True if recorded, False if dropped
        """
        try:
            # Convert labels to tuple for hashing
            label_tuple = tuple(sorted(labels.items())) if labels else ()
            
            # Check cardinality
            if len(self._label_sets[name]) >= self._optimization_config.max_cardinality:
                if label_tuple not in self._label_sets[name]:
                    return False
            
            # Record metric
            timestamp = time.time()
            metric_key = f"{name}_{label_tuple}"
            self._raw_metrics[metric_key].append((timestamp, value))
            self._label_sets[name].add(label_tuple)
            
            # Update metric value storage
            self._metric_values[name].append(value)
            
            # Update cardinality metric
            if isinstance(self.cardinality, Gauge):
                self.cardinality.labels(metric_name=name).set(
                    len(self._label_sets[name])
                )
            
            # Update storage metrics
            self._update_storage_size(name)
            
            # Record in Prometheus metric
            metric = self._metrics.get(name)
            if metric:
                self._update_metric_value(metric, value, labels)
            
            return True
            
        except Exception as e:
            logger.error("Error recording metric",
                        metric_name=name,
                        error=str(e))
            return False
    
    async def aggregate_metrics(self):
        """Aggregate metrics to reduce storage"""
        try:
            current_time = time.time()
            aggregation_start = current_time - self._optimization_config.aggregation_interval
            
            for metric_key, samples in self._raw_metrics.items():
                # Filter samples in aggregation window
                window_samples = [
                    (ts, val) for ts, val in samples
                    if ts >= aggregation_start
                ]
                
                if window_samples:
                    # Calculate aggregates
                    timestamps = np.array([ts for ts, _ in window_samples])
                    values = np.array([val for _, val in window_samples])
                    
                    aggregated = AggregatedValue(
                        timestamp=float(np.mean(timestamps)),
                        mean=float(np.mean(values)),
                        min=float(np.min(values)),
                        max=float(np.max(values)),
                        count=len(values)
                    )
                    
                    # Store aggregated data
                    self._aggregated_metrics[metric_key].append(
                        (current_time, aggregated)
                    )
            
            # Clear raw metrics
            self._raw_metrics.clear()
            
        except Exception as e:
            logger.error("Error aggregating metrics", error=str(e))
    
    async def analyze_trend(
        self,
        metric_name: str,
        window_size: Optional[int] = None
    ) -> Dict[str, float]:
        """Analyze trend components
        
        Args:
            metric_name: Name of metric to analyze
            window_size: Optional analysis window size
            
        Returns:
            Dictionary with trend analysis results
        """
        try:
            values = self._metric_values.get(metric_name, [])
            if len(values) < 2:
                return {}
            
            # Use specified or default window size
            window = window_size or self._optimization_config.trend_window
            values_array = np.array(values[-window:])
            
            # Perform seasonal decomposition
            result = seasonal_decompose(
                values_array,
                period=min(len(values_array) // 2, self._optimization_config.seasonality_window),
                extrapolate_trend='freq'
            )
            
            # Calculate trend strength
            trend_strength = float(1 - np.var(result.resid) / np.var(values_array - result.seasonal))
            if isinstance(self.trend_strength, Gauge):
                self.trend_strength.labels(metric_name=metric_name).set(trend_strength)
            
            # Calculate seasonality strength
            seasonality_strength = float(1 - np.var(result.resid) / np.var(values_array - result.trend))
            if isinstance(self.seasonality_strength, Gauge):
                self.seasonality_strength.labels(metric_name=metric_name).set(seasonality_strength)
            
            return {
                'trend_strength': trend_strength,
                'seasonality_strength': seasonality_strength,
                'trend_direction': float(np.mean(np.diff(result.trend))),
                'last_trend_value': float(result.trend[-1]),
                'last_seasonal_value': float(result.seasonal[-1])
            }
            
        except Exception as e:
            logger.error("Error analyzing trend",
                        metric_name=metric_name,
                        error=str(e))
            return {}
    
    async def forecast_metric(
        self,
        metric_name: str,
        horizon: int = 24
    ) -> Dict[str, List[float]]:
        """Generate metric forecasts
        
        Args:
            metric_name: Name of metric to forecast
            horizon: Forecast horizon
            
        Returns:
            Dictionary with forecast results
        """
        try:
            values = self._metric_values.get(metric_name, [])
            if len(values) < horizon:
                return {}
            
            # Fit model
            model = ExponentialSmoothing(
                values,
                seasonal_periods=min(len(values) // 2, self._optimization_config.seasonality_window),
                trend='add',
                seasonal='add'
            ).fit()
            
            # Generate forecast
            forecast = model.forecast(horizon)
            
            # Calculate forecast error
            mape = np.mean(np.abs(model.resid / values)) * 100
            if isinstance(self.forecast_error, Gauge):
                self.forecast_error.labels(metric_name=metric_name).set(mape)
            
            return {
                'forecast': forecast.tolist(),
                'mape': mape,
                'lower_bound': (forecast - 2 * model.resid.std()).tolist(),
                'upper_bound': (forecast + 2 * model.resid.std()).tolist()
            }
            
        except Exception as e:
            logger.error("Error generating forecast",
                        metric_name=metric_name,
                        error=str(e))
            return {}
    
    async def detect_anomalies(
        self,
        metric_name: str,
        window_size: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """Detect metric anomalies
        
        Args:
            metric_name: Name of metric to analyze
            window_size: Optional detection window size
            
        Returns:
            List of (index, score) tuples for anomalies
        """
        try:
            values = self._metric_values.get(metric_name, [])
            window = window_size or self._optimization_config.trend_window
            
            if len(values) < window:
                return []
            
            values_array = np.array(values)
            anomalies = []
            
            # Calculate rolling statistics
            for i in range(window, len(values)):
                window_data = values_array[i-window:i]
                mean = np.mean(window_data)
                std = np.std(window_data)
                
                # Check if current value is anomalous
                z_score = (values_array[i] - mean) / std if std > 0 else 0
                if abs(z_score) > 3:  # 3 sigma rule
                    anomalies.append((i, z_score))
            
            return anomalies
            
        except Exception as e:
            logger.error("Error detecting anomalies",
                        metric_name=metric_name,
                        error=str(e))
            return []
    
    def get_metric(
        self,
        name: str
    ) -> Optional[Union[Counter, Gauge, Histogram, Summary]]:
        """Get an existing metric
        
        Args:
            name: Metric name
            
        Returns:
            Metric object if exists, None otherwise
        """
        return self._metrics.get(name)
    
    def get_statistics(
        self,
        name: str,
        window: Optional[timedelta] = None
    ) -> Dict[str, float]:
        """Get metric statistics
        
        Args:
            name: Metric name
            window: Optional time window
            
        Returns:
            Dictionary with metric statistics
        """
        values = self._metric_values.get(name, [])
        if not values:
            return {}
        
        values_array = np.array(values)
        return {
            'count': float(len(values)),
            'mean': float(np.mean(values_array)),
            'std': float(np.std(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'p50': float(np.percentile(values_array, 50)),
            'p90': float(np.percentile(values_array, 90)),
            'p95': float(np.percentile(values_array, 95)),
            'p99': float(np.percentile(values_array, 99))
        }
    
    def _update_storage_size(self, metric_name: str):
        """Update storage size metrics"""
        try:
            # Estimate raw metrics size
            raw_size = sum(
                len(str((ts, val))) for metric_key, samples in self._raw_metrics.items()
                if metric_key.startswith(metric_name)
                for ts, val in samples
            )
            
            # Estimate aggregated metrics size
            agg_size = sum(
                len(str((ts, val))) for metric_key, samples in self._aggregated_metrics.items()
                if metric_key.startswith(metric_name)
                for ts, val in samples
            )
            
            if isinstance(self.storage_size, Gauge):
                self.storage_size.labels(
                    metric_name=metric_name,
                    type='raw'
                ).set(raw_size)
                
                self.storage_size.labels(
                    metric_name=metric_name,
                    type='aggregated'
                ).set(agg_size)
            
        except Exception as e:
            logger.error("Error updating storage size",
                        metric_name=metric_name,
                        error=str(e))
    
    def clear_metrics(self, name: Optional[str] = None) -> None:
        """Clear metric values
        
        Args:
            name: Optional metric name to clear. If None, clears all metrics.
        """
        if name:
            if name in self._metric_values:
                self._metric_values[name] = []
        else:
            for metric_name in self._metric_values:
                self._metric_values[metric_name] = []

# Global metrics manager instance
metrics_manager = MetricsManager() 