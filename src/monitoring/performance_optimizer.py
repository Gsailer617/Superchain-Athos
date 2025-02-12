import time
from typing import Dict, List, Optional, Set, Tuple
import structlog
from prometheus_client import Counter, Gauge, Histogram
import numpy as np
from collections import defaultdict
import asyncio

logger = structlog.get_logger(__name__)

class PerformanceOptimizer:
    def __init__(
        self,
        max_cardinality: int = 1000,
        aggregation_interval: int = 300,  # 5 minutes
        retention_hours: int = 24
    ):
        self.max_cardinality = max_cardinality
        self.aggregation_interval = aggregation_interval
        self.retention_hours = retention_hours
        
        # Metric storage
        self.raw_metrics: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self.aggregated_metrics: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self.label_sets: Dict[str, Set[Tuple]] = defaultdict(set)
        
        # Performance metrics
        self.cardinality = Gauge(
            'metric_cardinality',
            'Number of unique label combinations',
            ['metric_name']
        )
        
        self.dropped_samples = Counter(
            'dropped_samples_total',
            'Number of samples dropped due to cardinality limits',
            ['metric_name']
        )
        
        self.aggregation_lag = Gauge(
            'aggregation_lag_seconds',
            'Time since last aggregation',
            ['metric_name']
        )
        
        self.storage_size = Gauge(
            'metric_storage_bytes',
            'Estimated storage size of metrics',
            ['metric_name', 'type']
        )

    async def record_metric(
        self,
        name: str,
        value: float,
        labels: Dict[str, str]
    ) -> bool:
        """
        Record a metric value with cardinality control.
        Returns True if recorded, False if dropped.
        """
        try:
            # Convert labels to tuple for hashing
            label_tuple = tuple(sorted(labels.items()))
            
            # Check cardinality
            if len(self.label_sets[name]) >= self.max_cardinality:
                if label_tuple not in self.label_sets[name]:
                    self.dropped_samples.labels(metric_name=name).inc()
                    return False
            
            # Record metric
            timestamp = time.time()
            metric_key = f"{name}_{label_tuple}"
            self.raw_metrics[metric_key].append((timestamp, value))
            self.label_sets[name].add(label_tuple)
            
            # Update cardinality metric
            self.cardinality.labels(metric_name=name).set(
                len(self.label_sets[name])
            )
            
            # Update storage size metric
            self._update_storage_size(name)
            
            return True
            
        except Exception as e:
            logger.error("Error recording metric",
                        metric_name=name,
                        error=str(e))
            return False

    async def aggregate_metrics(self):
        """Aggregate metrics to reduce storage and improve query performance"""
        try:
            current_time = time.time()
            aggregation_start = current_time - self.aggregation_interval
            
            for metric_key, samples in self.raw_metrics.items():
                # Filter samples in aggregation window
                window_samples = [
                    (ts, val) for ts, val in samples
                    if ts >= aggregation_start
                ]
                
                if window_samples:
                    # Calculate aggregates
                    timestamps = [ts for ts, _ in window_samples]
                    values = [val for _, val in window_samples]
                    
                    aggregated_value = {
                        'timestamp': np.mean(timestamps),
                        'mean': np.mean(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }
                    
                    # Store aggregated data
                    self.aggregated_metrics[metric_key].append(
                        (current_time, aggregated_value)
                    )
                    
                    # Update lag metric
                    metric_name = metric_key.split('_')[0]
                    self.aggregation_lag.labels(
                        metric_name=metric_name
                    ).set(current_time - aggregated_value['timestamp'])
            
            # Clear raw metrics
            self.raw_metrics.clear()
            
        except Exception as e:
            logger.error("Error aggregating metrics", error=str(e))

    async def cleanup_old_data(self):
        """Remove old data based on retention policy"""
        try:
            retention_threshold = time.time() - (self.retention_hours * 3600)
            
            # Clean aggregated metrics
            for metric_key in list(self.aggregated_metrics.keys()):
                self.aggregated_metrics[metric_key] = [
                    (ts, val) for ts, val in self.aggregated_metrics[metric_key]
                    if ts >= retention_threshold
                ]
                
                # Remove empty metrics
                if not self.aggregated_metrics[metric_key]:
                    del self.aggregated_metrics[metric_key]
                    metric_name = metric_key.split('_')[0]
                    self._cleanup_label_sets(metric_name)
            
            # Update storage metrics
            for metric_key in self.aggregated_metrics:
                metric_name = metric_key.split('_')[0]
                self._update_storage_size(metric_name)
                
        except Exception as e:
            logger.error("Error cleaning up old data", error=str(e))

    def get_metric_value(
        self,
        name: str,
        labels: Dict[str, str],
        aggregation: str = 'mean'
    ) -> Optional[float]:
        """Get the latest value for a metric"""
        try:
            label_tuple = tuple(sorted(labels.items()))
            metric_key = f"{name}_{label_tuple}"
            
            # Check raw metrics first
            if metric_key in self.raw_metrics and self.raw_metrics[metric_key]:
                return self.raw_metrics[metric_key][-1][1]
            
            # Check aggregated metrics
            if metric_key in self.aggregated_metrics and self.aggregated_metrics[metric_key]:
                latest = self.aggregated_metrics[metric_key][-1][1]
                return latest.get(aggregation, latest['mean'])
            
            return None
            
        except Exception as e:
            logger.error("Error getting metric value",
                        metric_name=name,
                        error=str(e))
            return None

    def _update_storage_size(self, metric_name: str):
        """Update storage size metrics"""
        try:
            # Estimate raw metrics size
            raw_size = sum(
                len(str((ts, val))) for metric_key, samples in self.raw_metrics.items()
                if metric_key.startswith(metric_name)
                for ts, val in samples
            )
            
            # Estimate aggregated metrics size
            agg_size = sum(
                len(str((ts, val))) for metric_key, samples in self.aggregated_metrics.items()
                if metric_key.startswith(metric_name)
                for ts, val in samples
            )
            
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

    def _cleanup_label_sets(self, metric_name: str):
        """Clean up unused label combinations"""
        try:
            active_labels = set()
            
            # Collect active label combinations
            for metric_key in self.aggregated_metrics:
                if metric_key.startswith(metric_name):
                    label_str = metric_key[len(metric_name) + 1:]
                    if label_str:
                        active_labels.add(eval(label_str))
            
            # Update label sets
            self.label_sets[metric_name] = active_labels
            self.cardinality.labels(metric_name=metric_name).set(len(active_labels))
            
        except Exception as e:
            logger.error("Error cleaning up label sets",
                        metric_name=metric_name,
                        error=str(e))

    async def start_optimization_tasks(self):
        """Start periodic optimization tasks"""
        while True:
            await self.aggregate_metrics()
            await self.cleanup_old_data()
            await asyncio.sleep(self.aggregation_interval) 