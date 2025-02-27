"""
Unified Metrics System

A centralized metrics collection and reporting system that integrates:
- Performance metrics
- Risk metrics
- Bridge/transaction metrics
- Market metrics
- Execution metrics
with standardized export to Prometheus, logging, and dashboard integrations.
"""

from typing import Dict, Any, List, Optional, Union, Callable
import time
import logging
import threading
import json
from enum import Enum
from dataclasses import dataclass, field, asdict
from prometheus_client import Counter, Gauge, Histogram, Summary, CollectorRegistry, push_to_gateway
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

# Registry for Prometheus metrics
REGISTRY = CollectorRegistry()

class MetricType(Enum):
    """Types of metrics that can be tracked"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    COMPOSITE = "composite"

@dataclass
class MetricDefinition:
    """Definition of a metric to be tracked"""
    name: str
    description: str
    type: MetricType
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms
    percentiles: Optional[List[float]] = None  # For summaries
    unit: str = ""
    aggregation: str = "last"  # last, sum, avg, min, max

@dataclass
class MetricValue:
    """Value of a metric at a point in time"""
    name: str
    value: Union[float, int]
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

class MetricsManager:
    """Centralized metrics management system"""
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'MetricsManager':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = MetricsManager()
        return cls._instance
    
    def __init__(self):
        """Initialize metrics manager"""
        self.metrics_registry: Dict[str, MetricDefinition] = {}
        self.prometheus_metrics: Dict[str, Any] = {}
        self.time_series_data: Dict[str, List[MetricValue]] = {}
        self.callbacks: Dict[str, List[Callable]] = {}
        self.retention_limit = 10000  # Max points per metric
        self.export_interval = 60  # Seconds between automated exports
        self.lock = threading.RLock()
        self._setup_exporters()
    
    def _setup_exporters(self):
        """Set up automated metric exporters"""
        self.exporters = {
            "prometheus": {
                "enabled": False,
                "gateway": "localhost:9091",
                "job": "flashing_base"
            },
            "csv": {
                "enabled": False,
                "path": "./metrics",
                "interval": 3600  # Export hourly
            },
            "dashboard": {
                "enabled": True,
                "endpoint": "/api/metrics"
            }
        }
        
        # Start export thread if any exporters enabled
        if any(e["enabled"] for e in self.exporters.values()):
            self._start_export_thread()
    
    def _start_export_thread(self):
        """Start background thread for metric export"""
        def export_loop():
            while True:
                try:
                    time.sleep(self.export_interval)
                    self.export_metrics()
                except Exception as e:
                    logger.error(f"Error in metrics export: {str(e)}")
        
        thread = threading.Thread(target=export_loop, daemon=True)
        thread.start()
    
    def register_metric(self, definition: MetricDefinition):
        """Register a new metric"""
        with self.lock:
            # Store definition
            self.metrics_registry[definition.name] = definition
            
            # Initialize time series storage
            if definition.name not in self.time_series_data:
                self.time_series_data[definition.name] = []
            
            # Create Prometheus metric
            if definition.type == MetricType.COUNTER:
                self.prometheus_metrics[definition.name] = Counter(
                    definition.name,
                    definition.description,
                    definition.labels,
                    registry=REGISTRY
                )
            elif definition.type == MetricType.GAUGE:
                self.prometheus_metrics[definition.name] = Gauge(
                    definition.name,
                    definition.description,
                    definition.labels,
                    registry=REGISTRY
                )
            elif definition.type == MetricType.HISTOGRAM:
                self.prometheus_metrics[definition.name] = Histogram(
                    definition.name,
                    definition.description,
                    definition.labels,
                    buckets=definition.buckets or Histogram.DEFAULT_BUCKETS,
                    registry=REGISTRY
                )
            elif definition.type == MetricType.SUMMARY:
                self.prometheus_metrics[definition.name] = Summary(
                    definition.name,
                    definition.description,
                    definition.labels,
                    registry=REGISTRY
                )
        
    def update_metric(self, name: str, value: Union[float, int], labels: Dict[str, str] = None):
        """Update a metric value"""
        with self.lock:
            if name not in self.metrics_registry:
                logger.warning(f"Updating unregistered metric: {name}")
            return
            
            labels = labels or {}
            definition = self.metrics_registry[name]
            prometheus_metric = self.prometheus_metrics.get(name)
            
            # Store in time series
            metric_value = MetricValue(
                name=name,
                value=value,
                labels=labels,
                timestamp=time.time()
            )
            self.time_series_data[name].append(metric_value)
            
            # Enforce retention limit
            if len(self.time_series_data[name]) > self.retention_limit:
                self.time_series_data[name] = self.time_series_data[name][-self.retention_limit:]
            
            # Update Prometheus metric
            if prometheus_metric:
                if definition.type == MetricType.COUNTER:
                    # For counters, we increment by the value
                    if labels:
                        prometheus_metric.labels(**labels).inc(value)
                    else:
                        prometheus_metric.inc(value)
                elif definition.type == MetricType.GAUGE:
                    # For gauges, we set the value
                    if labels:
                        prometheus_metric.labels(**labels).set(value)
                    else:
                        prometheus_metric.set(value)
                elif definition.type == MetricType.HISTOGRAM:
                    # For histograms, we observe the value
                    if labels:
                        prometheus_metric.labels(**labels).observe(value)
                    else:
                        prometheus_metric.observe(value)
                elif definition.type == MetricType.SUMMARY:
                    # For summaries, we observe the value
                    if labels:
                        prometheus_metric.labels(**labels).observe(value)
                    else:
                        prometheus_metric.observe(value)
            
            # Notify callbacks
            if name in self.callbacks:
                for callback in self.callbacks[name]:
                    try:
                        callback(name, value, labels)
                    except Exception as e:
                        logger.error(f"Error in metric callback for {name}: {str(e)}")
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Increment a counter metric"""
        self.update_metric(name, value, labels)
            
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric"""
        self.update_metric(name, value, labels)
    
    def observe_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Observe a value for a histogram metric"""
        self.update_metric(name, value, labels)
    
    def observe_summary(self, name: str, value: float, labels: Dict[str, str] = None):
        """Observe a value for a summary metric"""
        self.update_metric(name, value, labels)
    
    def get_metric_values(
        self, 
        name: str, 
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        labels: Dict[str, str] = None
    ) -> List[MetricValue]:
        """Get time series values for a metric"""
        with self.lock:
            if name not in self.time_series_data:
                return []
            
            values = self.time_series_data[name]
            
            # Filter by time range
            if start_time is not None:
                values = [v for v in values if v.timestamp >= start_time]
            if end_time is not None:
                values = [v for v in values if v.timestamp <= end_time]
            
            # Filter by labels
            if labels:
                values = [
                    v for v in values if all(
                        k in v.labels and v.labels[k] == labels[k]
                        for k in labels
                    )
                ]
            
            return values
    
    def register_callback(self, metric_name: str, callback: Callable):
        """Register a callback for metric updates"""
        with self.lock:
            if metric_name not in self.callbacks:
                self.callbacks[metric_name] = []
            self.callbacks[metric_name].append(callback)
    
    def export_metrics(self):
        """Export metrics to configured exporters"""
        with self.lock:
            # Export to Prometheus if enabled
            if self.exporters["prometheus"]["enabled"]:
                push_to_gateway(
                    self.exporters["prometheus"]["gateway"],
                    job=self.exporters["prometheus"]["job"],
                    registry=REGISTRY
                )
                
            # Export to CSV if enabled
            if self.exporters["csv"]["enabled"]:
                self._export_to_csv()
    
    def _export_to_csv(self):
        """Export metrics to CSV files"""
        import os
        from pathlib import Path
        
        export_path = Path(self.exporters["csv"]["path"])
        os.makedirs(export_path, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for name, values in self.time_series_data.items():
            if not values:
                continue
                
            # Convert to DataFrame
            data = []
            for v in values:
                row = {
                    "timestamp": v.timestamp,
                    "value": v.value
                }
                row.update(v.labels)
                data.append(row)
            
            df = pd.DataFrame(data)
            
            # Save to CSV
            file_path = export_path / f"{name}_{timestamp}.csv"
            df.to_csv(file_path, index=False)
    
    def get_metric_stats(
        self,
        name: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        labels: Dict[str, str] = None
    ) -> Dict[str, float]:
        """Get statistical summary of a metric"""
        values = self.get_metric_values(name, start_time, end_time, labels)
        
        if not values:
            return {
                "count": 0,
                "sum": 0.0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "stddev": 0.0
            }
        
        raw_values = [v.value for v in values]
        
            return {
            "count": len(raw_values),
            "sum": sum(raw_values),
            "mean": np.mean(raw_values),
            "min": min(raw_values),
            "max": max(raw_values),
            "stddev": np.std(raw_values)
        }

# ---- Integration Points ----

# Bridge metrics
def register_bridge_metrics():
    """Register common bridge metrics"""
    manager = MetricsManager.get_instance()
    
    # Transaction metrics
    manager.register_metric(MetricDefinition(
        name="bridge_txs_total",
        description="Total number of bridge transactions",
        type=MetricType.COUNTER,
        labels=["bridge", "source_chain", "target_chain", "token", "status"]
    ))
    
    # Gas metrics
    manager.register_metric(MetricDefinition(
        name="bridge_gas_cost",
        description="Gas cost for bridge transactions",
        type=MetricType.HISTOGRAM,
        labels=["bridge", "source_chain", "target_chain"],
        buckets=[10, 50, 100, 200, 500, 1000, 2000, 5000]
    ))
    
    # Time metrics
    manager.register_metric(MetricDefinition(
        name="bridge_time_seconds",
        description="Time for bridge transactions in seconds",
        type=MetricType.HISTOGRAM,
        labels=["bridge", "source_chain", "target_chain"],
        buckets=[60, 300, 600, 1800, 3600, 7200, 14400]
    ))
    
    # Success rate
    manager.register_metric(MetricDefinition(
        name="bridge_success_rate",
        description="Success rate for bridge transactions",
        type=MetricType.GAUGE,
        labels=["bridge", "source_chain", "target_chain"]
    ))
    
    # Liquidity
    manager.register_metric(MetricDefinition(
        name="bridge_liquidity",
        description="Bridge liquidity",
        type=MetricType.GAUGE,
        labels=["bridge", "chain", "token"]
    ))

# Market metrics
def register_market_metrics():
    """Register common market metrics"""
    manager = MetricsManager.get_instance()
    
    # Price metrics
    manager.register_metric(MetricDefinition(
        name="token_price_usd",
        description="Token price in USD",
        type=MetricType.GAUGE,
        labels=["token", "chain"]
    ))
    
    # Volatility metrics
    manager.register_metric(MetricDefinition(
        name="token_volatility",
        description="Token price volatility",
        type=MetricType.GAUGE,
        labels=["token", "chain", "timeframe"]
    ))
    
    # Liquidity metrics
    manager.register_metric(MetricDefinition(
        name="market_liquidity",
        description="Market liquidity",
        type=MetricType.GAUGE,
        labels=["token", "chain", "dex"]
    ))

# Risk metrics
def register_risk_metrics():
    """Register common risk metrics"""
    manager = MetricsManager.get_instance()
    
    # Overall risk
    manager.register_metric(MetricDefinition(
        name="overall_risk_score",
        description="Overall risk score",
        type=MetricType.GAUGE,
        labels=["component", "entity"]
    ))
    
    # Component risks
    manager.register_metric(MetricDefinition(
        name="component_risk_score",
        description="Component risk score",
        type=MetricType.GAUGE,
        labels=["component", "entity", "risk_type"]
    ))
    
    # Risk events
    manager.register_metric(MetricDefinition(
        name="risk_events_total",
        description="Total risk events",
        type=MetricType.COUNTER,
        labels=["level", "type", "entity"]
    ))

# Initialize all standard metrics
def initialize_metrics():
    """Initialize all standard metrics"""
    register_bridge_metrics()
    register_market_metrics()
    register_risk_metrics()

# Get a shared metrics manager instance
def get_metrics_manager() -> MetricsManager:
    """Get the shared metrics manager instance"""
    return MetricsManager.get_instance() 