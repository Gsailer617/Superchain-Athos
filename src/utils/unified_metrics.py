"""
Unified metrics system combining Prometheus metrics, performance tracking, and monitoring
"""

from typing import Dict, List, Any, Optional, Union, TypeVar, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import threading
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, REGISTRY
import logging
import structlog
from pathlib import Path

logger = structlog.get_logger(__name__)

T = TypeVar('T')

@dataclass
class MetricConfig:
    """Configuration for a metric"""
    name: str
    description: str
    type: str = "gauge"  # gauge, counter, histogram
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None

@dataclass
class PerformanceMetrics:
    """Performance tracking metrics"""
    total_trades: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    total_profit: float = 0.0
    total_gas_spent: float = 0.0
    best_trade: Optional[Dict[str, Any]] = None
    worst_trade: Optional[Dict[str, Any]] = None

class UnifiedMetricsSystem:
    """Unified system for metrics, performance tracking, and monitoring"""
    
    def __init__(
        self,
        registry: Optional[CollectorRegistry] = None,
        max_points: int = 1000,
        enable_prometheus: bool = True
    ):
        """Initialize unified metrics system"""
        self._registry = registry or REGISTRY
        self._lock = threading.Lock()
        self.max_points = max_points
        self.enable_prometheus = enable_prometheus
        
        # Initialize components
        self._metrics: Dict[str, Union[Counter, Gauge, Histogram]] = {}
        self._initialize_metrics()
        
        # Real-time tracking
        self.tracking_metrics = {
            'timestamps': deque(maxlen=max_points),
            'profits': deque(maxlen=max_points),
            'confidence_scores': deque(maxlen=max_points),
            'risk_scores': deque(maxlen=max_points),
            'gas_prices': deque(maxlen=max_points),
            'volumes': deque(maxlen=max_points),
            'slippage': deque(maxlen=max_points),
            'price_impact': deque(maxlen=max_points),
            'execution_times': deque(maxlen=max_points)
        }
        
        # Performance tracking
        self.performance = PerformanceMetrics()
        
        # Analytics storage
        self.token_analytics: Dict[str, Dict[str, Any]] = {}
        self.dex_analytics: Dict[str, Dict[str, Any]] = {}
        
    def _initialize_metrics(self):
        """Initialize default Prometheus metrics"""
        if not self.enable_prometheus:
            return
            
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
        
        # Performance metrics
        self._add_metric(MetricConfig(
            name="trade_success_rate",
            description="Trade success rate percentage",
            type="gauge"
        ))
        self._add_metric(MetricConfig(
            name="total_profit",
            description="Total profit in USD",
            type="gauge"
        ))
        
        # System metrics
        self._add_metric(MetricConfig(
            name="system_resource_usage",
            description="System resource usage percentage",
            type="gauge",
            labels=["resource"]
        ))
        
    def _add_metric(self, config: MetricConfig):
        """Add a new Prometheus metric"""
        if not self.enable_prometheus:
            return
            
        name = f"defi_{config.name}"
        
        if config.type == "gauge":
            metric = Gauge(
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
        
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update real-time metrics"""
        with self._lock:
            timestamp = datetime.now()
            self.tracking_metrics['timestamps'].append(timestamp)
            
            for key, value in metrics.items():
                if key in self.tracking_metrics:
                    self.tracking_metrics[key].append(value)
                    
            # Update Prometheus metrics if enabled
            if self.enable_prometheus:
                self._update_prometheus_metrics(metrics)
                
    def record_trade(self, trade_data: Dict[str, Any]):
        """Record trade performance"""
        with self._lock:
            self.performance.total_trades += 1
            
            if trade_data.get('success', False):
                self.performance.successful_trades += 1
                profit = trade_data.get('profit', 0)
                self.performance.total_profit += profit
                
                if (self.performance.best_trade is None or 
                    profit > self.performance.best_trade.get('profit', float('-inf'))):
                    self.performance.best_trade = trade_data
                    
                if (self.performance.worst_trade is None or 
                    profit < self.performance.worst_trade.get('profit', float('inf'))):
                    self.performance.worst_trade = trade_data
            else:
                self.performance.failed_trades += 1
                
            self.performance.total_gas_spent += trade_data.get('gas_cost', 0)
            
            # Update analytics
            self._update_token_analytics(trade_data)
            self._update_dex_analytics(trade_data)
            
    def _update_token_analytics(self, trade_data: Dict[str, Any]):
        """Update token-specific analytics"""
        token = trade_data.get('token', '')
        if not token:
            return
            
        if token not in self.token_analytics:
            self.token_analytics[token] = {
                'trades': 0,
                'volume': 0,
                'profit': 0,
                'success_rate': 0
            }
            
        stats = self.token_analytics[token]
        stats['trades'] += 1
        stats['volume'] += trade_data.get('volume', 0)
        stats['profit'] += trade_data.get('profit', 0)
        stats['success_rate'] = (
            stats.get('success_rate', 0) * (stats['trades'] - 1) + 
            int(trade_data.get('success', False))
        ) / stats['trades']
        
    def _update_dex_analytics(self, trade_data: Dict[str, Any]):
        """Update DEX-specific analytics"""
        dex = trade_data.get('dex', '')
        if not dex:
            return
            
        if dex not in self.dex_analytics:
            self.dex_analytics[dex] = {
                'trades': 0,
                'volume': 0,
                'profit': 0,
                'success_rate': 0,
                'avg_gas': 0
            }
            
        stats = self.dex_analytics[dex]
        stats['trades'] += 1
        stats['volume'] += trade_data.get('volume', 0)
        stats['profit'] += trade_data.get('profit', 0)
        stats['success_rate'] = (
            stats.get('success_rate', 0) * (stats['trades'] - 1) + 
            int(trade_data.get('success', False))
        ) / stats['trades']
        stats['avg_gas'] = (
            stats.get('avg_gas', 0) * (stats['trades'] - 1) + 
            trade_data.get('gas_cost', 0)
        ) / stats['trades']
        
    def _update_prometheus_metrics(self, metrics: Dict[str, Any]):
        """Update Prometheus metrics"""
        if not self.enable_prometheus:
            return
            
        try:
            # Update success rate
            if 'trade_success_rate' in self._metrics:
                success_rate = (
                    self.performance.successful_trades / 
                    max(self.performance.total_trades, 1)
                ) * 100
                self._metrics['trade_success_rate'].set(success_rate)
                
            # Update total profit
            if 'total_profit' in self._metrics:
                self._metrics['total_profit'].set(self.performance.total_profit)
                
            # Update other metrics
            for name, value in metrics.items():
                if name in self._metrics:
                    metric = self._metrics[name]
                    if isinstance(metric, Gauge):
                        metric.set(value)
                    elif isinstance(metric, Counter):
                        metric.inc(value)
                    elif isinstance(metric, Histogram):
                        metric.observe(value)
                        
        except Exception as e:
            logger.error(f"Error updating Prometheus metrics: {str(e)}")
            
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary"""
        with self._lock:
            return {
                'total_trades': self.performance.total_trades,
                'success_rate': (
                    self.performance.successful_trades / 
                    max(self.performance.total_trades, 1)
                ) * 100,
                'total_profit': self.performance.total_profit,
                'total_gas_spent': self.performance.total_gas_spent,
                'net_profit': (
                    self.performance.total_profit - 
                    self.performance.total_gas_spent
                ),
                'best_trade': self.performance.best_trade,
                'worst_trade': self.performance.worst_trade
            }
            
    def get_token_analytics(self, token: Optional[str] = None) -> Dict[str, Any]:
        """Get token analytics"""
        with self._lock:
            if token:
                return self.token_analytics.get(token, {})
            return self.token_analytics
            
    def get_dex_analytics(self, dex: Optional[str] = None) -> Dict[str, Any]:
        """Get DEX analytics"""
        with self._lock:
            if dex:
                return self.dex_analytics.get(dex, {})
            return self.dex_analytics
            
    def get_tracking_metrics(self) -> Dict[str, Any]:
        """Get current tracking metrics"""
        with self._lock:
            return {
                key: list(value) for key, value in self.tracking_metrics.items()
            } 