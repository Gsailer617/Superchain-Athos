import threading
from collections import deque
from typing import Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PerformanceTracker:
    """Centralized performance tracking for the arbitrage system"""
    
    def __init__(self, max_points: int = 1000):
        self.max_points = max_points
        self._lock = threading.Lock()
        
        # Real-time metrics
        self.metrics = {
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
        
        # Aggregate performance
        self.performance = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_profit': 0.0,
            'total_gas_spent': 0.0,
            'best_trade': None,
            'worst_trade': None
        }
        
        # Analytics
        self.token_analytics = {}
        self.dex_analytics = {}
        
    def update_metrics(self, metrics: Dict):
        """Thread-safe update of real-time metrics"""
        with self._lock:
            timestamp = datetime.now()
            self.metrics['timestamps'].append(timestamp)
            
            for key, value in metrics.items():
                if key in self.metrics:
                    self.metrics[key].append(value)
                    
    def record_trade(self, trade_data: Dict):
        """Record trade performance"""
        with self._lock:
            self.performance['total_trades'] += 1
            
            if trade_data.get('success', False):
                self.performance['successful_trades'] += 1
                profit = trade_data.get('profit', 0)
                self.performance['total_profit'] += profit
                
                if (self.performance['best_trade'] is None or 
                    profit > self.performance['best_trade']['profit']):
                    self.performance['best_trade'] = trade_data
                    
                if (self.performance['worst_trade'] is None or 
                    profit < self.performance['worst_trade']['profit']):
                    self.performance['worst_trade'] = trade_data
            else:
                self.performance['failed_trades'] += 1
            
            self.performance['total_gas_spent'] += trade_data.get('gas_cost', 0)
            
    def update_token_analytics(self, token: str, data: Dict):
        """Update token-specific analytics"""
        with self._lock:
            if token not in self.token_analytics:
                self.token_analytics[token] = {
                    'trades': 0,
                    'volume': 0,
                    'profit': 0,
                    'success_rate': 0
                }
            
            stats = self.token_analytics[token]
            stats['trades'] += 1
            stats['volume'] += data.get('volume', 0)
            stats['profit'] += data.get('profit', 0)
            stats['success_rate'] = (
                stats.get('success_rate', 0) * (stats['trades'] - 1) + 
                int(data.get('success', False))
            ) / stats['trades']
            
    def update_dex_analytics(self, dex: str, data: Dict):
        """Update DEX-specific analytics"""
        with self._lock:
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
            stats['volume'] += data.get('volume', 0)
            stats['profit'] += data.get('profit', 0)
            stats['success_rate'] = (
                stats.get('success_rate', 0) * (stats['trades'] - 1) + 
                int(data.get('success', False))
            ) / stats['trades']
            stats['avg_gas'] = (
                stats.get('avg_gas', 0) * (stats['trades'] - 1) + 
                data.get('gas_cost', 0)
            ) / stats['trades']
            
    def get_performance_summary(self) -> Dict:
        """Get current performance summary"""
        with self._lock:
            return {
                'total_trades': self.performance['total_trades'],
                'success_rate': (
                    self.performance['successful_trades'] / 
                    max(self.performance['total_trades'], 1)
                ) * 100,
                'total_profit': self.performance['total_profit'],
                'total_gas_spent': self.performance['total_gas_spent'],
                'net_profit': (
                    self.performance['total_profit'] - 
                    self.performance['total_gas_spent']
                )
            } 