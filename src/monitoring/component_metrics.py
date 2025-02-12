import time
from typing import Dict, List, Optional
import structlog
from prometheus_client import Counter, Gauge, Histogram
from functools import wraps

logger = structlog.get_logger(__name__)

class ComponentMetrics:
    def __init__(self):
        # Strategy-specific metrics
        self.strategy_execution_time = Histogram(
            'strategy_execution_time_detailed',
            'Detailed execution time breakdown for strategies',
            ['strategy', 'phase'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
        )
        
        self.strategy_memory_usage = Gauge(
            'strategy_memory_usage_bytes',
            'Memory usage by strategy components',
            ['strategy', 'component']
        )
        
        self.strategy_cache_stats = Counter(
            'strategy_cache_operations_total',
            'Cache operation statistics by strategy',
            ['strategy', 'operation', 'status']
        )
        
        # Smart contract interaction metrics
        self.contract_call_duration = Histogram(
            'contract_call_duration_seconds',
            'Duration of smart contract calls',
            ['contract', 'method', 'status'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        self.contract_gas_usage = Histogram(
            'contract_gas_usage',
            'Gas usage by contract calls',
            ['contract', 'method'],
            buckets=[50000, 100000, 200000, 500000, 1000000]
        )
        
        # Price feed metrics
        self.price_update_interval = Histogram(
            'price_update_interval_seconds',
            'Time between price updates',
            ['pair', 'source'],
            buckets=[1.0, 5.0, 15.0, 30.0, 60.0]
        )
        
        self.price_deviation = Histogram(
            'price_deviation_percent',
            'Price deviation between updates',
            ['pair', 'source'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        # Memory pool metrics
        self.mempool_transaction_count = Gauge(
            'mempool_transactions',
            'Number of transactions in mempool by type',
            ['type']
        )
        
        self.mempool_gas_prices = Histogram(
            'mempool_gas_prices_gwei',
            'Distribution of gas prices in mempool',
            buckets=[10, 20, 50, 100, 200, 500]
        )

    def track_strategy_execution(self, strategy: str):
        """Decorator to track strategy execution time by phase"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    # Track pre-execution phase
                    self.strategy_execution_time.labels(
                        strategy=strategy,
                        phase='pre_execution'
                    ).observe(time.time() - start_time)
                    
                    # Execute strategy
                    result = await func(*args, **kwargs)
                    
                    # Track main execution phase
                    self.strategy_execution_time.labels(
                        strategy=strategy,
                        phase='execution'
                    ).observe(time.time() - start_time)
                    
                    return result
                    
                finally:
                    # Track total execution time
                    self.strategy_execution_time.labels(
                        strategy=strategy,
                        phase='total'
                    ).observe(time.time() - start_time)
            
            return wrapper
        return decorator

    async def record_contract_call(
        self,
        contract: str,
        method: str,
        duration: float,
        gas_used: int,
        status: str
    ):
        """Record metrics for a smart contract call"""
        self.contract_call_duration.labels(
            contract=contract,
            method=method,
            status=status
        ).observe(duration)
        
        self.contract_gas_usage.labels(
            contract=contract,
            method=method
        ).observe(gas_used)

    async def record_price_update(
        self,
        pair: str,
        source: str,
        interval: float,
        deviation: float
    ):
        """Record metrics for price feed updates"""
        self.price_update_interval.labels(
            pair=pair,
            source=source
        ).observe(interval)
        
        self.price_deviation.labels(
            pair=pair,
            source=source
        ).observe(deviation)

    async def update_mempool_metrics(
        self,
        transactions: Dict[str, List[Dict]],
        gas_prices: List[int]
    ):
        """Update mempool metrics"""
        # Update transaction counts
        for tx_type, txs in transactions.items():
            self.mempool_transaction_count.labels(
                type=tx_type
            ).set(len(txs))
        
        # Record gas prices
        for gas_price in gas_prices:
            self.mempool_gas_prices.observe(gas_price)

    async def record_cache_operation(
        self,
        strategy: str,
        operation: str,
        status: str
    ):
        """Record cache operation metrics"""
        self.strategy_cache_stats.labels(
            strategy=strategy,
            operation=operation,
            status=status
        ).inc()

    async def update_strategy_memory(
        self,
        strategy: str,
        component: str,
        bytes_used: int
    ):
        """Update strategy memory usage metrics"""
        self.strategy_memory_usage.labels(
            strategy=strategy,
            component=component
        ).set(bytes_used) 