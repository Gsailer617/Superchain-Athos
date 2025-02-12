"""
Gas Optimization Module

This module provides gas optimization strategies and analysis:
- Gas price estimation
- Historical analysis
- Optimization strategies
- Integration with monitoring
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any, cast
import structlog
from web3 import Web3
from web3.types import TxParams, Wei, BlockData
from prometheus_client import Histogram, Gauge, Counter
from ..utils.cache import cache
from datetime import datetime, timedelta
from hexbytes import HexBytes

logger = structlog.get_logger(__name__)

class GasOptimizer:
    def __init__(self, web3: Web3):
        self.web3 = web3
        self._setup_metrics()
        
    def _setup_metrics(self):
        """Initialize Prometheus metrics for gas optimization"""
        self._gas_price_gauge = Gauge(
            'gas_price_gwei',
            'Current gas price in Gwei',
            ['network']
        )
        self._gas_savings = Counter(
            'gas_savings_total',
            'Total gas saved through optimization',
            ['strategy']
        )
        self._optimization_time = Histogram(
            'gas_optimization_seconds',
            'Time spent on gas optimization',
            ['strategy']
        )

    @cache.memoize(ttl=60)  # Cache for 1 minute
    async def get_current_gas_price(self, network: str = "ethereum") -> Wei:
        """Get current gas price with caching"""
        gas_price = self.web3.eth.gas_price
        gwei_price = float(self.web3.from_wei(gas_price, 'gwei'))
        
        self._gas_price_gauge.labels(
            network=network
        ).set(gwei_price)
        
        return gas_price

    async def estimate_gas_cost(self, 
                              transaction: TxParams,
                              network: str = "ethereum") -> Tuple[Wei, Wei]:
        """Estimate gas cost for a transaction"""
        try:
            gas_estimate = await asyncio.to_thread(
                self.web3.eth.estimate_gas,
                transaction
            )
            gas_price = await self.get_current_gas_price(network)
            return gas_estimate, gas_price
        except Exception as e:
            logger.error("Gas estimation failed",
                        error=str(e),
                        transaction=transaction)
            raise

    @cache.memoize(ttl=3600)
    async def get_historical_gas_prices(self, 
                                      hours: int = 24,
                                      network: str = "ethereum") -> List[Tuple[datetime, Wei]]:
        """Get historical gas prices for analysis"""
        current_block = await asyncio.to_thread(
            lambda: self.web3.eth.block_number
        )
        blocks_per_hour = 240  # ~15 second blocks
        prices = []
        
        for i in range(hours * blocks_per_hour, 0, -blocks_per_hour):
            block = current_block - i
            try:
                block_data = await asyncio.to_thread(
                    self.web3.eth.get_block,
                    block
                )
                block_data = cast(BlockData, block_data)
                timestamp = datetime.fromtimestamp(block_data['timestamp'])
                base_fee = Wei(block_data.get('baseFeePerGas', 0))
                prices.append((timestamp, base_fee))
            except Exception as e:
                logger.warning("Failed to get historical gas price",
                             block=block,
                             error=str(e))
                continue
                
        return prices

    async def optimize_gas_timing(self, 
                                target_gas_price: Wei,
                                max_wait_time: int = 3600,
                                network: str = "ethereum") -> Optional[int]:
        """Optimize transaction timing based on gas price target"""
        start_time = asyncio.get_event_loop().time()
        try:
            historical_prices = await self.get_historical_gas_prices(24, network)
            if not historical_prices:
                return None
                
            # Analyze patterns
            hourly_averages = {}
            for timestamp, price in historical_prices:
                hour = timestamp.hour
                if hour not in hourly_averages:
                    hourly_averages[hour] = []
                hourly_averages[hour].append(int(price))
            
            # Find optimal hour
            optimal_hour = min(
                hourly_averages.keys(),
                key=lambda h: sum(hourly_averages[h]) / len(hourly_averages[h])
            )
            
            return optimal_hour
        finally:
            duration = asyncio.get_event_loop().time() - start_time
            self._optimization_time.labels(
                strategy="timing"
            ).observe(duration)

    async def optimize_gas_limit(self, 
                               transaction: TxParams,
                               network: str = "ethereum") -> Wei:
        """Optimize gas limit for a transaction"""
        start_time = asyncio.get_event_loop().time()
        try:
            # Get base estimate
            base_estimate = await asyncio.to_thread(
                self.web3.eth.estimate_gas,
                transaction
            )
            
            # Add safety margin (5%)
            optimized_limit = Wei(int(base_estimate * 1.05))
            
            # Track savings
            if 'gas' in transaction and transaction['gas'] > optimized_limit:
                saved_gas = transaction['gas'] - optimized_limit
                self._gas_savings.labels(
                    strategy="limit"
                ).inc(int(saved_gas))
            
            return optimized_limit
        finally:
            duration = asyncio.get_event_loop().time() - start_time
            self._optimization_time.labels(
                strategy="limit"
            ).observe(duration)

    async def should_replace_transaction(self,
                                       old_gas_price: Wei,
                                       tx_hash: str,
                                       network: str = "ethereum") -> bool:
        """Determine if transaction should be replaced with higher gas price"""
        current_price = await self.get_current_gas_price(network)
        tx_receipt = await asyncio.to_thread(
            self.web3.eth.get_transaction_receipt,
            HexBytes(tx_hash)
        )
        
        if tx_receipt is None:  # Transaction pending
            # Replace if current price is significantly higher (20%+)
            return current_price > (old_gas_price * 1.2)
            
        return False 