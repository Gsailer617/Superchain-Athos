import logging
import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from web3 import Web3
from web3.types import TxParams
from datetime import datetime, timedelta
from cachetools import TTLCache
import statistics
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GasMetrics:
    """Gas metrics for tracking and analysis"""
    timestamp: float
    base_fee: int
    priority_fee: int
    gas_price: int
    block_number: int
    network_congestion: float

class GasManager:
    """Manages gas price optimization and estimation"""
    
    def __init__(self, web3: Web3, config: Dict[str, Any]):
        self.web3 = web3
        self.config = config
        
        # Gas price limits from config
        gas_config = config.get('gas', {})
        self.max_priority_fee = gas_config.get('max_priority_fee', 2_000_000_000)  # 2 Gwei default
        self.max_fee_per_gas = gas_config.get('max_fee_per_gas', 100_000_000_000)  # 100 Gwei default
        self.min_priority_fee = gas_config.get('min_priority_fee', 100_000_000)  # 0.1 Gwei default
        self.gas_limit_buffer = gas_config.get('gas_limit_buffer', 1.2)  # 20% buffer
        self.base_fee_multiplier = gas_config.get('base_fee_multiplier', 1.5)  # 50% buffer on base fee
        
        # EIP-1559 support
        self.use_eip1559 = gas_config.get('use_eip1559', True)
        
        # Optimization modes
        self.optimization_modes = {
            'economy': {
                'base_fee_multiplier': 1.1,
                'priority_fee_percentile': 10,
                'gas_limit_buffer': 1.1
            },
            'normal': {
                'base_fee_multiplier': 1.5,
                'priority_fee_percentile': 50,
                'gas_limit_buffer': 1.2
            },
            'performance': {
                'base_fee_multiplier': 2.0,
                'priority_fee_percentile': 90,
                'gas_limit_buffer': 1.3
            },
            'urgent': {
                'base_fee_multiplier': 2.5,
                'priority_fee_percentile': 95,
                'gas_limit_buffer': 1.5
            }
        }
        
        # Historical tracking
        self.metrics_history: List[GasMetrics] = []
        self.max_history_size = gas_config.get('max_history_size', 100)
        self.history_window = gas_config.get('history_window_minutes', 30)
        
        # Cache for gas estimates
        self.cache = TTLCache(maxsize=100, ttl=30)  # 30 second TTL
        
        # Network congestion tracking
        self.last_congestion_check = 0
        self.congestion_check_interval = 10  # seconds
        self.network_congestion = 0.5  # default medium congestion
        
    async def optimize_gas_settings(
        self, 
        tx_params: TxParams, 
        mode: str = 'normal'
    ) -> Dict[str, Any]:
        """Optimize gas settings for a transaction
        
        Args:
            tx_params: Transaction parameters
            mode: Optimization mode ('economy', 'normal', 'performance', 'urgent')
            
        Returns:
            Optimized gas settings
        """
        try:
            # Update network congestion if needed
            await self._maybe_update_congestion()
            
            # Get optimization parameters for the selected mode
            opt_params = self.optimization_modes.get(mode, self.optimization_modes['normal'])
            
            # Get latest block
            block = await self.web3.eth.get_block('latest')
            block_number = block.get('number', 0)
            
            # Get base fee from block or fallback to gas price
            base_fee = block.get('baseFeePerGas', 0)
            if base_fee == 0:
                base_fee = await self.web3.eth.gas_price
            
            # Calculate optimal gas settings based on EIP-1559 support
            if self.use_eip1559 and base_fee > 0:
                # Get priority fee based on network conditions
                priority_fee = await self._get_optimal_priority_fee(opt_params['priority_fee_percentile'])
                
                # Calculate max fee per gas
            max_fee_per_gas = min(
                self.max_fee_per_gas,
                    int(base_fee * opt_params['base_fee_multiplier']) + priority_fee
                )
                
                # Ensure priority fee is within limits
                priority_fee = min(
                    max(priority_fee, self.min_priority_fee),
                    self.max_priority_fee
                )
                
                # Track metrics
                self._track_metrics(GasMetrics(
                    timestamp=time.time(),
                    base_fee=base_fee,
                    priority_fee=priority_fee,
                    gas_price=max_fee_per_gas,
                    block_number=block_number,
                    network_congestion=self.network_congestion
                ))
            
                # Estimate gas with buffer if 'to' and 'data' are provided
                gas_limit = 500000  # Conservative default
                if 'to' in tx_params and ('data' in tx_params or 'input' in tx_params):
                    try:
                        # Create a copy for estimation to avoid modifying original
                        est_tx = dict(tx_params)
                        gas_estimate = await self.web3.eth.estimate_gas(est_tx)
                        gas_limit = int(gas_estimate * opt_params['gas_limit_buffer'])
                    except Exception as e:
                        logger.warning(f"Error estimating gas, using default: {str(e)}")
            
            return {
                    'maxPriorityFeePerGas': priority_fee,
                'maxFeePerGas': max_fee_per_gas,
                'gas': gas_limit
            }
            else:
                # Legacy gas price for chains without EIP-1559
                gas_price = await self.web3.eth.gas_price
                gas_price = int(gas_price * opt_params['base_fee_multiplier'])
                
                # Track metrics
                self._track_metrics(GasMetrics(
                    timestamp=time.time(),
                    base_fee=gas_price,
                    priority_fee=0,
                    gas_price=gas_price,
                    block_number=block_number,
                    network_congestion=self.network_congestion
                ))
                
                # Estimate gas with buffer
                gas_limit = 500000  # Conservative default
                if 'to' in tx_params and ('data' in tx_params or 'input' in tx_params):
                    try:
                        # Create a copy for estimation to avoid modifying original
                        est_tx = dict(tx_params)
                        gas_estimate = await self.web3.eth.estimate_gas(est_tx)
                        gas_limit = int(gas_estimate * opt_params['gas_limit_buffer'])
                    except Exception as e:
                        logger.warning(f"Error estimating gas, using default: {str(e)}")
                
                return {
                    'gasPrice': gas_price,
                'gas': gas_limit
            }
            
        except Exception as e:
            logger.error(f"Error optimizing gas settings: {str(e)}")
            # Fallback to basic gas settings
            return {
                'gasPrice': await self.web3.eth.gas_price,
                'gas': 500000  # Conservative default
            }
            
    async def estimate_gas_cost(self, tx_params: TxParams, mode: str = 'normal') -> int:
        """Estimate total gas cost for a transaction
        
        Args:
            tx_params: Transaction parameters
            mode: Optimization mode
            
        Returns:
            Estimated gas cost in wei
        """
        try:
            gas_settings = await self.optimize_gas_settings(tx_params, mode)
            gas_limit = gas_settings.get('gas', 500000)
            
            # Calculate cost based on EIP-1559 or legacy
            if 'maxFeePerGas' in gas_settings:
                # For EIP-1559, use maxFeePerGas as worst-case
                gas_price = gas_settings['maxFeePerGas']
            else:
                gas_price = gas_settings.get('gasPrice', 0)
            
            return gas_limit * gas_price
            
        except Exception as e:
            logger.error(f"Error estimating gas cost: {str(e)}")
            return 0 
    
    async def bump_gas_price(
        self, 
        tx_params: TxParams, 
        multiplier: float = 1.2
    ) -> Dict[str, Any]:
        """Bump gas price for a stuck transaction
        
        Args:
            tx_params: Original transaction parameters
            multiplier: Multiplier for gas price increase
            
        Returns:
            Updated transaction parameters with bumped gas price
        """
        try:
            # Create a copy to avoid modifying the original
            new_params = dict(tx_params)
            
            # Handle EIP-1559 transactions
            if 'maxFeePerGas' in new_params and 'maxPriorityFeePerGas' in new_params:
                new_params['maxFeePerGas'] = int(new_params['maxFeePerGas'] * multiplier)
                new_params['maxPriorityFeePerGas'] = int(new_params['maxPriorityFeePerGas'] * multiplier)
                
                # Ensure we don't exceed maximum values
                new_params['maxFeePerGas'] = min(new_params['maxFeePerGas'], self.max_fee_per_gas)
                new_params['maxPriorityFeePerGas'] = min(new_params['maxPriorityFeePerGas'], self.max_priority_fee)
            
            # Handle legacy transactions
            elif 'gasPrice' in new_params:
                new_params['gasPrice'] = int(new_params['gasPrice'] * multiplier)
            
            return new_params
            
        except Exception as e:
            logger.error(f"Error bumping gas price: {str(e)}")
            return tx_params
    
    async def get_network_congestion(self) -> float:
        """Get current network congestion level (0.0 to 1.0)
        
        Returns:
            Congestion level where 0.0 is no congestion and 1.0 is full congestion
        """
        await self._maybe_update_congestion()
        return self.network_congestion
    
    def get_gas_price_stats(self) -> Dict[str, Any]:
        """Get statistics about gas prices
        
        Returns:
            Dictionary with gas price statistics
        """
        if not self.metrics_history:
            return {
                "avg_base_fee": 0,
                "avg_priority_fee": 0,
                "avg_gas_price": 0,
                "avg_congestion": 0,
                "samples": 0
            }
        
        # Filter to recent history
        cutoff = time.time() - (self.history_window * 60)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff]
        
        if not recent_metrics:
            recent_metrics = self.metrics_history
        
        return {
            "avg_base_fee": int(statistics.mean(m.base_fee for m in recent_metrics)),
            "avg_priority_fee": int(statistics.mean(m.priority_fee for m in recent_metrics)),
            "avg_gas_price": int(statistics.mean(m.gas_price for m in recent_metrics)),
            "avg_congestion": statistics.mean(m.network_congestion for m in recent_metrics),
            "samples": len(recent_metrics)
        }
    
    async def _get_optimal_priority_fee(self, percentile: int = 50) -> int:
        """Get optimal priority fee based on recent blocks
        
        Args:
            percentile: Percentile to use for priority fee (0-100)
            
        Returns:
            Priority fee in wei
        """
        try:
            # Try to get from cache
            cache_key = f"priority_fee_{percentile}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Get max priority fee from network
            priority_fee = await self.web3.eth.max_priority_fee
            
            # Adjust based on percentile
            adjusted_fee = int(priority_fee * percentile / 50)  # 50th percentile = 1x
            
            # Ensure it's within limits
            adjusted_fee = min(max(adjusted_fee, self.min_priority_fee), self.max_priority_fee)
            
            # Cache the result
            self.cache[cache_key] = adjusted_fee
            
            return adjusted_fee
            
        except Exception as e:
            logger.error(f"Error getting optimal priority fee: {str(e)}")
            return self.min_priority_fee
    
    async def _maybe_update_congestion(self) -> None:
        """Update network congestion if needed"""
        current_time = time.time()
        if current_time - self.last_congestion_check < self.congestion_check_interval:
            return
            
        try:
            # Get latest block
            block = await self.web3.eth.get_block('latest')
            
            # Calculate congestion based on gas used vs gas limit
            gas_used = block.get('gasUsed', 0)
            gas_limit = block.get('gasLimit', 30000000)
            
            if gas_limit > 0:
                self.network_congestion = min(gas_used / gas_limit, 1.0)
            
            # Update timestamp
            self.last_congestion_check = current_time
            
        except Exception as e:
            logger.error(f"Error updating network congestion: {str(e)}")
    
    def _track_metrics(self, metrics: GasMetrics) -> None:
        """Track gas metrics for analysis
        
        Args:
            metrics: Gas metrics to track
        """
        self.metrics_history.append(metrics)
        
        # Limit history size
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:] 