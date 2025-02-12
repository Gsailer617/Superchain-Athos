"""Gas optimization module for efficient transaction execution"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from web3 import Web3
from web3.types import TxParams
from datetime import datetime, timedelta
from cachetools import TTLCache
import statistics
import time

logger = logging.getLogger(__name__)

class AsyncGasOptimizer:
    """Asynchronous gas price optimizer for efficient transaction execution"""
    
    def __init__(
        self,
        web3: Optional[Web3] = None,
        config: Optional[Dict[str, Any]] = None,
        mode: str = 'normal'
    ):
        """Initialize gas optimizer with configuration
        
        Args:
            web3: Web3 instance for blockchain interaction
            config: Configuration dictionary with gas settings
            mode: Optimization mode ('normal', 'performance', or 'economy')
        """
        self.web3 = web3
        self.config = config or {}
        self.mode = mode
        
        # Gas price limits from config
        self.max_priority_fee = config.get('gas', {}).get('max_priority_fee', 2000000000)  # 2 Gwei default
        self.max_fee_per_gas = config.get('gas', {}).get('max_fee_per_gas', 100000000000)  # 100 Gwei default
        self.gas_limit_buffer = config.get('gas', {}).get('gas_limit_buffer', 1.2)  # 20% buffer
        
        # Performance tracking
        self.optimization_history = []
        self.last_base_fee = None
        self.last_priority_fee = None
        
        # Cache configuration
        self.cache = TTLCache(maxsize=1000, ttl=60)  # 1 minute TTL
        
        # Mode-specific multipliers
        self.mode_multipliers = {
            'economy': 0.8,
            'normal': 1.0,
            'performance': 1.3
        }
        
        # Cache for base settings
        self._cache = {}
        
    def clear_cache(self):
        """Clear the optimization cache"""
        self.cache.clear()
        
    async def optimize_gas_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize gas parameters based on current conditions
        
        Args:
            params: Parameters including urgency level
            
        Returns:
            Optimized gas parameters
        """
        # Check cache first
        cache_key = str(params)
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        try:
            # Get base settings
            base_settings = await self._get_base_settings()
            
            # Apply mode-specific optimizations
            if self.mode == 'performance':
                settings = self._optimize_for_performance(base_settings, params)
            elif self.mode == 'economy':
                settings = self._optimize_for_economy(base_settings, params)
            else:
                settings = self._optimize_for_normal(base_settings, params)
                
            # Apply urgency adjustments
            urgency = params.get('urgency', 'normal')
            settings = self._adjust_for_urgency(settings, urgency)
            
            # Cache result
            self.cache[cache_key] = settings
            
            # Track optimization
            self._track_optimization(base_settings, settings)
            
            return settings
            
        except Exception as e:
            logger.error(f"Error optimizing gas parameters: {str(e)}")
            return await self._get_fallback_settings()
            
    async def optimize_gas_settings(self, tx_params: TxParams) -> Dict[str, Any]:
        """Optimize gas settings for a transaction
        
        Args:
            tx_params: Transaction parameters to optimize
            
        Returns:
            Dict containing optimized gas settings
        """
        try:
            # Get latest block
            block = await self.web3.eth.get_block('latest')
            base_fee = block.get('baseFeePerGas', await self.web3.eth.gas_price)
            
            # Calculate optimal gas settings
            max_priority_fee = min(
                self.max_priority_fee,
                await self.web3.eth.max_priority_fee
            )
            
            max_fee_per_gas = min(
                self.max_fee_per_gas,
                base_fee * 2 + max_priority_fee  # Double base fee plus priority fee
            )
            
            # Estimate gas with buffer
            gas_estimate = await self.web3.eth.estimate_gas(tx_params)
            gas_limit = int(gas_estimate * self.gas_limit_buffer)
            
            # Update history
            self.last_base_fee = base_fee
            self.last_priority_fee = max_priority_fee
            
            return {
                'maxPriorityFeePerGas': max_priority_fee,
                'maxFeePerGas': max_fee_per_gas,
                'gas': gas_limit
            }
            
        except Exception as e:
            logger.error(f"Error optimizing gas settings: {str(e)}")
            # Fallback to basic gas settings
            return {
                'gasPrice': await self.web3.eth.gas_price,
                'gas': 500000  # Conservative default
            }
            
    async def estimate_gas_cost(self, tx_params: TxParams) -> int:
        """Estimate total gas cost for a transaction
        
        Args:
            tx_params: Transaction parameters to estimate
            
        Returns:
            Estimated gas cost in wei
        """
        try:
            gas_settings = await self.optimize_gas_settings(tx_params)
            gas_limit = gas_settings.get('gas', 500000)
            gas_price = gas_settings.get('gasPrice') or gas_settings.get('maxFeePerGas')
            
            return gas_limit * gas_price
            
        except Exception as e:
            logger.error(f"Error estimating gas cost: {str(e)}")
            return 0
            
    async def get_network_congestion(self) -> float:
        """Get current network congestion level
        
        Returns:
            Float between 0 and 1 indicating network congestion
        """
        try:
            block = await self.web3.eth.get_block('latest')
            gas_used = block.get('gasUsed', 0)
            gas_limit = block.get('gasLimit', 30000000)
            
            return min(gas_used / gas_limit, 1.0)
            
        except Exception as e:
            logger.error(f"Error getting network congestion: {str(e)}")
            return 1.0  # Conservative estimate
            
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get gas optimization statistics
        
        Returns:
            Dict containing optimization statistics
        """
        return {
            'last_base_fee': self.last_base_fee,
            'last_priority_fee': self.last_priority_fee,
            'optimization_count': len(self.optimization_history),
            'average_savings': self._calculate_average_savings()
        }
        
    async def _get_base_settings(self) -> Dict[str, Any]:
        """Get base gas settings"""
        try:
            # Check cache first
            cache_key = f"base_settings_{int(time.time() / 30)}"  # 30-second cache
            if cache_key in self._cache:
                return self._cache[cache_key]

            if not self.web3:
                return {'gasPrice': 50000000000}  # 50 Gwei default
                
            # Get latest block
            block = await self.web3.eth.get_block('latest')
            base_fee = block.get('baseFeePerGas', await self.web3.eth.gas_price)
            
            # Calculate optimal gas settings
            max_priority_fee = min(
                self.max_priority_fee,
                int(await self.web3.eth.max_priority_fee)  # Convert to int
            )
            
            max_fee_per_gas = min(
                self.max_fee_per_gas,
                base_fee * 2 + max_priority_fee  # Double base fee plus priority fee
            )
            
            settings = {
                'maxPriorityFeePerGas': max_priority_fee,
                'maxFeePerGas': max_fee_per_gas,
                'baseFee': base_fee
            }
            
            # Cache the result
            self._cache[cache_key] = settings
            return settings
            
        except Exception as e:
            logger.error(f"Error getting base settings: {str(e)}")
            return {'gasPrice': 50000000000}  # 50 Gwei fallback
            
    def _optimize_for_performance(self, base_settings: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize for high performance (faster confirmations)"""
        settings = base_settings.copy()
        
        # Increase priority fee for faster inclusion
        if 'maxPriorityFeePerGas' in settings:
            settings['maxPriorityFeePerGas'] *= 1.5
            settings['maxFeePerGas'] = settings['baseFee'] * 2 + settings['maxPriorityFeePerGas']
        else:
            settings['gasPrice'] *= 1.5
            
        return settings
        
    def _optimize_for_economy(self, base_settings: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize for lower costs"""
        settings = base_settings.copy()
        
        # Reduce fees for cost savings
        if 'maxPriorityFeePerGas' in settings:
            settings['maxPriorityFeePerGas'] *= 0.8
            settings['maxFeePerGas'] = settings['baseFee'] * 1.5 + settings['maxPriorityFeePerGas']
        else:
            settings['gasPrice'] *= 0.8
            
        return settings
        
    def _optimize_for_normal(self, base_settings: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Standard optimization"""
        return base_settings.copy()
        
    def _adjust_for_urgency(self, settings: Dict[str, Any], urgency: str) -> Dict[str, Any]:
        """Adjust settings based on urgency level"""
        multipliers = {
            'low': 0.8,
            'normal': 1.0,
            'high': 1.3
        }
        
        multiplier = multipliers.get(urgency, 1.0)
        adjusted = settings.copy()
        
        if 'maxPriorityFeePerGas' in adjusted:
            adjusted['maxPriorityFeePerGas'] *= multiplier
            adjusted['maxFeePerGas'] = adjusted['baseFee'] * 2 + adjusted['maxPriorityFeePerGas']
        else:
            adjusted['gasPrice'] *= multiplier
            
        return adjusted
        
    async def _get_fallback_settings(self) -> Dict[str, Any]:
        """Get fallback gas settings"""
        try:
            if self.web3:
                return {'gasPrice': await self.web3.eth.gas_price}
            return {'gasPrice': 50000000000}  # 50 Gwei default
        except Exception:
            return {'gasPrice': 50000000000}  # 50 Gwei default
            
    def _track_optimization(self, original: Dict[str, Any], optimized: Dict[str, Any]):
        """Track optimization for performance analysis"""
        original_cost = original.get('gasPrice', 0) or original.get('maxFeePerGas', 0)
        optimized_cost = optimized.get('gasPrice', 0) or optimized.get('maxFeePerGas', 0)
        
        self.optimization_history.append({
            'timestamp': datetime.now(),
            'original_cost': original_cost,
            'optimized_cost': optimized_cost,
            'savings': original_cost - optimized_cost
        })
        
        # Trim history to last 1000 optimizations
        if len(self.optimization_history) > 1000:
            self.optimization_history = self.optimization_history[-1000:]
            
    def _calculate_average_savings(self) -> float:
        """Calculate average gas savings from optimization history"""
        if not self.optimization_history:
            return 0.0
            
        total_savings = sum(
            opt['original_cost'] - opt['optimized_cost'] 
            for opt in self.optimization_history
        )
        return total_savings / len(self.optimization_history) 