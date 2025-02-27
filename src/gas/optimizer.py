"""Gas optimization module for efficient transaction execution"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from web3 import Web3
from web3.types import TxParams, Wei
from datetime import datetime, timedelta
from cachetools import TTLCache
import statistics
import time
import math
from dataclasses import dataclass
from functools import lru_cache
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.gas.gas_manager import GasManager, GasMetrics

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Result of gas optimization"""
    success: bool
    gas_params: Dict[str, Any]
    estimated_cost: int
    estimated_wait_time: int  # seconds
    network_congestion: float
    error: Optional[str] = None

class AsyncGasOptimizer:
    """Asynchronous gas price optimizer for efficient transaction execution"""
    
    def __init__(
        self,
        web3: Optional[Web3] = None,
        config: Optional[Dict[str, Any]] = None,
        gas_manager: Optional[GasManager] = None,
        mode: str = 'normal'
    ):
        """Initialize gas optimizer with configuration
        
        Args:
            web3: Web3 instance for blockchain interaction
            config: Configuration dictionary with gas settings
            gas_manager: Optional GasManager instance to use
            mode: Optimization mode ('economy', 'normal', 'performance', or 'urgent')
        """
        self.web3 = web3
        self.config = config or {}
        self.mode = mode
        
        # Use provided gas manager or create a new one
        if gas_manager:
            self.gas_manager = gas_manager
        elif web3:
            self.gas_manager = GasManager(web3, config or {})
        else:
            raise ValueError("Either gas_manager or web3 must be provided")
        
        # Gas price limits from config
        gas_config = config.get('gas', {}) if config else {}
        self.max_priority_fee = gas_config.get('max_priority_fee', 2_000_000_000)  # 2 Gwei default
        self.max_fee_per_gas = gas_config.get('max_fee_per_gas', 100_000_000_000)  # 100 Gwei default
        self.min_priority_fee = gas_config.get('min_priority_fee', 100_000_000)  # 0.1 Gwei default
        self.gas_limit_buffer = gas_config.get('gas_limit_buffer', 1.2)  # 20% buffer
        
        # EIP-1559 support
        self.use_eip1559 = gas_config.get('use_eip1559', True)
        
        # Performance tracking
        self.optimization_history: List[Dict[str, Any]] = []
        self.max_history_size = gas_config.get('max_history_size', 100)
        
        # Cache configuration
        self.cache = TTLCache(maxsize=1000, ttl=60)  # 1 minute TTL
        
        # Mode-specific multipliers
        self.mode_multipliers = {
            'economy': 0.8,
            'normal': 1.0,
            'performance': 1.3,
            'urgent': 1.5
        }
        
        # External gas price API
        self.use_external_api = gas_config.get('use_external_api', False)
        self.api_url = gas_config.get('gas_api_url', 'https://api.etherscan.io/api?module=gastracker&action=gasoracle')
        self.api_key = gas_config.get('etherscan_api_key', '')
        self._session: Optional[aiohttp.ClientSession] = None
        
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": "FlashingBase/1.0.0"}
            )
        return self._session
        
    def clear_cache(self):
        """Clear the optimization cache"""
        self.cache.clear()
        
    async def optimize_gas_params(
        self, 
        params: Dict[str, Any]
    ) -> OptimizationResult:
        """Optimize gas parameters based on current conditions
        
        Args:
            params: Parameters including urgency level
            
        Returns:
            OptimizationResult with optimized gas parameters
        """
        # Check cache first
        cache_key = str(params)
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        try:
            # Get urgency level
            urgency = params.get('urgency', self.mode)
            
            # Get network congestion
            congestion = await self.gas_manager.get_network_congestion()
            
            # Get base settings from gas manager
            gas_params = await self.gas_manager.optimize_gas_settings(
                params.get('tx_params', {}),
                mode=urgency
            )
            
            # Estimate wait time based on congestion and urgency
            wait_time = self._estimate_wait_time(congestion, urgency)
            
            # Estimate cost
            estimated_cost = 0
            if 'gas' in gas_params:
                if 'maxFeePerGas' in gas_params:
                    estimated_cost = gas_params['gas'] * gas_params['maxFeePerGas']
                elif 'gasPrice' in gas_params:
                    estimated_cost = gas_params['gas'] * gas_params['gasPrice']
            
            # Create result
            result = OptimizationResult(
                success=True,
                gas_params=gas_params,
                estimated_cost=estimated_cost,
                estimated_wait_time=wait_time,
                network_congestion=congestion
            )
            
            # Cache result
            self.cache[cache_key] = result
            
            # Track optimization
            self._track_optimization(params, gas_params, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing gas parameters: {str(e)}")
            return OptimizationResult(
                success=False,
                gas_params={},
                estimated_cost=0,
                estimated_wait_time=0,
                network_congestion=0,
                error=str(e)
            )
    
    async def optimize_gas_settings(
        self, 
        tx_params: TxParams
    ) -> Dict[str, Any]:
        """Optimize gas settings for a transaction
        
        Args:
            tx_params: Transaction parameters
            
        Returns:
            Optimized gas settings
        """
        try:
            # Use the gas manager to optimize settings
            return await self.gas_manager.optimize_gas_settings(tx_params, mode=self.mode)
            
        except Exception as e:
            logger.error(f"Error optimizing gas settings: {str(e)}")
            # Fallback to basic gas settings
            if self.web3:
                gas_price = await self.web3.eth.gas_price
                return {
                    'gasPrice': gas_price,
                    'gas': 500000  # Conservative default
                }
            return {
                'gasPrice': 50_000_000_000,  # 50 Gwei fallback
                'gas': 500000  # Conservative default
            }
            
    async def estimate_gas_cost(
        self, 
        tx_params: TxParams
    ) -> int:
        """Estimate total gas cost for a transaction
        
        Args:
            tx_params: Transaction parameters
            
        Returns:
            Estimated gas cost in wei
        """
        try:
            return await self.gas_manager.estimate_gas_cost(tx_params, mode=self.mode)
            
        except Exception as e:
            logger.error(f"Error estimating gas cost: {str(e)}")
            return 0
            
    async def get_network_congestion(self) -> float:
        """Get current network congestion level
        
        Returns:
            Congestion level from 0.0 (no congestion) to 1.0 (full congestion)
        """
        try:
            return await self.gas_manager.get_network_congestion()
            
        except Exception as e:
            logger.error(f"Error getting network congestion: {str(e)}")
            return 0.5  # Default to medium congestion
            
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get statistics about gas optimizations
        
        Returns:
            Dictionary with optimization statistics
        """
        if not self.optimization_history:
            return {
                "avg_gas_price": 0,
                "avg_priority_fee": 0,
                "avg_gas_limit": 0,
                "avg_congestion": 0,
                "avg_wait_time": 0,
                "total_optimizations": 0
            }
        
        # Calculate averages
        total = len(self.optimization_history)
        
        # Get gas price stats based on EIP-1559 or legacy
        gas_prices = []
        priority_fees = []
        gas_limits = []
        congestions = []
        wait_times = []
        
        for opt in self.optimization_history:
            result = opt.get('result', {})
            gas_params = opt.get('gas_params', {})
            
            # Get gas price
            if 'maxFeePerGas' in gas_params:
                gas_prices.append(gas_params['maxFeePerGas'])
                priority_fees.append(gas_params.get('maxPriorityFeePerGas', 0))
            elif 'gasPrice' in gas_params:
                gas_prices.append(gas_params['gasPrice'])
                priority_fees.append(0)
                
            # Get gas limit
            if 'gas' in gas_params:
                gas_limits.append(gas_params['gas'])
                
            # Get congestion and wait time
            congestions.append(result.network_congestion if hasattr(result, 'network_congestion') else 0)
            wait_times.append(result.estimated_wait_time if hasattr(result, 'estimated_wait_time') else 0)
        
        return {
            "avg_gas_price": int(statistics.mean(gas_prices)) if gas_prices else 0,
            "avg_priority_fee": int(statistics.mean(priority_fees)) if priority_fees else 0,
            "avg_gas_limit": int(statistics.mean(gas_limits)) if gas_limits else 0,
            "avg_congestion": statistics.mean(congestions) if congestions else 0,
            "avg_wait_time": int(statistics.mean(wait_times)) if wait_times else 0,
            "total_optimizations": total
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ValueError, ConnectionError))
    )
    async def get_external_gas_prices(self) -> Dict[str, int]:
        """Get gas prices from external API
        
        Returns:
            Dictionary with gas prices for different speeds
        """
        if not self.use_external_api:
            return {}
            
        try:
            session = await self.get_session()
            url = f"{self.api_url}&apikey={self.api_key}"
            
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Error fetching gas prices: {response.status}")
                    return {}
                    
                data = await response.json()
                
                if data.get('status') != '1':
                    logger.error(f"API error: {data.get('message')}")
                    return {}
                    
                result = data.get('result', {})
                
                return {
                    'slow': int(float(result.get('SafeGasPrice', '5')) * 1e9),
                    'normal': int(float(result.get('ProposeGasPrice', '10')) * 1e9),
                    'fast': int(float(result.get('FastGasPrice', '20')) * 1e9)
                }
                
        except Exception as e:
            logger.error(f"Error fetching external gas prices: {str(e)}")
            return {}
    
    def _estimate_wait_time(self, congestion: float, urgency: str) -> int:
        """Estimate transaction wait time based on congestion and urgency
        
        Args:
            congestion: Network congestion level (0.0 to 1.0)
            urgency: Urgency level ('economy', 'normal', 'performance', 'urgent')
            
        Returns:
            Estimated wait time in seconds
        """
        # Base wait time in seconds
        base_wait = 15  # 15 seconds for normal conditions
        
        # Adjust for congestion (higher congestion = longer wait)
        congestion_factor = 1 + (congestion * 5)  # 1x to 6x multiplier
        
        # Adjust for urgency (higher urgency = shorter wait)
        urgency_factor = {
            'economy': 2.0,
            'normal': 1.0,
            'performance': 0.5,
            'urgent': 0.2
        }.get(urgency, 1.0)
        
        # Calculate wait time
        wait_time = base_wait * congestion_factor * urgency_factor
        
        return int(wait_time)
    
    def _track_optimization(
        self, 
        params: Dict[str, Any], 
        gas_params: Dict[str, Any],
        result: OptimizationResult
    ):
        """Track optimization for analysis
        
        Args:
            params: Original parameters
            gas_params: Optimized gas parameters
            result: Optimization result
        """
        self.optimization_history.append({
            'timestamp': time.time(),
            'params': params,
            'gas_params': gas_params,
            'result': result
        })
        
        # Limit history size
        if len(self.optimization_history) > self.max_history_size:
            self.optimization_history = self.optimization_history[-self.max_history_size:]
            
    def _calculate_average_savings(self) -> float:
        """Calculate average gas savings from optimizations
        
        Returns:
            Average savings percentage
        """
        if not self.optimization_history:
            return 0.0
            
        savings = []
        
        for opt in self.optimization_history:
            params = opt.get('params', {})
            gas_params = opt.get('gas_params', {})
            
            # Skip if we don't have original gas price
            if 'original_gas_price' not in params:
                continue
                
            original_price = params['original_gas_price']
            
            # Get optimized price
            if 'maxFeePerGas' in gas_params:
                optimized_price = gas_params['maxFeePerGas']
            elif 'gasPrice' in gas_params:
                optimized_price = gas_params['gasPrice']
            else:
                continue
                
            # Calculate savings
            if original_price > 0:
                saving = (original_price - optimized_price) / original_price
                savings.append(saving)
                
        return statistics.mean(savings) if savings else 0.0
        
    async def cleanup(self):
        """Clean up resources"""
        if self._session and not self._session.closed:
            await self._session.close() 