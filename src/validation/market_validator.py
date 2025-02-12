"""Market validation module"""

import logging
from typing import Dict, Any, cast
from web3 import Web3, AsyncWeb3
from web3.types import BlockData
from src.gas.optimizer import AsyncGasOptimizer
from src.core.web3_config import get_async_web3

logger = logging.getLogger(__name__)

class MarketValidator:
    """Validates market conditions before execution"""
    
    def __init__(self, web3: Web3, config: Dict[str, Any]):
        self.web3 = web3
        self.async_web3 = get_async_web3()
        self.config = config
        self.gas_optimizer = AsyncGasOptimizer(web3=web3, config=config)
        
    async def validate_conditions(self) -> bool:
        """Validate current market conditions"""
        try:
            # Check network health
            if not await self._check_network_health():
                return False
                
            # Check gas conditions
            if not await self._check_gas_conditions():
                return False
                
            # All validations passed
            return True
            
        except Exception as e:
            logger.error(f"Error validating market conditions: {str(e)}")
            return False
            
    async def _check_network_health(self) -> bool:
        """Check if network is healthy"""
        try:
            # Get latest block
            latest_block = cast(BlockData, await self.async_web3.eth.get_block('latest'))
            block_timestamp = latest_block.get('timestamp', 0)
            
            # Check if block is recent (within last minute)
            current_block = cast(BlockData, await self.async_web3.eth.get_block('latest'))
            if block_timestamp < current_block['timestamp'] - 60:
                logger.warning("Network might be stale - old block timestamp")
                return False
                
            # Check pending transaction count
            pending_tx_count = await self.async_web3.eth.get_block_transaction_count('pending')
            if pending_tx_count > 10000:  # Arbitrary threshold
                logger.warning("High pending transaction count")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking network health: {str(e)}")
            return False
            
    async def _check_gas_conditions(self) -> bool:
        """Check if gas conditions are acceptable"""
        try:
            # Get optimized gas settings
            gas_params = {'urgency': 'normal'}
            gas_settings = await self.gas_optimizer.optimize_gas_params(gas_params)
            
            # Check if gas price is within acceptable range
            max_fee = gas_settings.get('maxFeePerGas', float('inf'))
            if max_fee > self.config.get('gas', {}).get('max_fee_per_gas', float('inf')):
                logger.warning("Gas price too high")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking gas conditions: {str(e)}")
            return False 