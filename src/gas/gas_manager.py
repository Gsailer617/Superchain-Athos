import logging
from typing import Dict, Any
from web3 import Web3
from web3.types import TxParams

logger = logging.getLogger(__name__)

class GasManager:
    """Manages gas price optimization and estimation"""
    
    def __init__(self, web3: Web3, config: Dict[str, Any]):
        self.web3 = web3
        self.config = config
        self.max_priority_fee = config.get('gas', {}).get('max_priority_fee', 2000000000)  # 2 Gwei default
        self.max_fee_per_gas = config.get('gas', {}).get('max_fee_per_gas', 100000000000)  # 100 Gwei default
        self.gas_limit_buffer = config.get('gas', {}).get('gas_limit_buffer', 1.2)  # 20% buffer
        
    async def optimize_gas_settings(self, tx_params: TxParams) -> Dict[str, Any]:
        """Optimize gas settings for a transaction"""
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
        """Estimate total gas cost for a transaction"""
        try:
            gas_settings = await self.optimize_gas_settings(tx_params)
            gas_limit = gas_settings.get('gas', 500000)
            gas_price = gas_settings.get('gasPrice') or gas_settings.get('maxFeePerGas')
            
            return gas_limit * gas_price
            
        except Exception as e:
            logger.error(f"Error estimating gas cost: {str(e)}")
            return 0 