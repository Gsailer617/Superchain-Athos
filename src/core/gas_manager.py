from typing import Dict, Optional, Any, Tuple, cast
from web3 import Web3, AsyncWeb3
from web3.types import Wei, TxParams
import logging
import time
from dataclasses import dataclass
from decimal import Decimal

from ..config.chain_specs import (
    ChainSpec,
    GasModel,
    GasConfig,
    ChainConfig,
    GasFeeModel
)
from ..utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)

@dataclass
class GasMetrics:
    """Gas price metrics for a chain"""
    base_fee: Optional[Wei] = None
    priority_fee: Optional[Wei] = None
    max_fee: Optional[Wei] = None
    last_updated: float = 0.0
    block_number: int = 0
    
    def is_stale(self, max_age: float = 30.0) -> bool:
        """Check if metrics are stale"""
        return time.time() - self.last_updated > max_age

class GasManager:
    """Gas price estimation and optimization across chains"""
    
    def __init__(self):
        self._metrics: Dict[str, GasMetrics] = {}
    
    @retry_with_backoff(max_retries=3)
    async def estimate_gas_price(
        self,
        web3: AsyncWeb3,
        chain_spec: ChainSpec,
        speed: str = 'average'
    ) -> Dict[str, Wei]:
        """Estimate gas price based on chain's gas model"""
        if chain_spec.gas_fee_model == GasFeeModel.EIP1559:
            return await self._estimate_eip1559_fees(web3, speed)
        elif chain_spec.gas_fee_model == GasFeeModel.OPTIMISTIC:
            return await self._estimate_optimistic_fees(web3, chain_spec, speed)
        elif chain_spec.gas_fee_model == GasFeeModel.ARBITRUM:
            return await self._estimate_arbitrum_fees(web3, speed)
        else:
            return await self._estimate_legacy_fees(web3, speed)
    
    async def _estimate_eip1559_fees(
        self,
        web3: AsyncWeb3,
        speed: str
    ) -> Dict[str, Wei]:
        """Estimate EIP-1559 gas fees"""
        block = await web3.eth.get_block('latest')
        base_fee = Wei(int(block['baseFeePerGas']))
        
        # Get max priority fee
        priority_fee = await web3.eth.max_priority_fee
        
        # Adjust priority fee based on speed
        if speed == 'fast':
            priority_fee = Wei(int(priority_fee * 1.5))
        elif speed == 'fastest':
            priority_fee = Wei(int(priority_fee * 2))
        
        # Calculate max fee
        max_fee = Wei(base_fee + priority_fee)
        
        return {
            'base_fee': base_fee,
            'max_priority_fee': priority_fee,
            'max_fee_per_gas': max_fee
        }
    
    async def _estimate_optimistic_fees(
        self,
        web3: AsyncWeb3,
        chain_spec: ChainSpec,
        speed: str
    ) -> Dict[str, Wei]:
        """Estimate gas fees for Optimistic rollups"""
        # Get L1 data fee
        l1_fee = await self._get_l1_data_fee(web3, chain_spec)
        
        # Get L2 execution fee
        l2_fee = await web3.eth.gas_price
        
        # Adjust based on speed
        multiplier = 1.0
        if speed == 'fast':
            multiplier = 1.2
        elif speed == 'fastest':
            multiplier = 1.5
        
        total_fee = Wei(int((l1_fee + l2_fee) * multiplier))
        
        return {
            'gas_price': total_fee,
            'l1_fee': l1_fee,
            'l2_fee': l2_fee
        }
    
    async def _estimate_arbitrum_fees(
        self,
        web3: AsyncWeb3,
        speed: str
    ) -> Dict[str, Wei]:
        """Estimate gas fees for Arbitrum"""
        # Get current gas price which includes L1+L2 costs
        gas_price = await web3.eth.gas_price
        
        # Adjust based on speed
        multiplier = 1.0
        if speed == 'fast':
            multiplier = 1.2
        elif speed == 'fastest':
            multiplier = 1.5
        
        return {
            'gas_price': Wei(int(gas_price * multiplier))
        }
    
    async def _estimate_legacy_fees(
        self,
        web3: AsyncWeb3,
        speed: str
    ) -> Dict[str, Wei]:
        """Estimate legacy gas fees"""
        gas_price = await web3.eth.gas_price
        
        # Adjust based on speed
        multiplier = 1.0
        if speed == 'fast':
            multiplier = 1.2
        elif speed == 'fastest':
            multiplier = 1.5
        
        return {
            'gas_price': Wei(int(gas_price * multiplier))
        }
    
    async def _get_l1_data_fee(
        self,
        web3: AsyncWeb3,
        chain_spec: ChainSpec
    ) -> Wei:
        """Get L1 data fee for Optimistic rollups"""
        # Different implementations for different chains
        if chain_spec.name == 'optimism':
            return await self._get_optimism_l1_fee(web3)
        elif chain_spec.name == 'base':
            return await self._get_base_l1_fee(web3)
        elif chain_spec.name in ['mode', 'sonic']:
            return await self._get_mode_l1_fee(web3)
        else:
            return Wei(0)
    
    async def _get_optimism_l1_fee(self, web3: AsyncWeb3) -> Wei:
        """Get L1 data fee for Optimism"""
        gas_price_oracle = web3.eth.contract(
            address=web3.to_checksum_address('0x420000000000000000000000000000000000000F'),
            abi=[{
                "inputs": [],
                "name": "l1BaseFee",
                "outputs": [{"type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            }]
        )
        return Wei(await gas_price_oracle.functions.l1BaseFee().call())
    
    async def _get_base_l1_fee(self, web3: AsyncWeb3) -> Wei:
        """Get L1 data fee for Base"""
        gas_price_oracle = web3.eth.contract(
            address=web3.to_checksum_address('0x420000000000000000000000000000000000000F'),
            abi=[{
                "inputs": [],
                "name": "l1BaseFee",
                "outputs": [{"type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            }]
        )
        return Wei(await gas_price_oracle.functions.l1BaseFee().call())
    
    async def _get_mode_l1_fee(self, web3: AsyncWeb3) -> Wei:
        """Get L1 data fee for Mode/Sonic"""
        gas_price_oracle = web3.eth.contract(
            address=web3.to_checksum_address('0x420000000000000000000000000000000000000F'),
            abi=[{
                "inputs": [],
                "name": "l1BaseFee",
                "outputs": [{"type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            }]
        )
        return Wei(await gas_price_oracle.functions.l1BaseFee().call())
    
    @retry_with_backoff(max_retries=3)
    async def estimate_gas_limit(
        self,
        web3: AsyncWeb3,
        tx: TxParams,
        chain_spec: ChainSpec,
        buffer: float = 1.1
    ) -> int:
        """Estimate gas limit with safety buffer"""
        try:
            # Get base estimate
            gas_estimate = await web3.eth.estimate_gas(tx)
            
            # Add buffer based on chain type
            if chain_spec.is_l2:
                # L2s often need larger buffer due to state proof costs
                buffer = 1.2
            
            # Apply buffer and round up
            return int(gas_estimate * buffer)
            
        except Exception as e:
            logger.error(f"Gas estimation failed: {str(e)}")
            raise
    
    def update_metrics(
        self,
        chain_name: str,
        metrics: GasMetrics
    ) -> None:
        """Update gas metrics for chain"""
        self._metrics[chain_name] = metrics
    
    def get_metrics(
        self,
        chain_name: str
    ) -> Optional[GasMetrics]:
        """Get gas metrics for chain"""
        return self._metrics.get(chain_name)
    
    def clear_metrics(self) -> None:
        """Clear all metrics"""
        self._metrics.clear() 