from typing import Dict, Any, Optional, cast
from web3 import Web3, AsyncWeb3
from web3.types import TxParams, Wei, HexStr
from eth_typing import ChecksumAddress
import logging
import time

from .bridge_adapter import BridgeAdapter, BridgeConfig, BridgeState, BridgeMetrics

logger = logging.getLogger(__name__)

class StargateAdapter(BridgeAdapter):
    """Adapter implementation for Stargate protocol"""
    
    def __init__(self, config: BridgeConfig, web3: Web3):
        super().__init__(config, web3)
        self.stargate_config = config.get('stargate_config', {})
        self.pool_ids = self.stargate_config.get('pool_ids', {})
        self.router_version = self.stargate_config.get('router_version', 'v2')
        
        # Cache for contract instances
        self._contracts = {}
        self._initialize_protocol()
    
    def _initialize_protocol(self) -> None:
        """Initialize Stargate protocol contracts and settings"""
        try:
            # Initialize router and pool contracts
            self._init_contracts()
            
            # Validate configuration
            if not self.config.supported_chains:
                raise ValueError("No supported chains configured for Stargate")
                
            logger.info(f"Initialized Stargate adapter with {len(self.config.supported_chains)} supported chains")
            
        except Exception as e:
            logger.error(f"Failed to initialize Stargate adapter: {str(e)}")
            raise
    
    def validate_transfer(
        self,
        source_chain: str,
        target_chain: str,
        token: str,
        amount: float
    ) -> bool:
        """Validate if transfer is possible via Stargate"""
        try:
            # Check chain support
            if source_chain not in self.config.supported_chains or target_chain not in self.config.supported_chains:
                logger.warning(f"Chain pair {source_chain}->{target_chain} not supported")
                return False
            
            # Check token support
            if token not in self.pool_ids:
                logger.warning(f"Token {token} not supported")
                return False
            
            # Check amount limits
            if amount < self.config.min_amount or amount > self.config.max_amount:
                logger.warning(f"Amount {amount} outside limits [{self.config.min_amount}, {self.config.max_amount}]")
                return False
            
            # Check liquidity
            if not self._has_sufficient_liquidity(token, target_chain, amount):
                logger.warning(f"Insufficient liquidity for {amount} {token} on {target_chain}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating transfer: {str(e)}")
            return False
    
    def estimate_fees(
        self,
        source_chain: str,
        target_chain: str,
        token: str,
        amount: float
    ) -> Dict[str, float]:
        """Estimate Stargate transfer fees"""
        try:
            # Get LayerZero message fee
            lz_fee = self._get_lz_fee(source_chain, target_chain)
            
            # Get protocol fee
            protocol_fee = self._get_protocol_fee(amount)
            
            # Get LP fee
            lp_fee = self._get_lp_fee(amount)
            
            total = lz_fee + protocol_fee + lp_fee
            
            return {
                'lz_fee': lz_fee,
                'protocol_fee': protocol_fee,
                'lp_fee': lp_fee,
                'total': total
            }
            
        except Exception as e:
            logger.error(f"Error estimating fees: {str(e)}")
            return {
                'lz_fee': 0,
                'protocol_fee': 0,
                'lp_fee': 0,
                'total': 0
            }
    
    def estimate_time(
        self,
        source_chain: str,
        target_chain: str
    ) -> int:
        """Estimate transfer time in seconds"""
        try:
            # Base confirmation time
            base_time = self.config.confirmation_blocks * 15  # Assuming 15s block time
            
            # Add LayerZero message delivery time
            lz_time = self._get_lz_delivery_time(source_chain, target_chain)
            
            # Add protocol overhead
            protocol_overhead = 60  # 1 minute for Stargate processing
            
            total_time = base_time + lz_time + protocol_overhead
            
            return total_time
            
        except Exception as e:
            logger.error(f"Error estimating time: {str(e)}")
            return 600  # Default 10 minutes
    
    def prepare_transfer(
        self,
        source_chain: str,
        target_chain: str,
        token: str,
        amount: float,
        recipient: str
    ) -> TxParams:
        """Prepare Stargate transfer transaction"""
        try:
            # Get router contract
            router = self._get_router_contract(source_chain)
            
            # Get pool ID for token
            pool_id = self.pool_ids[token]
            
            # Estimate fees
            fees = self.estimate_fees(source_chain, target_chain, token, amount)
            
            # Prepare transfer parameters
            transfer_params = {
                'dstChainId': self._get_lz_chain_id(target_chain),
                'srcPoolId': pool_id,
                'dstPoolId': pool_id,
                'refundAddress': recipient,
                'amountLD': self.web3.to_wei(amount, 'ether'),
                'minAmountLD': self.web3.to_wei(amount * 0.995, 'ether'),  # 0.5% slippage
                'dstGasForCall': 0,
                'lzTxParams': self._encode_lz_params(target_chain),
                'to': recipient,
                'payload': b''
            }
            
            # Encode transaction data
            tx_data = router.functions.swap(**transfer_params).build_transaction({
                'from': self.web3.eth.default_account,
                'value': Wei(fees['total']),
                'gas': self._estimate_gas_limit(source_chain),
                'nonce': self.web3.eth.get_transaction_count(self.web3.eth.default_account)
            })
            
            return tx_data
            
        except Exception as e:
            logger.error(f"Error preparing transfer: {str(e)}")
            raise
    
    def verify_message(
        self,
        source_chain: str,
        target_chain: str,
        message_hash: str,
        proof: bytes
    ) -> bool:
        """Verify Stargate message"""
        try:
            endpoint = self._get_lz_endpoint(target_chain)
            
            # Verify the message using LayerZero endpoint
            is_valid = endpoint.functions.validateProof(
                self._get_lz_chain_id(source_chain),
                message_hash,
                proof
            ).call()
            
            if not is_valid:
                logger.warning(f"Invalid message proof for hash {message_hash}")
                
            return is_valid
            
        except Exception as e:
            logger.error(f"Error verifying message: {str(e)}")
            return False
    
    def get_bridge_state(
        self,
        source_chain: str,
        target_chain: str
    ) -> BridgeState:
        """Get Stargate operational state"""
        try:
            # Check if chains are supported
            if not self._are_chains_supported(source_chain, target_chain):
                return BridgeState.OFFLINE
            
            # Check if router contracts are deployed and active
            if not self._are_contracts_active(source_chain, target_chain):
                return BridgeState.OFFLINE
            
            # Check if bridge is paused
            if self._is_bridge_paused(source_chain, target_chain):
                return BridgeState.PAUSED
            
            # Check LayerZero endpoint status
            if not self._is_lz_endpoint_active(source_chain, target_chain):
                return BridgeState.OFFLINE
            
            # Check congestion
            if self._is_congested(source_chain, target_chain):
                return BridgeState.CONGESTED
            
            # Check liquidity
            if self._is_low_liquidity(source_chain, target_chain):
                return BridgeState.LOW_LIQUIDITY
            
            return BridgeState.ACTIVE
            
        except Exception as e:
            logger.error(f"Error getting bridge state: {str(e)}")
            return BridgeState.OFFLINE
    
    def monitor_liquidity(
        self,
        chain: str,
        token: str
    ) -> float:
        """Monitor Stargate liquidity"""
        try:
            pool_id = self.pool_ids[token]
            pool = self._get_pool_contract(chain, pool_id)
            
            # Get pool balance
            balance = pool.functions.totalLiquidity().call()
            
            # Convert to float
            liquidity = float(self.web3.from_wei(balance, 'ether'))
            
            # Update metrics
            self.metrics.liquidity = liquidity
            self.metrics.last_updated = time.time()
            
            return liquidity
            
        except Exception as e:
            logger.error(f"Error monitoring liquidity: {str(e)}")
            return 0.0
    
    def recover_failed_transfer(
        self,
        source_chain: str,
        target_chain: str,
        tx_hash: str
    ) -> Optional[str]:
        """Recover failed Stargate transfer"""
        try:
            # Get original transaction
            tx = self.web3.eth.get_transaction(tx_hash)
            if not tx:
                logger.error(f"Transaction {tx_hash} not found")
                return None
            
            # Decode transaction data
            router = self._get_router_contract(source_chain)
            decoded = router.decode_function_input(tx['input'])
            
            # Prepare recovery transaction
            recovery_tx = self.prepare_transfer(
                source_chain,
                target_chain,
                self._get_token_from_pool_id(decoded[1]['srcPoolId']),
                float(self.web3.from_wei(decoded[1]['amountLD'], 'ether')),
                decoded[1]['refundAddress']
            )
            
            # Add recovery parameters
            recovery_tx['nonce'] = tx['nonce']
            recovery_tx['gasPrice'] = int(tx['gasPrice'] * 1.2)  # 20% higher gas
            
            # Send recovery transaction
            recovery_hash = self.web3.eth.send_transaction(recovery_tx)
            return recovery_hash.hex()
            
        except Exception as e:
            logger.error(f"Error recovering transfer: {str(e)}")
            return None
    
    # Helper methods
    def _init_contracts(self) -> None:
        """Initialize Stargate contracts"""
        pass  # Implementation details
    
    def _get_lz_fee(self, source_chain: str, target_chain: str) -> float:
        """Get LayerZero message fee"""
        pass  # Implementation details
    
    def _get_protocol_fee(self, amount: float) -> float:
        """Get Stargate protocol fee"""
        pass  # Implementation details
    
    def _get_lp_fee(self, amount: float) -> float:
        """Get LP fee"""
        pass  # Implementation details
    
    def _get_lz_delivery_time(self, source_chain: str, target_chain: str) -> int:
        """Get LayerZero message delivery time"""
        pass  # Implementation details
    
    def _get_router_contract(self, chain: str) -> Any:
        """Get router contract instance"""
        pass  # Implementation details
    
    def _get_pool_contract(self, chain: str, pool_id: int) -> Any:
        """Get pool contract instance"""
        pass  # Implementation details
    
    def _get_lz_endpoint(self, chain: str) -> Any:
        """Get LayerZero endpoint contract"""
        pass  # Implementation details
    
    def _get_lz_chain_id(self, chain: str) -> int:
        """Get LayerZero chain ID"""
        pass  # Implementation details
    
    def _encode_lz_params(self, target_chain: str) -> bytes:
        """Encode LayerZero parameters"""
        pass  # Implementation details
    
    def _estimate_gas_limit(self, chain: str) -> int:
        """Estimate gas limit"""
        pass  # Implementation details
    
    def _are_chains_supported(self, source_chain: str, target_chain: str) -> bool:
        """Check if chains are supported"""
        pass  # Implementation details
    
    def _are_contracts_active(self, source_chain: str, target_chain: str) -> bool:
        """Check if contracts are active"""
        pass  # Implementation details
    
    def _is_bridge_paused(self, source_chain: str, target_chain: str) -> bool:
        """Check if bridge is paused"""
        pass  # Implementation details
    
    def _is_lz_endpoint_active(self, source_chain: str, target_chain: str) -> bool:
        """Check if LayerZero endpoint is active"""
        pass  # Implementation details
    
    def _is_congested(self, source_chain: str, target_chain: str) -> bool:
        """Check if bridge is congested"""
        pass  # Implementation details
    
    def _is_low_liquidity(self, source_chain: str, target_chain: str) -> bool:
        """Check if liquidity is low"""
        pass  # Implementation details
    
    def _get_token_from_pool_id(self, pool_id: int) -> str:
        """Get token symbol from pool ID"""
        pass  # Implementation details 