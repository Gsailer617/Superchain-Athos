from typing import Dict, Any, Optional
from web3 import Web3
from web3.types import TxParams, Wei
from eth_typing import ChecksumAddress
import logging
import time

from .bridge_adapter import BridgeAdapter, BridgeConfig, BridgeState, BridgeMetrics

logger = logging.getLogger(__name__)

class LineaBridgeAdapter(BridgeAdapter):
    """Adapter implementation for Linea bridge protocol"""
    
    def __init__(self, config: BridgeConfig, web3: Web3):
        super().__init__(config, web3)
        self.linea_config = config.get('linea_config', {})
        self.message_service = self.linea_config.get('message_service', '')
        self.token_bridge = self.linea_config.get('token_bridge', '')
        
        # Cache for contract instances
        self._contracts = {}
        self._initialize_protocol()
    
    def _initialize_protocol(self) -> None:
        """Initialize Linea protocol contracts and settings"""
        try:
            # Initialize message service and token bridge contracts
            self._init_contracts()
            
            # Validate configuration
            if not self.config.supported_chains:
                raise ValueError("No supported chains configured for Linea")
                
            logger.info(f"Initialized Linea bridge adapter with {len(self.config.supported_chains)} supported chains")
            
        except Exception as e:
            logger.error(f"Failed to initialize Linea bridge adapter: {str(e)}")
            raise
    
    def validate_transfer(
        self,
        source_chain: str,
        target_chain: str,
        token: str,
        amount: float
    ) -> bool:
        """Validate if transfer is possible via Linea bridge"""
        try:
            # Check chain support (Linea only bridges with Ethereum mainnet)
            if not (
                (source_chain == "ethereum" and target_chain == "linea") or
                (source_chain == "linea" and target_chain == "ethereum")
            ):
                logger.warning(f"Chain pair {source_chain}->{target_chain} not supported")
                return False
            
            # Check amount limits
            if amount < self.config.min_amount or amount > self.config.max_amount:
                logger.warning(f"Amount {amount} outside limits [{self.config.min_amount}, {self.config.max_amount}]")
                return False
            
            # Check token support
            if not self._is_token_supported(token, source_chain):
                logger.warning(f"Token {token} not supported on {source_chain}")
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
        """Estimate Linea bridge fees"""
        try:
            # Get L1 gas cost for message verification
            l1_gas_cost = self._estimate_l1_gas_cost(source_chain, target_chain)
            
            # Get L2 execution cost
            l2_gas_cost = self._estimate_l2_gas_cost(source_chain, target_chain)
            
            # Get protocol fee (if any)
            protocol_fee = self._get_protocol_fee(amount)
            
            total = l1_gas_cost + l2_gas_cost + protocol_fee
            
            return {
                'l1_gas_cost': l1_gas_cost,
                'l2_gas_cost': l2_gas_cost,
                'protocol_fee': protocol_fee,
                'total': total
            }
            
        except Exception as e:
            logger.error(f"Error estimating fees: {str(e)}")
            return {
                'l1_gas_cost': 0,
                'l2_gas_cost': 0,
                'protocol_fee': 0,
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
            base_time = self.config.confirmation_blocks * 12  # 12s block time for Ethereum
            
            # Add message verification time
            verification_time = 1800  # 30 minutes for message verification
            
            # Add protocol overhead
            protocol_overhead = 300  # 5 minutes for processing
            
            total_time = base_time + verification_time + protocol_overhead
            
            return total_time
            
        except Exception as e:
            logger.error(f"Error estimating time: {str(e)}")
            return 3600  # Default 1 hour
    
    def prepare_transfer(
        self,
        source_chain: str,
        target_chain: str,
        token: str,
        amount: float,
        recipient: str
    ) -> TxParams:
        """Prepare Linea bridge transfer"""
        try:
            # Get bridge contract
            bridge = self._get_bridge_contract(source_chain)
            
            # Estimate fees
            fees = self.estimate_fees(source_chain, target_chain, token, amount)
            
            # Prepare transfer parameters
            transfer_params = {
                'token': self._get_token_address(token, source_chain),
                'amount': self.web3.to_wei(amount, 'ether'),
                'recipient': recipient,
                'targetChainId': self._get_chain_id(target_chain),
                'data': b''  # Optional payload data
            }
            
            # Encode transaction data
            tx_data = bridge.functions.bridgeToken(**transfer_params).build_transaction({
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
        """Verify Linea bridge message"""
        try:
            message_service = self._get_message_service(target_chain)
            
            # Verify the message proof
            is_valid = message_service.functions.verifyMessage(
                self._get_chain_id(source_chain),
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
        """Get Linea bridge operational state"""
        try:
            # Check if chains are supported
            if not self._are_chains_supported(source_chain, target_chain):
                return BridgeState.OFFLINE
            
            # Check if bridge contracts are deployed and active
            if not self._are_contracts_active(source_chain, target_chain):
                return BridgeState.OFFLINE
            
            # Check if bridge is paused
            if self._is_bridge_paused(source_chain, target_chain):
                return BridgeState.PAUSED
            
            # Check message service status
            if not self._is_message_service_active(source_chain, target_chain):
                return BridgeState.OFFLINE
            
            # Check congestion
            if self._is_congested(source_chain, target_chain):
                return BridgeState.CONGESTED
            
            return BridgeState.ACTIVE
            
        except Exception as e:
            logger.error(f"Error getting bridge state: {str(e)}")
            return BridgeState.OFFLINE
    
    def monitor_liquidity(
        self,
        chain: str,
        token: str
    ) -> float:
        """Monitor Linea bridge liquidity"""
        try:
            bridge = self._get_bridge_contract(chain)
            token_address = self._get_token_address(token, chain)
            
            # Get token balance
            balance = bridge.functions.getTokenBalance(token_address).call()
            
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
        """Recover failed Linea transfer"""
        try:
            # Get original transaction
            tx = self.web3.eth.get_transaction(tx_hash)
            if not tx:
                logger.error(f"Transaction {tx_hash} not found")
                return None
            
            # Decode transaction data
            bridge = self._get_bridge_contract(source_chain)
            decoded = bridge.decode_function_input(tx['input'])
            
            # Prepare recovery transaction
            recovery_tx = self.prepare_transfer(
                source_chain,
                target_chain,
                decoded[1]['token'],
                float(self.web3.from_wei(decoded[1]['amount'], 'ether')),
                decoded[1]['recipient']
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
        """Initialize Linea contracts"""
        pass  # Implementation details
    
    def _is_token_supported(self, token: str, chain: str) -> bool:
        """Check if token is supported on chain"""
        pass  # Implementation details
    
    def _has_sufficient_liquidity(self, token: str, chain: str, amount: float) -> bool:
        """Check if there's sufficient liquidity"""
        pass  # Implementation details
    
    def _estimate_l1_gas_cost(self, source_chain: str, target_chain: str) -> float:
        """Estimate L1 gas cost"""
        pass  # Implementation details
    
    def _estimate_l2_gas_cost(self, source_chain: str, target_chain: str) -> float:
        """Estimate L2 gas cost"""
        pass  # Implementation details
    
    def _get_protocol_fee(self, amount: float) -> float:
        """Get protocol fee"""
        pass  # Implementation details
    
    def _get_bridge_contract(self, chain: str) -> Any:
        """Get bridge contract instance"""
        pass  # Implementation details
    
    def _get_message_service(self, chain: str) -> Any:
        """Get message service contract instance"""
        pass  # Implementation details
    
    def _get_token_address(self, token: str, chain: str) -> ChecksumAddress:
        """Get token address on chain"""
        pass  # Implementation details
    
    def _get_chain_id(self, chain: str) -> int:
        """Get chain ID"""
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
    
    def _is_message_service_active(self, source_chain: str, target_chain: str) -> bool:
        """Check if message service is active"""
        pass  # Implementation details
    
    def _is_congested(self, source_chain: str, target_chain: str) -> bool:
        """Check if bridge is congested"""
        pass  # Implementation details 