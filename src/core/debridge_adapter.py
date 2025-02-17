from typing import Dict, Any, Optional
from web3 import Web3
from web3.types import TxParams
from eth_typing import HexAddress
import logging
import time

from .bridge_adapter import BridgeAdapter, BridgeConfig, BridgeState, BridgeMetrics

logger = logging.getLogger(__name__)

class DeBridgeAdapter(BridgeAdapter):
    """Adapter implementation for deBridge protocol"""
    
    def __init__(self, config: BridgeConfig, web3: Web3):
        super().__init__(config, web3)
        self.debridge_config = config.debridge_config
        self.execution_fee_multiplier = self.debridge_config.get('execution_fee_multiplier', 1.2)
        self.claim_timeout = self.debridge_config.get('claim_timeout', 7200)  # 2 hours default
        self.auto_claim = self.debridge_config.get('auto_claim', True)
        
        # Cache for contract instances
        self._contracts = {}
        self._initialize_protocol()
    
    def _initialize_protocol(self) -> None:
        """Initialize deBridge protocol contracts and settings"""
        try:
            # Initialize core contracts based on chain
            self._init_core_contracts()
            
            # Validate configuration
            if not self.config.supported_chains:
                raise ValueError("No supported chains configured for deBridge")
                
            logger.info(f"Initialized deBridge adapter with {len(self.config.supported_chains)} supported chains")
            
        except Exception as e:
            logger.error(f"Failed to initialize deBridge adapter: {str(e)}")
            raise
    
    def validate_transfer(
        self,
        source_chain: str,
        target_chain: str,
        token: str,
        amount: float
    ) -> bool:
        """Validate if transfer is possible via deBridge"""
        try:
            # Check chain support
            if source_chain not in self.config.supported_chains or target_chain not in self.config.supported_chains:
                logger.warning(f"Chain pair {source_chain}->{target_chain} not supported")
                return False
                
            # Check amount limits
            if amount < self.config.min_amount or amount > self.config.max_amount:
                logger.warning(f"Amount {amount} outside limits [{self.config.min_amount}, {self.config.max_amount}]")
                return False
                
            # Check token support on both chains
            if not self._is_token_supported(token, source_chain) or not self._is_token_supported(token, target_chain):
                logger.warning(f"Token {token} not supported on {source_chain} or {target_chain}")
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
        """Estimate deBridge transfer fees"""
        try:
            # Get base protocol fee
            base_fee = self._get_protocol_fee(source_chain, target_chain, token, amount)
            
            # Get execution fee for target chain
            execution_fee = self._get_execution_fee(target_chain) * self.execution_fee_multiplier
            
            # Get gas fee for source chain
            gas_fee = self._estimate_gas_fee(source_chain, target_chain)
            
            total = base_fee + execution_fee + gas_fee
            
            return {
                'base_fee': base_fee,
                'execution_fee': execution_fee,
                'gas_fee': gas_fee,
                'total': total
            }
            
        except Exception as e:
            logger.error(f"Error estimating fees: {str(e)}")
            return {
                'base_fee': 0,
                'execution_fee': 0,
                'gas_fee': 0,
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
            
            # Add chain-specific delays
            source_delay = self._get_chain_delay(source_chain)
            target_delay = self._get_chain_delay(target_chain)
            
            # Add protocol overhead
            protocol_overhead = 60  # 1 minute for deBridge processing
            
            total_time = base_time + source_delay + target_delay + protocol_overhead
            
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
        """Prepare deBridge transfer transaction"""
        try:
            # Get protocol contracts
            bridge_contract = self._get_bridge_contract(source_chain)
            
            # Estimate fees
            fees = self.estimate_fees(source_chain, target_chain, token, amount)
            
            # Prepare transfer parameters
            transfer_params = {
                'token': self._get_token_address(token, source_chain),
                'amount': self.web3.to_wei(amount, 'ether'),
                'receiver': recipient,
                'targetChainId': self._get_chain_id(target_chain),
                'executionFee': self.web3.to_wei(fees['execution_fee'], 'ether'),
                'flags': self._get_transfer_flags(),
                'data': b''  # Optional payload data
            }
            
            # Encode transaction data
            tx_data = bridge_contract.functions.send(**transfer_params).build_transaction({
                'from': self.web3.eth.default_account,
                'value': self.web3.to_wei(fees['total'], 'ether'),
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
        """Verify deBridge message"""
        try:
            verifier_contract = self._get_verifier_contract(target_chain)
            
            # Verify the message proof
            is_valid = verifier_contract.functions.verifyProof(
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
        """Get deBridge operational state"""
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
        """Monitor deBridge liquidity"""
        try:
            bridge_contract = self._get_bridge_contract(chain)
            token_address = self._get_token_address(token, chain)
            
            # Get token balance
            balance = bridge_contract.functions.getTokenBalance(token_address).call()
            
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
        """Recover failed deBridge transfer"""
        try:
            # Get original transaction
            tx = self.web3.eth.get_transaction(tx_hash)
            if not tx:
                logger.error(f"Transaction {tx_hash} not found")
                return None
                
            # Decode transaction data
            bridge_contract = self._get_bridge_contract(source_chain)
            decoded = bridge_contract.decode_function_input(tx['input'])
            
            # Prepare recovery transaction
            recovery_tx = self.prepare_transfer(
                source_chain,
                target_chain,
                decoded[1]['token'],
                float(self.web3.from_wei(decoded[1]['amount'], 'ether')),
                decoded[1]['receiver']
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
    def _init_core_contracts(self) -> None:
        """Initialize core deBridge contracts"""
        pass  # Implementation details
    
    def _is_token_supported(self, token: str, chain: str) -> bool:
        """Check if token is supported on chain"""
        pass  # Implementation details
    
    def _has_sufficient_liquidity(self, token: str, chain: str, amount: float) -> bool:
        """Check if there's sufficient liquidity"""
        pass  # Implementation details
    
    def _get_protocol_fee(self, source_chain: str, target_chain: str, token: str, amount: float) -> float:
        """Get deBridge protocol fee"""
        pass  # Implementation details
    
    def _get_execution_fee(self, chain: str) -> float:
        """Get execution fee for chain"""
        pass  # Implementation details
    
    def _estimate_gas_fee(self, source_chain: str, target_chain: str) -> float:
        """Estimate gas fee"""
        pass  # Implementation details
    
    def _get_chain_delay(self, chain: str) -> int:
        """Get chain-specific delay"""
        pass  # Implementation details
    
    def _get_bridge_contract(self, chain: str) -> Any:
        """Get bridge contract instance"""
        pass  # Implementation details
    
    def _get_verifier_contract(self, chain: str) -> Any:
        """Get verifier contract instance"""
        pass  # Implementation details
    
    def _get_token_address(self, token: str, chain: str) -> HexAddress:
        """Get token address on chain"""
        pass  # Implementation details
    
    def _get_chain_id(self, chain: str) -> int:
        """Get chain ID"""
        pass  # Implementation details
    
    def _get_transfer_flags(self) -> int:
        """Get transfer flags"""
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
    
    def _is_congested(self, source_chain: str, target_chain: str) -> bool:
        """Check if bridge is congested"""
        pass  # Implementation details
    
    def _is_low_liquidity(self, source_chain: str, target_chain: str) -> bool:
        """Check if liquidity is low"""
        pass  # Implementation details 