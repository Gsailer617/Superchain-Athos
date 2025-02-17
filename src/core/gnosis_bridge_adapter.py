from typing import Dict, Any, Optional, cast, Mapping
from web3 import Web3
from web3.types import TxParams, TxData, Wei, _Hash32
from hexbytes import HexBytes
from eth_typing import ChecksumAddress, Address, HexStr
from dataclasses import dataclass
import logging
import time

from .bridge_adapter import BridgeAdapter, BridgeConfig, BridgeState, BridgeMetrics

logger = logging.getLogger(__name__)

@dataclass
class GnosisBridgeConfig:
    """Gnosis-specific bridge configuration"""
    omni_bridge: str
    amb_bridge: Optional[str] = None
    token_registry: Optional[str] = None

class GnosisBridgeAdapter(BridgeAdapter):
    """Adapter implementation for Gnosis bridge protocol"""
    
    def __init__(self, config: BridgeConfig, web3: Web3):
        super().__init__(config, web3)
        bridge_contracts = getattr(config, 'bridge_contracts', {}) or {}
        self.gnosis_config = GnosisBridgeConfig(
            omni_bridge=bridge_contracts.get('omni_bridge', ''),
            amb_bridge=bridge_contracts.get('amb_bridge'),
            token_registry=bridge_contracts.get('token_registry')
        )
        
        # Cache for contract instances
        self._contracts: Dict[str, Any] = {}
        self._initialize_protocol()
    
    def _initialize_protocol(self) -> None:
        """Initialize Gnosis protocol contracts and settings"""
        try:
            # Initialize bridge contracts
            self._init_contracts()
            
            # Validate configuration
            if not self.config.supported_chains:
                raise ValueError("No supported chains configured for Gnosis")
                
            logger.info(f"Initialized Gnosis bridge adapter with {len(self.config.supported_chains)} supported chains")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gnosis bridge adapter: {str(e)}")
            raise
    
    def validate_transfer(
        self,
        source_chain: str,
        target_chain: str,
        token: str,
        amount: float
    ) -> bool:
        """Validate if transfer is possible via Gnosis bridge"""
        try:
            # Check chain support (Gnosis bridges with Ethereum and other supported chains)
            if not (
                source_chain == "gnosis" or
                target_chain == "gnosis" or
                (source_chain in self.config.supported_chains and target_chain in self.config.supported_chains)
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
        """Estimate Gnosis bridge fees"""
        try:
            # Get bridge fee
            bridge_fee = self._get_bridge_fee(amount)
            
            # Get gas costs
            gas_cost = self._estimate_gas_cost(source_chain, target_chain)
            
            # Get AMB fee if applicable
            amb_fee = self._get_amb_fee(source_chain, target_chain) if self._is_amb_transfer(source_chain, target_chain) else 0
            
            total = bridge_fee + gas_cost + amb_fee
            
            return {
                'bridge_fee': bridge_fee,
                'gas_cost': gas_cost,
                'amb_fee': amb_fee,
                'total': total
            }
            
        except Exception as e:
            logger.error(f"Error estimating fees: {str(e)}")
            return {
                'bridge_fee': 0,
                'gas_cost': 0,
                'amb_fee': 0,
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
            base_time = self.config.confirmation_blocks * 5  # 5s block time for Gnosis
            
            # Add finality time
            finality_time = 60  # 1 minute for finality
            
            # Add protocol overhead
            protocol_overhead = 300  # 5 minutes for processing
            
            # Add AMB overhead if applicable
            if self._is_amb_transfer(source_chain, target_chain):
                protocol_overhead += 300  # Additional 5 minutes for AMB
            
            total_time = base_time + finality_time + protocol_overhead
            
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
        """Prepare Gnosis bridge transfer"""
        try:
            # Get appropriate bridge contract
            bridge = self._get_bridge_contract(source_chain)
            
            # Estimate fees
            fees = self.estimate_fees(source_chain, target_chain, token, amount)
            
            # Convert amount to Wei
            amount_wei = Wei(int(self.web3.to_wei(amount, 'ether')))
            
            # Prepare transfer parameters
            transfer_params = {
                'token': self._get_token_address(token, source_chain),
                'amount': amount_wei,
                'recipient': recipient,
                'targetChainId': self._get_chain_id(target_chain),
                'messageData': b''  # Optional data
            }
            
            # Get account nonce
            nonce = self.web3.eth.get_transaction_count(
                cast(Address, self.web3.eth.default_account)
            )
            
            # Encode transaction data
            tx_data = bridge.functions.relayTokens(**transfer_params).build_transaction({
                'from': self.web3.eth.default_account,
                'value': Wei(int(fees['total'])),
                'gas': self._estimate_gas_limit(source_chain),
                'nonce': nonce
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
        """Verify Gnosis bridge message"""
        try:
            bridge = self._get_bridge_contract(target_chain)
            
            # Convert message hash to bytes
            message_hash_bytes = HexBytes(message_hash)
            
            # Verify the message proof
            is_valid = bridge.functions.verifyMessage(
                self._get_chain_id(source_chain),
                message_hash_bytes,
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
        """Get Gnosis bridge operational state"""
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
            
            # Check AMB status if applicable
            if self._is_amb_transfer(source_chain, target_chain):
                if not self._is_amb_active(source_chain, target_chain):
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
        """Monitor Gnosis bridge liquidity"""
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
    
    # Helper methods
    def _init_contracts(self) -> None:
        """Initialize Gnosis contracts"""
        try:
            # Implementation details
            pass
        except Exception as e:
            logger.error(f"Error initializing contracts: {str(e)}")
    
    def _is_token_supported(self, token: str, chain: str) -> bool:
        """Check if token is supported on chain"""
        try:
            # Implementation details
            return True  # Placeholder
        except Exception:
            return False
    
    def _has_sufficient_liquidity(self, token: str, chain: str, amount: float) -> bool:
        """Check if there's sufficient liquidity"""
        try:
            # Implementation details
            return True  # Placeholder
        except Exception:
            return False
    
    def _get_bridge_fee(self, amount: float) -> float:
        """Get bridge fee"""
        try:
            # Implementation details
            return 0.0  # Placeholder
        except Exception as e:
            logger.error(f"Error getting bridge fee: {str(e)}")
            return 0.0
    
    def _estimate_gas_cost(self, source_chain: str, target_chain: str) -> float:
        """Estimate gas cost"""
        try:
            # Implementation details
            return 0.0  # Placeholder
        except Exception as e:
            logger.error(f"Error estimating gas cost: {str(e)}")
            return 0.0
    
    def _get_amb_fee(self, source_chain: str, target_chain: str) -> float:
        """Get AMB fee if applicable"""
        try:
            # Implementation details
            return 0.0  # Placeholder
        except Exception as e:
            logger.error(f"Error getting AMB fee: {str(e)}")
            return 0.0
    
    def _is_amb_transfer(self, source_chain: str, target_chain: str) -> bool:
        """Check if transfer uses AMB bridge"""
        try:
            # Implementation details
            return False  # Placeholder
        except Exception:
            return False
    
    def _get_bridge_contract(self, chain: str) -> Any:
        """Get bridge contract instance"""
        try:
            # Implementation details
            return self._contracts.get(chain)  # Placeholder
        except Exception as e:
            logger.error(f"Error getting bridge contract: {str(e)}")
            return None
    
    def _get_token_address(self, token: str, chain: str) -> ChecksumAddress:
        """Get token address on chain"""
        try:
            # Implementation details
            return self.web3.to_checksum_address('0x0000000000000000000000000000000000000000')  # Placeholder
        except Exception as e:
            logger.error(f"Error getting token address: {str(e)}")
            return self.web3.to_checksum_address('0x0000000000000000000000000000000000000000')
    
    def _get_chain_id(self, chain: str) -> int:
        """Get chain ID"""
        try:
            # Implementation details
            return 1  # Placeholder
        except Exception as e:
            logger.error(f"Error getting chain ID: {str(e)}")
            return 1
    
    def _estimate_gas_limit(self, chain: str) -> int:
        """Estimate gas limit"""
        try:
            # Implementation details
            return 300000  # Placeholder
        except Exception as e:
            logger.error(f"Error estimating gas limit: {str(e)}")
            return 300000
    
    def _are_chains_supported(self, source_chain: str, target_chain: str) -> bool:
        """Check if chains are supported"""
        try:
            # Implementation details
            return True  # Placeholder
        except Exception:
            return False
    
    def _are_contracts_active(self, source_chain: str, target_chain: str) -> bool:
        """Check if contracts are active"""
        try:
            # Implementation details
            return True  # Placeholder
        except Exception:
            return False
    
    def _is_bridge_paused(self, source_chain: str, target_chain: str) -> bool:
        """Check if bridge is paused"""
        try:
            # Implementation details
            return False  # Placeholder
        except Exception:
            return False
    
    def _is_amb_active(self, source_chain: str, target_chain: str) -> bool:
        """Check if AMB bridge is active"""
        try:
            # Implementation details
            return True  # Placeholder
        except Exception:
            return False
    
    def _is_congested(self, source_chain: str, target_chain: str) -> bool:
        """Check if bridge is congested"""
        try:
            # Implementation details
            return False  # Placeholder
        except Exception:
            return False 