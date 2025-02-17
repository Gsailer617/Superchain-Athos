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
class ModeBridgeConfig:
    """Mode-specific bridge configuration"""
    l1_bridge: str
    l2_bridge: str
    message_service: str

class ModeBridgeAdapter(BridgeAdapter):
    """Adapter implementation for Mode bridge protocol"""
    
    def __init__(self, config: BridgeConfig, web3: Web3):
        super().__init__(config, web3)
        bridge_contracts = getattr(config, 'bridge_contracts', {}) or {}
        self.mode_config = ModeBridgeConfig(
            l1_bridge=bridge_contracts.get('l1_bridge', ''),
            l2_bridge=bridge_contracts.get('l2_bridge', ''),
            message_service=bridge_contracts.get('message_service', '')
        )
        
        # Cache for contract instances
        self._contracts: Dict[str, Any] = {}
        self._initialize_protocol()
    
    def _initialize_protocol(self) -> None:
        """Initialize Mode protocol contracts and settings"""
        try:
            # Initialize bridge contracts
            self._init_contracts()
            
            # Validate configuration
            if not self.config.supported_chains:
                raise ValueError("No supported chains configured for Mode")
                
            logger.info(f"Initialized Mode bridge adapter with {len(self.config.supported_chains)} supported chains")
            
        except Exception as e:
            logger.error(f"Failed to initialize Mode bridge adapter: {str(e)}")
            raise
    
    def validate_transfer(
        self,
        source_chain: str,
        target_chain: str,
        token: str,
        amount: float
    ) -> bool:
        """Validate if transfer is possible via Mode bridge"""
        try:
            # Check chain support (Mode only bridges with Ethereum mainnet)
            if not (
                (source_chain == "ethereum" and target_chain == "mode") or
                (source_chain == "mode" and target_chain == "ethereum")
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
        """Estimate Mode bridge fees"""
        try:
            # Get L1 data availability cost
            l1_da_cost = self._estimate_l1_da_cost(source_chain, target_chain)
            
            # Get L2 execution cost
            l2_execution_cost = self._estimate_l2_execution_cost(source_chain, target_chain)
            
            # Get bridge fee
            bridge_fee = self._get_bridge_fee(amount)
            
            total = l1_da_cost + l2_execution_cost + bridge_fee
            
            return {
                'l1_da_cost': l1_da_cost,
                'l2_execution_cost': l2_execution_cost,
                'bridge_fee': bridge_fee,
                'total': total
            }
            
        except Exception as e:
            logger.error(f"Error estimating fees: {str(e)}")
            return {
                'l1_da_cost': 0,
                'l2_execution_cost': 0,
                'bridge_fee': 0,
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
        """Prepare Mode bridge transfer"""
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
                'data': b''  # Optional data
            }
            
            # Get account nonce
            nonce = self.web3.eth.get_transaction_count(
                cast(Address, self.web3.eth.default_account)
            )
            
            # Encode transaction data
            tx_data = bridge.functions.bridgeToken(**transfer_params).build_transaction({
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
        """Verify Mode bridge message"""
        try:
            message_service = self._get_message_service(target_chain)
            
            # Convert message hash to bytes
            message_hash_bytes = HexBytes(message_hash)
            
            # Verify the message proof
            is_valid = message_service.functions.verifyMessage(
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
        """Get Mode bridge operational state"""
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
        """Monitor Mode bridge liquidity"""
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
        """Recover failed Mode transfer"""
        try:
            # Convert transaction hash to bytes
            tx_hash_bytes = HexBytes(tx_hash)
            
            # Get original transaction
            tx = cast(TxData, self.web3.eth.get_transaction(tx_hash_bytes))
            if not tx:
                logger.error(f"Transaction {tx_hash} not found")
                return None
            
            # Get bridge contract
            bridge = self._get_bridge_contract(source_chain)
            
            # Safely access transaction data
            tx_input = tx.get('input')
            if not tx_input:
                logger.error("Transaction input data not found")
                return None
            
            # Decode transaction data
            decoded = bridge.decode_function_input(tx_input)
            
            # Safely get nonce and gas price
            tx_nonce = tx.get('nonce')
            tx_gas_price = tx.get('gasPrice')
            
            if tx_nonce is None or tx_gas_price is None:
                logger.error("Transaction nonce or gas price not found")
                return None
            
            # Prepare recovery transaction
            recovery_tx = self.prepare_transfer(
                source_chain,
                target_chain,
                decoded[1]['token'],
                float(self.web3.from_wei(decoded[1]['amount'], 'ether')),
                decoded[1]['recipient']
            )
            
            # Add recovery parameters
            recovery_tx['nonce'] = tx_nonce
            recovery_tx['gasPrice'] = Wei(int(tx_gas_price * 1.2))  # 20% higher gas
            
            # Send recovery transaction
            recovery_hash = self.web3.eth.send_transaction(recovery_tx)
            return recovery_hash.hex()
            
        except Exception as e:
            logger.error(f"Error recovering transfer: {str(e)}")
            return None
    
    # Helper methods
    def _init_contracts(self) -> None:
        """Initialize Mode contracts"""
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
    
    def _estimate_l1_da_cost(self, source_chain: str, target_chain: str) -> float:
        """Estimate L1 data availability cost"""
        try:
            # Implementation details
            return 0.0  # Placeholder
        except Exception as e:
            logger.error(f"Error estimating L1 DA cost: {str(e)}")
            return 0.0
    
    def _estimate_l2_execution_cost(self, source_chain: str, target_chain: str) -> float:
        """Estimate L2 execution cost"""
        try:
            # Implementation details
            return 0.0  # Placeholder
        except Exception as e:
            logger.error(f"Error estimating L2 execution cost: {str(e)}")
            return 0.0
    
    def _get_bridge_fee(self, amount: float) -> float:
        """Get bridge fee"""
        try:
            # Implementation details
            return 0.0  # Placeholder
        except Exception as e:
            logger.error(f"Error getting bridge fee: {str(e)}")
            return 0.0
    
    def _get_bridge_contract(self, chain: str) -> Any:
        """Get bridge contract instance"""
        try:
            # Implementation details
            return self._contracts.get(chain)  # Placeholder
        except Exception as e:
            logger.error(f"Error getting bridge contract: {str(e)}")
            return None
    
    def _get_message_service(self, chain: str) -> Any:
        """Get message service contract instance"""
        try:
            # Implementation details
            return self._contracts.get(f"{chain}_message_service")  # Placeholder
        except Exception as e:
            logger.error(f"Error getting message service: {str(e)}")
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
    
    def _is_message_service_active(self, source_chain: str, target_chain: str) -> bool:
        """Check if message service is active"""
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