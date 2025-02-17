from typing import Dict, Any, Optional, cast, TypedDict, Union
from web3 import Web3
from web3.types import TxParams, TxData, Wei, _Hash32, Nonce, ENS
from eth_typing import HexAddress, Address, ChecksumAddress
from hexbytes import HexBytes
import logging
import time
import json
from pathlib import Path

from .bridge_adapter import BridgeAdapter, BridgeConfig, BridgeState, BridgeMetrics
from .bridge_config import AcrossConfig, BridgeGlobalConfig, ChainConfig, TokenConfig

logger = logging.getLogger(__name__)

class TxDataExtended(TypedDict):
    """Extended transaction data type with required fields"""
    input: str
    nonce: Nonce
    gasPrice: Wei
    hash: HexBytes
    blockHash: HexBytes
    blockNumber: int
    from_: ChecksumAddress
    to: Optional[ChecksumAddress]
    value: Wei

class AcrossAdapter(BridgeAdapter):
    """Adapter implementation for Across protocol"""
    
    def __init__(self, config: BridgeConfig, web3: Web3):
        super().__init__(config, web3)
        if not config.across_config:
            raise ValueError("Across config is required")
            
        self.across_config = config.across_config
        self.global_config = config.global_config
        self.chain_configs = config.chain_configs
        self.token_configs = config.token_configs
        
        self.relayer_fee_pct = self.across_config.relayer_fee_pct
        self.lp_fee_pct = self.across_config.lp_fee_pct
        self.verification_gas_limit = self.across_config.verification_gas_limit
        
        # Cache for contract instances and ABIs
        self._contracts: Dict[str, Any] = {}
        self._abis: Dict[str, Any] = {}
        self._initialize_protocol()
    
    def _initialize_protocol(self) -> None:
        """Initialize Across protocol contracts and settings"""
        try:
            # Initialize router contracts for each chain
            self._init_router_contracts()
            
            # Validate configuration
            if not self.across_config.supported_chains:
                raise ValueError("No supported chains configured for Across")
                
            logger.info(f"Initialized Across adapter with {len(self.across_config.supported_chains)} supported chains")
            
        except Exception as e:
            logger.error(f"Failed to initialize Across adapter: {str(e)}")
            raise
    
    def validate_transfer(
        self,
        source_chain: str,
        target_chain: str,
        token: str,
        amount: float
    ) -> bool:
        """Validate if transfer is possible via Across"""
        try:
            # Check chain support
            if (source_chain not in self.across_config.supported_chains or 
                target_chain not in self.across_config.supported_chains):
                logger.warning(f"Chain pair {source_chain}->{target_chain} not supported")
                return False
                
            # Check token support
            if token not in self.across_config.supported_tokens:
                logger.warning(f"Token {token} not supported")
                return False
                
            # Check amount limits
            min_amount = self.across_config.min_deposit_amounts.get(token, 0)
            max_amount = self.across_config.max_deposit_amounts.get(token, float('inf'))
            
            if amount < min_amount:
                logger.warning(f"Amount {amount} below minimum {min_amount} for {token}")
                return False
                
            if amount > max_amount:
                logger.warning(f"Amount {amount} above maximum {max_amount} for {token}")
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
        """Estimate Across transfer fees"""
        try:
            # Calculate relayer fee
            relayer_fee = self._calculate_relayer_fee(amount, token, target_chain)
            
            # Calculate LP fee
            lp_fee = self._calculate_lp_fee(amount, token, target_chain)
            
            # Get gas fee for source chain
            gas_fee = self._estimate_gas_fee(source_chain, target_chain)
            
            total = relayer_fee + lp_fee + gas_fee
            
            return {
                'relayer_fee': relayer_fee,
                'lp_fee': lp_fee,
                'gas_fee': gas_fee,
                'total': total
            }
            
        except Exception as e:
            logger.error(f"Error estimating fees: {str(e)}")
            return {
                'relayer_fee': 0,
                'lp_fee': 0,
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
            
            # Add protocol overhead for relayer processing
            protocol_overhead = 300  # 5 minutes for Across processing
            
            total_time = base_time + source_delay + target_delay + protocol_overhead
            
            return total_time
            
        except Exception as e:
            logger.error(f"Error estimating time: {str(e)}")
            return 900  # Default 15 minutes
    
    def prepare_transfer(
        self,
        source_chain: str,
        target_chain: str,
        token: str,
        amount: float,
        recipient: str
    ) -> TxParams:
        """Prepare Across transfer transaction"""
        try:
            # Get router contract
            router = self._get_router_contract(source_chain)
            
            # Estimate fees
            fees = self.estimate_fees(source_chain, target_chain, token, amount)
            
            # Get token address
            token_address = self._get_token_address(token, source_chain)
            
            # Prepare deposit parameters
            deposit_params = {
                'token': token_address,
                'amount': self.web3.to_wei(amount, 'ether'),
                'destinationChainId': self._get_chain_id(target_chain),
                'relayerFeePct': int(self.relayer_fee_pct * 10000),  # Convert to basis points
                'recipient': recipient,
                'quoteTimestamp': int(time.time()),
                'message': b'',  # Optional message data
                'maxCount': 1    # Number of relayer submissions to wait for
            }
            
            # Encode transaction data
            tx_data = router.functions.deposit(**deposit_params).build_transaction({
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
        """Verify Across message"""
        try:
            router = self._get_router_contract(target_chain)
            
            # Verify the message using Across router
            is_valid = router.functions.verifyMessageHash(
                self._get_chain_id(source_chain),
                message_hash,
                proof
            ).call({'gas': self.verification_gas_limit})
            
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
        """Get Across operational state"""
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
        """Monitor Across liquidity"""
        try:
            router = self._get_router_contract(chain)
            token_address = self._get_token_address(token, chain)
            
            # Get pool liquidity
            liquidity = router.functions.getPoolLiquidity(token_address).call()
            
            # Convert to float
            liquidity_float = float(self.web3.from_wei(liquidity, 'ether'))
            
            # Update metrics
            self.metrics.liquidity = liquidity_float
            self.metrics.last_updated = time.time()
            
            return liquidity_float
            
        except Exception as e:
            logger.error(f"Error monitoring liquidity: {str(e)}")
            return 0.0
    
    def recover_failed_transfer(
        self,
        source_chain: str,
        target_chain: str,
        tx_hash: str
    ) -> Optional[str]:
        """Recover failed Across transfer"""
        try:
            # Get original transaction
            tx = self.web3.eth.get_transaction(HexBytes(tx_hash))
            if not tx:
                logger.error(f"Transaction {tx_hash} not found")
                return None
                
            # Decode transaction data
            router = self._get_router_contract(source_chain)
            tx_data = cast(TxDataExtended, tx)
            decoded = router.decode_function_input(tx_data['input'])
            
            # Prepare recovery transaction
            recovery_tx = self.prepare_transfer(
                source_chain,
                target_chain,
                decoded[1]['token'],
                float(self.web3.from_wei(decoded[1]['amount'], 'ether')),
                decoded[1]['recipient']
            )
            
            # Add recovery parameters
            recovery_tx['nonce'] = tx_data['nonce']
            recovery_tx['gasPrice'] = Wei(int(tx_data['gasPrice'] * 1.2))  # 20% higher gas
            
            # Send recovery transaction
            recovery_hash = self.web3.eth.send_transaction(recovery_tx)
            return recovery_hash.hex()
            
        except Exception as e:
            logger.error(f"Error recovering transfer: {str(e)}")
            return None
    
    # Helper methods
    def _load_contract_abi(self, name: str) -> Dict:
        """Load contract ABI from JSON file"""
        if name not in self._abis:
            abi_path = Path(__file__).parent / 'abis' / f'across_{name}.json'
            try:
                with open(abi_path) as f:
                    self._abis[name] = json.load(f)
            except FileNotFoundError:
                # Fallback to hardcoded minimal ABI if file not found
                self._abis[name] = [
                    {
                        "inputs": [
                            {"name": "token", "type": "address"},
                            {"name": "amount", "type": "uint256"},
                            {"name": "destinationChainId", "type": "uint256"},
                            {"name": "relayerFeePct", "type": "uint256"},
                            {"name": "recipient", "type": "address"},
                            {"name": "quoteTimestamp", "type": "uint256"},
                            {"name": "message", "type": "bytes"},
                            {"name": "maxCount", "type": "uint256"}
                        ],
                        "name": "deposit",
                        "outputs": [],
                        "stateMutability": "payable",
                        "type": "function"
                    },
                    {
                        "inputs": [{"name": "token", "type": "address"}],
                        "name": "getPoolLiquidity",
                        "outputs": [{"name": "", "type": "uint256"}],
                        "stateMutability": "view",
                        "type": "function"
                    },
                    {
                        "inputs": [],
                        "name": "isPaused",
                        "outputs": [{"name": "", "type": "bool"}],
                        "stateMutability": "view",
                        "type": "function"
                    }
                ]
        return self._abis[name]

    def _init_router_contracts(self) -> None:
        """Initialize Across router contracts"""
        try:
            # Load router ABI
            router_abi = self._load_contract_abi('router')
            
            # Initialize router contracts for each supported chain
            for chain in self.across_config.supported_chains:
                if chain not in self._contracts:
                    router_address = self.across_config.router_addresses.get(chain)
                    if not router_address:
                        logger.warning(f"No router address configured for chain {chain}")
                        continue
                        
                    self._contracts[chain] = self.web3.eth.contract(
                        address=router_address,
                        abi=router_abi
                    )
                    
            logger.info(f"Initialized {len(self._contracts)} router contracts")
            
        except Exception as e:
            logger.error(f"Error initializing router contracts: {str(e)}")
            raise

    def _calculate_relayer_fee(self, amount: float, token: str, target_chain: str) -> float:
        """Calculate relayer fee based on current network conditions"""
        try:
            # Get base relayer fee percentage
            base_fee_pct = self.relayer_fee_pct
            
            # Add chain-specific premium based on gas prices
            chain_config = self._get_chain_config(target_chain)
            current_gas_price = self.web3.eth.gas_price
            max_gas_price = self.web3.to_wei(chain_config.max_gas_price, 'gwei')
            
            # Calculate gas price premium (0-50% additional fee)
            gas_premium = min(0.5, current_gas_price / max_gas_price)
            
            # Add congestion premium if network is busy
            congestion_premium = 0.2 if self._is_congested(target_chain, target_chain) else 0
            
            # Calculate final fee percentage
            total_fee_pct = base_fee_pct * (1 + gas_premium + congestion_premium)
            
            # Calculate fee amount
            fee_amount = amount * total_fee_pct
            
            # Ensure minimum fee
            min_fee = self.across_config.min_deposit_amounts.get(token, 0) * 0.001
            return max(fee_amount, min_fee)
            
        except Exception as e:
            logger.error(f"Error calculating relayer fee: {str(e)}")
            return amount * self.relayer_fee_pct  # Fallback to base fee

    def _calculate_lp_fee(self, amount: float, token: str, target_chain: str) -> float:
        """Calculate liquidity provider fee based on pool conditions"""
        try:
            # Get base LP fee percentage
            base_fee_pct = self.lp_fee_pct
            
            # Check pool utilization
            token_address = self._get_token_address(token, target_chain)
            router = self._get_router_contract(target_chain)
            
            pool_liquidity = float(self.web3.from_wei(
                router.functions.getPoolLiquidity(token_address).call(),
                'ether'
            ))
            
            # Calculate utilization ratio (0-1)
            utilization = min(1.0, amount / pool_liquidity if pool_liquidity > 0 else 1.0)
            
            # Add premium based on utilization (0-100% additional fee)
            utilization_premium = utilization
            
            # Calculate final fee percentage
            total_fee_pct = base_fee_pct * (1 + utilization_premium)
            
            # Calculate fee amount
            fee_amount = amount * total_fee_pct
            
            # Ensure minimum fee
            min_fee = self.across_config.min_deposit_amounts.get(token, 0) * 0.001
            return max(fee_amount, min_fee)
            
        except Exception as e:
            logger.error(f"Error calculating LP fee: {str(e)}")
            return amount * self.lp_fee_pct  # Fallback to base fee

    def _has_sufficient_liquidity(self, token: str, chain: str, amount: float) -> bool:
        """Check if there's sufficient liquidity"""
        try:
            router = self._get_router_contract(chain)
            token_address = self._get_token_address(token, chain)
            
            # Get pool liquidity
            liquidity = router.functions.getPoolLiquidity(token_address).call()
            liquidity_float = float(self.web3.from_wei(liquidity, 'ether'))
            
            # Check against minimum liquidity ratio
            required_liquidity = amount * (1 + self.global_config.min_liquidity_ratio)
            if liquidity_float < required_liquidity:
                return False
                
            # Check against minimum USD liquidity
            if liquidity_float < self.global_config.min_liquidity_usd:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking liquidity: {str(e)}")
            return False
    
    def _estimate_gas_fee(self, source_chain: str, target_chain: str) -> float:
        """Estimate gas fee"""
        try:
            # Get chain configurations
            source_config = self._get_chain_config(source_chain)
            target_config = self._get_chain_config(target_chain)
            
            # Get current gas price
            gas_price = min(
                self.web3.eth.gas_price,
                self.web3.to_wei(source_config.max_gas_price, 'gwei')
            )
            
            # Estimate gas limit with multiplier
            base_gas = 250000  # Base gas for Across deposit
            gas_limit = int(base_gas * source_config.gas_limit_multiplier)
            
            # Calculate fee in ETH
            return float(self.web3.from_wei(gas_price * gas_limit, 'ether'))
            
        except Exception as e:
            logger.error(f"Error estimating gas fee: {str(e)}")
            return 0.0
    
    def _get_chain_delay(self, chain: str) -> int:
        """Get chain-specific delay"""
        try:
            chain_config = self._get_chain_config(chain)
            
            # Base delay from confirmation blocks
            base_delay = chain_config.confirmation_blocks * chain_config.block_time
            
            # Add chain-specific overhead
            chain_overhead = {
                'ethereum': 60,   # 1 minute
                'base': 30,       # 30 seconds
                'polygon': 90,    # 1.5 minutes
                'arbitrum': 45,   # 45 seconds
                'optimism': 45,   # 45 seconds
                'zksync': 60      # 1 minute
            }
            
            return base_delay + chain_overhead.get(chain, 60)
            
        except Exception as e:
            logger.error(f"Error getting chain delay: {str(e)}")
            return 300  # Default 5 minutes
    
    def _get_router_contract(self, chain: str) -> Any:
        """Get router contract instance"""
        try:
            if chain not in self._contracts:
                self._init_router_contracts()
            
            contract = self._contracts.get(chain)
            if not contract:
                raise ValueError(f"No router contract initialized for chain {chain}")
                
            return contract
            
        except Exception as e:
            logger.error(f"Error getting router contract: {str(e)}")
            raise
    
    def _get_token_address(self, token: str, chain: str) -> ChecksumAddress:
        """Get token address for chain"""
        token_config = self._get_token_config(token)
        return token_config.addresses[chain]
    
    def _get_chain_id(self, chain: str) -> int:
        """Get chain ID"""
        try:
            chain_config = self._get_chain_config(chain)
            return chain_config.chain_id
            
        except Exception as e:
            logger.error(f"Error getting chain ID: {str(e)}")
            raise
    
    def _estimate_gas_limit(self, chain: str) -> int:
        """Estimate gas limit"""
        try:
            chain_config = self._get_chain_config(chain)
            
            # Base gas limits for different operations
            base_limits = {
                'deposit': 250000,
                'message': 100000,
                'verification': self.verification_gas_limit
            }
            
            # Apply chain-specific multiplier
            return int(base_limits['deposit'] * chain_config.gas_limit_multiplier)
            
        except Exception as e:
            logger.error(f"Error estimating gas limit: {str(e)}")
            return 300000  # Default safe limit
    
    def _are_chains_supported(self, source_chain: str, target_chain: str) -> bool:
        """Check if chains are supported"""
        try:
            return (
                source_chain in self.across_config.supported_chains and
                target_chain in self.across_config.supported_chains and
                source_chain != target_chain
            )
            
        except Exception as e:
            logger.error(f"Error checking chain support: {str(e)}")
            return False
    
    def _are_contracts_active(self, source_chain: str, target_chain: str) -> bool:
        """Check if contracts are active"""
        try:
            # Check source chain contract
            source_router = self._get_router_contract(source_chain)
            source_code = self.web3.eth.get_code(source_router.address)
            if len(source_code) <= 2:  # "0x" or empty
                return False
                
            # Check target chain contract
            target_router = self._get_router_contract(target_chain)
            target_code = self.web3.eth.get_code(target_router.address)
            if len(target_code) <= 2:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking contract status: {str(e)}")
            return False
    
    def _is_bridge_paused(self, source_chain: str, target_chain: str) -> bool:
        """Check if bridge is paused"""
        try:
            # Check if either source or target router is paused
            source_router = self._get_router_contract(source_chain)
            if source_router.functions.isPaused().call():
                return True
                
            target_router = self._get_router_contract(target_chain)
            if target_router.functions.isPaused().call():
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking bridge pause status: {str(e)}")
            return True  # Assume paused on error for safety
    
    def _is_congested(self, source_chain: str, target_chain: str) -> bool:
        """Check if bridge is congested"""
        try:
            # Check pending transfers count (this would need to be implemented based on Across's API)
            # For now, using a simple gas price heuristic
            source_config = self._get_chain_config(source_chain)
            current_gas_price = self.web3.eth.gas_price
            
            # Consider congested if gas price is above 80% of max
            congestion_threshold = self.web3.to_wei(source_config.max_gas_price * 0.8, 'gwei')
            return current_gas_price > congestion_threshold
            
        except Exception as e:
            logger.error(f"Error checking congestion: {str(e)}")
            return True  # Assume congested on error for safety
    
    def _is_low_liquidity(self, source_chain: str, target_chain: str) -> bool:
        """Check if liquidity is low"""
        try:
            # Check liquidity for all supported tokens
            for token in self.across_config.supported_tokens:
                # Check source chain liquidity
                if not self._has_sufficient_liquidity(token, source_chain, 0):
                    return True
                    
                # Check target chain liquidity
                if not self._has_sufficient_liquidity(token, target_chain, 0):
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error checking liquidity status: {str(e)}")
            return True  # Assume low liquidity on error for safety
    
    def _get_chain_config(self, chain: str) -> ChainConfig:
        """Get chain configuration"""
        return self.chain_configs[chain]
        
    def _get_token_config(self, token: str) -> TokenConfig:
        """Get token configuration"""
        return self.token_configs[token]

    def get_transaction_count(
        self,
        address: Union[Address, ChecksumAddress, ENS],
        block_identifier: Optional[str] = None
    ) -> int:
        """Get transaction count for address"""
        return self.web3.eth.get_transaction_count(address, block_identifier)

    def _verify_message_proof(
        self,
        source_chain: str,
        target_chain: str,
        message_hash: str,
        proof: bytes
    ) -> bool:
        """Verify message proof using Across protocol verification"""
        try:
            router = self._get_router_contract(target_chain)
            source_chain_id = self._get_chain_id(source_chain)
            
            # Get verification parameters
            verification_params = {
                'sourceChainId': source_chain_id,
                'messageHash': message_hash,
                'proof': proof
            }
            
            # Call verification with gas limit
            is_valid = router.functions.verifyMessageHash(
                verification_params['sourceChainId'],
                verification_params['messageHash'],
                verification_params['proof']
            ).call({'gas': self.verification_gas_limit})
            
            if not is_valid:
                logger.warning(
                    f"Invalid message proof for hash {message_hash} "
                    f"from chain {source_chain} to {target_chain}"
                )
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Error verifying message proof: {str(e)}")
            return False
    
    def _monitor_pool_metrics(self, chain: str, token: str) -> Dict[str, float]:
        """Monitor detailed pool metrics"""
        try:
            router = self._get_router_contract(chain)
            token_address = self._get_token_address(token, chain)
            
            # Get current pool liquidity
            liquidity = router.functions.getPoolLiquidity(token_address).call()
            liquidity_float = float(self.web3.from_wei(liquidity, 'ether'))
            
            # Get minimum required liquidity
            min_required = self.across_config.min_deposit_amounts.get(token, 0) * 10  # 10x minimum deposit
            
            # Calculate utilization (if historical data available)
            utilization = self.metrics.utilization
            
            # Calculate liquidity ratio
            liquidity_ratio = (
                liquidity_float / min_required 
                if min_required > 0 
                else float('inf')
            )
            
            # Update metrics
            self.metrics.liquidity = liquidity_float
            self.metrics.last_updated = time.time()
            
            return {
                'liquidity': liquidity_float,
                'min_required': min_required,
                'utilization': utilization,
                'liquidity_ratio': liquidity_ratio
            }
            
        except Exception as e:
            logger.error(f"Error monitoring pool metrics: {str(e)}")
            return {
                'liquidity': 0.0,
                'min_required': 0.0,
                'utilization': 0.0,
                'liquidity_ratio': 0.0
            }
    
    def _update_transfer_metrics(
        self,
        success: bool,
        amount: float,
        duration: float,
        error: Optional[str] = None
    ) -> None:
        """Update transfer metrics"""
        try:
            # Update success/failure counts
            self.metrics.total_transfers += 1
            if not success:
                self.metrics.failed_transfers += 1
            
            # Update success rate
            self.metrics.success_rate = (
                (self.metrics.total_transfers - self.metrics.failed_transfers) /
                self.metrics.total_transfers
                if self.metrics.total_transfers > 0
                else 1.0
            )
            
            # Update average transfer time
            if success:
                self.metrics.avg_transfer_time = (
                    (self.metrics.avg_transfer_time * (self.metrics.total_transfers - 1) + duration) /
                    self.metrics.total_transfers
                )
            
            # Log metrics update
            logger.info(
                f"Updated transfer metrics - "
                f"Success Rate: {self.metrics.success_rate:.2%}, "
                f"Avg Time: {self.metrics.avg_transfer_time:.1f}s, "
                f"Total: {self.metrics.total_transfers}"
            )
            
            if error:
                logger.error(f"Transfer error: {error}")
            
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")
    
    def _validate_pool_health(
        self,
        chain: str,
        token: str,
        amount: float
    ) -> Dict[str, bool]:
        """Validate pool health status"""
        try:
            metrics = self._monitor_pool_metrics(chain, token)
            
            # Check liquidity ratio
            has_sufficient_liquidity = (
                metrics['liquidity_ratio'] >= self.global_config.min_liquidity_ratio
            )
            
            # Check absolute liquidity
            meets_min_liquidity = (
                metrics['liquidity'] >= self.global_config.min_liquidity_usd
            )
            
            # Check utilization
            is_not_overutilized = metrics['utilization'] < 0.8  # 80% utilization threshold
            
            # Check amount against liquidity
            can_handle_amount = amount <= metrics['liquidity'] * 0.2  # 20% of liquidity
            
            return {
                'has_sufficient_liquidity': has_sufficient_liquidity,
                'meets_min_liquidity': meets_min_liquidity,
                'is_not_overutilized': is_not_overutilized,
                'can_handle_amount': can_handle_amount,
                'is_healthy': all([
                    has_sufficient_liquidity,
                    meets_min_liquidity,
                    is_not_overutilized,
                    can_handle_amount
                ])
            }
            
        except Exception as e:
            logger.error(f"Error validating pool health: {str(e)}")
            return {
                'has_sufficient_liquidity': False,
                'meets_min_liquidity': False,
                'is_not_overutilized': False,
                'can_handle_amount': False,
                'is_healthy': False
            } 