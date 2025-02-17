import logging
from typing import Dict, Union, Optional, Tuple, List, cast, Any, TypedDict
from web3 import Web3
from web3.types import TxParams, TxReceipt, Wei, Nonce
from eth_typing import Address, ChecksumAddress
import time
import os
from dataclasses import dataclass

from src.core.types import OpportunityType, FlashLoanOpportunityType
from src.core.register_adapters import get_registered_adapters
from src.core.bridge_adapter import BridgeConfig, BridgeState

logger = logging.getLogger(__name__)

class CrossChainOpportunityType(TypedDict):
    """Type definition for cross-chain opportunity"""
    source_chain: str
    target_chain: str
    token_pair: Tuple[str, str]
    amount: float
    recipient: str

@dataclass
class CrossChainTxResult:
    """Result of cross-chain transaction preparation"""
    source_tx: Optional[TxParams]
    target_tx: Optional[TxParams]
    bridge_name: str
    estimated_time: int
    total_fee: float
    success: bool
    error: Optional[str] = None

class CrossChainTransactionBuilder:
    """Builds transactions for cross-chain arbitrage"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.web3_connections: Dict[str, Web3] = {}
        self._initialize_connections()
        
    def _initialize_connections(self) -> None:
        """Initialize Web3 connections for each chain"""
        for chain in self.config['supported_chains']:
            try:
                alchemy_key = os.getenv(f'{chain.upper()}_ALCHEMY_KEY')
                if not alchemy_key:
                    logger.error(f"Missing Alchemy key for {chain}")
                    continue
                    
                rpc_url = f"https://{chain}-mainnet.g.alchemy.com/v2/{alchemy_key}"
                web3 = Web3(Web3.HTTPProvider(
                    rpc_url,
                    request_kwargs={
                        'timeout': 30,
                        'headers': {'User-Agent': 'FlashingBase/1.0.0'}
                    }
                ))
                
                if web3.is_connected():
                    self.web3_connections[chain] = web3
                    logger.info(f"Connected to {chain}")
                else:
                    logger.error(f"Failed to connect to {chain}")
                    
            except Exception as e:
                logger.error(f"Error initializing {chain} connection: {str(e)}")
                
    async def build_cross_chain_transaction(
        self,
        opportunity: CrossChainOpportunityType
    ) -> CrossChainTxResult:
        """Build transactions for cross-chain arbitrage
        
        Args:
            opportunity: Cross-chain arbitrage opportunity
            
        Returns:
            CrossChainTxResult containing source and target transactions
        """
        try:
            source_chain = opportunity['source_chain']
            target_chain = opportunity['target_chain']
            
            # Get Web3 connections
            source_web3 = self.web3_connections.get(source_chain)
            target_web3 = self.web3_connections.get(target_chain)
            
            if not source_web3 or not target_web3:
                return CrossChainTxResult(
                    source_tx=None,
                    target_tx=None,
                    bridge_name="",
                    estimated_time=0,
                    total_fee=0,
                    success=False,
                    error="Missing Web3 connection"
                )
            
            # Get available bridges
            bridge_analysis = await self._analyze_bridges(
                opportunity['token_pair'],
                opportunity['amount'],
                source_chain,
                target_chain,
                source_web3
            )
            
            if not bridge_analysis['success']:
                return CrossChainTxResult(
                    source_tx=None,
                    target_tx=None,
                    bridge_name="",
                    estimated_time=0,
                    total_fee=0,
                    success=False,
                    error=bridge_analysis['error']
                )
            
            # Prepare transactions
            source_tx = await self._build_source_transaction(
                opportunity,
                bridge_analysis,
                source_web3
            )
            
            target_tx = await self._build_target_transaction(
                opportunity,
                bridge_analysis,
                target_web3
            )
            
            return CrossChainTxResult(
                source_tx=source_tx,
                target_tx=target_tx,
                bridge_name=bridge_analysis['recommended_bridge'],
                estimated_time=bridge_analysis['estimated_time'],
                total_fee=bridge_analysis['total_fee'],
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error building cross-chain transaction: {str(e)}")
            return CrossChainTxResult(
                source_tx=None,
                target_tx=None,
                bridge_name="",
                estimated_time=0,
                total_fee=0,
                success=False,
                error=str(e)
            )
            
    async def _analyze_bridges(
        self,
        token_pair: Tuple[str, str],
        amount: float,
        source_chain: str,
        target_chain: str,
        web3: Web3
    ) -> Dict[str, Any]:
        """Analyze available bridges and select optimal one"""
        try:
            registered_adapters = get_registered_adapters()
            
            best_bridge = None
            lowest_fee = float('inf')
            best_time = float('inf')
            
            results = {
                'success': False,
                'recommended_bridge': "",
                'estimated_time': 0,
                'total_fee': 0,
                'error': None
            }
            
            for bridge_name, adapter_class in registered_adapters.items():
                try:
                    # Create bridge config
                    config = self._create_bridge_config(bridge_name, source_chain, target_chain)
                    
                    # Initialize adapter
                    adapter = adapter_class(config, web3)
                    
                    # Check if bridge is active and supports transfer
                    if (adapter.get_bridge_state(source_chain, target_chain) == BridgeState.ACTIVE and
                        adapter.validate_transfer(source_chain, target_chain, token_pair[0], amount)):
                        
                        # Get fees and time estimate
                        fees = adapter.estimate_fees(source_chain, target_chain, token_pair[0], amount)
                        time_estimate = adapter.estimate_time(source_chain, target_chain)
                        
                        # Update best bridge if this one has lower fees
                        total_fee = fees.get('total', float('inf'))
                        if total_fee < lowest_fee:
                            best_bridge = bridge_name
                            lowest_fee = total_fee
                            best_time = time_estimate
                            
                except Exception as e:
                    logger.error(f"Error analyzing bridge {bridge_name}: {str(e)}")
                    continue
                    
            if best_bridge:
                results.update({
                    'success': True,
                    'recommended_bridge': best_bridge,
                    'estimated_time': best_time,
                    'total_fee': lowest_fee
                })
            else:
                results['error'] = "No suitable bridge found"
                
            return results
            
        except Exception as e:
            logger.error(f"Error in bridge analysis: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    def _create_bridge_config(
        self,
        bridge_name: str,
        source_chain: str,
        target_chain: str
    ) -> BridgeConfig:
        """Create bridge configuration"""
        base_config = BridgeConfig(
            name=bridge_name,
            supported_chains=[source_chain, target_chain],
            min_amount=self.config.get('min_amount', 0.0),
            max_amount=self.config.get('max_amount', float('inf')),
            fee_multiplier=self.config.get('fee_multiplier', 1.0),
            gas_limit_multiplier=self.config.get('gas_limit_multiplier', 1.2),
            confirmation_blocks=self.config.get('confirmation_blocks', 1)
        )
        
        # Add chain-specific bridge contracts
        if bridge_name == 'mode':
            base_config.bridge_contracts = {
                'l1_bridge': self.config.get('mode_config', {}).get('l1_bridge', ''),
                'l2_bridge': self.config.get('mode_config', {}).get('l2_bridge', ''),
                'message_service': self.config.get('mode_config', {}).get('message_service', '')
            }
        elif bridge_name == 'sonic':
            base_config.bridge_contracts = {
                'bridge_router': self.config.get('sonic_config', {}).get('bridge_router', ''),
                'token_bridge': self.config.get('sonic_config', {}).get('token_bridge', ''),
                'liquidity_pool': self.config.get('sonic_config', {}).get('liquidity_pool', '')
            }
        
        return base_config
        
    async def _build_source_transaction(
        self,
        opportunity: CrossChainOpportunityType,
        bridge_analysis: Dict[str, Any],
        web3: Web3
    ) -> Optional[TxParams]:
        """Build transaction for source chain"""
        try:
            # Get bridge adapter
            adapter_class = get_registered_adapters()[bridge_analysis['recommended_bridge']]
            config = self._create_bridge_config(
                bridge_analysis['recommended_bridge'],
                opportunity['source_chain'],
                opportunity['target_chain']
            )
            
            adapter = adapter_class(config, web3)
            
            # Prepare bridge transfer
            tx_params = adapter.prepare_transfer(
                opportunity['source_chain'],
                opportunity['target_chain'],
                opportunity['token_pair'][0],
                opportunity['amount'],
                opportunity['recipient']
            )
            
            # Add chain-specific parameters
            if opportunity['source_chain'] == 'mode':
                # Mode uses optimized gas parameters
                tx_params['maxFeePerGas'] = Wei(int(tx_params.get('gasPrice', 0) * 0.8))  # 20% lower than standard
                if 'gasPrice' in tx_params:
                    del tx_params['gasPrice']  # Remove legacy gas price when using EIP-1559
            elif opportunity['source_chain'] == 'sonic':
                # Sonic uses fixed priority fee
                tx_params['maxPriorityFeePerGas'] = Wei(1_000_000_000)  # 1 gwei
                if 'gasPrice' in tx_params:
                    del tx_params['gasPrice']  # Remove legacy gas price when using EIP-1559
            
            return tx_params
            
        except Exception as e:
            logger.error(f"Error building source transaction: {str(e)}")
            return None
            
    async def _build_target_transaction(
        self,
        opportunity: CrossChainOpportunityType,
        bridge_analysis: Dict[str, Any],
        web3: Web3
    ) -> Optional[TxParams]:
        """Build transaction for target chain"""
        try:
            # For target chain, we typically need to prepare a transaction that will
            # execute once the bridged funds arrive. This might involve:
            # 1. Swapping the received tokens
            # 2. Sending them to a specific contract
            # 3. Executing an arbitrage
            
            # This is a placeholder - implement based on your specific needs
            return None
            
        except Exception as e:
            logger.error(f"Error building target transaction: {str(e)}")
            return None
            
    async def _get_optimal_gas_price(self, web3: Web3) -> int:
        """Get optimal gas price for chain"""
        try:
            block = web3.eth.get_block('latest')
            base_fee = block.get('baseFeePerGas', web3.eth.gas_price)
            priority_fee = web3.eth.max_priority_fee
            
            return int(base_fee * 1.2) + priority_fee
            
        except Exception as e:
            logger.error(f"Error getting gas price: {str(e)}")
            return web3.eth.gas_price
            
    async def _estimate_gas(self, web3: Web3, tx_params: TxParams) -> int:
        """Estimate gas for transaction"""
        return web3.eth.estimate_gas(tx_params)

class TransactionBuilder:
    """Builds and signs transactions for execution"""
    
    def __init__(self, config: Dict):
        """Initialize transaction builder with configuration"""
        # Get Alchemy key from environment
        alchemy_key = os.getenv('ALCHEMY_API_KEY')
        if not alchemy_key:
            raise ValueError("ALCHEMY_API_KEY environment variable is not set")
            
        # Initialize Web3 with Alchemy
        self.web3 = Web3(Web3.HTTPProvider(
            f"https://base-mainnet.g.alchemy.com/v2/{alchemy_key}",
            request_kwargs={
                'timeout': 30,
                'headers': {'User-Agent': 'FlashingBase/1.0.0'}
            }
        ))
        
        if not self.web3.is_connected():
            raise ValueError("Failed to connect to Base mainnet via Alchemy")
            
        # Set default account from private key
        private_key = os.getenv('MAINNET_PRIVATE_KEY')
        if not private_key:
            raise ValueError("MAINNET_PRIVATE_KEY not set")
        account = self.web3.eth.account.from_key(private_key)
        self.web3.eth.default_account = self.web3.to_checksum_address(account.address)
            
        self.config = config
        
    async def build_transaction(
        self,
        opportunity: Union[OpportunityType, FlashLoanOpportunityType]
    ) -> Optional[TxParams]:
        """Build transaction for arbitrage execution
        
        Args:
            opportunity: Arbitrage opportunity to execute
            
        Returns:
            TxParams containing transaction parameters or None if build fails
        """
        try:
            # Get optimal gas price
            gas_price = await self._get_optimal_gas_price()
            
            # Build base transaction
            tx: TxParams = {
                'from': self.web3.to_checksum_address(str(self.web3.eth.default_account)),
                'gasPrice': Wei(gas_price),
                'nonce': Nonce(await self._get_nonce()),
                'chainId': self.web3.eth.chain_id,
                'value': Wei(0),
            }
            
            # Add opportunity-specific parameters
            if opportunity['type'] == 'Flash Loan Arbitrage':
                tx.update(await self._build_flash_loan_tx(opportunity))
            else:
                tx.update(await self._build_regular_arb_tx(opportunity))
                
            # Estimate gas and add buffer
            try:
                gas_estimate = await self._estimate_gas(tx)
                tx['gas'] = int(gas_estimate * 1.2)  # Add 20% buffer
            except Exception as e:
                logger.error(f"Error estimating gas: {str(e)}")
                return None
                
            # Sign transaction
            signed_tx = await self._sign_transaction(tx)
            if not signed_tx:
                return None
                
            return tx
            
        except Exception as e:
            logger.error(f"Error building transaction: {str(e)}")
            return None
            
    async def _build_flash_loan_tx(
        self,
        opportunity: FlashLoanOpportunityType
    ) -> TxParams:
        """Build flash loan specific transaction parameters"""
        try:
            provider = opportunity['flash_loan_provider']
            provider_config = self._get_provider_config(provider)
            
            return {
                'to': provider_config['router'],
                'data': self._encode_flash_loan_data(
                    opportunity['token_pair'][0],
                    opportunity['amount'],
                    opportunity['dex_weights']
                ),
                'value': Wei(0)  # Flash loans don't require ETH
            }
            
        except Exception as e:
            logger.error(f"Error building flash loan tx: {str(e)}")
            raise
            
    async def _build_regular_arb_tx(
        self,
        opportunity: OpportunityType
    ) -> TxParams:
        """Build regular arbitrage transaction parameters"""
        try:
            return {
                'to': self._get_router_address(),
                'data': self._encode_swap_data(
                    opportunity['token_pair'],
                    opportunity['amount'],
                    opportunity['path']
                ),
                'value': Wei(opportunity.get('value', 0))
            }
            
        except Exception as e:
            logger.error(f"Error building regular arb tx: {str(e)}")
            raise
            
    async def _get_optimal_gas_price(self) -> int:
        """Get optimal gas price with current network conditions"""
        try:
            block = self.web3.eth.get_block('latest')
            base_fee = block.get('baseFeePerGas', self.web3.eth.gas_price)
            priority_fee = self.web3.eth.max_priority_fee
            
            # Add 20% buffer to base fee
            return int(base_fee * 1.2) + priority_fee
            
        except Exception as e:
            logger.error(f"Error getting gas price: {str(e)}")
            # Fallback to network gas price
            return self.web3.eth.gas_price
            
    async def _get_nonce(self) -> int:
        """Get next nonce for the account"""
        account = self.web3.to_checksum_address(str(self.web3.eth.default_account))
        return self.web3.eth.get_transaction_count(
            account,
            'pending'
        )
        
    async def _estimate_gas(self, tx: TxParams) -> int:
        """Estimate gas for transaction"""
        return self.web3.eth.estimate_gas(tx)
        
    async def _sign_transaction(self, tx: TxParams) -> Optional[bytes]:
        """Sign transaction with account key"""
        try:
            private_key = os.getenv('MAINNET_PRIVATE_KEY')
            if not private_key:
                raise ValueError("MAINNET_PRIVATE_KEY not set")
            return self.web3.eth.account.sign_transaction(
                tx,
                private_key
            ).rawTransaction
            
        except Exception as e:
            logger.error(f"Error signing transaction: {str(e)}")
            return None
            
    def _encode_flash_loan_data(
        self,
        token: Address,
        amount: int,
        dex_weights: Dict[str, float]
    ) -> bytes:
        """Encode flash loan function call data"""
        return self.web3.eth.contract(
            abi=self._get_flash_loan_abi()
        ).encode_abi(
            fn_name='flashLoan',
            args=[token, amount, dex_weights]
        )
        
    def _encode_swap_data(
        self,
        token_pair: Tuple[Address, Address],
        amount: int,
        path: List[Address]
    ) -> bytes:
        """Encode swap function call data"""
        return self.web3.eth.contract(
            abi=self._get_router_abi()
        ).encode_abi(
            fn_name='swapExactTokensForTokens',
            args=[amount, 0, path, self.web3.eth.default_account, int(time.time()) + 300]
        )
        
    def _get_provider_config(self, provider: str) -> Dict:
        """Get configuration for flash loan provider"""
        # This would be loaded from config in practice
        return {
            'aave': {
                'router': '0x...',
                'fee': 0.09
            },
            'balancer': {
                'router': '0x...',
                'fee': 0.08
            },
            'radiant': {
                'router': '0x...',
                'fee': 0.1
            }
        }[provider]
        
    def _get_router_address(self) -> ChecksumAddress:
        """Get router contract address"""
        # This would be loaded from config in practice
        return Web3.to_checksum_address('0x...')
        
    def _get_flash_loan_abi(self) -> List:
        """Get flash loan contract ABI"""
        # This would be loaded from file in practice
        return []
        
    def _get_router_abi(self) -> List:
        """Get router contract ABI"""
        # This would be loaded from file in practice
        return [] 