import logging
from typing import Dict, Union, Optional, Tuple, List
from web3 import Web3
from web3.types import TxParams, TxReceipt, Wei, Nonce
from eth_typing import Address, ChecksumAddress
import time
import os

from src.core.types import OpportunityType, FlashLoanOpportunityType

logger = logging.getLogger(__name__)

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