from typing import Dict, Any, Optional, Protocol, Type
from abc import ABC, abstractmethod
import logging
from web3 import Web3
from web3.types import TxParams, Wei
from eth_typing import HexAddress
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ChainConfig:
    """Chain-specific configuration"""
    chain_id: int
    native_token: str
    block_time: int
    gas_limit_multiplier: float = 1.1
    max_priority_fee: int = 2000000000  # 2 Gwei
    base_fee_multiplier: float = 1.2
    # Added chain-specific configurations
    block_finality: int = 12  # Number of blocks for finality
    rpc_timeout: int = 10  # RPC timeout in seconds
    max_reorg_depth: int = 64  # Maximum reorg depth
    nonce_offset: int = 0  # Offset for nonce calculation
    retry_count: int = 3  # Number of RPC retries

class ChainAdapter(ABC):
    """Base class for chain-specific adapters"""
    
    def __init__(self, config: ChainConfig, web3: Web3):
        self.config = config
        self.web3 = web3
    
    @abstractmethod
    def adapt_transaction(self, tx: TxParams) -> TxParams:
        """Adapt transaction parameters for specific chain"""
        pass
    
    @abstractmethod
    def estimate_gas_limit(self, base_limit: int) -> int:
        """Estimate appropriate gas limit for chain"""
        pass
    
    @abstractmethod
    def calculate_gas_price(self, base_fee: int) -> Dict[str, int]:
        """Calculate appropriate gas price for chain"""
        pass
    
    @abstractmethod
    def validate_address(self, address: str) -> bool:
        """Validate address format for chain"""
        pass
    
    @abstractmethod
    def get_finality_blocks(self) -> int:
        """Get number of blocks needed for finality"""
        pass
    
    @abstractmethod
    def handle_rpc_error(self, error: Exception) -> Optional[str]:
        """Handle chain-specific RPC errors"""
        pass
    
    @abstractmethod
    def calculate_nonce(self, address: str, pending: bool = True) -> int:
        """Calculate appropriate nonce for chain"""
        pass
    
    @abstractmethod
    def validate_transaction_state(self, tx_hash: str) -> Dict[str, Any]:
        """Validate transaction state considering chain quirks"""
        pass

class EthereumAdapter(ChainAdapter):
    """Adapter for Ethereum mainnet"""
    
    def __init__(self, config: ChainConfig, web3: Web3):
        super().__init__(config, web3)
        self.rpc_endpoints = []  # List of fallback RPC endpoints
        self.error_patterns = {
            'nonce too low': self._handle_nonce_error,
            'insufficient funds': self._handle_balance_error,
            'gas required exceeds allowance': self._handle_gas_error
        }
    
    def adapt_transaction(self, tx: TxParams) -> TxParams:
        """Adapt transaction for Ethereum"""
        # Handle EIP-1559 parameters
        if 'gasPrice' in tx:
            del tx['gasPrice']
        if 'maxFeePerGas' not in tx:
            tx['maxFeePerGas'] = Wei(int(tx.get('gasPrice', 0) * 1.5))
        if 'maxPriorityFeePerGas' not in tx:
            tx['maxPriorityFeePerGas'] = Wei(self.config.max_priority_fee)
        
        # Add chain-specific parameters
        tx['chainId'] = self.config.chain_id
        return tx
    
    def estimate_gas_limit(self, base_limit: int) -> int:
        """Estimate gas limit for Ethereum"""
        return int(base_limit * self.config.gas_limit_multiplier)
    
    def calculate_gas_price(self, base_fee: int) -> Dict[str, int]:
        """Calculate gas price for Ethereum"""
        max_fee = int(base_fee * self.config.base_fee_multiplier)
        return {
            'maxFeePerGas': max_fee,
            'maxPriorityFeePerGas': self.config.max_priority_fee
        }
    
    def validate_address(self, address: str) -> bool:
        """Validate Ethereum address"""
        try:
            return Web3.is_address(address)
        except ValueError:
            return False
    
    def get_finality_blocks(self) -> int:
        """Get Ethereum finality blocks"""
        return self.config.block_finality
    
    def handle_rpc_error(self, error: Exception) -> Optional[str]:
        """Handle Ethereum RPC errors"""
        error_msg = str(error)
        for pattern, handler in self.error_patterns.items():
            if pattern in error_msg:
                return handler(error_msg)
        return None
    
    def calculate_nonce(self, address: str, pending: bool = True) -> int:
        """Calculate nonce for Ethereum"""
        base_nonce = self.web3.eth.get_transaction_count(
            address,
            'pending' if pending else 'latest'
        )
        return base_nonce + self.config.nonce_offset
    
    def validate_transaction_state(self, tx_hash: str) -> Dict[str, Any]:
        """Validate Ethereum transaction state"""
        receipt = self.web3.eth.get_transaction_receipt(tx_hash)
        block_number = receipt.get('blockNumber')
        current_block = self.web3.eth.block_number
        
        return {
            'confirmed': current_block - block_number >= self.get_finality_blocks(),
            'success': receipt.get('status') == 1,
            'reorged': self._check_reorg(receipt),
            'finality_blocks': self.get_finality_blocks(),
            'confirmations': current_block - block_number
        }
    
    def _handle_nonce_error(self, error_msg: str) -> str:
        """Handle Ethereum nonce errors"""
        return "Nonce synchronization error - retry with updated nonce"
    
    def _handle_balance_error(self, error_msg: str) -> str:
        """Handle Ethereum balance errors"""
        return "Insufficient balance for transaction and gas"
    
    def _handle_gas_error(self, error_msg: str) -> str:
        """Handle Ethereum gas errors"""
        return "Gas limit too low for transaction"
    
    def _check_reorg(self, receipt: Dict[str, Any]) -> bool:
        """Check if transaction was affected by reorg"""
        tx_hash = receipt.get('transactionHash')
        block_hash = receipt.get('blockHash')
        
        # Check if transaction is in a different block
        current_block = self.web3.eth.get_block(receipt['blockNumber'])
        return current_block['hash'] != block_hash

class PolygonAdapter(ChainAdapter):
    """Adapter for Polygon network"""
    
    def __init__(self, config: ChainConfig, web3: Web3):
        super().__init__(config, web3)
        self.checkpoint_interval = 10000  # Polygon checkpoint interval
        self.bor_chain_id = 137  # Polygon Bor chain ID
    
    def adapt_transaction(self, tx: TxParams) -> TxParams:
        """Adapt transaction for Polygon"""
        tx = super().adapt_transaction(tx)
        
        # Polygon-specific adaptations
        if 'gas' in tx:
            tx['gas'] = Wei(self.estimate_gas_limit(int(tx['gas'])))
        if 'maxPriorityFeePerGas' in tx:
            tx['maxPriorityFeePerGas'] = Wei(max(
                int(tx['maxPriorityFeePerGas']),
                self.config.max_priority_fee * 2  # Double the priority fee
            ))
        
        # Add Polygon-specific parameters
        tx['chainId'] = self.bor_chain_id
        return tx
    
    def estimate_gas_limit(self, base_limit: int) -> int:
        """Estimate gas limit for Polygon"""
        return int(base_limit * 1.5)  # Polygon needs higher limits
    
    def calculate_gas_price(self, base_fee: int) -> Dict[str, int]:
        """Calculate gas price for Polygon"""
        max_fee = int(base_fee * 2)  # Polygon needs higher multiplier
        return {
            'maxFeePerGas': max_fee,
            'maxPriorityFeePerGas': self.config.max_priority_fee * 2
        }
    
    def validate_address(self, address: str) -> bool:
        """Validate Polygon address"""
        return Web3.is_address(address)  # Same as Ethereum
    
    def get_finality_blocks(self) -> int:
        """Get Polygon finality blocks - checkpoint based"""
        return min(self.config.block_finality, self.checkpoint_interval)
    
    def handle_rpc_error(self, error: Exception) -> Optional[str]:
        """Handle Polygon RPC errors"""
        error_msg = str(error)
        if "checkpoint not found" in error_msg:
            return "Wait for next checkpoint"
        if "bor receipt not found" in error_msg:
            return "Transaction pending on Bor chain"
        return super().handle_rpc_error(error)
    
    def calculate_nonce(self, address: str, pending: bool = True) -> int:
        """Calculate nonce for Polygon - handle Bor chain"""
        nonce = super().calculate_nonce(address, pending)
        # Add offset for Bor transactions
        return nonce + (1 if self._is_bor_transaction() else 0)
    
    def validate_transaction_state(self, tx_hash: str) -> Dict[str, Any]:
        """Validate Polygon transaction state"""
        state = super().validate_transaction_state(tx_hash)
        
        # Add Polygon-specific validations
        state['checkpoint_confirmed'] = self._is_checkpoint_confirmed(
            state.get('blockNumber', 0)
        )
        state['bor_confirmed'] = self._is_bor_confirmed(tx_hash)
        
        return state
    
    def _is_checkpoint_confirmed(self, block_number: int) -> bool:
        """Check if block is confirmed by checkpoint"""
        latest_checkpoint = self._get_latest_checkpoint()
        return block_number <= latest_checkpoint
    
    def _is_bor_confirmed(self, tx_hash: str) -> bool:
        """Check if transaction is confirmed on Bor chain"""
        try:
            receipt = self.web3.eth.get_transaction_receipt(tx_hash)
            return receipt.get('status') == 1
        except Exception:
            return False
    
    def _is_bor_transaction(self) -> bool:
        """Check if transaction is on Bor chain"""
        return True  # All Polygon transactions are on Bor
    
    def _get_latest_checkpoint(self) -> int:
        """Get latest Polygon checkpoint"""
        # Implementation would get actual checkpoint from Polygon
        return self.web3.eth.block_number - self.checkpoint_interval

class BaseAdapter(ChainAdapter):
    """Adapter for Base network"""
    
    def __init__(self, config: ChainConfig, web3: Web3):
        super().__init__(config, web3)
        self.l1_chain_id = 1  # Ethereum mainnet
        self.l2_chain_id = 8453  # Base chain ID
    
    def adapt_transaction(self, tx: TxParams) -> TxParams:
        """Adapt transaction for Base"""
        tx = super().adapt_transaction(tx)
        
        # Base-specific adaptations
        if 'maxFeePerGas' in tx:
            tx['maxFeePerGas'] = Wei(int(int(tx['maxFeePerGas']) * 0.85))
        if 'maxPriorityFeePerGas' in tx:
            tx['maxPriorityFeePerGas'] = Wei(int(self.config.max_priority_fee * 0.5))
        
        # Add Base-specific parameters
        tx['chainId'] = self.l2_chain_id
        return tx
    
    def estimate_gas_limit(self, base_limit: int) -> int:
        """Estimate gas limit for Base"""
        return int(base_limit * 1.1)  # Base needs slightly higher limits
    
    def calculate_gas_price(self, base_fee: int) -> Dict[str, int]:
        """Calculate gas price for Base"""
        max_fee = int(base_fee * 1.1)  # Base has lower fees
        return {
            'maxFeePerGas': max_fee,
            'maxPriorityFeePerGas': int(self.config.max_priority_fee * 0.5)
        }
    
    def validate_address(self, address: str) -> bool:
        """Validate Base address"""
        return Web3.is_address(address)  # Same as Ethereum
    
    def get_finality_blocks(self) -> int:
        """Get Base finality blocks - L2 specific"""
        return self.config.block_finality
    
    def handle_rpc_error(self, error: Exception) -> Optional[str]:
        """Handle Base RPC errors"""
        error_msg = str(error)
        if "sequencer down" in error_msg:
            return "Base sequencer is down"
        if "l1 confirmation pending" in error_msg:
            return "Waiting for L1 confirmation"
        return super().handle_rpc_error(error)
    
    def calculate_nonce(self, address: str, pending: bool = True) -> int:
        """Calculate nonce for Base - handle L2 specifics"""
        nonce = super().calculate_nonce(address, pending)
        # Add offset for L2 transactions
        return nonce + (1 if self._is_l2_transaction() else 0)
    
    def validate_transaction_state(self, tx_hash: str) -> Dict[str, Any]:
        """Validate Base transaction state"""
        state = super().validate_transaction_state(tx_hash)
        
        # Add Base-specific validations
        state['l1_confirmed'] = self._is_l1_confirmed(tx_hash)
        state['sequencer_confirmed'] = self._is_sequencer_confirmed(tx_hash)
        
        return state
    
    def _is_l1_confirmed(self, tx_hash: str) -> bool:
        """Check if transaction is confirmed on L1"""
        try:
            receipt = self._get_l1_receipt(tx_hash)
            return receipt.get('status') == 1
        except Exception:
            return False
    
    def _is_sequencer_confirmed(self, tx_hash: str) -> bool:
        """Check if transaction is confirmed by sequencer"""
        try:
            receipt = self.web3.eth.get_transaction_receipt(tx_hash)
            return receipt.get('status') == 1
        except Exception:
            return False
    
    def _is_l2_transaction(self) -> bool:
        """Check if transaction is on L2"""
        return True  # All Base transactions are on L2
    
    def _get_l1_receipt(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        """Get L1 receipt for L2 transaction"""
        # Implementation would get L1 receipt from Base bridge
        return None

class ChainAdapterFactory:
    """Factory for creating chain-specific adapters"""
    
    _adapters: Dict[str, Type[ChainAdapter]] = {
        'ethereum': EthereumAdapter,
        'polygon': PolygonAdapter,
        'base': BaseAdapter
    }
    
    @classmethod
    def get_adapter(cls, chain: str, config: ChainConfig, web3: Web3) -> ChainAdapter:
        """Get appropriate adapter for chain"""
        adapter_class = cls._adapters.get(chain.lower())
        if not adapter_class:
            raise ValueError(f"No adapter available for chain: {chain}")
        return adapter_class(config, web3)
    
    @classmethod
    def register_adapter(cls, chain: str, adapter: Type[ChainAdapter]) -> None:
        """Register new chain adapter"""
        cls._adapters[chain.lower()] = adapter 