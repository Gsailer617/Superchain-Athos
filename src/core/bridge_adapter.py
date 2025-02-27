from typing import Dict, Any, Optional, Protocol, Type, List, Union
from abc import ABC, abstractmethod
import logging
from web3 import Web3
from web3.types import TxParams, Wei
from eth_typing import HexAddress
from dataclasses import dataclass, field
from enum import Enum
import time
import functools
from urllib.error import URLError
import json

logger = logging.getLogger(__name__)

# Error handling decorator
def handle_bridge_errors(method):
    """Decorator for handling common bridge-related errors"""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except URLError as e:
            logger.error(f"Connection error in {self.__class__.__name__}.{method.__name__}: {str(e)}")
            self._update_metrics(success=False, error=f"Connection error: {str(e)}")
            raise
        except TimeoutError as e:
            logger.error(f"Timeout error in {self.__class__.__name__}.{method.__name__}: {str(e)}")
            self._update_metrics(success=False, error=f"Timeout error: {str(e)}")
            raise
        except ValueError as e:
            logger.error(f"Value error in {self.__class__.__name__}.{method.__name__}: {str(e)}")
            self._update_metrics(success=False, error=f"Value error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {self.__class__.__name__}.{method.__name__}: {str(e)}")
            self._update_metrics(success=False, error=f"Unexpected error: {str(e)}")
            raise
    return wrapper

@dataclass
class BridgeConfig:
    """Enhanced bridge configuration incorporating all protocol requirements"""
    name: str
    supported_chains: List[str]
    min_amount: float
    max_amount: float
    
    # Base configuration
    fee_multiplier: float = 1.0
    gas_limit_multiplier: float = 1.2
    confirmation_blocks: int = 1
    
    # Retry and timeout settings
    retry_interval: int = 60
    max_retries: int = 3
    timeout_seconds: int = 1800  # 30 minutes
    
    # Liquidity settings
    min_liquidity_ratio: float = 0.1
    min_liquidity_usd: float = 10000  # $10k minimum
    
    # Message verification
    message_verification_blocks: int = 5
    verification_mode: str = "optimistic"  # optimistic/strict
    
    # Protocol-specific settings
    layerzero_config: Dict[str, Any] = field(default_factory=lambda: {
        "version": "v2",
        "executor_config": {},
        "uln_config": {},
        "message_library": "latest"
    })
    
    stargate_config: Dict[str, Any] = field(default_factory=lambda: {
        "pool_ids": {
            "USDC": 1,
            "USDT": 2,
            "ETH": 3
        },
        "router_version": "v2"
    })
    
    across_config: Dict[str, Any] = field(default_factory=lambda: {
        "relayer_fee_pct": 0.04,
        "lp_fee_pct": 0.02,
        "verification_gas_limit": 2000000
    })
    
    debridge_config: Dict[str, Any] = field(default_factory=lambda: {
        "execution_fee_multiplier": 1.2,
        "claim_timeout": 7200,  # 2 hours
        "auto_claim": True
    })
    
    superbridge_config: Dict[str, Any] = field(default_factory=lambda: {
        "lz_endpoint": "",
        "custom_adapter": "",
        "fee_tier": "standard"
    })

class BridgeState(Enum):
    """Enhanced bridge states based on all protocols"""
    ACTIVE = "active"
    CONGESTED = "congested"
    PAUSED = "paused"
    OFFLINE = "offline"
    VERIFICATION_PENDING = "verification_pending"
    LOW_LIQUIDITY = "low_liquidity"
    MAINTENANCE = "maintenance"
    RELAYER_UNAVAILABLE = "relayer_unavailable"
    MESSAGE_FAILED = "message_failed"

@dataclass
class BridgeMetrics:
    """Enhanced metrics tracking for all bridge protocols"""
    # Basic metrics
    liquidity: float
    utilization: float
    success_rate: float
    avg_transfer_time: float
    failed_transfers: int
    total_transfers: int
    
    # Message verification metrics
    verification_success_rate: float = 1.0
    avg_verification_time: float = 0.0
    pending_messages: int = 0
    
    # Performance metrics
    avg_gas_cost: float = 0.0
    avg_fee_cost: float = 0.0
    relayer_response_time: float = 0.0
    
    # Liquidity metrics
    liquidity_utilization: float = 0.0
    liquidity_depth: float = 0.0
    
    # Error tracking
    last_error: Optional[str] = None
    last_updated: float = 0.0
    error_count: Dict[str, int] = field(default_factory=dict)

class BridgeAdapter(ABC):
    """Abstract base class for bridge adapters with common implementation"""
    
    def __init__(self, config: BridgeConfig, web3: Web3):
        """Initialize the bridge adapter
        
        Args:
            config: Bridge configuration
            web3: Web3 instance for interacting with the blockchain
        """
        self.config = config
        self.web3 = web3
        self.metrics = BridgeMetrics(
            liquidity=0.0,
            utilization=0.0,
            success_rate=1.0,  # Start optimistic
            avg_transfer_time=0.0,
            failed_transfers=0,
            total_transfers=0
        )
        self.last_updated = time.time()
        self._initialize_protocol()
    
    @abstractmethod
    def _initialize_protocol(self) -> None:
        """Initialize protocol-specific components
        
        This method should be implemented by each adapter to set up
        protocol-specific components like contracts, endpoints, etc.
        """
        pass
    
    @handle_bridge_errors
    def validate_transfer(
        self,
        source_chain: str,
        target_chain: str,
        token: str,
        amount: float
    ) -> bool:
        """Validate if a cross-chain transfer is possible
        
        This base implementation performs common validation checks.
        Subclasses should call super().validate_transfer() and then
        perform protocol-specific validation.
        
        Args:
            source_chain: Source chain ID
            target_chain: Target chain ID
            token: Token address or symbol
            amount: Amount to transfer
            
        Returns:
            True if transfer is valid, False otherwise
        """
        # Common validation logic
        if source_chain not in self.config.supported_chains:
            logger.warning(f"Source chain {source_chain} not supported")
            return False
            
        if target_chain not in self.config.supported_chains:
            logger.warning(f"Target chain {target_chain} not supported")
            return False
            
        if amount < self.config.min_amount:
            logger.warning(f"Amount {amount} below minimum {self.config.min_amount}")
            return False
            
        if amount > self.config.max_amount:
            logger.warning(f"Amount {amount} above maximum {self.config.max_amount}")
            return False
            
        # Protocol-specific validation to be implemented by subclasses
        return True
    
    @abstractmethod
    def estimate_fees(
        self,
        source_chain: str,
        target_chain: str,
        token: str,
        amount: float
    ) -> Dict[str, float]:
        """Estimate fees for a cross-chain transfer"""
        pass
    
    @abstractmethod
    def estimate_time(
        self,
        source_chain: str,
        target_chain: str
    ) -> int:
        """Estimate time for cross-chain message delivery in seconds"""
        pass
    
    @abstractmethod
    def prepare_transfer(
        self,
        source_chain: str,
        target_chain: str,
        token: str,
        amount: float,
        recipient: str
    ) -> TxParams:
        """Prepare transaction parameters for a cross-chain transfer"""
        pass
    
    @abstractmethod
    def verify_message(
        self,
        source_chain: str,
        target_chain: str,
        message_hash: str,
        proof: bytes
    ) -> bool:
        """Verify a cross-chain message delivery"""
        pass
    
    @handle_bridge_errors
    def get_bridge_state(
        self,
        source_chain: str,
        target_chain: str
    ) -> BridgeState:
        """Get the current state of the bridge
        
        Base implementation that can be extended by subclasses for
        protocol-specific state checks.
        
        Args:
            source_chain: Source chain ID
            target_chain: Target chain ID
            
        Returns:
            Current bridge state
        """
        # Check if chains are supported
        if source_chain not in self.config.supported_chains or target_chain not in self.config.supported_chains:
            return BridgeState.OFFLINE
            
        # Base implementation just returns ACTIVE
        # Subclasses should override with protocol-specific checks
        return BridgeState.ACTIVE
    
    @abstractmethod
    def monitor_liquidity(
        self,
        chain: str,
        token: str
    ) -> float:
        """Monitor liquidity for a specific token on a chain"""
        pass
    
    @abstractmethod
    def recover_failed_transfer(
        self,
        source_chain: str,
        target_chain: str,
        tx_hash: str
    ) -> Optional[str]:
        """Attempt to recover a failed transfer"""
        pass
    
    def _update_metrics(
        self,
        success: bool,
        error: Optional[str] = None,
        transfer_time: Optional[float] = None,
        gas_cost: Optional[float] = None,
        fee_cost: Optional[float] = None
    ) -> None:
        """Update bridge metrics"""
        # Update counters
        self.metrics.total_transfers += 1
        if not success:
            self.metrics.failed_transfers += 1
            if error:
                if error not in self.metrics.error_count:
                    self.metrics.error_count[error] = 0
                self.metrics.error_count[error] += 1
                self.metrics.last_error = error
        
        # Update times
        if transfer_time is not None:
            # Weighted average for transfer time
            if self.metrics.avg_transfer_time == 0:
                self.metrics.avg_transfer_time = transfer_time
            else:
                self.metrics.avg_transfer_time = (
                    self.metrics.avg_transfer_time * 0.9 + transfer_time * 0.1
                )
        
        # Update costs
        if gas_cost is not None:
            if self.metrics.avg_gas_cost == 0:
                self.metrics.avg_gas_cost = gas_cost
            else:
                self.metrics.avg_gas_cost = (
                    self.metrics.avg_gas_cost * 0.9 + gas_cost * 0.1
                )
                
        if fee_cost is not None:
            if self.metrics.avg_fee_cost == 0:
                self.metrics.avg_fee_cost = fee_cost
            else:
                self.metrics.avg_fee_cost = (
                    self.metrics.avg_fee_cost * 0.9 + fee_cost * 0.1
                )
        
        # Update success rate
        self.metrics.success_rate = 1.0 - (
            self.metrics.failed_transfers / self.metrics.total_transfers
            if self.metrics.total_transfers > 0 else 0
        )
        
        # Update timestamp
        self.metrics.last_updated = time.time()

class LayerZeroAdapter(BridgeAdapter):
    """Adapter for LayerZero bridge"""
    
    def __init__(self, config: BridgeConfig, web3: Web3):
        super().__init__(config, web3)
        self.chain_ids = {
            'ethereum': 101,
            'polygon': 109,
            'base': 184
        }
        self.endpoints = {}  # Cache for endpoints
        self.message_library = {}  # Cache for message libraries
    
    def validate_transfer(
        self,
        source_chain: str,
        target_chain: str,
        token: str,
        amount: float
    ) -> bool:
        """Validate LayerZero transfer"""
        if source_chain not in self.chain_ids or target_chain not in self.chain_ids:
            return False
        if amount < self.config.min_amount or amount > self.config.max_amount:
            return False
        return True
    
    def estimate_fees(
        self,
        source_chain: str,
        target_chain: str,
        token: str,
        amount: float
    ) -> Dict[str, float]:
        """Estimate LayerZero fees"""
        base_fee = amount * self.config.fee_multiplier
        gas_fee = self._estimate_gas_fee(source_chain, target_chain)
        return {
            'base_fee': base_fee,
            'gas_fee': gas_fee,
            'total': base_fee + gas_fee
        }
    
    def estimate_time(
        self,
        source_chain: str,
        target_chain: str
    ) -> int:
        """Estimate LayerZero transfer time"""
        # Base time + block confirmations
        return 300 + (self.config.confirmation_blocks * 15)
    
    def prepare_transfer(
        self,
        source_chain: str,
        target_chain: str,
        token: str,
        amount: float,
        recipient: str
    ) -> TxParams:
        """Prepare LayerZero transfer"""
        return {
            'to': self._get_endpoint(source_chain),
            'data': self._encode_transfer_data(
                self.chain_ids[target_chain],
                token,
                amount,
                recipient
            ),
            'value': Wei(self.estimate_fees(source_chain, target_chain, token, amount)['total'])
        }
    
    def verify_message(
        self,
        source_chain: str,
        target_chain: str,
        message_hash: str,
        proof: bytes
    ) -> bool:
        """Verify LayerZero message"""
        try:
            endpoint = self._get_endpoint(target_chain)
            library = self._get_message_library(target_chain)
            
            # Verify message with library
            return library.verifyMessage(
                self.chain_ids[source_chain],
                message_hash,
                proof
            )
        except Exception as e:
            self._update_metrics(success=False, error=str(e))
            return False
    
    def get_bridge_state(
        self,
        source_chain: str,
        target_chain: str
    ) -> BridgeState:
        """Get LayerZero bridge state"""
        try:
            # Check endpoint status
            endpoint = self._get_endpoint(source_chain)
            if not self._is_endpoint_active(endpoint):
                return BridgeState.OFFLINE
            
            # Check congestion
            if self._is_congested(source_chain, target_chain):
                return BridgeState.CONGESTED
            
            # Check if paused
            if self._is_paused(source_chain, target_chain):
                return BridgeState.PAUSED
            
            return BridgeState.ACTIVE
            
        except Exception as e:
            self._update_metrics(success=False, error=str(e))
            return BridgeState.OFFLINE
    
    def monitor_liquidity(
        self,
        chain: str,
        token: str
    ) -> float:
        """Monitor LayerZero liquidity"""
        try:
            endpoint = self._get_endpoint(chain)
            pool = self._get_liquidity_pool(chain, token)
            
            # Get pool balance
            balance = self.web3.eth.get_balance(pool)
            
            # Update metrics
            self.metrics.liquidity = float(balance)
            self.metrics.last_updated = time.time()
            
            return float(balance)
            
        except Exception as e:
            self._update_metrics(success=False, error=str(e))
            return 0.0
    
    def recover_failed_transfer(
        self,
        source_chain: str,
        target_chain: str,
        tx_hash: str
    ) -> Optional[str]:
        """Recover failed LayerZero transfer"""
        try:
            # Get original transaction
            tx = self.web3.eth.get_transaction(tx_hash)
            if not tx:
                return None
            
            # Decode transaction data
            decoded = self._decode_transfer_data(tx['input'])
            
            # Prepare recovery transaction
            recovery_tx = self.prepare_transfer(
                source_chain,
                target_chain,
                decoded['token'],
                decoded['amount'],
                decoded['recipient']
            )
            
            # Add recovery parameters
            recovery_tx['nonce'] = tx['nonce']
            recovery_tx['gasPrice'] = int(tx['gasPrice'] * 1.2)  # 20% higher gas
            
            # Send recovery transaction
            recovery_hash = self.web3.eth.send_transaction(recovery_tx)
            return recovery_hash.hex()
            
        except Exception as e:
            self._update_metrics(success=False, error=str(e))
            return None
    
    def _is_endpoint_active(self, endpoint: str) -> bool:
        """Check if LayerZero endpoint is active"""
        try:
            code = self.web3.eth.get_code(endpoint)
            return len(code) > 0
        except Exception:
            return False
    
    def _is_congested(
        self,
        source_chain: str,
        target_chain: str
    ) -> bool:
        """Check if bridge path is congested"""
        try:
            # Check pending messages
            endpoint = self._get_endpoint(source_chain)
            pending = self._get_pending_messages(
                endpoint,
                self.chain_ids[target_chain]
            )
            return pending > 1000  # Example threshold
        except Exception:
            return True
    
    def _is_paused(
        self,
        source_chain: str,
        target_chain: str
    ) -> bool:
        """Check if bridge path is paused"""
        try:
            endpoint = self._get_endpoint(source_chain)
            return self._check_pause_status(
                endpoint,
                self.chain_ids[target_chain]
            )
        except Exception:
            return True
    
    def _update_metrics(
        self,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """Update bridge metrics"""
        self.metrics.total_transfers += 1
        if not success:
            self.metrics.failed_transfers += 1
            self.metrics.last_error = error
        self.metrics.success_rate = (
            (self.metrics.total_transfers - self.metrics.failed_transfers) /
            self.metrics.total_transfers
        )
        self.metrics.last_updated = time.time()

class StargateAdapter(BridgeAdapter):
    """Adapter for Stargate bridge"""
    
    def __init__(self, config: BridgeConfig, web3: Web3):
        super().__init__(config, web3)
        self.pool_ids = {
            'USDC': 1,
            'USDT': 2,
            'ETH': 3
        }
        self.routers = {}  # Cache for routers
        self.pools = {}  # Cache for pools
    
    def validate_transfer(
        self,
        source_chain: str,
        target_chain: str,
        token: str,
        amount: float
    ) -> bool:
        """Validate Stargate transfer"""
        if token not in self.pool_ids:
            return False
        if amount < self.config.min_amount or amount > self.config.max_amount:
            return False
        return True
    
    def estimate_fees(
        self,
        source_chain: str,
        target_chain: str,
        token: str,
        amount: float
    ) -> Dict[str, float]:
        """Estimate Stargate fees"""
        base_fee = amount * self.config.fee_multiplier * 1.2  # Stargate has higher fees
        gas_fee = self._estimate_gas_fee(source_chain, target_chain)
        return {
            'base_fee': base_fee,
            'gas_fee': gas_fee,
            'total': base_fee + gas_fee
        }
    
    def estimate_time(
        self,
        source_chain: str,
        target_chain: str
    ) -> int:
        """Estimate Stargate transfer time"""
        # Base time + block confirmations
        return 180 + (self.config.confirmation_blocks * 15)
    
    def prepare_transfer(
        self,
        source_chain: str,
        target_chain: str,
        token: str,
        amount: float,
        recipient: str
    ) -> TxParams:
        """Prepare Stargate transfer"""
        return {
            'to': self._get_router(source_chain),
            'data': self._encode_transfer_data(
                self.pool_ids[token],
                amount,
                recipient
            ),
            'value': Wei(self.estimate_fees(source_chain, target_chain, token, amount)['total'])
        }
    
    def verify_message(
        self,
        source_chain: str,
        target_chain: str,
        message_hash: str,
        proof: bytes
    ) -> bool:
        """Verify Stargate message"""
        try:
            router = self._get_router(target_chain)
            
            # Verify message with router
            return router.verifyMessage(
                source_chain,
                message_hash,
                proof
            )
        except Exception as e:
            self._update_metrics(success=False, error=str(e))
            return False
    
    def get_bridge_state(
        self,
        source_chain: str,
        target_chain: str
    ) -> BridgeState:
        """Get Stargate bridge state"""
        try:
            # Check router status
            router = self._get_router(source_chain)
            if not self._is_router_active(router):
                return BridgeState.OFFLINE
            
            # Check congestion
            if self._is_congested(source_chain, target_chain):
                return BridgeState.CONGESTED
            
            # Check if paused
            if self._is_paused(source_chain, target_chain):
                return BridgeState.PAUSED
            
            return BridgeState.ACTIVE
            
        except Exception as e:
            self._update_metrics(success=False, error=str(e))
            return BridgeState.OFFLINE
    
    def monitor_liquidity(
        self,
        chain: str,
        token: str
    ) -> float:
        """Monitor Stargate liquidity"""
        try:
            pool = self._get_pool(chain, token)
            
            # Get pool balance
            balance = self.web3.eth.get_balance(pool)
            
            # Update metrics
            self.metrics.liquidity = float(balance)
            self.metrics.last_updated = time.time()
            
            return float(balance)
            
        except Exception as e:
            self._update_metrics(success=False, error=str(e))
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
                return None
            
            # Decode transaction data
            decoded = self._decode_transfer_data(tx['input'])
            
            # Prepare recovery transaction
            recovery_tx = self.prepare_transfer(
                source_chain,
                target_chain,
                decoded['token'],
                decoded['amount'],
                decoded['recipient']
            )
            
            # Add recovery parameters
            recovery_tx['nonce'] = tx['nonce']
            recovery_tx['gasPrice'] = int(tx['gasPrice'] * 1.2)  # 20% higher gas
            
            # Send recovery transaction
            recovery_hash = self.web3.eth.send_transaction(recovery_tx)
            return recovery_hash.hex()
            
        except Exception as e:
            self._update_metrics(success=False, error=str(e))
            return None
    
    def _is_router_active(self, router: str) -> bool:
        """Check if Stargate router is active"""
        try:
            code = self.web3.eth.get_code(router)
            return len(code) > 0
        except Exception:
            return False
    
    def _is_congested(
        self,
        source_chain: str,
        target_chain: str
    ) -> bool:
        """Check if bridge path is congested"""
        try:
            # Check queue length
            router = self._get_router(source_chain)
            queue = self._get_queue_length(
                router,
                target_chain
            )
            return queue > 100  # Example threshold
        except Exception:
            return True
    
    def _is_paused(
        self,
        source_chain: str,
        target_chain: str
    ) -> bool:
        """Check if bridge path is paused"""
        try:
            router = self._get_router(source_chain)
            return self._check_pause_status(
                router,
                target_chain
            )
        except Exception:
            return True
    
    def _update_metrics(
        self,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """Update bridge metrics"""
        self.metrics.total_transfers += 1
        if not success:
            self.metrics.failed_transfers += 1
            self.metrics.last_error = error
        self.metrics.success_rate = (
            (self.metrics.total_transfers - self.metrics.failed_transfers) /
            self.metrics.total_transfers
        )
        self.metrics.last_updated = time.time()

class BridgeAdapterFactory:
    """Factory for creating bridge adapter instances"""
    
    _adapters: Dict[str, Type[BridgeAdapter]] = {}
    
    @classmethod
    def register_adapter(cls, name: str, adapter_class: Type[BridgeAdapter]) -> None:
        """Register a new bridge adapter
        
        Args:
            name: Unique identifier for the adapter
            adapter_class: The adapter class to register
            
        Raises:
            ValueError: If adapter name is already registered
        """
        if name in cls._adapters:
            raise ValueError(f"Adapter {name} is already registered")
        cls._adapters[name] = adapter_class
    
    @classmethod
    def get_adapter(cls, name: str, config: BridgeConfig, web3: Web3) -> BridgeAdapter:
        """Get an instance of a registered adapter
        
        Args:
            name: Name of the adapter to instantiate
            config: Bridge configuration
            web3: Web3 instance
            
        Returns:
            BridgeAdapter: Instance of the requested adapter
            
        Raises:
            KeyError: If adapter name is not registered
        """
        if name not in cls._adapters:
            raise KeyError(f"No adapter registered with name: {name}")
        return cls._adapters[name](config, web3)
    
    @classmethod
    def get_registered_adapters(cls) -> Dict[str, Type[BridgeAdapter]]:
        """Get all registered adapters
        
        Returns:
            Dict[str, Type[BridgeAdapter]]: Mapping of adapter names to their classes
        """
        return cls._adapters.copy()
    
    @classmethod
    def unregister_adapter(cls, name: str) -> None:
        """Unregister an adapter
        
        Args:
            name: Name of the adapter to unregister
            
        Raises:
            KeyError: If adapter name is not registered
        """
        if name not in cls._adapters:
            raise KeyError(f"No adapter registered with name: {name}")
        del cls._adapters[name]
    
    @classmethod
    def clear_adapters(cls) -> None:
        """Remove all registered adapters"""
        cls._adapters.clear() 