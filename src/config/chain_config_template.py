from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from enum import Enum
from decimal import Decimal

class ConsensusType(Enum):
    """Consensus mechanisms used by different chains"""
    POW = "proof_of_work"
    POS = "proof_of_stake"
    POSA = "proof_of_stake_authority"
    OPTIMISTIC = "optimistic_rollup"
    ZK = "zk_rollup"
    AVALANCHE = "snow_protocol"

class RollupType(Enum):
    """Types of rollup solutions"""
    NONE = "none"  # For L1s
    OPTIMISTIC = "optimistic"
    ZK = "zk"
    VALIDIUM = "validium"

class GasModel(Enum):
    """Gas pricing models"""
    LEGACY = "legacy"
    EIP1559 = "eip1559"
    L2_ROLLUP = "l2_rollup"
    ARBITRUM_STYLE = "arbitrum"
    OPTIMISM_STYLE = "optimism"

@dataclass
class RPCConfig:
    """RPC endpoint configuration"""
    http: List[str]  # Multiple endpoints for redundancy
    ws: Optional[List[str]] = None
    rate_limit: Optional[int] = None  # Requests per second
    batch_size: Optional[int] = None  # Max batch request size
    timeout: int = 10  # Seconds
    retry_count: int = 3
    provider_weights: Optional[Dict[str, float]] = None  # For load balancing

@dataclass
class GasConfig:
    """Gas-related configuration"""
    model: GasModel
    base_fee_enabled: bool = True
    priority_fee_enabled: bool = True
    max_fee_per_gas: Optional[int] = None  # in wei
    max_priority_fee_per_gas: Optional[int] = None  # in wei
    min_gas_price: Optional[int] = None  # for legacy transactions
    l1_fee_overhead: Optional[int] = None  # for L2s
    l1_fee_scalar: Optional[int] = None  # for L2s
    gas_limit_multiplier: float = 1.2
    gas_price_multiplier: float = 1.1
    base_fee_max_change: Optional[float] = None  # per block
    suggested_fee_history_blocks: int = 10

@dataclass
class BlockConfig:
    """Block-related parameters"""
    target_block_time: float  # in seconds
    block_size_limit: Optional[int] = None  # in bytes
    max_gas_per_block: Optional[int] = None
    safe_confirmations: int = 1  # Number of blocks for "safe" tx
    finalized_confirmations: int = 1  # Number of blocks for "finalized" tx
    reorg_protection_blocks: int = 0  # Extra blocks to wait for reorg protection
    max_reorg_depth: Optional[int] = None  # Maximum expected reorg depth

@dataclass
class BridgeConfig:
    """Bridge-specific configuration"""
    address: str
    type: str  # official, fast, canonical, etc.
    token_list: Optional[List[str]] = None
    min_amount: Optional[Dict[str, Decimal]] = None
    max_amount: Optional[Dict[str, Decimal]] = None
    withdrawal_delay: Optional[int] = None  # in seconds
    challenge_period: Optional[int] = None  # in seconds
    verification_blocks: Optional[int] = None
    supported_tokens: Optional[List[str]] = None

@dataclass
class SecurityConfig:
    """Security-related parameters"""
    is_permissionless: bool = True
    validator_set_size: Optional[int] = None
    min_stake_amount: Optional[Decimal] = None
    challenge_period: Optional[int] = None  # in seconds
    fraud_proof_window: Optional[int] = None  # in seconds
    sequencer_publication_window: Optional[int] = None  # in seconds
    trusted_validators: Optional[List[str]] = None

@dataclass
class NetworkConfig:
    """Network-specific configuration"""
    is_testnet: bool = False
    is_local: bool = False
    supports_eip1559: bool = True
    supports_eth_call_by_hash: bool = False
    supports_revert_reason: bool = True
    supports_pending_transactions: bool = True
    supports_eth_get_logs_by_hash: bool = False
    trace_api_available: bool = False
    debug_api_available: bool = False
    archive_node_available: bool = False

@dataclass
class APIConfig:
    """API and service configuration"""
    explorer_url: Optional[str] = None
    explorer_api_url: Optional[str] = None
    explorer_api_key: Optional[str] = None
    graph_node_url: Optional[str] = None
    index_service_url: Optional[str] = None
    trace_api_url: Optional[str] = None
    debug_api_url: Optional[str] = None

@dataclass
class PerformanceConfig:
    """Performance-related parameters"""
    tx_pool_size: Optional[int] = None
    max_pending_transactions: Optional[int] = None
    max_queued_transactions: Optional[int] = None
    mempool_max_size: Optional[int] = None
    max_peers: Optional[int] = None
    target_gas_utilization: Optional[float] = None
    cache_size: Optional[int] = None

@dataclass
class ChainFeatures:
    """Chain-specific features and capabilities"""
    eips_supported: List[int] = field(default_factory=list)
    opcodes_enabled: List[str] = field(default_factory=list)
    precompiles: Dict[str, str] = field(default_factory=dict)
    custom_extensions: Dict[str, Any] = field(default_factory=dict)
    smart_contract_languages: List[str] = field(default_factory=list)
    vm_version: Optional[str] = None

@dataclass
class ChainConfig:
    """Complete chain configuration template"""
    # Basic Information
    name: str
    chain_id: int
    native_currency: str
    native_currency_decimals: int
    consensus_type: ConsensusType
    rollup_type: RollupType
    
    # Network Configuration
    rpc: RPCConfig
    network: NetworkConfig
    
    # Performance and Timing
    block: BlockConfig
    performance: PerformanceConfig
    
    # Gas and Fees
    gas: GasConfig
    
    # Security and Protocol
    security: SecurityConfig
    
    # Bridges and Cross-chain
    bridges: Dict[str, BridgeConfig] = field(default_factory=dict)
    
    # APIs and Services
    api: APIConfig = field(default_factory=APIConfig)
    
    # Features and Capabilities
    features: ChainFeatures = field(default_factory=ChainFeatures)
    
    # Additional Metadata
    description: Optional[str] = None
    website: Optional[str] = None
    docs_url: Optional[str] = None
    github_url: Optional[str] = None
    support_url: Optional[str] = None
    
    def validate(self) -> bool:
        """Validate chain configuration"""
        try:
            # Basic validation
            assert self.chain_id > 0, "Chain ID must be positive"
            assert len(self.rpc.http) > 0, "At least one RPC endpoint required"
            
            # Gas model validation
            if self.gas.model == GasModel.EIP1559:
                assert self.gas.base_fee_enabled, "EIP-1559 requires base fee"
            
            # L2-specific validation
            if self.rollup_type != RollupType.NONE:
                assert self.gas.l1_fee_overhead is not None, "L2 requires l1_fee_overhead"
                assert self.gas.l1_fee_scalar is not None, "L2 requires l1_fee_scalar"
            
            # Bridge validation
            for bridge in self.bridges.values():
                if bridge.min_amount:
                    assert all(amt > 0 for amt in bridge.min_amount.values())
                if bridge.max_amount:
                    assert all(amt > 0 for amt in bridge.max_amount.values())
            
            return True
            
        except AssertionError as e:
            print(f"Validation failed: {str(e)}")
            return False
    
    def get_safe_gas_limit(self, transaction_type: str = "transfer") -> int:
        """Get safe gas limit for different transaction types"""
        base_limits = {
            "transfer": 21000,
            "erc20_transfer": 65000,
            "swap": 200000,
            "bridge": 300000,
        }
        
        base_limit = base_limits.get(transaction_type, 21000)
        return int(base_limit * self.gas.gas_limit_multiplier)
    
    def get_recommended_confirmations(self, security_level: str = "safe") -> int:
        """Get recommended confirmations based on security level"""
        if security_level == "instant":
            return 1
        elif security_level == "safe":
            return self.block.safe_confirmations
        elif security_level == "finalized":
            return self.block.finalized_confirmations
        else:
            return self.block.safe_confirmations
    
    def estimate_transaction_time(self, confirmations: int) -> float:
        """Estimate transaction time in seconds"""
        return self.block.target_block_time * confirmations
    
    def get_explorer_url(self, tx_hash: str) -> Optional[str]:
        """Get explorer URL for transaction"""
        if self.api.explorer_url:
            return f"{self.api.explorer_url}/tx/{tx_hash}"
        return None
    
    def is_l2(self) -> bool:
        """Check if chain is an L2"""
        return self.rollup_type != RollupType.NONE 