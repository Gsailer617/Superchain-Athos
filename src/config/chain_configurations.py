from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional
from web3 import Web3
from eth_typing import ChecksumAddress

from .chain_config_template import (
    ChainConfig, RPCConfig, GasConfig, BlockConfig,
    BridgeConfig, SecurityConfig, NetworkConfig,
    APIConfig, PerformanceConfig, ChainFeatures,
    ConsensusType, RollupType, GasModel
)
from .chain_specs import ChainSpec, GasFeeModel

# Ethereum Mainnet Configuration
ETHEREUM_CONFIG = ChainConfig(
    # Basic Information
    name="Ethereum",
    chain_id=1,
    native_currency="ETH",
    native_currency_decimals=18,
    consensus_type=ConsensusType.POS,
    rollup_type=RollupType.NONE,
    
    # Network Configuration
    rpc=RPCConfig(
        http=[
            "https://eth-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}",
            "https://mainnet.infura.io/v3/${INFURA_KEY}",
            "https://ethereum.publicnode.com",
        ],
        ws=[
            "wss://eth-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}",
            "wss://mainnet.infura.io/ws/v3/${INFURA_KEY}",
        ],
        rate_limit=1000,  # Depends on provider
        batch_size=100,
        timeout=30,
        retry_count=3
    ),
    
    network=NetworkConfig(
        is_testnet=False,
        supports_eip1559=True,
        supports_eth_call_by_hash=True,
        supports_revert_reason=True,
        supports_pending_transactions=True,
        trace_api_available=True,
        debug_api_available=True,
        archive_node_available=True
    ),
    
    # Performance and Timing
    block=BlockConfig(
        target_block_time=12.0,  # seconds
        block_size_limit=30_000_000,  # bytes
        max_gas_per_block=30_000_000,
        safe_confirmations=12,
        finalized_confirmations=64,  # ~15 minutes
        reorg_protection_blocks=12,
        max_reorg_depth=64
    ),
    
    performance=PerformanceConfig(
        tx_pool_size=5000,
        max_pending_transactions=5000,
        max_queued_transactions=1000,
        mempool_max_size=5000,
        max_peers=50,
        target_gas_utilization=0.9,
        cache_size=2048  # MB
    ),
    
    # Gas and Fees
    gas=GasConfig(
        model=GasModel.EIP1559,
        base_fee_enabled=True,
        priority_fee_enabled=True,
        max_fee_per_gas=500_000_000_000,  # 500 gwei
        max_priority_fee_per_gas=10_000_000_000,  # 10 gwei
        gas_limit_multiplier=1.2,
        gas_price_multiplier=1.1,
        base_fee_max_change=0.125,  # 12.5% per block
        suggested_fee_history_blocks=10
    ),
    
    # Security and Protocol
    security=SecurityConfig(
        is_permissionless=True,
        validator_set_size=None,  # Dynamic based on stake
        min_stake_amount=Decimal("32"),  # 32 ETH
        challenge_period=None,
        fraud_proof_window=None,
        sequencer_publication_window=None
    ),
    
    # APIs and Services
    api=APIConfig(
        explorer_url="https://etherscan.io",
        explorer_api_url="https://api.etherscan.io/api",
        graph_node_url="https://api.thegraph.com/subgraphs/name/graphprotocol/graph-network-mainnet",
        trace_api_url="https://api.etherscan.io/api",
        debug_api_url="https://api.etherscan.io/api"
    ),
    
    # Features and Capabilities
    features=ChainFeatures(
        eips_supported=[155, 1559, 4895],  # Key EIPs
        opcodes_enabled=[
            "PUSH0",
            "CREATE",
            "DELEGATECALL",
            "STATICCALL",
            "REVERT"
        ],
        precompiles={
            "0x01": "ecrecover",
            "0x02": "sha256",
            "0x03": "ripemd160",
            "0x04": "identity",
            "0x05": "modexp",
            "0x06": "ecadd",
            "0x07": "ecmul",
            "0x08": "ecpairing",
            "0x09": "blake2f"
        },
        smart_contract_languages=["Solidity", "Vyper", "Yul"],
        vm_version="paris"
    ),
    
    # Additional Metadata
    description="Ethereum is a decentralized, open-source blockchain with smart contract functionality",
    website="https://ethereum.org",
    docs_url="https://ethereum.org/developers",
    github_url="https://github.com/ethereum",
    support_url="https://ethereum.org/community"
)

# Base Configuration
BASE_CONFIG = ChainConfig(
    # Basic Information
    name="Base",
    chain_id=8453,
    native_currency="ETH",
    native_currency_decimals=18,
    consensus_type=ConsensusType.OPTIMISTIC,
    rollup_type=RollupType.OPTIMISTIC,
    
    # Network Configuration
    rpc=RPCConfig(
        http=[
            "https://mainnet.base.org",
            "https://base-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}",
            "https://base.gateway.tenderly.co",
        ],
        ws=[
            "wss://base-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}",
        ],
        rate_limit=1000,
        batch_size=100,
        timeout=30,
        retry_count=3
    ),
    
    network=NetworkConfig(
        is_testnet=False,
        supports_eip1559=True,
        supports_eth_call_by_hash=True,
        supports_revert_reason=True,
        supports_pending_transactions=True,
        trace_api_available=True,
        debug_api_available=True,
        archive_node_available=True
    ),
    
    # Performance and Timing
    block=BlockConfig(
        target_block_time=2.0,  # seconds
        block_size_limit=5_000_000,  # bytes
        max_gas_per_block=15_000_000,
        safe_confirmations=5,
        finalized_confirmations=10,
        reorg_protection_blocks=0,  # L2 doesn't have reorgs
        max_reorg_depth=0
    ),
    
    performance=PerformanceConfig(
        tx_pool_size=5000,
        max_pending_transactions=5000,
        max_queued_transactions=1000,
        mempool_max_size=5000,
        max_peers=50,
        target_gas_utilization=0.9,
        cache_size=1024  # MB
    ),
    
    # Gas and Fees
    gas=GasConfig(
        model=GasModel.OPTIMISM_STYLE,
        base_fee_enabled=True,
        priority_fee_enabled=True,
        l1_fee_overhead=2100,  # L1 data fee overhead
        l1_fee_scalar=1_000_000,  # L1 fee scalar
        gas_limit_multiplier=1.2,
        gas_price_multiplier=1.1,
        base_fee_max_change=0.1,  # 10% per block
        suggested_fee_history_blocks=10
    ),
    
    # Security and Protocol
    security=SecurityConfig(
        is_permissionless=False,  # Uses sequencer
        validator_set_size=1,  # Single sequencer
        challenge_period=604800,  # 7 days
        fraud_proof_window=604800,  # 7 days
        sequencer_publication_window=3600  # 1 hour
    ),
    
    # Bridges and Cross-chain
    bridges={
        "official": BridgeConfig(
            address="0x3154Cf16ccdb4C6d922629664174b904d80F2C35",
            type="canonical",
            withdrawal_delay=604800,  # 7 days
            challenge_period=604800,  # 7 days
            verification_blocks=0
        )
    },
    
    # APIs and Services
    api=APIConfig(
        explorer_url="https://basescan.org",
        explorer_api_url="https://api.basescan.org/api",
        graph_node_url="https://api.studio.thegraph.com/query/base",
        trace_api_url="https://api.basescan.org/api",
        debug_api_url="https://api.basescan.org/api"
    ),
    
    # Features and Capabilities
    features=ChainFeatures(
        eips_supported=[155, 1559],
        opcodes_enabled=[
            "PUSH0",
            "CREATE",
            "DELEGATECALL",
            "STATICCALL",
            "REVERT"
        ],
        precompiles={
            "0x01": "ecrecover",
            "0x02": "sha256",
            "0x03": "ripemd160",
            "0x04": "identity"
        },
        smart_contract_languages=["Solidity", "Vyper"],
        vm_version="bedrock"
    ),
    
    # Additional Metadata
    description="Base is a secure, low-cost, builder-friendly Ethereum L2 built to bring the next billion users onchain",
    website="https://base.org",
    docs_url="https://docs.base.org",
    github_url="https://github.com/base-org",
    support_url="https://base.org/support"
)

# Arbitrum Configuration
ARBITRUM_CONFIG = ChainConfig(
    # Basic Information
    name="Arbitrum One",
    chain_id=42161,
    native_currency="ETH",
    native_currency_decimals=18,
    consensus_type=ConsensusType.OPTIMISTIC,
    rollup_type=RollupType.OPTIMISTIC,
    
    # Network Configuration
    rpc=RPCConfig(
        http=[
            "https://arb1.arbitrum.io/rpc",
            "https://arbitrum-mainnet.infura.io/v3/${INFURA_KEY}",
            "https://arb-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}",
        ],
        ws=[
            "wss://arb-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}",
            "wss://arbitrum-mainnet.infura.io/ws/v3/${INFURA_KEY}",
        ],
        rate_limit=1000,
        batch_size=100,
        timeout=30,
        retry_count=3
    ),
    
    network=NetworkConfig(
        is_testnet=False,
        supports_eip1559=True,
        supports_eth_call_by_hash=True,
        supports_revert_reason=True,
        supports_pending_transactions=True,
        trace_api_available=True,
        debug_api_available=True,
        archive_node_available=True
    ),
    
    # Performance and Timing
    block=BlockConfig(
        target_block_time=0.25,  # 250ms
        block_size_limit=128_000,  # bytes
        max_gas_per_block=32_000_000,
        safe_confirmations=20,
        finalized_confirmations=64,
        reorg_protection_blocks=0,  # L2 doesn't have reorgs
        max_reorg_depth=0
    ),
    
    performance=PerformanceConfig(
        tx_pool_size=50000,
        max_pending_transactions=50000,
        max_queued_transactions=10000,
        mempool_max_size=50000,
        max_peers=50,
        target_gas_utilization=0.9,
        cache_size=2048  # MB
    ),
    
    # Gas and Fees
    gas=GasConfig(
        model=GasModel.ARBITRUM_STYLE,
        base_fee_enabled=True,
        priority_fee_enabled=True,
        l1_fee_overhead=50000,  # Arbitrum-specific
        l1_fee_scalar=1_000_000,
        gas_limit_multiplier=1.2,
        gas_price_multiplier=1.1,
        base_fee_max_change=0.1,  # 10% per block
        suggested_fee_history_blocks=10
    ),
    
    # Security and Protocol
    security=SecurityConfig(
        is_permissionless=False,  # Uses sequencer
        validator_set_size=1,  # Single sequencer
        challenge_period=604800,  # 7 days
        fraud_proof_window=604800,  # 7 days
        sequencer_publication_window=3600  # 1 hour
    ),
    
    # Bridges and Cross-chain
    bridges={
        "official": BridgeConfig(
            address="0x72Ce9c846789fdB6fC1f34aC4AD25Dd9ef7031ef",
            type="canonical",
            withdrawal_delay=604800,  # 7 days
            challenge_period=604800,  # 7 days
            verification_blocks=0
        )
    },
    
    # APIs and Services
    api=APIConfig(
        explorer_url="https://arbiscan.io",
        explorer_api_url="https://api.arbiscan.io/api",
        graph_node_url="https://api.thegraph.com/subgraphs/name/arbitrum",
        trace_api_url="https://api.arbiscan.io/api",
        debug_api_url="https://api.arbiscan.io/api"
    ),
    
    # Features and Capabilities
    features=ChainFeatures(
        eips_supported=[155, 1559],
        opcodes_enabled=[
            "PUSH0",
            "CREATE",
            "DELEGATECALL",
            "STATICCALL",
            "REVERT",
            "ARBGAS",  # Arbitrum-specific
            "ARBADDRESS"  # Arbitrum-specific
        ],
        precompiles={
            "0x01": "ecrecover",
            "0x02": "sha256",
            "0x03": "ripemd160",
            "0x04": "identity",
            "0x64": "ArbSys",  # Arbitrum system contract
            "0x65": "ArbGas",  # Arbitrum gas info
            "0x66": "ArbAddressTable"  # Arbitrum address compression
        },
        smart_contract_languages=["Solidity", "Vyper"],
        vm_version="nitro"
    ),
    
    # Additional Metadata
    description="Arbitrum One is a Layer 2 scaling solution for Ethereum that uses optimistic rollups",
    website="https://arbitrum.io",
    docs_url="https://docs.arbitrum.io",
    github_url="https://github.com/OffchainLabs/arbitrum",
    support_url="https://arbitrum.io/support"
)

# Optimism Configuration
OPTIMISM_CONFIG = ChainConfig(
    # Basic Information
    name="Optimism",
    chain_id=10,
    native_currency="ETH",
    native_currency_decimals=18,
    consensus_type=ConsensusType.OPTIMISTIC,
    rollup_type=RollupType.OPTIMISTIC,
    
    # Network Configuration
    rpc=RPCConfig(
        http=[
            "https://mainnet.optimism.io",
            "https://opt-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}",
            "https://optimism-mainnet.infura.io/v3/${INFURA_KEY}",
        ],
        ws=[
            "wss://opt-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}",
            "wss://optimism-mainnet.infura.io/ws/v3/${INFURA_KEY}",
        ],
        rate_limit=1000,
        batch_size=100,
        timeout=30,
        retry_count=3
    ),
    
    network=NetworkConfig(
        is_testnet=False,
        supports_eip1559=True,
        supports_eth_call_by_hash=True,
        supports_revert_reason=True,
        supports_pending_transactions=True,
        trace_api_available=True,
        debug_api_available=True,
        archive_node_available=True
    ),
    
    # Performance and Timing
    block=BlockConfig(
        target_block_time=2.0,  # seconds
        block_size_limit=11_000_000,  # bytes
        max_gas_per_block=30_000_000,
        safe_confirmations=10,
        finalized_confirmations=50,
        reorg_protection_blocks=0,  # L2 doesn't have reorgs
        max_reorg_depth=0
    ),
    
    performance=PerformanceConfig(
        tx_pool_size=5000,
        max_pending_transactions=5000,
        max_queued_transactions=1000,
        mempool_max_size=5000,
        max_peers=50,
        target_gas_utilization=0.9,
        cache_size=1024  # MB
    ),
    
    # Gas and Fees
    gas=GasConfig(
        model=GasModel.OPTIMISM_STYLE,
        base_fee_enabled=True,
        priority_fee_enabled=True,
        l1_fee_overhead=2100,  # L1 data fee overhead
        l1_fee_scalar=1_000_000,  # L1 fee scalar
        gas_limit_multiplier=1.2,
        gas_price_multiplier=1.1,
        base_fee_max_change=0.1,  # 10% per block
        suggested_fee_history_blocks=10
    ),
    
    # Security and Protocol
    security=SecurityConfig(
        is_permissionless=False,  # Uses sequencer
        validator_set_size=1,  # Single sequencer
        challenge_period=604800,  # 7 days
        fraud_proof_window=604800,  # 7 days
        sequencer_publication_window=3600  # 1 hour
    ),
    
    # Bridges and Cross-chain
    bridges={
        "official": BridgeConfig(
            address="0x99C9fc46f92E8a1c0deC1b1747d010903E884bE1",
            type="canonical",
            withdrawal_delay=604800,  # 7 days
            challenge_period=604800,  # 7 days
            verification_blocks=0
        )
    },
    
    # APIs and Services
    api=APIConfig(
        explorer_url="https://optimistic.etherscan.io",
        explorer_api_url="https://api-optimistic.etherscan.io/api",
        graph_node_url="https://api.thegraph.com/subgraphs/name/optimism",
        trace_api_url="https://api-optimistic.etherscan.io/api",
        debug_api_url="https://api-optimistic.etherscan.io/api"
    ),
    
    # Features and Capabilities
    features=ChainFeatures(
        eips_supported=[155, 1559],
        opcodes_enabled=[
            "PUSH0",
            "CREATE",
            "DELEGATECALL",
            "STATICCALL",
            "REVERT"
        ],
        precompiles={
            "0x01": "ecrecover",
            "0x02": "sha256",
            "0x03": "ripemd160",
            "0x04": "identity",
            "0x4200000000000000000000000000000000000015": "OVM_L2CrossDomainMessenger",
            "0x4200000000000000000000000000000000000016": "OVM_L2ToL1MessagePasser"
        },
        smart_contract_languages=["Solidity", "Vyper"],
        vm_version="bedrock"
    ),
    
    # Additional Metadata
    description="Optimism is a Layer 2 scaling solution for Ethereum that uses optimistic rollups",
    website="https://optimism.io",
    docs_url="https://docs.optimism.io",
    github_url="https://github.com/ethereum-optimism/optimism",
    support_url="https://optimism.io/support"
)

# Polygon Configuration
POLYGON_CONFIG = ChainConfig(
    # Basic Information
    name="Polygon",
    chain_id=137,
    native_currency="MATIC",
    native_currency_decimals=18,
    consensus_type=ConsensusType.POS,
    rollup_type=RollupType.NONE,
    
    # Network Configuration
    rpc=RPCConfig(
        http=[
            "https://polygon-rpc.com",
            "https://polygon-mainnet.infura.io/v3/${INFURA_KEY}",
            "https://polygon-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}",
            "https://polygon.llamarpc.com",
        ],
        ws=[
            "wss://polygon-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}",
            "wss://polygon-mainnet.infura.io/ws/v3/${INFURA_KEY}",
        ],
        rate_limit=1000,
        batch_size=100,
        timeout=30,
        retry_count=3
    ),
    
    network=NetworkConfig(
        is_testnet=False,
        supports_eip1559=True,
        supports_eth_call_by_hash=True,
        supports_revert_reason=True,
        supports_pending_transactions=True,
        trace_api_available=True,
        debug_api_available=True,
        archive_node_available=True
    ),
    
    # Performance and Timing
    block=BlockConfig(
        target_block_time=2.0,  # seconds
        block_size_limit=20_000_000,  # bytes
        max_gas_per_block=30_000_000,
        safe_confirmations=128,  # Recommended for finality
        finalized_confirmations=256,
        reorg_protection_blocks=128,
        max_reorg_depth=256
    ),
    
    performance=PerformanceConfig(
        tx_pool_size=10000,
        max_pending_transactions=10000,
        max_queued_transactions=2000,
        mempool_max_size=10000,
        max_peers=100,
        target_gas_utilization=0.9,
        cache_size=2048  # MB
    ),
    
    # Gas and Fees
    gas=GasConfig(
        model=GasModel.EIP1559,
        base_fee_enabled=True,
        priority_fee_enabled=True,
        max_fee_per_gas=1000_000_000_000,  # 1000 gwei
        max_priority_fee_per_gas=100_000_000_000,  # 100 gwei
        gas_limit_multiplier=1.2,
        gas_price_multiplier=1.1,
        base_fee_max_change=0.25,  # 25% per block
        suggested_fee_history_blocks=10
    ),
    
    # Security and Protocol
    security=SecurityConfig(
        is_permissionless=False,  # Permissioned validator set
        validator_set_size=100,  # Maximum validator slots
        min_stake_amount=Decimal("1"),  # 1 MATIC
        challenge_period=None,
        fraud_proof_window=None,
        sequencer_publication_window=None,
        trusted_validators=[
            "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",  # Example validator
        ]
    ),
    
    # Bridges and Cross-chain
    bridges={
        "pos": BridgeConfig(
            address="0x7D1AfA7B718fb893dB30A3aBc0Cfc608AaCfeBB0",
            type="pos",
            token_list=["MATIC", "ETH", "USDC", "USDT"],
            withdrawal_delay=None,  # Instant for most tokens
            challenge_period=None,
            verification_blocks=0
        ),
        "plasma": BridgeConfig(
            address="0x401F6c983eA34274ec46f84D70b31C151321188b",
            type="plasma",
            token_list=["MATIC", "ETH", "ERC20"],
            withdrawal_delay=604800,  # 7 days for Plasma
            challenge_period=604800,
            verification_blocks=0
        )
    },
    
    # APIs and Services
    api=APIConfig(
        explorer_url="https://polygonscan.com",
        explorer_api_url="https://api.polygonscan.com/api",
        graph_node_url="https://api.thegraph.com/subgraphs/name/polygon",
        trace_api_url="https://api.polygonscan.com/api",
        debug_api_url="https://api.polygonscan.com/api"
    ),
    
    # Features and Capabilities
    features=ChainFeatures(
        eips_supported=[155, 1559],
        opcodes_enabled=[
            "PUSH0",
            "CREATE",
            "DELEGATECALL",
            "STATICCALL",
            "REVERT"
        ],
        precompiles={
            "0x01": "ecrecover",
            "0x02": "sha256",
            "0x03": "ripemd160",
            "0x04": "identity",
            "0x05": "modexp",
            "0x06": "ecadd",
            "0x07": "ecmul",
            "0x08": "ecpairing"
        },
        smart_contract_languages=["Solidity", "Vyper"],
        vm_version="london"
    ),
    
    # Additional Metadata
    description="Polygon PoS is a Layer 2 scaling solution for Ethereum that uses a Proof of Stake consensus mechanism",
    website="https://polygon.technology",
    docs_url="https://docs.polygon.technology",
    github_url="https://github.com/maticnetwork",
    support_url="https://support.polygon.technology"
)

# BNB Chain Configuration
BNB_CHAIN_CONFIG = ChainConfig(
    # Basic Information
    name="BNB Chain",
    chain_id=56,
    native_currency="BNB",
    native_currency_decimals=18,
    consensus_type=ConsensusType.POSA,
    rollup_type=RollupType.NONE,
    
    # Network Configuration
    rpc=RPCConfig(
        http=[
            "https://bsc-dataseed.binance.org",
            "https://bsc-dataseed1.defibit.io",
            "https://bsc-dataseed1.ninicoin.io",
            "https://bsc.nodereal.io",
        ],
        ws=[
            "wss://bsc-ws-node.nariox.org",
            "wss://bsc.nodereal.io/ws",
        ],
        rate_limit=1000,
        batch_size=100,
        timeout=30,
        retry_count=3
    ),
    
    network=NetworkConfig(
        is_testnet=False,
        supports_eip1559=False,  # Uses legacy gas model
        supports_eth_call_by_hash=True,
        supports_revert_reason=True,
        supports_pending_transactions=True,
        trace_api_available=True,
        debug_api_available=True,
        archive_node_available=True
    ),
    
    # Performance and Timing
    block=BlockConfig(
        target_block_time=3.0,  # seconds
        block_size_limit=30_000_000,  # bytes
        max_gas_per_block=100_000_000,
        safe_confirmations=15,
        finalized_confirmations=30,
        reorg_protection_blocks=15,
        max_reorg_depth=50
    ),
    
    performance=PerformanceConfig(
        tx_pool_size=20000,
        max_pending_transactions=20000,
        max_queued_transactions=5000,
        mempool_max_size=20000,
        max_peers=100,
        target_gas_utilization=0.9,
        cache_size=4096  # MB
    ),
    
    # Gas and Fees
    gas=GasConfig(
        model=GasModel.LEGACY,
        base_fee_enabled=False,
        priority_fee_enabled=False,
        min_gas_price=3_000_000_000,  # 3 gwei
        gas_limit_multiplier=1.2,
        gas_price_multiplier=1.1,
        suggested_fee_history_blocks=10
    ),
    
    # Security and Protocol
    security=SecurityConfig(
        is_permissionless=False,  # Permissioned validator set
        validator_set_size=21,  # Fixed validator set size
        min_stake_amount=Decimal("2000"),  # 2000 BNB minimum stake
        challenge_period=None,
        fraud_proof_window=None,
        sequencer_publication_window=None,
        trusted_validators=[
            "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",  # Example validator
        ]
    ),
    
    # Bridges and Cross-chain
    bridges={
        "official": BridgeConfig(
            address="0x0000000000000000000000000000000000001004",
            type="canonical",
            token_list=["BNB", "ETH", "USDC", "USDT", "BUSD"],
            withdrawal_delay=None,
            challenge_period=None,
            verification_blocks=15
        )
    },
    
    # APIs and Services
    api=APIConfig(
        explorer_url="https://bscscan.com",
        explorer_api_url="https://api.bscscan.com/api",
        graph_node_url="https://api.thegraph.com/subgraphs/name/bnb",
        trace_api_url="https://api.bscscan.com/api",
        debug_api_url="https://api.bscscan.com/api"
    ),
    
    # Features and Capabilities
    features=ChainFeatures(
        eips_supported=[155],  # Limited EIP support
        opcodes_enabled=[
            "PUSH0",
            "CREATE",
            "DELEGATECALL",
            "STATICCALL",
            "REVERT"
        ],
        precompiles={
            "0x01": "ecrecover",
            "0x02": "sha256",
            "0x03": "ripemd160",
            "0x04": "identity",
            "0x05": "modexp",
            "0x06": "ecadd",
            "0x07": "ecmul",
            "0x08": "ecpairing",
            "0x09": "blake2f"
        },
        smart_contract_languages=["Solidity", "Vyper"],
        vm_version="london"
    ),
    
    # Additional Metadata
    description="BNB Chain (formerly BSC) is a blockchain network built for running smart contract-based applications",
    website="https://www.bnbchain.org",
    docs_url="https://docs.bnbchain.org",
    github_url="https://github.com/bnb-chain",
    support_url="https://www.bnbchain.org/en/support"
)

# Mode
MODE_CONFIG = ChainConfig(
    name="mode",
    chain_id=34443,
    native_currency="ETH",
    native_currency_decimals=18,
    consensus_type=ConsensusType.OPTIMISTIC,
    rollup_type=RollupType.OPTIMISTIC,
    
    rpc=RPCConfig(
        http=["https://mainnet.mode.network"],
        ws=None,
        rate_limit=1000,
        batch_size=100,
        timeout=30,
        retry_count=3
    ),
    
    network=NetworkConfig(
        is_testnet=False,
        supports_eip1559=True,
        supports_eth_call_by_hash=True,
        supports_revert_reason=True,
        supports_pending_transactions=True
    ),
    
    block=BlockConfig(
        target_block_time=2.0,
        safe_confirmations=5,
        finalized_confirmations=10,
        reorg_protection_blocks=20
    ),
    
    gas=GasConfig(
        model=GasModel.L2_ROLLUP,
        base_fee_enabled=True,
        priority_fee_enabled=True,
        max_fee_per_gas=500_000_000_000,  # 500 gwei
        max_priority_fee_per_gas=10_000_000_000,  # 10 gwei
        gas_limit_multiplier=1.2,
        gas_price_multiplier=0.8  # 20% lower than standard
    ),
    
    security=SecurityConfig(
        is_permissionless=False,
        validator_set_size=1,
        challenge_period=604800,  # 7 days
        fraud_proof_window=604800,  # 7 days
        sequencer_publication_window=3600  # 1 hour
    ),
    
    api=APIConfig(
        explorer_url="https://explorer.mode.network",
        explorer_api_url="https://api.explorer.mode.network"
    ),
    
    performance=PerformanceConfig(
        tx_pool_size=5000,
        max_pending_transactions=1000,
        target_gas_utilization=0.8
    ),
    
    features=ChainFeatures(
        eips_supported=[1559, 2718, 2930],
        opcodes_enabled=["CREATE2", "STATICCALL"],
        precompiles={
            "0x01": "ecrecover",
            "0x02": "sha256",
            "0x03": "ripemd160",
            "0x04": "identity",
            "0x05": "modexp",
            "0x06": "ecadd",
            "0x07": "ecmul",
            "0x08": "ecpairing"
        },
        smart_contract_languages=["Solidity", "Vyper"],
        vm_version="paris"
    ),
    
    description="Mode is a high-performance Ethereum L2 network focused on DeFi applications",
    website="https://mode.network",
    docs_url="https://docs.mode.network",
    github_url="https://github.com/mode-network",
    support_url="https://support.mode.network"
)

# Sonic
SONIC_CONFIG = ChainConfig(
    name="sonic",
    chain_id=8899,
    native_currency="SONIC",
    native_currency_decimals=18,
    consensus_type=ConsensusType.POS,
    rollup_type=RollupType.NONE,
    
    rpc=RPCConfig(
        http=["https://mainnet.sonic.network"],
        ws=None,
        rate_limit=1000,
        batch_size=100,
        timeout=30,
        retry_count=3
    ),
    
    network=NetworkConfig(
        is_testnet=False,
        supports_eip1559=True,
        supports_eth_call_by_hash=True,
        supports_revert_reason=True,
        supports_pending_transactions=True
    ),
    
    block=BlockConfig(
        target_block_time=1.0,
        safe_confirmations=3,
        finalized_confirmations=6,
        reorg_protection_blocks=12
    ),
    
    gas=GasConfig(
        model=GasModel.EIP1559,
        base_fee_enabled=True,
        priority_fee_enabled=True,
        max_fee_per_gas=1_000_000_000_000,  # 1000 gwei
        max_priority_fee_per_gas=1_000_000_000,  # 1 gwei fixed
        gas_limit_multiplier=1.2,
        gas_price_multiplier=1.0
    ),
    
    security=SecurityConfig(
        is_permissionless=True,
        validator_set_size=100,
        min_stake_amount=Decimal("1000"),  # 1000 SONIC
        challenge_period=None,
        fraud_proof_window=None,
        sequencer_publication_window=None
    ),
    
    api=APIConfig(
        explorer_url="https://explorer.sonic.network",
        explorer_api_url="https://api.explorer.sonic.network"
    ),
    
    performance=PerformanceConfig(
        tx_pool_size=10000,
        max_pending_transactions=2000,
        target_gas_utilization=0.7
    ),
    
    features=ChainFeatures(
        eips_supported=[1559, 2718, 2930],
        opcodes_enabled=["CREATE2", "STATICCALL"],
        precompiles={
            "0x01": "ecrecover",
            "0x02": "sha256",
            "0x03": "ripemd160",
            "0x04": "identity",
            "0x05": "modexp",
            "0x06": "ecadd",
            "0x07": "ecmul",
            "0x08": "ecpairing"
        },
        smart_contract_languages=["Solidity", "Vyper"],
        vm_version="paris"
    ),
    
    description="Sonic is a high-performance blockchain network focused on DeFi applications",
    website="https://sonic.network",
    docs_url="https://docs.soniclabs.com",
    github_url="https://github.com/soniclabs",
    support_url="https://sonic.network/support"
)

# Chain Registry
CHAIN_CONFIGS: Dict[str, ChainConfig] = {
    "ethereum": ETHEREUM_CONFIG,
    "base": BASE_CONFIG,
    "arbitrum": ARBITRUM_CONFIG,
    "optimism": OPTIMISM_CONFIG,
    "polygon": POLYGON_CONFIG,
    "bnb": BNB_CHAIN_CONFIG,
    "mode": MODE_CONFIG,
    "sonic": SONIC_CONFIG,
}

def get_chain_config(chain_name: str) -> ChainConfig:
    """Get chain configuration by name"""
    if chain_name not in CHAIN_CONFIGS:
        raise ValueError(f"Chain {chain_name} not supported")
    return CHAIN_CONFIGS[chain_name]

def get_all_supported_chains() -> List[str]:
    """Get list of all supported chain names"""
    return list(CHAIN_CONFIGS.keys())

def get_l2_chains() -> List[str]:
    """Get list of L2 chain names"""
    return [name for name, spec in CHAIN_CONFIGS.items() if spec.rollup_type != RollupType.NONE]

def get_l1_chains() -> List[str]:
    """Get list of L1 chain names"""
    return [name for name, spec in CHAIN_CONFIGS.items() if spec.rollup_type == RollupType.NONE]

# Update bridge configurations for all chains
for chain_name, config in CHAIN_CONFIGS.items():
    if chain_name == "ethereum":
        config.bridges.update({
            "arbitrum": BridgeConfig(
                address="0x8315177aB297bA92A06054cE80a67Ed4DBd7ed3a",
                type="canonical",
                withdrawal_delay=604800  # 7 days
            ),
            "optimism": BridgeConfig(
                address="0x99C9fc46f92E8a1c0deC1b1747d010903E884bE1",
                type="canonical",
                withdrawal_delay=604800  # 7 days
            ),
            "base": BridgeConfig(
                address="0x3154Cf16ccdb4C6d922629664174b904d80F2C35",
                type="canonical",
                withdrawal_delay=604800  # 7 days
            ),
            "linea": BridgeConfig(
                address="0xE87d317eB8dcc9afE24d9f63D6C760e52Bc18A40",
                type="canonical",
                withdrawal_delay=604800  # 7 days
            ),
            "mantle": BridgeConfig(
                address="0x0000000000000000000000000000000000001010",
                type="canonical",
                withdrawal_delay=604800  # 7 days
            ),
            "mode": BridgeConfig(
                address="0x0000000000000000000000000000000000001010",
                type="canonical",
                withdrawal_delay=604800  # 7 days
            ),
            "gnosis": BridgeConfig(
                address="0x88ad09518695c6c3712AC10a214bE5109a655671",
                type="omni",
                withdrawal_delay=0  # Instant for xDAI
            ),
            "polygon": BridgeConfig(
                address="0x7D1AfA7B718fb893dB30A3aBc0Cfc608AaCfeBB0",
                type="pos",
                withdrawal_delay=0  # Instant for PoS
            ),
            "avalanche": BridgeConfig(
                address="0x1a44076050125825900e736c501f859c50fe728c",
                type="canonical",
                withdrawal_delay=0
            ),
            "sonic": BridgeConfig(
                address="0x0000000000000000000000000000000000001010",
                type="canonical",
                withdrawal_delay=0
            ),
        })
    elif chain_name == "base":
        config.bridges.update({
            "ethereum": BridgeConfig(
                address="0x4200000000000000000000000000000000000010",
                type="canonical",
                withdrawal_delay=604800  # 7 days
            )
        })
    elif chain_name == "arbitrum":
        config.bridges.update({
            "ethereum": BridgeConfig(
                address="0x5288c571Fd7aD117beA99bF60FE0846C4E84F933",
                type="canonical",
                withdrawal_delay=604800  # 7 days
            )
        })
    elif chain_name == "optimism":
        config.bridges.update({
            "ethereum": BridgeConfig(
                address="0x4200000000000000000000000000000000000010",
                type="canonical",
                withdrawal_delay=604800  # 7 days
            )
        })
    elif chain_name == "polygon":
        config.bridges.update({
            "ethereum": BridgeConfig(
                address="0x401F6c983eA34274ec46f84D70b31C151321188b",
                type="plasma",
                withdrawal_delay=604800  # 7 days for Plasma
            ),
            "ethereum_pos": BridgeConfig(
                address="0x7D1AfA7B718fb893dB30A3aBc0Cfc608AaCfeBB0",
                type="pos",
                withdrawal_delay=0  # Instant for PoS
            )
        })
    elif chain_name == "linea":
        config.bridges.update({
            "ethereum": BridgeConfig(
                address="0x0000000000000000000000000000000000001010",
                type="canonical",
                withdrawal_delay=604800  # 7 days
            )
        })
    elif chain_name == "mantle":
        config.bridges.update({
            "ethereum": BridgeConfig(
                address="0x0000000000000000000000000000000000001010",
                type="canonical",
                withdrawal_delay=604800  # 7 days
            )
        })
    elif chain_name == "mode":
        config.bridges.update({
            "ethereum": BridgeConfig(
                address="0x0000000000000000000000000000000000001010",
                type="canonical",
                withdrawal_delay=604800  # 7 days
            )
        })
    elif chain_name == "gnosis":
        config.bridges.update({
            "ethereum": BridgeConfig(
                address="0x88ad09518695c6c3712AC10a214bE5109a655671",
                type="omni",
                withdrawal_delay=0  # Instant for xDAI
            ),
            "ethereum_amb": BridgeConfig(
                address="0x75Df5AF045d91108662D8080fD1FEFAd6aA0bb59",
                type="amb",
                withdrawal_delay=0
            )
        })
    elif chain_name == "avalanche":
        config.bridges.update({
            "ethereum": BridgeConfig(
                address="0x1a44076050125825900e736c501f859c50fe728c",
                type="canonical",
                withdrawal_delay=0
            ),
            "subnet": BridgeConfig(
                address="0x0000000000000000000000000000000000001010",
                type="subnet",
                withdrawal_delay=0
            )
        })
    elif chain_name == "sonic":
        config.bridges.update({
            "ethereum": BridgeConfig(
                address="0x0000000000000000000000000000000000001010",
                type="canonical",
                withdrawal_delay=0
            )
        }) 