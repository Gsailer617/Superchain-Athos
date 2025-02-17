from dataclasses import dataclass
from typing import Optional, Dict, List
from enum import Enum

class GasFeeModel(Enum):
    """Gas fee models supported by different chains"""
    EIP1559 = "EIP-1559"      # Base fee + priority fee
    LEGACY = "legacy"         # Gas price only
    OPTIMISTIC = "optimistic" # L2-specific fee model
    ARBITRUM = "arbitrum"     # Arbitrum-specific fee model

class GasModel(Enum):
    """Gas pricing models"""
    LEGACY = "legacy"
    EIP1559 = "eip1559"
    L2_ROLLUP = "l2_rollup"
    ARBITRUM_STYLE = "arbitrum"
    MODE_STYLE = "mode"  # Mode's gas model
    SONIC_STYLE = "sonic"  # Sonic's gas model

@dataclass
class ChainSpec:
    """Detailed chain specifications"""
    name: str
    chain_id: int
    native_currency: str
    block_time: float  # seconds
    confirmation_blocks: int
    gas_fee_model: GasFeeModel
    is_l2: bool
    rpc_urls: List[str]  # Multiple URLs for fallback
    ws_urls: Optional[List[str]] = None
    explorer_url: Optional[str] = None
    explorer_api_url: Optional[str] = None
    bridge_contracts: Optional[Dict[str, str]] = None
    max_gas_price: Optional[int] = None  # in gwei
    safe_gas_limit: Optional[int] = None
    features: Optional[List[str]] = None

# Chain Specifications based on official documentation
CHAIN_SPECS = {
    # Ethereum Mainnet
    "ethereum": ChainSpec(
        name="Ethereum",
        chain_id=1,
        native_currency="ETH",
        block_time=12.0,  # Average block time
        confirmation_blocks=12,  # Recommended for high-value transactions
        gas_fee_model=GasFeeModel.EIP1559,
        is_l2=False,
        rpc_urls=[
            "https://eth-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}",
            "https://mainnet.infura.io/v3/${INFURA_KEY}",
            "https://ethereum.publicnode.com",
        ],
        ws_urls=[
            "wss://eth-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}",
            "wss://mainnet.infura.io/ws/v3/${INFURA_KEY}",
        ],
        explorer_url="https://etherscan.io",
        explorer_api_url="https://api.etherscan.io/api",
        max_gas_price=300,  # 300 gwei max
        safe_gas_limit=21000,  # Base transfer
        features=[
            "EIP1559",
            "ENS",
            "FLASHBOTS",
            "MEV_BOOST",
        ]
    ),

    # Base
    "base": ChainSpec(
        name="Base",
        chain_id=8453,
        native_currency="ETH",
        block_time=2.0,
        confirmation_blocks=5,  # Recommended by Base docs
        gas_fee_model=GasFeeModel.OPTIMISTIC,
        is_l2=True,
        rpc_urls=[
            "https://mainnet.base.org",
            "https://base-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}",
            "https://base.gateway.tenderly.co",
        ],
        ws_urls=[
            "wss://base-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}",
        ],
        explorer_url="https://basescan.org",
        explorer_api_url="https://api.basescan.org/api",
        bridge_contracts={
            "l1_bridge": "0x3154Cf16ccdb4C6d922629664174b904d80F2C35",
            "l2_bridge": "0x4200000000000000000000000000000000000010",
        },
        max_gas_price=150,
        safe_gas_limit=21000,
        features=[
            "EIP1559",
            "OPTIMISTIC_ROLLUP",
            "FAST_WITHDRAWALS",
        ]
    ),

    # Arbitrum
    "arbitrum": ChainSpec(
        name="Arbitrum One",
        chain_id=42161,
        native_currency="ETH",
        block_time=0.25,  # 250ms block time
        confirmation_blocks=64,  # Recommended for finality
        gas_fee_model=GasFeeModel.ARBITRUM,
        is_l2=True,
        rpc_urls=[
            "https://arb1.arbitrum.io/rpc",
            "https://arbitrum-mainnet.infura.io/v3/${INFURA_KEY}",
            "https://arb-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}",
        ],
        ws_urls=[
            "wss://arb-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}",
            "wss://arbitrum-mainnet.infura.io/ws/v3/${INFURA_KEY}",
        ],
        explorer_url="https://arbiscan.io",
        explorer_api_url="https://api.arbiscan.io/api",
        bridge_contracts={
            "l1_gateway": "0x72Ce9c846789fdB6fC1f34aC4AD25Dd9ef7031ef",
            "l2_gateway": "0x5288c571Fd7aD117beA99bF60FE0846C4E84F933",
        },
        max_gas_price=200,
        safe_gas_limit=21000,
        features=[
            "NITRO",
            "FAST_WITHDRAWALS",
            "CALLDATA_COMPRESSION",
        ]
    ),

    # Polygon
    "polygon": ChainSpec(
        name="Polygon",
        chain_id=137,
        native_currency="MATIC",
        block_time=2.0,
        confirmation_blocks=128,  # Recommended for finality
        gas_fee_model=GasFeeModel.EIP1559,
        is_l2=False,
        rpc_urls=[
            "https://polygon-rpc.com",
            "https://polygon-mainnet.infura.io/v3/${INFURA_KEY}",
            "https://polygon-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}",
        ],
        ws_urls=[
            "wss://polygon-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}",
            "wss://polygon-mainnet.infura.io/ws/v3/${INFURA_KEY}",
        ],
        explorer_url="https://polygonscan.com",
        explorer_api_url="https://api.polygonscan.com/api",
        bridge_contracts={
            "pos_bridge": "0x7D1AfA7B718fb893dB30A3aBc0Cfc608AaCfeBB0",
            "plasma_bridge": "0x401F6c983eA34274ec46f84D70b31C151321188b",
        },
        max_gas_price=500,  # Higher due to MATIC pricing
        safe_gas_limit=21000,
        features=[
            "EIP1559",
            "POS",
            "PLASMA",
            "FAST_EXITS",
        ]
    ),

    # BNB Chain
    "bnb": ChainSpec(
        name="BNB Chain",
        chain_id=56,
        native_currency="BNB",
        block_time=3.0,
        confirmation_blocks=15,  # Recommended by BSC docs
        gas_fee_model=GasFeeModel.LEGACY,
        is_l2=False,
        rpc_urls=[
            "https://bsc-dataseed.binance.org",
            "https://bsc-dataseed1.defibit.io",
            "https://bsc-dataseed1.ninicoin.io",
        ],
        ws_urls=[
            "wss://bsc-ws-node.nariox.org",
        ],
        explorer_url="https://bscscan.com",
        explorer_api_url="https://api.bscscan.com/api",
        bridge_contracts={
            "bsc_bridge": "0x0000000000000000000000000000000000001004",
        },
        max_gas_price=300,
        safe_gas_limit=21000,
        features=[
            "PARLIA_CONSENSUS",
            "CROSS_CHAIN",
        ]
    ),

    # Optimism
    "optimism": ChainSpec(
        name="Optimism",
        chain_id=10,
        native_currency="ETH",
        block_time=2.0,
        confirmation_blocks=50,  # Recommended for finality
        gas_fee_model=GasFeeModel.OPTIMISTIC,
        is_l2=True,
        rpc_urls=[
            "https://mainnet.optimism.io",
            "https://opt-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}",
            "https://optimism-mainnet.infura.io/v3/${INFURA_KEY}",
        ],
        ws_urls=[
            "wss://opt-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}",
            "wss://optimism-mainnet.infura.io/ws/v3/${INFURA_KEY}",
        ],
        explorer_url="https://optimistic.etherscan.io",
        explorer_api_url="https://api-optimistic.etherscan.io/api",
        bridge_contracts={
            "l1_bridge": "0x99C9fc46f92E8a1c0deC1b1747d010903E884bE1",
            "l2_bridge": "0x4200000000000000000000000000000000000010",
        },
        max_gas_price=200,
        safe_gas_limit=21000,
        features=[
            "BEDROCK",
            "FAST_WITHDRAWALS",
            "OP_STACK",
        ]
    ),

    # Linea
    "linea": ChainSpec(
        name="Linea",
        chain_id=59144,
        native_currency="ETH",
        block_time=12.0,
        confirmation_blocks=10,
        gas_fee_model=GasFeeModel.EIP1559,
        is_l2=True,
        rpc_urls=[
            "https://linea-mainnet.infura.io/v3/${INFURA_KEY}",
            "https://rpc.linea.build",
        ],
        ws_urls=[
            "wss://linea-mainnet.infura.io/ws/v3/${INFURA_KEY}",
        ],
        explorer_url="https://lineascan.build",
        explorer_api_url="https://api.lineascan.build/api",
        bridge_contracts={
            "l1_bridge": "0xE87d317eB8dcc9afE24d9f63D6C760e52Bc18A40",
            "l2_bridge": "0x0000000000000000000000000000000000000000",  # TBD
        },
        max_gas_price=150,
        safe_gas_limit=21000,
        features=[
            "EIP1559",
            "ZK_ROLLUP",
        ]
    ),

    # Mantle
    "mantle": ChainSpec(
        name="Mantle",
        chain_id=5000,
        native_currency="MNT",
        block_time=2.0,
        confirmation_blocks=10,
        gas_fee_model=GasFeeModel.OPTIMISTIC,
        is_l2=True,
        rpc_urls=[
            "https://rpc.mantle.xyz",
        ],
        explorer_url="https://explorer.mantle.xyz",
        bridge_contracts={
            "l1_bridge": "0x0000000000000000000000000000000000001010",
        },
        max_gas_price=100,
        safe_gas_limit=21000,
        features=[
            "OPTIMISTIC_ROLLUP",
            "DATA_AVAILABILITY",
        ]
    ),

    # Avalanche
    "avalanche": ChainSpec(
        name="Avalanche",
        chain_id=43114,
        native_currency="AVAX",
        block_time=2.0,
        confirmation_blocks=12,
        gas_fee_model=GasFeeModel.EIP1559,
        is_l2=False,
        rpc_urls=[
            "https://api.avax.network/ext/bc/C/rpc",
            "https://avalanche-mainnet.infura.io/v3/${INFURA_KEY}",
        ],
        ws_urls=[
            "wss://api.avax.network/ext/bc/C/ws",
        ],
        explorer_url="https://snowtrace.io",
        explorer_api_url="https://api.snowtrace.io/api",
        max_gas_price=225,
        safe_gas_limit=21000,
        features=[
            "EIP1559",
            "SUBNET",
            "X_CHAIN",
            "P_CHAIN",
        ]
    ),

    # Gnosis Chain (formerly xDai)
    "gnosis": ChainSpec(
        name="Gnosis Chain",
        chain_id=100,
        native_currency="xDAI",
        block_time=5.0,
        confirmation_blocks=12,
        gas_fee_model=GasFeeModel.EIP1559,
        is_l2=False,
        rpc_urls=[
            "https://rpc.gnosischain.com",
            "https://gnosis-mainnet.public.blastapi.io",
        ],
        explorer_url="https://gnosisscan.io",
        explorer_api_url="https://api.gnosisscan.io/api",
        bridge_contracts={
            "omni_bridge": "0x88ad09518695c6c3712AC10a214bE5109a655671",
        },
        max_gas_price=100,
        safe_gas_limit=21000,
        features=[
            "EIP1559",
            "POS",
            "OMNI_BRIDGE",
        ]
    ),
}

def get_chain_spec(chain_name: str) -> ChainSpec:
    """Get chain specification by name"""
    if chain_name not in CHAIN_SPECS:
        raise ValueError(f"Chain {chain_name} not supported")
    return CHAIN_SPECS[chain_name]

def get_all_supported_chains() -> List[str]:
    """Get list of all supported chain names"""
    return list(CHAIN_SPECS.keys())

def get_l2_chains() -> List[str]:
    """Get list of L2 chain names"""
    return [name for name, spec in CHAIN_SPECS.items() if spec.is_l2]

def get_l1_chains() -> List[str]:
    """Get list of L1 chain names"""
    return [name for name, spec in CHAIN_SPECS.items() if not spec.is_l2]

# Mode
MODE_CONFIG = ChainSpec(
    name="Mode",
    chain_id=34443,
    native_currency="ETH",
    native_currency_decimals=18,
    consensus_type=ConsensusType.OPTIMISTIC,
    rollup_type=RollupType.OPTIMISTIC,
    
    rpc=RPCConfig(
        http=[
            "https://mainnet.mode.network",
            "https://mode-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}",
        ],
        ws=[
            "wss://mode-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}",
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
    
    block=BlockConfig(
        target_block_time=2.0,
        block_size_limit=11_000_000,
        max_gas_per_block=30_000_000,
        safe_confirmations=10,
        finalized_confirmations=50,
        reorg_protection_blocks=0,
        max_reorg_depth=0
    ),
    
    performance=PerformanceConfig(
        tx_pool_size=5000,
        max_pending_transactions=5000,
        max_queued_transactions=1000,
        mempool_max_size=5000,
        max_peers=50,
        target_gas_utilization=0.9,
        cache_size=1024
    ),
    
    gas=GasConfig(
        model=GasModel.OPTIMISTIC,
        base_fee_enabled=True,
        priority_fee_enabled=True,
        max_fee_per_gas=500_000_000_000,
        max_priority_fee_per_gas=10_000_000_000,
        gas_limit_multiplier=1.2,
        gas_price_multiplier=1.1,
        base_fee_max_change=0.125,
        suggested_fee_history_blocks=10
    ),
    
    security=SecurityConfig(
        is_permissionless=False,
        validator_set_size=1,
        challenge_period=604800,  # 7 days
        fraud_proof_window=604800,  # 7 days
        sequencer_publication_window=3600  # 1 hour
    ),
    
    bridges={
        "official": BridgeConfig(
            address="0x0000000000000000000000000000000000001010",
            type="canonical",
            withdrawal_delay=604800,  # 7 days
            challenge_period=604800,  # 7 days
            verification_blocks=0
        )
    },
    
    api=APIConfig(
        explorer_url="https://explorer.mode.network",
        explorer_api_url="https://api.mode.network",
        graph_node_url="https://api.thegraph.com/subgraphs/name/mode",
        trace_api_url="https://api.mode.network/trace",
        debug_api_url="https://api.mode.network/debug"
    ),
    
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
    
    description="Mode is an Optimistic rollup built on Ethereum",
    website="https://mode.network",
    docs_url="https://docs.mode.network",
    github_url="https://github.com/mode-network",
    support_url="https://mode.network/support"
)

# Sonic
SONIC_CONFIG = ChainSpec(
    name="Sonic",
    chain_id=8899,  # Example chain ID
    native_currency="SONIC",
    native_currency_decimals=18,
    consensus_type=ConsensusType.POS,
    rollup_type=RollupType.NONE,
    
    rpc=RPCConfig(
        http=[
            "https://mainnet.sonic.network",
            "https://rpc.sonic.network",
        ],
        ws=[
            "wss://ws.sonic.network",
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
    
    block=BlockConfig(
        target_block_time=1.0,  # Fast block time
        block_size_limit=30_000_000,
        max_gas_per_block=30_000_000,
        safe_confirmations=20,
        finalized_confirmations=100,
        reorg_protection_blocks=20,
        max_reorg_depth=100
    ),
    
    performance=PerformanceConfig(
        tx_pool_size=10000,
        max_pending_transactions=10000,
        max_queued_transactions=2000,
        mempool_max_size=10000,
        max_peers=100,
        target_gas_utilization=0.9,
        cache_size=2048
    ),
    
    gas=GasConfig(
        model=GasModel.EIP1559,
        base_fee_enabled=True,
        priority_fee_enabled=True,
        max_fee_per_gas=1000_000_000_000,
        max_priority_fee_per_gas=20_000_000_000,
        gas_limit_multiplier=1.2,
        gas_price_multiplier=1.1,
        base_fee_max_change=0.125,
        suggested_fee_history_blocks=10
    ),
    
    security=SecurityConfig(
        is_permissionless=True,
        validator_set_size=100,
        min_stake_amount=Decimal("1000"),  # 1000 SONIC
        challenge_period=None,
        fraud_proof_window=None,
        sequencer_publication_window=None
    ),
    
    bridges={
        "official": BridgeConfig(
            address="0x0000000000000000000000000000000000001010",
            type="canonical",
            token_list=["SONIC", "ETH", "USDC", "USDT"],
            withdrawal_delay=None,
            challenge_period=None,
            verification_blocks=20
        )
    },
    
    api=APIConfig(
        explorer_url="https://explorer.sonic.network",
        explorer_api_url="https://api.sonic.network",
        graph_node_url="https://api.thegraph.com/subgraphs/name/sonic",
        trace_api_url="https://api.sonic.network/trace",
        debug_api_url="https://api.sonic.network/debug"
    ),
    
    features=ChainFeatures(
        eips_supported=[155, 1559, 4895],
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
        vm_version="paris"
    ),
    
    description="Sonic is a high-performance blockchain network focused on DeFi applications",
    website="https://sonic.network",
    docs_url="https://docs.soniclabs.com",
    github_url="https://github.com/soniclabs",
    support_url="https://sonic.network/support"
)

# Update CHAIN_SPECS dictionary
CHAIN_SPECS.update({
    "mode": MODE_CONFIG,
    "sonic": SONIC_CONFIG
}) 