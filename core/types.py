from typing import Dict, List, Tuple, Union, Optional, Any, Literal
from dataclasses import dataclass
from enum import Enum, auto
import torch

# Basic type aliases
TokenAddress = str
DexName = str
FlashLoanProviderType = Literal['aave', 'balancer', 'radiant']
TokenPair = Tuple[TokenAddress, TokenAddress]

# Complex types
MarketDataType = Dict[str, Union[float, str, Dict[str, Any]]]
OpportunityType = Dict[str, Union[str, float, TokenPair, Dict[str, Any]]]
FlashLoanOpportunityType = Dict[str, Union[
    str,  # id, type, timestamp
    TokenPair,  # token_pair
    float,  # amount, provider_score, fees, profits
    FlashLoanProviderType,  # flash_loan_provider
    Dict[str, Any]  # provider_metrics
]]

# Training types
@dataclass
class TrainingBatchType:
    """Training batch data structure"""
    features: torch.Tensor
    labels: torch.Tensor
    market_conditions: List[MarketDataType]
    timestamps: List[float]
    execution_results: List[Dict[str, Any]]
    weights: Optional[torch.Tensor] = None

# Network status types
@dataclass
class NetworkStatusType:
    """Network health and status information"""
    is_healthy: bool
    block_time: float
    gas_price: int
    pending_transactions: int
    network_load: float
    reason: Optional[str] = None

# Market metrics types
@dataclass
class MarketMetricsType:
    """Market performance and health metrics"""
    price: float
    volume_24h: float
    liquidity: float
    volatility: float
    tvl: float
    fees_24h: float
    price_impact: float
    slippage: float

class ExecutionStatus(Enum):
    """Execution status codes for arbitrage operations"""
    SUCCESS = auto()
    INVALID_OPPORTUNITY = auto()
    MARKET_CONDITIONS_CHANGED = auto()
    EXECUTION_ERROR = auto()
    NETWORK_ERROR = auto()

@dataclass
class ExecutionResult:
    """Result of an arbitrage execution attempt"""
    status: ExecutionStatus
    success: bool
    error_message: Optional[str] = None
    gas_used: Optional[int] = None
    execution_time: Optional[float] = None
    tx_hash: Optional[str] = None

@dataclass
class MarketValidationResult:
    """Result of market condition validation"""
    is_valid: bool
    reason: Optional[str] = None
    current_price: Optional[float] = None
    price_change: Optional[float] = None
    current_liquidity: Optional[float] = None
    current_gas: Optional[int] = None
    network_status: Optional[Dict[str, Any]] = None 