"""Risk management type definitions"""

from typing import Dict, Any, TypedDict, List
from decimal import Decimal
from dataclasses import dataclass
from datetime import datetime

@dataclass
class RiskMetrics:
    """Risk metrics data class"""
    volatility: float
    liquidity_risk: float
    market_risk: float
    network_risk: float
    overall_risk: float

@dataclass
class ProtocolMetrics:
    """Protocol metrics data class"""
    tvl: Decimal
    total_borrowed: Decimal
    total_supplied: Decimal
    health_score: float
    audit_score: float
    bug_bounty: float
    age_days: float

class MarketData(TypedDict):
    """Market data type definition"""
    price: Decimal
    volume_24h: Decimal
    liquidity: Decimal
    volatility: float
    correlations: Dict[str, Dict[str, float]]
    sentiment: Dict[str, Any]
    gas_price: float
    pending_tx_count: int
    block_time: float
    total_value: Decimal

class RiskAssessment(TypedDict):
    """Risk assessment result type"""
    health_risk: float
    liquidation_risk: float
    market_risk: float
    concentration_risk: float
    overall_risk: float

class ProtocolRisk(TypedDict):
    """Protocol risk assessment type"""
    tvl_risk: float
    utilization_risk: float
    health_risk: float
    contract_risk: float
    overall_risk: float

@dataclass
class RiskConfig:
    """Configuration for risk management."""
    max_drawdown: float
    var_confidence: float
    max_leverage: float
    correlation_threshold: float
    position_limits: Dict[str, Any]

@dataclass
class RiskThresholds:
    """Risk thresholds for strategy validation."""
    max_drawdown: float
    var_confidence: float
    max_leverage: float
    correlation_threshold: float 