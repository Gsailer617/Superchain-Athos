"""Risk management package

Provides comprehensive risk assessment and management functionality:
- Position risk calculation
- Market risk evaluation
- Protocol risk assessment
- Portfolio risk analysis
- Risk metrics and types
"""

from .risk_manager import RiskManager
from .risk_analysis import RiskAnalysis
from .fallback import FallbackStrategy, ConservativeGasStrategy, LiquidityPreservationStrategy
from .types import (
    RiskMetrics,
    ProtocolMetrics,
    MarketData,
    RiskAssessment,
    ProtocolRisk
)

__all__ = [
    'RiskManager',
    'RiskAnalysis',
    'FallbackStrategy',
    'ConservativeGasStrategy',
    'LiquidityPreservationStrategy',
    'RiskMetrics',
    'ProtocolMetrics',
    'MarketData',
    'RiskAssessment',
    'ProtocolRisk'
]

__version__ = '1.0.0' 