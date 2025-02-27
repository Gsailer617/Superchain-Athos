"""
Analysis Module

This module provides various analysis tools for market data, cross-chain opportunities,
flash loan arbitrage, and yield farming strategies.
"""

from .analyzer import MarketAnalyzer
from .cross_chain import CrossChainAnalyzer
from .cross_chain_analyzer import CrossChainAnalyzer as CrossChainOpportunityAnalyzer
from .cross_chain_analyzer import ChainMetrics, PerformanceMetrics, OpportunityMetrics
from .yield_farming import YieldFarmingAnalyzer, PoolData, FarmingOpportunity

__all__ = [
    'MarketAnalyzer',
    'CrossChainAnalyzer',
    'CrossChainOpportunityAnalyzer',
    'ChainMetrics',
    'PerformanceMetrics',
    'OpportunityMetrics',
    'YieldFarmingAnalyzer',
    'PoolData',
    'FarmingOpportunity'
]
