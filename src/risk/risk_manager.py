"""
Risk Manager Service

Provides consolidated risk assessment functionality:
- Position risk calculation
- Market risk evaluation
- Protocol risk assessment
- Portfolio risk analysis
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
import numpy as np

from ..market.market_metrics import MarketMetrics
from ..market.types import Position, PortfolioState
from .risk_analysis import RiskAnalysis

logger = logging.getLogger(__name__)

class RiskManager:
    """Consolidated risk management service"""
    
    def __init__(self, market_metrics: MarketMetrics):
        self.market_metrics = market_metrics
        self.risk_analysis = RiskAnalysis()
        
    async def calculate_position_risk(
        self,
        position: Position,
        market_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate comprehensive position risk metrics"""
        try:
            # Calculate base risk metrics using RiskAnalysis
            health_risk = RiskAnalysis.calculate_position_health_risk(position)
            liquidation_risk = RiskAnalysis.calculate_liquidation_risk(position)
            
            # Calculate market-based risks
            market_risks = await self._get_market_risks(position.token, market_data)
            
            # Calculate concentration risk
            concentration_risk = self._calculate_concentration_risk(
                position.supplied,
                market_data['total_value']
            )
            
            # Calculate overall risk score
            overall_risk = (
                health_risk * 0.3 +
                liquidation_risk * 0.2 +
                market_risks['overall_risk'] * 0.35 +
                concentration_risk * 0.15
            )
            
            return {
                'health_risk': float(health_risk),
                'liquidation_risk': float(liquidation_risk),
                'market_risk': float(market_risks['overall_risk']),
                'concentration_risk': float(concentration_risk),
                'overall_risk': float(overall_risk)
            }
            
        except Exception as e:
            logger.error(f"Error calculating position risk: {str(e)}")
            raise
            
    async def evaluate_market_risk(
        self,
        market_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Evaluate market-wide risk factors"""
        try:
            return RiskAnalysis.calculate_market_risk(market_data)
        except Exception as e:
            logger.error(f"Error evaluating market risk: {str(e)}")
            raise
            
    async def assess_protocol_risk(
        self,
        protocol: str,
        metrics: Dict[str, Any]
    ) -> Dict[str, float]:
        """Assess protocol-specific risks"""
        try:
            return RiskAnalysis.calculate_protocol_risk(metrics)
        except Exception as e:
            logger.error(f"Error assessing protocol risk: {str(e)}")
            raise

    async def _get_market_risks(
        self,
        token: str,
        market_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Get market risks for a specific token"""
        try:
            # Enrich market data with token-specific info
            enriched_data = {
                **market_data,
                'token': token,
                'amount': market_data.get('position_size', 0)
            }
            return RiskAnalysis.calculate_market_risk(enriched_data)
        except Exception as e:
            logger.error(f"Error getting market risks: {str(e)}")
            return {
                'volatility': 1.0,
                'liquidity_risk': 1.0,
                'network_risk': 1.0,
                'overall_risk': 1.0
            }
            
    def _calculate_concentration_risk(
        self,
        position_value: Decimal,
        total_value: Decimal
    ) -> float:
        """Calculate position concentration risk"""
        try:
            if total_value == 0:
                return 1.0
            concentration = float(position_value / total_value)
            return min(concentration, 1.0)
        except Exception as e:
            logger.error(f"Error calculating concentration risk: {str(e)}")
            return 1.0 