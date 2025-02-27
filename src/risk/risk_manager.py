"""
Risk Manager Service

Provides consolidated risk assessment functionality:
- Position risk calculation
- Market risk evaluation
- Protocol risk assessment
- Portfolio risk analysis
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from decimal import Decimal
from datetime import datetime, timedelta
import numpy as np
import asyncio

from ..market.market_metrics import MarketMetrics
from ..market.types import Position, PortfolioState
from .risk_analysis import RiskAnalysis
from .types import RiskLevel, RiskThreshold, RiskReportType, RiskReport
from ..core.circuit_breaker import CircuitBreaker
from ..monitoring.notification_manager import NotificationManager

logger = logging.getLogger(__name__)

class RiskManager:
    """Enhanced Risk Management System
    
    Centralized risk management with:
    - Event-driven risk analysis
    - Integration with monitoring and notification systems
    - Circuit breaker integration for emergency stops
    - Historical risk data for trend analysis
    """

    def __init__(self, market_metrics: MarketMetrics):
        """Initialize the risk manager"""
        self.market_metrics = market_metrics
        self.risk_analysis = RiskAnalysis()
        
        self.risk_thresholds: Dict[str, RiskThreshold] = {
            'market': RiskThreshold(low=0.3, medium=0.6, high=0.8),
            'liquidity': RiskThreshold(low=0.3, medium=0.6, high=0.8),
            'execution': RiskThreshold(low=0.3, medium=0.6, high=0.8),
            'volatility': RiskThreshold(low=0.3, medium=0.6, high=0.8),
            'bridge': RiskThreshold(low=0.3, medium=0.6, high=0.8),
        }
        
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_time=300,  # 5 minutes
            half_open_timeout=60  # 1 minute
        )
        
        self.notification_manager = NotificationManager()
        self.risk_history: Dict[str, List[RiskReport]] = {}
        
        # Register event handlers for market/execution events
        self._register_event_handlers()
    
    def _register_event_handlers(self) -> None:
        """Register event handlers for system events"""
        try:
            from ..cqrs.events import event_bus
            
            # Register for market events
            event_bus.subscribe("market.price_change", self._handle_price_change)
            event_bus.subscribe("market.volatility_change", self._handle_volatility_change)
            event_bus.subscribe("market.liquidity_change", self._handle_liquidity_change)
            
            # Register for execution events
            event_bus.subscribe("execution.transaction_submitted", self._handle_transaction_submitted)
            event_bus.subscribe("execution.transaction_failed", self._handle_transaction_failed)
            
            # Register for bridge events
            event_bus.subscribe("bridge.status_change", self._handle_bridge_status_change)
            
            logger.info("Risk manager event handlers registered")
        except ImportError:
            logger.warning("Event bus not available, risk analysis will be request-driven only")
    
    async def _handle_price_change(self, event_data: Dict[str, Any]) -> None:
        """Handle price change events"""
        token = event_data.get('token')
        market_data = event_data.get('market_data', {})
        
        # Analyze risk based on price change
        risk_level = await self.analyze_market_risk(market_data)
        
        # Record and potentially notify
        self._record_risk(
            risk_type="market.price",
            entity=token,
            level=risk_level,
            data=event_data
        )
    
    async def _handle_volatility_change(self, event_data: Dict[str, Any]) -> None:
        """Handle volatility change events"""
        token = event_data.get('token')
        volatility = event_data.get('volatility', 0.0)
        
        # Determine risk level based on volatility
        threshold = self.risk_thresholds['volatility']
        risk_level = self._determine_risk_level(volatility, threshold)
        
        # Record and potentially notify
        self._record_risk(
            risk_type="market.volatility",
            entity=token,
            level=risk_level,
            data=event_data
        )
    
    async def _handle_liquidity_change(self, event_data: Dict[str, Any]) -> None:
        """Handle liquidity change events"""
        token = event_data.get('token')
        liquidity_risk = event_data.get('liquidity_risk', 0.0)
        
        # Determine risk level based on liquidity
        threshold = self.risk_thresholds['liquidity']
        risk_level = self._determine_risk_level(liquidity_risk, threshold)
        
        # Record and potentially notify
        self._record_risk(
            risk_type="market.liquidity",
            entity=token,
            level=risk_level,
            data=event_data
        )
    
    async def _handle_transaction_submitted(self, event_data: Dict[str, Any]) -> None:
        """Handle transaction submission events"""
        tx_hash = event_data.get('tx_hash')
        
        # Reset circuit breaker on successful submission
        if self.circuit_breaker.state != 'closed':
            self.circuit_breaker.succeed()
            logger.info(f"Circuit breaker reset on successful transaction submission")
    
    async def _handle_transaction_failed(self, event_data: Dict[str, Any]) -> None:
        """Handle transaction failure events"""
        tx_hash = event_data.get('tx_hash')
        error = event_data.get('error')
        
        # Increment circuit breaker failure counter
        self.circuit_breaker.fail()
        
        if self.circuit_breaker.state == 'open':
            # Alert on circuit breaker open
            await self.notification_manager.send_alert(
                level="high",
                title="Circuit Breaker Opened",
                message=f"Transaction failures exceeded threshold. Latest error: {error}",
                data=event_data
            )
    
    async def _handle_bridge_status_change(self, event_data: Dict[str, Any]) -> None:
        """Handle bridge status change events"""
        bridge = event_data.get('bridge')
        status = event_data.get('status')
        
        # Map status to risk level
        risk_map = {
            'active': RiskLevel.LOW,
            'congested': RiskLevel.MEDIUM,
            'low_liquidity': RiskLevel.MEDIUM,
            'verification_pending': RiskLevel.MEDIUM,
            'maintenance': RiskLevel.HIGH,
            'paused': RiskLevel.HIGH,
            'offline': RiskLevel.CRITICAL,
            'relayer_unavailable': RiskLevel.HIGH,
            'message_failed': RiskLevel.HIGH
        }
        
        risk_level = risk_map.get(status, RiskLevel.MEDIUM)
        
        # Record and potentially notify
        self._record_risk(
            risk_type="bridge.status",
            entity=bridge,
            level=risk_level,
            data=event_data
        )
    
    def _determine_risk_level(self, value: float, threshold: RiskThreshold) -> RiskLevel:
        """Determine risk level based on value and threshold"""
        if value >= threshold.high:
            return RiskLevel.HIGH
        elif value >= threshold.medium:
            return RiskLevel.MEDIUM
        elif value >= threshold.low:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL
    
    def _record_risk(self, risk_type: str, entity: str, level: RiskLevel, data: Dict[str, Any]) -> None:
        """Record risk event and send notifications if needed"""
        import time
        
        report = RiskReport(
            type=risk_type,
            entity=entity,
            level=level,
            timestamp=time.time(),
            data=data
        )
        
        # Store in history
        key = f"{risk_type}:{entity}"
        if key not in self.risk_history:
            self.risk_history[key] = []
        
        self.risk_history[key].append(report)
        
        # Keep history limited to recent reports
        if len(self.risk_history[key]) > 100:
            self.risk_history[key] = self.risk_history[key][-100:]
        
        # Send notification for high risk events
        if level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            asyncio.create_task(
                self.notification_manager.send_alert(
                    level=level.name.lower(),
                    title=f"High Risk Event: {risk_type}",
                    message=f"Risk level {level.name} detected for {entity}",
                    data=data
                )
            )
    
    async def analyze_market_risk(self, market_data: Dict[str, Any]) -> RiskLevel:
        """Analyze market risk and determine risk level"""
        risk_metrics = RiskAnalysis.calculate_market_risk(market_data)
        overall_risk = risk_metrics.get('overall_risk', 0.0)
        
        threshold = self.risk_thresholds['market']
        return self._determine_risk_level(overall_risk, threshold)
    
    async def analyze_execution_risk(self, tx_data: Dict[str, Any]) -> RiskLevel:
        """Analyze execution risk for a transaction"""
        # Enrich data with protocol information if available
        if 'token' in tx_data and 'protocol' not in tx_data:
            # Get protocol info for this token if available
            pass
            
        # Calculate risk including protocol-specific factors
        risk_metrics = RiskAnalysis.calculate_protocol_risk(tx_data)
        overall_risk = risk_metrics.get('overall_risk', 0.0)
        
        threshold = self.risk_thresholds['execution']
        return self._determine_risk_level(overall_risk, threshold)

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