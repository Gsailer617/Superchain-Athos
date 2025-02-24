import logging
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class FallbackStrategy:
    """Base class for fallback strategies"""
    
    def __init__(self, name: str, priority: int):
        self.name = name
        self.priority = priority
        self.is_active = False
        self.activation_time = None
        
    async def activate(self, context: Dict[str, Any]) -> bool:
        """Activate the fallback strategy
        
        Args:
            context: Context data for activation
            
        Returns:
            Success status
        """
        try:
            self.is_active = True
            self.activation_time = time.time()
            return True
        except Exception as e:
            logger.error(f"Error activating fallback strategy {self.name}: {str(e)}")
            return False
            
    async def deactivate(self) -> bool:
        """Deactivate the fallback strategy
        
        Returns:
            Success status
        """
        try:
            self.is_active = False
            self.activation_time = None
            return True
        except Exception as e:
            logger.error(f"Error deactivating fallback strategy {self.name}: {str(e)}")
            return False

class ConservativeGasStrategy(FallbackStrategy):
    """Conservative gas price strategy"""
    
    def __init__(self):
        super().__init__("conservative_gas", 1)
        self.base_multiplier = 1.2
        
    async def activate(self, context: Dict[str, Any]) -> bool:
        """Activate conservative gas strategy
        
        Args:
            context: Context data including gas prices
            
        Returns:
            Success status
        """
        try:
            await super().activate(context)
            
            # Increase gas price estimates
            base_gas = context.get('base_gas_price', 0)
            context['adjusted_gas_price'] = base_gas * self.base_multiplier
            
            return True
            
        except Exception as e:
            logger.error(f"Error activating conservative gas strategy: {str(e)}")
            return False

class LiquidityPreservationStrategy(FallbackStrategy):
    """Liquidity preservation strategy"""
    
    def __init__(self):
        super().__init__("liquidity_preservation", 2)
        self.min_ratio = 0.2  # Minimum liquidity ratio
        
    async def activate(self, context: Dict[str, Any]) -> bool:
        """Activate liquidity preservation strategy
        
        Args:
            context: Context data including liquidity info
            
        Returns:
            Success status
        """
        try:
            await super().activate(context)
            
            # Adjust transaction amount
            liquidity = context.get('liquidity', 0)
            original_amount = context.get('amount', 0)
            
            max_amount = liquidity * self.min_ratio
            context['adjusted_amount'] = min(original_amount, max_amount)
            
            return True
            
        except Exception as e:
            logger.error(f"Error activating liquidity preservation strategy: {str(e)}")
            return False

class TimeoutExtensionStrategy(FallbackStrategy):
    """Timeout extension strategy"""
    
    def __init__(self):
        super().__init__("timeout_extension", 3)
        self.extension_factor = 2.0
        
    async def activate(self, context: Dict[str, Any]) -> bool:
        """Activate timeout extension strategy
        
        Args:
            context: Context data including timeout settings
            
        Returns:
            Success status
        """
        try:
            await super().activate(context)
            
            # Extend timeouts
            base_timeout = context.get('timeout', 60)
            context['adjusted_timeout'] = base_timeout * self.extension_factor
            
            return True
            
        except Exception as e:
            logger.error(f"Error activating timeout extension strategy: {str(e)}")
            return False

class CircuitBreakerStrategy(FallbackStrategy):
    """Circuit breaker strategy"""
    
    def __init__(self):
        super().__init__("circuit_breaker", 4)
        self.cool_down_period = 300  # 5 minutes
        
    async def activate(self, context: Dict[str, Any]) -> bool:
        """Activate circuit breaker
        
        Args:
            context: Context data
            
        Returns:
            Success status
        """
        try:
            await super().activate(context)
            
            # Set circuit breaker state
            context['circuit_breaker_active'] = True
            context['resume_time'] = time.time() + self.cool_down_period
            
            return True
            
        except Exception as e:
            logger.error(f"Error activating circuit breaker: {str(e)}")
            return False

class FallbackManager:
    """Manages fallback strategies and their activation"""
    
    def __init__(self):
        # Initialize strategies
        self.strategies = {
            'conservative_gas': ConservativeGasStrategy(),
            'liquidity_preservation': LiquidityPreservationStrategy(),
            'timeout_extension': TimeoutExtensionStrategy(),
            'circuit_breaker': CircuitBreakerStrategy()
        }
        
        # Activation history
        self.activation_history: List[Dict[str, Any]] = []
        
        # Current state
        self.active_strategies: Dict[str, FallbackStrategy] = {}
        
    async def handle_risk_event(
        self,
        risk_profile: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Handle risk event by activating appropriate fallback strategies
        
        Args:
            risk_profile: Current risk profile
            context: Execution context
            
        Returns:
            Tuple of (success status, updated context)
        """
        try:
            # Check which strategies should be activated
            strategies_to_activate = self._determine_needed_strategies(risk_profile)
            
            # Activate strategies in priority order
            activated = []
            for strategy_name in strategies_to_activate:
                strategy = self.strategies[strategy_name]
                
                if await strategy.activate(context):
                    self.active_strategies[strategy_name] = strategy
                    activated.append(strategy_name)
                    
                    # Record activation
                    self.activation_history.append({
                        'strategy': strategy_name,
                        'timestamp': time.time(),
                        'risk_profile': risk_profile,
                        'context': context
                    })
                    
            # Check if circuit breaker was activated
            if 'circuit_breaker' in activated:
                return False, context
                
            return len(activated) > 0, context
            
        except Exception as e:
            logger.error(f"Error handling risk event: {str(e)}")
            return False, context
            
    def _determine_needed_strategies(
        self,
        risk_profile: Dict[str, Any]
    ) -> List[str]:
        """Determine which strategies should be activated
        
        Args:
            risk_profile: Current risk profile
            
        Returns:
            List of strategy names to activate
        """
        try:
            needed_strategies = []
            
            # Check gas price risk
            if risk_profile.get('network_risks', {}).get('gas_risk', 0) > 0.7:
                needed_strategies.append('conservative_gas')
                
            # Check liquidity risk
            if risk_profile.get('market_risks', {}).get('liquidity_risk', 0) > 0.7:
                needed_strategies.append('liquidity_preservation')
                
            # Check network congestion
            if risk_profile.get('network_risks', {}).get('congestion_risk', 0) > 0.7:
                needed_strategies.append('timeout_extension')
                
            # Check for critical conditions
            if self._check_critical_conditions(risk_profile):
                needed_strategies.append('circuit_breaker')
                
            return sorted(
                needed_strategies,
                key=lambda x: self.strategies[x].priority
            )
            
        except Exception as e:
            logger.error(f"Error determining needed strategies: {str(e)}")
            return []
            
    def _check_critical_conditions(self, risk_profile: Dict[str, Any]) -> bool:
        """Check if conditions warrant circuit breaker activation
        
        Args:
            risk_profile: Current risk profile
            
        Returns:
            Whether critical conditions exist
        """
        try:
            # Check various critical conditions
            critical_conditions = [
                risk_profile.get('token_risks', {}).get('overall_risk', 0) > 0.9,
                risk_profile.get('market_risks', {}).get('price_impact', 0) > 0.1,
                risk_profile.get('execution_risks', {}).get('error_rate', 0) > 0.2
            ]
            
            return any(critical_conditions)
            
        except Exception as e:
            logger.error(f"Error checking critical conditions: {str(e)}")
            return True  # Conservative approach
            
    async def deactivate_strategy(self, strategy_name: str) -> bool:
        """Deactivate a specific strategy
        
        Args:
            strategy_name: Name of strategy to deactivate
            
        Returns:
            Success status
        """
        try:
            if strategy_name in self.active_strategies:
                strategy = self.active_strategies[strategy_name]
                if await strategy.deactivate():
                    del self.active_strategies[strategy_name]
                    return True
            return False
            
        except Exception as e:
            logger.error(f"Error deactivating strategy {strategy_name}: {str(e)}")
            return False
            
    def get_active_strategies(self) -> Dict[str, Any]:
        """Get currently active strategies
        
        Returns:
            Dict of active strategies and their states
        """
        return {
            name: {
                'activation_time': strategy.activation_time,
                'priority': strategy.priority
            }
            for name, strategy in self.active_strategies.items()
        }
        
    def get_activation_history(
        self,
        start_time: Optional[float] = None,
        strategy_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get activation history with filtering
        
        Args:
            start_time: Optional start time filter
            strategy_name: Optional strategy name filter
            
        Returns:
            Filtered activation history
        """
        try:
            filtered_history = self.activation_history
            
            if start_time is not None:
                filtered_history = [
                    entry for entry in filtered_history
                    if entry['timestamp'] >= start_time
                ]
                
            if strategy_name is not None:
                filtered_history = [
                    entry for entry in filtered_history
                    if entry['strategy'] == strategy_name
                ]
                
            return sorted(
                filtered_history,
                key=lambda x: x['timestamp'],
                reverse=True
            )
            
        except Exception as e:
            logger.error(f"Error getting activation history: {str(e)}")
            return [] 