from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from torch import Tensor
from ..nn import _Module

class MarketImpactModel(_Module):
    """Advanced market impact modeling.
    
    Models complex market impact effects including
    temporary and permanent price impacts.
    """
    def __init__(
        self,
        impact_estimator: _Module,
        liquidity_model: _Module,
        decay_model: Optional[_Module] = None
    ) -> None: ...
    
    def estimate_impact(
        self,
        action: Dict[str, Tensor],
        market_state: Dict[str, Tensor],
        time_horizon: Optional[int] = None
    ) -> Tuple[Dict[str, Tensor], Dict[str, float]]: ...
    
    def compute_decay_profile(
        self,
        impact: Dict[str, Tensor],
        market_conditions: Dict[str, Any]
    ) -> Dict[str, Tensor]: ...
    
    def estimate_permanent_impact(
        self,
        action_sequence: List[Dict[str, Tensor]],
        market_history: Dict[str, Tensor]
    ) -> Dict[str, float]: ...

class LiquidityImpactModel(_Module):
    """Liquidity-aware impact modeling.
    
    Models how trades affect liquidity pools and
    cross-pool dynamics.
    """
    def __init__(
        self,
        pool_models: Dict[str, _Module],
        cross_pool_model: _Module,
        rebalance_threshold: float = 0.1
    ) -> None: ...
    
    def compute_pool_impact(
        self,
        trade_size: Tensor,
        pool_state: Dict[str, Tensor],
        include_rebalancing: bool = True
    ) -> Tuple[Dict[str, Tensor], Dict[str, float]]: ...
    
    def estimate_rebalancing_flows(
        self,
        pool_states: Dict[str, Tensor],
        price_impacts: Dict[str, float]
    ) -> Dict[str, Tensor]: ...
    
    def simulate_pool_dynamics(
        self,
        initial_state: Dict[str, Tensor],
        action_sequence: List[Dict[str, Tensor]]
    ) -> List[Dict[str, Tensor]]: ...

class CrossProtocolImpact:
    """Cross-protocol impact analyzer.
    
    Analyzes how actions in one protocol affect
    other protocols and overall market state.
    """
    def __init__(
        self,
        protocol_models: Dict[str, _Module],
        correlation_model: _Module,
        impact_threshold: float = 0.05
    ) -> None: ...
    
    def analyze_cross_effects(
        self,
        action: Dict[str, Tensor],
        protocol_states: Dict[str, Dict[str, Tensor]]
    ) -> Dict[str, Dict[str, Any]]: ...
    
    def estimate_contagion_risk(
        self,
        impact_event: Dict[str, Any],
        protocol_exposures: Dict[str, float]
    ) -> Dict[str, float]: ...
    
    def simulate_cascade_effects(
        self,
        initial_impact: Dict[str, Tensor],
        propagation_paths: List[List[str]]
    ) -> List[Dict[str, Dict[str, Tensor]]]: ...

class SlippageOptimizer:
    """Advanced slippage optimization.
    
    Optimizes trade execution to minimize slippage
    across multiple pools and routes.
    """
    def __init__(
        self,
        routing_model: _Module,
        impact_model: MarketImpactModel,
        optimization_params: Dict[str, Any]
    ) -> None: ...
    
    def optimize_execution(
        self,
        trade_intent: Dict[str, Any],
        market_state: Dict[str, Tensor],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Tensor]], Dict[str, float]]: ...
    
    def find_optimal_route(
        self,
        amount: float,
        token_pair: Tuple[str, str],
        available_pools: Dict[str, Dict[str, Tensor]]
    ) -> Tuple[List[Dict[str, Any]], float]: ...
    
    def estimate_execution_cost(
        self,
        route: List[Dict[str, Any]],
        amounts: List[float]
    ) -> Dict[str, float]: ...

class ImpactAwareController:
    """Impact-aware control system.
    
    Controls trading actions with explicit consideration
    of market impact and slippage.
    """
    def __init__(
        self,
        impact_model: MarketImpactModel,
        slippage_optimizer: SlippageOptimizer,
        control_params: Dict[str, Any]
    ) -> None: ...
    
    def compute_control_action(
        self,
        desired_action: Dict[str, Tensor],
        market_state: Dict[str, Tensor],
        risk_limits: Dict[str, float]
    ) -> Tuple[Dict[str, Tensor], Dict[str, Any]]: ...
    
    def validate_action_impact(
        self,
        action: Dict[str, Tensor],
        impact_estimates: Dict[str, float],
        thresholds: Dict[str, float]
    ) -> Tuple[bool, Dict[str, str]]: ...
    
    def adjust_for_impact(
        self,
        original_action: Dict[str, Tensor],
        impact_metrics: Dict[str, float]
    ) -> Dict[str, Tensor]: ... 