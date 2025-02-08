from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from torch import Tensor
from ..nn import _Module

class MarketPredictor(_Module):
    """Predictive model for market behavior.
    
    Forecasts short-term market movements and liquidity
    conditions for decision making.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        horizon: int = 10,
        ensemble_size: int = 5,
        uncertainty_estimation: bool = True
    ) -> None: ...
    
    def forward(
        self,
        current_state: Tensor,
        actions: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]: ...
    
    def predict_trajectory(
        self,
        initial_state: Tensor,
        action_sequence: Tensor,
        include_uncertainty: bool = True
    ) -> Tuple[Tensor, Dict[str, Tensor]]: ...
    
    def update_model(
        self,
        states: Tensor,
        actions: Tensor,
        next_states: Tensor
    ) -> Dict[str, float]: ...

class MPCOptimizer(_Module):
    """Model Predictive Control optimizer.
    
    Optimizes trading actions using predictive models
    and scenario analysis.
    """
    def __init__(
        self,
        predictor: MarketPredictor,
        planning_horizon: int = 10,
        num_scenarios: int = 100,
        optimization_method: str = 'cem'
    ) -> None: ...
    
    def optimize_actions(
        self,
        current_state: Tensor,
        objective_fn: Callable[[Tensor, Tensor], Tensor],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Tuple[Tensor, Dict[str, Any]]: ...
    
    def evaluate_trajectory(
        self,
        state_sequence: Tensor,
        action_sequence: Tensor,
        objective_fn: Callable[[Tensor, Tensor], Tensor]
    ) -> Dict[str, Tensor]: ...

class ScenarioGenerator:
    """Generates market scenarios for simulation.
    
    Creates diverse market scenarios for testing and
    optimization of trading strategies.
    """
    def __init__(
        self,
        base_market_state: Dict[str, Tensor],
        scenario_params: Dict[str, Any],
        num_scenarios: int = 100
    ) -> None: ...
    
    def generate_scenarios(
        self,
        horizon: int,
        include_adversarial: bool = True
    ) -> Tuple[List[Dict[str, Tensor]], Dict[str, Any]]: ...
    
    def create_stress_scenario(
        self,
        stress_type: str,
        severity: float
    ) -> Dict[str, Tensor]: ...

class ImpactSimulator:
    """Simulates market impact of actions.
    
    Models how trading actions affect market prices
    and liquidity conditions.
    """
    def __init__(
        self,
        impact_model: _Module,
        liquidity_model: _Module,
        slippage_params: Dict[str, float]
    ) -> None: ...
    
    def simulate_impact(
        self,
        action: Tensor,
        market_state: Dict[str, Tensor]
    ) -> Tuple[Dict[str, Tensor], Dict[str, float]]: ...
    
    def estimate_slippage(
        self,
        trade_size: Tensor,
        liquidity_state: Dict[str, Tensor]
    ) -> Tuple[float, Dict[str, float]]: ...

class RiskAwareController:
    """Risk-aware control system.
    
    Implements control strategies with explicit
    risk consideration and constraints.
    """
    def __init__(
        self,
        mpc_optimizer: MPCOptimizer,
        risk_constraints: Dict[str, float],
        adaptation_rate: float = 0.1
    ) -> None: ...
    
    def compute_control_action(
        self,
        current_state: Dict[str, Tensor],
        risk_metrics: Dict[str, float]
    ) -> Tuple[Tensor, Dict[str, Any]]: ...
    
    def validate_action(
        self,
        action: Tensor,
        state: Dict[str, Tensor],
        constraints: Dict[str, float]
    ) -> Tuple[bool, Dict[str, float]]: ...

class TrajectoryOptimizer:
    """Optimizes trading trajectories.
    
    Plans optimal sequences of trading actions
    considering multiple objectives.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        horizon: int,
        objective_weights: Dict[str, float]
    ) -> None: ...
    
    def optimize_trajectory(
        self,
        initial_state: Tensor,
        target_state: Optional[Tensor] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Tuple[Tensor, Tensor, Dict[str, Any]]: ...
    
    def evaluate_trajectory_cost(
        self,
        states: Tensor,
        actions: Tensor,
        objectives: Dict[str, Callable]
    ) -> Dict[str, Tensor]: ... 