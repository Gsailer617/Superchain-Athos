from typing import Any, Optional, Union, List, Dict, Tuple, Protocol, TypeVar, Generic, Callable
from torch import Tensor
from .nn import _Module

State = TypeVar('State')
Action = TypeVar('Action')
Reward = TypeVar('Reward', bound=float)

class Environment(Protocol[State, Action, Reward]):
    """Protocol defining the interface for a simulation environment.
    
    This protocol ensures that any environment can be used with the MPC controller
    as long as it implements these methods.
    """
    def reset(self) -> State: ...
    def step(self, action: Action) -> Tuple[State, Reward, bool, Dict[str, Any]]: ...
    def render(self) -> None: ...
    def close(self) -> None: ...

class MarketSimulator(Environment[Tensor, Tensor, float]):
    """Simulates market behavior for predictive analysis.
    
    Attributes:
        state_dim: Dimension of the market state space
        action_dim: Dimension of the action space
        history_length: Number of past states to consider
    """
    state_dim: int
    action_dim: int
    history_length: int
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        history_length: int = 10,
        volatility: float = 0.1,
        transaction_cost: float = 0.001
    ) -> None: ...
    
    def reset(self) -> Tensor: ...
    def step(self, action: Tensor) -> Tuple[Tensor, float, bool, Dict[str, Any]]: ...
    def render(self) -> None: ...
    def close(self) -> None: ...

class MPCController(Generic[State, Action]):
    """Model Predictive Control for optimizing trading decisions.
    
    Attributes:
        horizon: Number of future steps to consider
        num_simulations: Number of Monte Carlo simulations per decision
        model: Dynamics model for state prediction
    """
    horizon: int
    num_simulations: int
    model: _Module
    
    def __init__(
        self,
        model: _Module,
        horizon: int = 10,
        num_simulations: int = 100,
        optimization_method: str = 'CEM'
    ) -> None: ...
    
    def predict_trajectory(
        self,
        initial_state: State,
        actions: List[Action]
    ) -> Tuple[List[State], float]: ...
    
    def optimize_actions(
        self,
        current_state: State,
        objective_fn: Callable[[List[State], List[Action]], float]
    ) -> List[Action]: ...
    
    def update_model(self, states: List[State], actions: List[Action], next_states: List[State]) -> None: ...

class BacktestEngine:
    """Continuous backtesting engine for strategy validation.
    
    Attributes:
        data_pipeline: Pipeline for historical and synthetic data
        risk_metrics: Collection of risk assessment metrics
        performance_metrics: Collection of performance metrics
    """
    def __init__(
        self,
        data_pipeline: Any,
        risk_metrics: List[str],
        performance_metrics: List[str],
        simulation_params: Dict[str, Any]
    ) -> None: ...
    
    def run_backtest(
        self,
        strategy: Any,
        start_time: str,
        end_time: str,
        initial_capital: float
    ) -> Dict[str, Any]: ...
    
    def generate_synthetic_scenarios(
        self,
        base_scenario: Any,
        num_scenarios: int,
        perturbation_params: Dict[str, Any]
    ) -> List[Any]: ...
    
    def analyze_results(
        self,
        backtest_results: Dict[str, Any],
        confidence_level: float = 0.95
    ) -> Dict[str, Any]: ...

class ScenarioGenerator:
    """Generates synthetic market scenarios for testing.
    
    Attributes:
        scenario_types: Types of scenarios to generate
        market_params: Market-specific parameters
    """
    def __init__(
        self,
        scenario_types: List[str],
        market_params: Dict[str, Any],
        random_seed: Optional[int] = None
    ) -> None: ...
    
    def generate_normal_scenario(
        self,
        duration: int,
        volatility: float
    ) -> Dict[str, Tensor]: ...
    
    def generate_stress_scenario(
        self,
        duration: int,
        stress_params: Dict[str, Any]
    ) -> Dict[str, Tensor]: ...
    
    def generate_flash_crash_scenario(
        self,
        duration: int,
        crash_params: Dict[str, Any]
    ) -> Dict[str, Tensor]: ...
    
    def generate_manipulation_scenario(
        self,
        duration: int,
        manipulation_params: Dict[str, Any]
    ) -> Dict[str, Tensor]: ... 