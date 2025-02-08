from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Generic
from torch import Tensor
from ..nn import _Module

State = TypeVar('State')
Action = TypeVar('Action')

class Trajectory(Generic[State, Action]):
    """Represents a trajectory of states and actions.
    
    Attributes:
        states: List of states in the trajectory
        actions: List of actions taken
        rewards: List of rewards received
        total_reward: Sum of all rewards in trajectory
    """
    states: List[State]
    actions: List[Action]
    rewards: List[float]
    total_reward: float
    
    def __init__(
        self,
        states: List[State],
        actions: List[Action],
        rewards: List[float]
    ) -> None: ...
    
    def append(
        self,
        state: State,
        action: Action,
        reward: float
    ) -> None: ...
    
    def to_tensor(self) -> Tuple[Tensor, Tensor, Tensor]: ...

class CEMOptimizer:
    """Cross-Entropy Method optimizer for MPC.
    
    Implements the Cross-Entropy Method for optimizing action sequences
    in Model Predictive Control.
    """
    def __init__(
        self,
        num_iterations: int = 5,
        population_size: int = 400,
        elite_fraction: float = 0.1,
        noise_factor: float = 0.5
    ) -> None: ...
    
    def optimize(
        self,
        objective_fn: Callable[[Tensor], Tensor],
        initial_mean: Tensor,
        initial_std: Tensor,
        bounds: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, float]: ...

class MPCOptimizer:
    """Optimizer for Model Predictive Control problems.
    
    Handles the optimization of action sequences using various
    methods like CEM, MPPI, or gradient-based optimization.
    """
    def __init__(
        self,
        method: str = 'CEM',
        horizon: int = 10,
        num_simulations: int = 1000,
        optimization_params: Dict[str, Any] = None
    ) -> None: ...
    
    def optimize_actions(
        self,
        initial_state: State,
        dynamics_model: _Module,
        cost_fn: Callable[[State, Action], float],
        action_bounds: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[List[Action], float]: ...
    
    def update_parameters(
        self,
        optimization_results: Dict[str, Any]
    ) -> None: ...

class MPPIOptimizer:
    """Model Predictive Path Integral optimizer.
    
    Implements the MPPI algorithm for optimizing action sequences
    in continuous action spaces.
    """
    def __init__(
        self,
        num_samples: int = 1000,
        temperature: float = 1.0,
        noise_sigma: float = 1.0
    ) -> None: ...
    
    def optimize(
        self,
        cost_fn: Callable[[Tensor], Tensor],
        initial_actions: Tensor,
        action_bounds: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, float]: ...

class TrajectoryOptimizer:
    """Optimizes trajectories for multi-step planning.
    
    Handles optimization of entire trajectories considering
    both immediate and future costs.
    """
    def __init__(
        self,
        dynamics_model: _Module,
        cost_model: _Module,
        horizon: int = 10,
        optimization_method: str = 'gradient'
    ) -> None: ...
    
    def optimize_trajectory(
        self,
        initial_state: State,
        goal_state: Optional[State] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Tuple[Trajectory[State, Action], Dict[str, Any]]: ...
    
    def evaluate_trajectory(
        self,
        trajectory: Trajectory[State, Action]
    ) -> Dict[str, float]: ...
    
    def check_constraints(
        self,
        trajectory: Trajectory[State, Action],
        constraints: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]: ... 