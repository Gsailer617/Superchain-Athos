from typing import Any, Dict, List, Optional, Tuple, Union, Callable, TypeVar
from torch import Tensor
from ..nn import _Module

T = TypeVar('T')

class MultiObjectiveOptimizer:
    """Multi-objective optimization system.
    
    Optimizes trading strategies considering multiple
    competing objectives and constraints.
    """
    def __init__(
        self,
        objectives: Dict[str, Callable[[Dict[str, Tensor]], float]],
        constraints: Dict[str, Callable[[Dict[str, Tensor]], bool]],
        weights: Optional[Dict[str, float]] = None
    ) -> None: ...
    
    def optimize(
        self,
        initial_state: Dict[str, Tensor],
        action_space: Dict[str, Tensor],
        method: str = 'pareto'
    ) -> Tuple[Dict[str, Tensor], Dict[str, float]]: ...
    
    def compute_pareto_front(
        self,
        solutions: List[Dict[str, Tensor]]
    ) -> List[Dict[str, Tensor]]: ...
    
    def rank_solutions(
        self,
        solutions: List[Dict[str, Tensor]],
        preferences: Optional[Dict[str, float]] = None
    ) -> List[Tuple[Dict[str, Tensor], float]]: ...

class AdaptiveOptimizer:
    """Adaptive optimization strategy.
    
    Dynamically adjusts optimization parameters based
    on market conditions and performance.
    """
    def __init__(
        self,
        base_optimizer: Any,
        adaptation_rules: Dict[str, Callable],
        learning_params: Dict[str, float]
    ) -> None: ...
    
    def optimize_step(
        self,
        current_state: Dict[str, Tensor],
        performance_history: List[Dict[str, float]]
    ) -> Tuple[Dict[str, Any], Dict[str, float]]: ...
    
    def adapt_parameters(
        self,
        market_conditions: Dict[str, Any],
        performance_metrics: Dict[str, float]
    ) -> Dict[str, float]: ...
    
    def reset_adaptation(self) -> None: ...

class ConstrainedOptimizer:
    """Constrained optimization system.
    
    Handles complex constraints in DeFi optimization
    problems with dynamic bounds.
    """
    def __init__(
        self,
        objective_fn: Callable[[Dict[str, Tensor]], float],
        constraint_fns: List[Callable[[Dict[str, Tensor]], bool]],
        barrier_params: Dict[str, float]
    ) -> None: ...
    
    def optimize(
        self,
        initial_point: Dict[str, Tensor],
        bounds: Dict[str, Tuple[float, float]]
    ) -> Tuple[Dict[str, Tensor], Dict[str, float]]: ...
    
    def check_constraints(
        self,
        point: Dict[str, Tensor]
    ) -> Tuple[bool, List[str]]: ...
    
    def compute_barrier_gradient(
        self,
        point: Dict[str, Tensor]
    ) -> Dict[str, Tensor]: ...

class RobustOptimizer:
    """Robust optimization under uncertainty.
    
    Implements robust optimization techniques for
    handling uncertainty in DeFi markets.
    """
    def __init__(
        self,
        nominal_optimizer: Any,
        uncertainty_sets: Dict[str, Tensor],
        robustness_level: float = 0.95
    ) -> None: ...
    
    def optimize_robust(
        self,
        objective_fn: Callable[[Dict[str, Tensor]], float],
        constraints: List[Callable],
        initial_point: Dict[str, Tensor]
    ) -> Tuple[Dict[str, Tensor], Dict[str, float]]: ...
    
    def evaluate_worst_case(
        self,
        solution: Dict[str, Tensor],
        uncertainty_samples: int = 1000
    ) -> Tuple[float, Dict[str, Tensor]]: ...
    
    def compute_uncertainty_bounds(
        self,
        point: Dict[str, Tensor],
        confidence_level: float
    ) -> Dict[str, Tuple[float, float]]: ...

class CompositeOptimizer:
    """Composite optimization strategy.
    
    Combines multiple optimization strategies for
    different market conditions and objectives.
    """
    def __init__(
        self,
        optimizers: Dict[str, Any],
        selector_model: _Module,
        combination_weights: Optional[Dict[str, float]] = None
    ) -> None: ...
    
    def select_strategy(
        self,
        market_state: Dict[str, Tensor],
        performance_history: Optional[Dict[str, List[float]]] = None
    ) -> Tuple[str, float]: ...
    
    def optimize_composite(
        self,
        problem: Dict[str, Any],
        market_conditions: Dict[str, Tensor]
    ) -> Tuple[Dict[str, Tensor], Dict[str, Any]]: ...
    
    def update_weights(
        self,
        performance_metrics: Dict[str, Dict[str, float]]
    ) -> None: ... 