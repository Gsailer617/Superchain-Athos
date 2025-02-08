from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from torch import Tensor
from datetime import datetime
from ..nn import _Module

class FeedbackLoop:
    """Closed-loop feedback integration system.
    
    Manages the complete feedback loop from execution
    to strategy refinement and adaptation.
    """
    def __init__(
        self,
        strategy: _Module,
        feedback_collector: Any,
        optimizer: Any,
        update_config: Dict[str, Any]
    ) -> None: ...
    
    def process_feedback(
        self,
        execution_data: Dict[str, Any],
        market_context: Dict[str, Tensor]
    ) -> Dict[str, Any]: ...
    
    def update_strategy(
        self,
        feedback_metrics: Dict[str, float]
    ) -> Tuple[bool, Dict[str, Any]]: ...
    
    def validate_changes(
        self,
        proposed_updates: Dict[str, Any],
        validation_data: Dict[str, Tensor]
    ) -> bool: ...

class MetricAggregator:
    """Performance metric aggregation system.
    
    Aggregates and analyzes various performance metrics
    for strategy improvement.
    """
    def __init__(
        self,
        metric_definitions: Dict[str, Callable],
        aggregation_windows: List[str]
    ) -> None: ...
    
    def aggregate_metrics(
        self,
        raw_metrics: List[Dict[str, float]],
        window: str
    ) -> Dict[str, float]: ...
    
    def compute_trends(
        self,
        metric_history: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]: ...
    
    def detect_metric_shifts(
        self,
        recent_metrics: Dict[str, float],
        baseline: Dict[str, float]
    ) -> Dict[str, bool]: ...

class StrategyEvaluator:
    """Strategy evaluation system.
    
    Evaluates strategy performance and suggests
    improvements based on feedback.
    """
    def __init__(
        self,
        evaluation_criteria: Dict[str, Callable],
        performance_thresholds: Dict[str, float]
    ) -> None: ...
    
    def evaluate_strategy(
        self,
        performance_data: Dict[str, Any],
        market_conditions: Dict[str, Any]
    ) -> Dict[str, float]: ...
    
    def identify_weaknesses(
        self,
        evaluation_results: Dict[str, float]
    ) -> List[Dict[str, Any]]: ...
    
    def suggest_improvements(
        self,
        weaknesses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]: ...

class AdaptationManager:
    """Strategy adaptation management.
    
    Manages the process of adapting strategies based
    on feedback and performance.
    """
    def __init__(
        self,
        adaptation_rules: Dict[str, Callable],
        validation_criteria: Dict[str, Callable]
    ) -> None: ...
    
    def assess_adaptation_needs(
        self,
        performance_metrics: Dict[str, float],
        market_state: Dict[str, Any]
    ) -> Tuple[bool, List[str]]: ...
    
    def plan_adaptations(
        self,
        identified_needs: List[str],
        current_state: Dict[str, Any]
    ) -> Dict[str, Any]: ...
    
    def execute_adaptations(
        self,
        adaptation_plan: Dict[str, Any],
        strategy: _Module
    ) -> Tuple[bool, Dict[str, Any]]: ...

class ContinuousValidator:
    """Continuous validation system.
    
    Validates strategy changes and adaptations
    before deployment.
    """
    def __init__(
        self,
        validation_metrics: List[str],
        acceptance_criteria: Dict[str, Any]
    ) -> None: ...
    
    def validate_changes(
        self,
        current_state: Dict[str, Any],
        proposed_changes: Dict[str, Any],
        validation_data: Dict[str, Tensor]
    ) -> Tuple[bool, Dict[str, Any]]: ...
    
    def assess_risk(
        self,
        changes: Dict[str, Any],
        risk_metrics: Dict[str, float]
    ) -> Dict[str, float]: ...
    
    def verify_consistency(
        self,
        strategy_state: Dict[str, Any]
    ) -> Tuple[bool, List[str]]: ...

class PerformanceOptimizer:
    """Performance optimization system.
    
    Optimizes strategy performance based on
    feedback and execution data.
    """
    def __init__(
        self,
        optimization_objectives: Dict[str, Callable],
        constraints: Dict[str, Callable]
    ) -> None: ...
    
    def optimize_performance(
        self,
        execution_history: List[Dict[str, Any]],
        current_config: Dict[str, Any]
    ) -> Dict[str, Any]: ...
    
    def evaluate_improvement(
        self,
        baseline_metrics: Dict[str, float],
        new_metrics: Dict[str, float]
    ) -> Dict[str, float]: ...
    
    def generate_optimization_plan(
        self,
        performance_analysis: Dict[str, Any]
    ) -> Dict[str, Any]: ... 