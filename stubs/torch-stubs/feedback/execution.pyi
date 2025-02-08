from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from torch import Tensor
from datetime import datetime
from ..nn import _Module

class ExecutionFeedback:
    """Execution feedback collection and analysis.
    
    Collects and processes execution outcomes for
    strategy refinement and parameter adjustment.
    """
    def __init__(
        self,
        metrics_config: Dict[str, Any],
        update_frequency: str = '1h',
        history_window: str = '7d'
    ) -> None: ...
    
    def record_execution(
        self,
        trade_intent: Dict[str, Any],
        execution_result: Dict[str, Any],
        market_state: Dict[str, Tensor]
    ) -> None: ...
    
    def analyze_performance(
        self,
        time_window: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]: ...
    
    def compute_slippage_metrics(
        self,
        executions: List[Dict[str, Any]]
    ) -> Dict[str, float]: ...

class StrategyRefiner:
    """Dynamic strategy refinement system.
    
    Refines trading strategies based on execution
    feedback and performance metrics.
    """
    def __init__(
        self,
        base_strategy: _Module,
        refinement_config: Dict[str, Any],
        adaptation_rate: float = 0.1
    ) -> None: ...
    
    def update_strategy(
        self,
        feedback_metrics: Dict[str, float],
        market_conditions: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]: ...
    
    def adjust_parameters(
        self,
        performance_data: Dict[str, float]
    ) -> Dict[str, float]: ...
    
    def validate_refinement(
        self,
        proposed_changes: Dict[str, float],
        validation_data: Dict[str, Tensor]
    ) -> bool: ...

class PerformanceMonitor:
    """Real-time performance monitoring.
    
    Monitors and analyzes strategy performance for
    continuous improvement.
    """
    def __init__(
        self,
        monitoring_config: Dict[str, Any],
        alert_thresholds: Dict[str, float]
    ) -> None: ...
    
    def track_metrics(
        self,
        execution_data: Dict[str, Any],
        timestamp: datetime
    ) -> Dict[str, float]: ...
    
    def detect_anomalies(
        self,
        recent_metrics: Dict[str, float]
    ) -> List[Dict[str, Any]]: ...
    
    def generate_report(
        self,
        time_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]: ...

class FeedbackOptimizer:
    """Feedback-based parameter optimization.
    
    Optimizes strategy parameters based on execution
    feedback and performance metrics.
    """
    def __init__(
        self,
        parameter_space: Dict[str, Tuple[float, float]],
        objective_weights: Dict[str, float]
    ) -> None: ...
    
    def optimize_parameters(
        self,
        feedback_history: List[Dict[str, Any]],
        current_params: Dict[str, float]
    ) -> Dict[str, float]: ...
    
    def evaluate_parameter_set(
        self,
        parameters: Dict[str, float],
        validation_data: Dict[str, Tensor]
    ) -> float: ...
    
    def suggest_adjustments(
        self,
        performance_metrics: Dict[str, float]
    ) -> Dict[str, Tuple[float, str]]: ...

class AdaptiveThresholds:
    """Dynamic threshold adjustment system.
    
    Adjusts strategy thresholds based on execution
    outcomes and market conditions.
    """
    def __init__(
        self,
        initial_thresholds: Dict[str, float],
        adjustment_rules: Dict[str, Callable]
    ) -> None: ...
    
    def update_thresholds(
        self,
        performance_data: Dict[str, float],
        market_metrics: Dict[str, Any]
    ) -> Dict[str, float]: ...
    
    def validate_thresholds(
        self,
        proposed_thresholds: Dict[str, float],
        risk_metrics: Dict[str, float]
    ) -> Tuple[bool, Dict[str, str]]: ...
    
    def compute_adaptive_bounds(
        self,
        metric_history: Dict[str, List[float]]
    ) -> Dict[str, Tuple[float, float]]: ...

class RealTimeTrainer:
    """Real-time model training system.
    
    Continuously updates model parameters using
    execution feedback and market data.
    """
    def __init__(
        self,
        model: _Module,
        training_config: Dict[str, Any],
        validation_freq: str = '1h'
    ) -> None: ...
    
    def update_model(
        self,
        feedback_batch: List[Dict[str, Any]],
        market_context: Dict[str, Tensor]
    ) -> Dict[str, float]: ...
    
    def validate_update(
        self,
        model_update: Dict[str, Tensor],
        validation_data: Dict[str, Tensor]
    ) -> Tuple[bool, Dict[str, float]]: ...
    
    def compute_training_metrics(
        self,
        training_history: List[Dict[str, float]]
    ) -> Dict[str, Any]: ... 