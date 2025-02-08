from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from torch import Tensor
from datetime import datetime
from ..nn import _Module

class DynamicRiskManager:
    """Dynamic risk management system.
    
    Adjusts risk thresholds and limits based on market
    conditions and uncertainty estimates.
    """
    def __init__(
        self,
        base_risk_threshold: float = 0.95,
        min_risk_threshold: float = 0.8,
        max_risk_threshold: float = 0.99,
        adaptation_rate: float = 0.1
    ) -> None: ...
    
    def update_risk_threshold(
        self,
        market_metrics: Dict[str, float],
        uncertainty_metrics: Dict[str, float]
    ) -> float: ...
    
    def compute_position_limits(
        self,
        uncertainty: Tensor,
        current_exposure: Dict[str, float]
    ) -> Dict[str, float]: ...
    
    def evaluate_risk_metrics(
        self,
        portfolio_state: Dict[str, Any],
        market_state: Dict[str, Any]
    ) -> Dict[str, float]: ...

class UncertaintyAwarePortfolio:
    """Portfolio management with uncertainty awareness.
    
    Incorporates uncertainty estimates in portfolio
    optimization and risk management.
    """
    def __init__(
        self,
        assets: List[str],
        uncertainty_model: _Module,
        risk_tolerance: float = 0.95
    ) -> None: ...
    
    def optimize_weights(
        self,
        returns: Tensor,
        uncertainties: Tensor,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Tuple[Tensor, Dict[str, float]]: ...
    
    def compute_risk_metrics(
        self,
        weights: Tensor,
        uncertainties: Tensor
    ) -> Dict[str, float]: ...
    
    def rebalance_portfolio(
        self,
        current_weights: Tensor,
        target_weights: Tensor,
        uncertainty_threshold: float
    ) -> Dict[str, Any]: ...

class MarketUncertaintyMonitor:
    """Monitors market uncertainty conditions.
    
    Tracks and analyzes various sources of market
    uncertainty for risk management.
    """
    def __init__(
        self,
        uncertainty_sources: List[str],
        window_sizes: Dict[str, int],
        update_frequency: str = '1m'
    ) -> None: ...
    
    def compute_market_uncertainty(
        self,
        market_data: Dict[str, Tensor]
    ) -> Dict[str, float]: ...
    
    def detect_regime_change(
        self,
        uncertainty_history: Dict[str, Tensor]
    ) -> Tuple[bool, Dict[str, float]]: ...
    
    def get_uncertainty_forecast(
        self,
        current_state: Dict[str, Tensor],
        horizon: int = 10
    ) -> Dict[str, Tensor]: ...

class AdaptiveRiskLimits:
    """Adaptive risk limits based on uncertainty.
    
    Dynamically adjusts risk limits based on market
    conditions and uncertainty levels.
    """
    def __init__(
        self,
        base_limits: Dict[str, float],
        adjustment_factors: Dict[str, float],
        min_limits: Dict[str, float],
        max_limits: Dict[str, float]
    ) -> None: ...
    
    def compute_adjusted_limits(
        self,
        uncertainty_metrics: Dict[str, float],
        market_conditions: Dict[str, Any]
    ) -> Dict[str, float]: ...
    
    def validate_limits(
        self,
        proposed_limits: Dict[str, float],
        current_exposure: Dict[str, float]
    ) -> Tuple[bool, Dict[str, str]]: ...

class RiskAdjustedOptimizer:
    """Optimizer with dynamic risk adjustment.
    
    Modifies optimization objectives based on
    uncertainty and risk metrics.
    """
    def __init__(
        self,
        base_optimizer: Any,
        risk_weight: float = 0.5,
        uncertainty_penalty: float = 0.1
    ) -> None: ...
    
    def compute_risk_adjusted_objective(
        self,
        base_objective: Tensor,
        uncertainty: Tensor,
        risk_metrics: Dict[str, float]
    ) -> Tensor: ...
    
    def optimize_with_uncertainty(
        self,
        objective_fn: Callable[[Tensor], Tensor],
        initial_params: Tensor,
        uncertainty_fn: Callable[[Tensor], Tensor]
    ) -> Tuple[Tensor, Dict[str, float]]: ...

class ConfidenceCalibration:
    """Calibrates confidence levels for decisions.
    
    Adjusts confidence thresholds based on historical
    performance and uncertainty estimates.
    """
    def __init__(
        self,
        base_confidence: float = 0.95,
        calibration_window: str = '7d',
        min_samples: int = 1000
    ) -> None: ...
    
    def update_calibration(
        self,
        predictions: Tensor,
        outcomes: Tensor,
        uncertainties: Tensor
    ) -> None: ...
    
    def get_calibrated_threshold(
        self,
        uncertainty_level: float,
        risk_tolerance: float
    ) -> float: ...
    
    def evaluate_calibration_metrics(
        self,
        window: Optional[str] = None
    ) -> Dict[str, float]: ... 