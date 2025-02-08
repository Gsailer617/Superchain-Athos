from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from torch import Tensor, distributions
from ..nn import _Module
from torch.distributions import Normal, Distribution

class BayesianLayer(_Module):
    """Bayesian neural network layer with weight uncertainty.
    
    Implements variational inference for neural network weights,
    allowing uncertainty estimation in predictions.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        posterior_std_init: float = 0.1
    ) -> None: ...
    
    def forward(
        self,
        x: Tensor,
        num_samples: int = 1
    ) -> Tuple[Tensor, Distribution]: ...
    
    def kl_divergence(self) -> Tensor: ...
    
    def sample_weights(self) -> Tuple[Tensor, Tensor]: ...

class BayesianEnsemble:
    """Ensemble of Bayesian neural networks.
    
    Combines multiple Bayesian networks for more robust
    uncertainty estimation and prediction.
    """
    def __init__(
        self,
        model_constructor: Callable[[], _Module],
        num_models: int = 5,
        diversity_weight: float = 0.1
    ) -> None: ...
    
    def forward_ensemble(
        self,
        x: Tensor,
        num_samples: int = 10
    ) -> Tuple[Tensor, Tensor, Tensor]: ...  # mean, epistemic, aleatoric
    
    def compute_uncertainty(
        self,
        predictions: List[Tensor]
    ) -> Dict[str, Tensor]: ...

class MCDropout(_Module):
    """Monte Carlo Dropout for uncertainty estimation.
    
    Implements Bayesian approximation through dropout
    at inference time.
    """
    def __init__(
        self,
        base_model: _Module,
        dropout_rate: float = 0.1,
        num_samples: int = 20
    ) -> None: ...
    
    def forward_with_uncertainty(
        self,
        x: Tensor
    ) -> Tuple[Tensor, Tensor]: ...  # prediction, uncertainty
    
    def sample_predictions(
        self,
        x: Tensor,
        num_samples: Optional[int] = None
    ) -> List[Tensor]: ...

class UncertaintyCalibration:
    """Calibrates model uncertainty estimates.
    
    Ensures uncertainty estimates are well-calibrated
    with observed errors.
    """
    def __init__(
        self,
        num_bins: int = 10,
        recalibration_method: str = 'isotonic'
    ) -> None: ...
    
    def calibrate(
        self,
        predictions: Tensor,
        uncertainties: Tensor,
        targets: Tensor
    ) -> Tuple[Tensor, Tensor]: ...
    
    def evaluate_calibration(
        self,
        predictions: Tensor,
        uncertainties: Tensor,
        targets: Tensor
    ) -> Dict[str, float]: ...

class RiskAwareInference:
    """Risk-aware inference using uncertainty estimates.
    
    Incorporates uncertainty in decision making process
    for risk-adjusted predictions.
    """
    def __init__(
        self,
        base_model: _Module,
        risk_tolerance: float = 0.95,
        uncertainty_threshold: float = 0.1
    ) -> None: ...
    
    def predict_with_risk(
        self,
        x: Tensor,
        risk_level: Optional[float] = None
    ) -> Tuple[Tensor, Dict[str, float]]: ...
    
    def compute_value_at_risk(
        self,
        predictions: Tensor,
        uncertainties: Tensor,
        confidence_level: float = 0.95
    ) -> Tensor: ...

class ProbabilisticEnsemble:
    """Probabilistic ensemble for uncertainty estimation.
    
    Combines multiple probabilistic models for robust
    uncertainty quantification.
    """
    def __init__(
        self,
        models: List[_Module],
        aggregation_method: str = 'mixture',
        temperature: float = 1.0
    ) -> None: ...
    
    def forward(
        self,
        x: Tensor
    ) -> Tuple[Tensor, Distribution]: ...
    
    def compute_ensemble_stats(
        self,
        predictions: List[Distribution]
    ) -> Dict[str, Tensor]: ...
    
    def sample_predictions(
        self,
        x: Tensor,
        num_samples: int = 100
    ) -> Tensor: ... 