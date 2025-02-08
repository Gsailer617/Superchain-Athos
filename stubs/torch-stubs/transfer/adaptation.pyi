from typing import Any, Dict, List, Optional, Tuple, Union, Callable, TypeVar, Generic, Protocol
from torch import Tensor
from ..nn import _Module
from ..optim import Optimizer
from datetime import datetime

Model = TypeVar('Model', bound=_Module)
Domain = TypeVar('Domain')

class AdaptationMetrics(Protocol):
    """Protocol for adaptation metric computation."""
    def compute(self, source_data: Domain, target_data: Domain) -> Dict[str, float]: ...
    def update(self, new_data: Domain) -> None: ...

class DomainAdapter(Generic[Model, Domain]):
    """Adapts models across different market domains.
    
    Implements domain adaptation techniques for transferring
    knowledge between different market conditions or chains.
    """
    source_model: Model
    target_model: Model
    discriminator: Optional[_Module]
    metrics: List[AdaptationMetrics]
    
    def __init__(
        self,
        source_model: Model,
        adaptation_method: str = 'gradual',
        discriminator_config: Optional[Dict[str, Any]] = None,
        metrics: Optional[List[AdaptationMetrics]] = None
    ) -> None: ...
    
    def adapt(
        self,
        source_data: Domain,
        target_data: Domain,
        num_epochs: int = 10,
        validation_freq: int = 1,
        early_stopping: bool = True
    ) -> Tuple[Model, Dict[str, float]]: ...
    
    def validate_adaptation(
        self,
        validation_data: Domain,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]: ...
    
    def compute_domain_distance(
        self,
        source_features: Tensor,
        target_features: Tensor,
        distance_type: str = 'wasserstein'
    ) -> float: ...

class FeatureAlignment:
    """Aligns feature distributions across domains.
    
    Implements various feature alignment techniques for
    effective transfer learning.
    """
    def __init__(
        self,
        alignment_method: str = 'mmd',
        kernel_type: str = 'gaussian',
        num_kernels: int = 5,
        adversarial_training: bool = False
    ) -> None: ...
    
    def align_features(
        self,
        source_features: Tensor,
        target_features: Tensor,
        weights: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Dict[str, float]]: ...
    
    def compute_distance_matrix(
        self,
        source_features: Tensor,
        target_features: Tensor,
        metric: str = 'euclidean'
    ) -> Tensor: ...
    
    def optimize_kernel_parameters(
        self,
        source_features: Tensor,
        target_features: Tensor
    ) -> Dict[str, float]: ...

class AdversarialAdapter:
    """Implements adversarial domain adaptation.
    
    Uses adversarial training to align source and target domains
    while preserving task-relevant features.
    """
    def __init__(
        self,
        feature_extractor: Model,
        domain_discriminator: Model,
        task_classifier: Model,
        lambda_schedule: Callable[[int], float]
    ) -> None: ...
    
    def train_step(
        self,
        source_batch: Tuple[Tensor, Tensor],
        target_batch: Tensor,
        step: int
    ) -> Dict[str, float]: ...
    
    def compute_gradient_penalty(
        self,
        source_features: Tensor,
        target_features: Tensor
    ) -> Tensor: ...

class DynamicWeightAdjuster:
    """Adjusts domain adaptation weights dynamically.
    
    Implements various strategies for dynamic weight adjustment
    based on domain similarity and task performance.
    """
    def __init__(
        self,
        initial_weights: Dict[str, float],
        adaptation_rate: float = 0.1,
        min_weight: float = 0.0,
        max_weight: float = 1.0
    ) -> None: ...
    
    def update_weights(
        self,
        performance_metrics: Dict[str, float],
        domain_distances: Dict[str, float]
    ) -> Dict[str, float]: ...
    
    def compute_importance_weights(
        self,
        source_features: Tensor,
        target_features: Tensor
    ) -> Tensor: ...

class AdaptationMonitor:
    """Monitors and logs domain adaptation progress.
    
    Tracks various metrics and provides early stopping
    and adaptation scheduling capabilities.
    """
    def __init__(
        self,
        metrics: List[str],
        patience: int = 5,
        min_delta: float = 1e-4,
        logging_dir: Optional[str] = None
    ) -> None: ...
    
    def update(
        self,
        current_metrics: Dict[str, float],
        step: int,
        model_state: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Dict[str, Any]]: ...
    
    def should_stop(self) -> bool: ...
    
    def get_best_model(self) -> Dict[str, Any]: ...
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = ''
    ) -> None: ...

class GradualFinetuning:
    """Implements gradual fine-tuning strategies.
    
    Manages progressive adaptation of model layers
    for smooth knowledge transfer.
    """
    def __init__(
        self,
        layer_groups: List[str],
        lr_schedule: Dict[str, float],
        freeze_batch_norm: bool = True
    ) -> None: ...
    
    def create_layer_schedule(
        self,
        model: Model
    ) -> List[Dict[str, Any]]: ...
    
    def adjust_learning_rates(
        self,
        optimizer: Optimizer,
        current_phase: int
    ) -> None: ...

class DomainClassifier:
    """Classifies and quantifies domain differences.
    
    Used for adversarial domain adaptation and
    measuring domain shifts.
    """
    def __init__(
        self,
        feature_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.1
    ) -> None: ...
    
    def forward(
        self,
        features: Tensor
    ) -> Tensor: ...
    
    def compute_domain_loss(
        self,
        source_features: Tensor,
        target_features: Tensor
    ) -> Tensor: ...

class KnowledgeDistillation:
    """Implements knowledge distillation techniques.
    
    Transfers knowledge from larger models to smaller ones
    or between different architectures.
    """
    def __init__(
        self,
        teacher_model: Model,
        student_model: Model,
        temperature: float = 2.0,
        alpha: float = 0.5
    ) -> None: ...
    
    def compute_distillation_loss(
        self,
        teacher_logits: Tensor,
        student_logits: Tensor,
        labels: Optional[Tensor] = None
    ) -> Tensor: ...
    
    def train_step(
        self,
        input_data: Tensor,
        labels: Optional[Tensor] = None
    ) -> Dict[str, float]: ... 