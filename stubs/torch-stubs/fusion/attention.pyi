from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from torch import Tensor
from ..nn import _Module

class MultiHeadDeFiAttention(_Module):
    """Multi-head attention for DeFi data fusion.
    
    Implements specialized attention mechanisms for processing
    multiple streams of DeFi market data.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        attention_type: str = 'scaled_dot_product'
    ) -> None: ...
    
    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Optional[Tensor] = None,
        attention_weights: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]: ...
    
    def compute_attention_scores(
        self,
        query: Tensor,
        key: Tensor,
        scale_factor: Optional[float] = None
    ) -> Tensor: ...

class MarketDataTransformer(_Module):
    """Transformer architecture for market data processing.
    
    Processes temporal and cross-sectional market data
    using self-attention mechanisms.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu'
    ) -> None: ...
    
    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]: ...
    
    def encode_market_data(
        self,
        market_data: Dict[str, Tensor]
    ) -> Tensor: ...

class CrossPoolAttention(_Module):
    """Cross-pool attention mechanism.
    
    Captures relationships and dependencies between
    different liquidity pools and trading pairs.
    """
    def __init__(
        self,
        num_pools: int,
        feature_dim: int,
        num_heads: int = 4
    ) -> None: ...
    
    def forward(
        self,
        pool_features: Tensor,
        pool_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]: ...
    
    def compute_pool_correlations(
        self,
        pool_states: Dict[str, Tensor]
    ) -> Tensor: ...

class TemporalAttention(_Module):
    """Temporal attention for time series data.
    
    Processes temporal dependencies in market data
    with varying time scales.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        max_seq_length: int = 1000
    ) -> None: ...
    
    def forward(
        self,
        sequence: Tensor,
        timestamps: Optional[Tensor] = None,
        mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]: ...
    
    def compute_temporal_weights(
        self,
        time_diffs: Tensor
    ) -> Tensor: ...

class HierarchicalFeatureFusion(_Module):
    """Hierarchical feature fusion for market data.
    
    Combines features from multiple sources and scales
    using hierarchical attention.
    """
    def __init__(
        self,
        feature_dims: Dict[str, int],
        fusion_dim: int,
        num_levels: int = 3
    ) -> None: ...
    
    def forward(
        self,
        features: Dict[str, Tensor]
    ) -> Tuple[Tensor, Dict[str, Dict[str, Tensor]]]: ...
    
    def fuse_features(
        self,
        features: List[Tensor],
        level: int
    ) -> Tensor: ...

class AdaptiveFeatureAggregation(_Module):
    """Adaptive feature aggregation mechanism.
    
    Dynamically adjusts feature importance based on
    market conditions and trading context.
    """
    def __init__(
        self,
        feature_dim: int,
        context_dim: int,
        num_heads: int = 4
    ) -> None: ...
    
    def forward(
        self,
        features: Tensor,
        context: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]: ...
    
    def compute_feature_importance(
        self,
        features: Tensor,
        context: Tensor
    ) -> Tensor: ...

class MultiScaleAttention(_Module):
    """Multi-scale attention for different time horizons.
    
    Processes market data at multiple time scales
    simultaneously.
    """
    def __init__(
        self,
        input_dim: int,
        scale_factors: List[int],
        num_heads: int = 4
    ) -> None: ...
    
    def forward(
        self,
        input_sequence: Tensor,
        scale_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]: ...
    
    def process_scale(
        self,
        sequence: Tensor,
        scale: int
    ) -> Tensor: ... 