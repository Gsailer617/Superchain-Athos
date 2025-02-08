from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from torch import Tensor
from ..nn import _Module

class DeFiGraphConv(_Module):
    """Graph convolution layer for DeFi networks.
    
    Processes graph-structured DeFi data with specialized
    convolution operations for market relationships.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggregation: str = 'mean',
        add_self_loops: bool = True
    ) -> None: ...
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None
    ) -> Tensor: ...
    
    def message_passing(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None
    ) -> Tensor: ...

class LiquidityPoolGraph(_Module):
    """Graph neural network for liquidity pool analysis.
    
    Models relationships between liquidity pools,
    tokens, and trading activities.
    """
    def __init__(
        self,
        num_token_features: int,
        num_pool_features: int,
        hidden_channels: List[int],
        num_layers: int = 3
    ) -> None: ...
    
    def forward(
        self,
        token_features: Tensor,
        pool_features: Tensor,
        connections: Tensor
    ) -> Tuple[Tensor, Tensor]: ...
    
    def compute_pool_embeddings(
        self,
        pool_data: Dict[str, Tensor]
    ) -> Tensor: ...

class TokenRelationNetwork(_Module):
    """Network for modeling token relationships.
    
    Captures token-to-token relationships through
    trading patterns and pool interactions.
    """
    def __init__(
        self,
        num_tokens: int,
        feature_dim: int,
        edge_dim: int,
        hidden_dim: int = 64
    ) -> None: ...
    
    def forward(
        self,
        token_features: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]: ...
    
    def update_token_embeddings(
        self,
        token_states: Dict[str, Tensor]
    ) -> Tensor: ...

class MarketGraphAttention(_Module):
    """Graph attention for market structure analysis.
    
    Implements attention mechanisms over market graphs
    to capture complex dependencies.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        concat: bool = True,
        dropout: float = 0.1
    ) -> None: ...
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        return_attention: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]: ...
    
    def compute_attention_weights(
        self,
        query: Tensor,
        key: Tensor
    ) -> Tensor: ...

class CrossChainGraphNetwork(_Module):
    """Graph network for cross-chain analysis.
    
    Models relationships and interactions between
    different blockchain networks.
    """
    def __init__(
        self,
        num_chains: int,
        chain_feature_dim: int,
        bridge_feature_dim: int,
        hidden_dim: int = 128
    ) -> None: ...
    
    def forward(
        self,
        chain_features: Tensor,
        bridge_connections: Tensor,
        bridge_features: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]: ...
    
    def analyze_bridge_flows(
        self,
        bridge_data: Dict[str, Tensor]
    ) -> Dict[str, Tensor]: ...

class DynamicGraphConv(_Module):
    """Dynamic graph convolution for evolving markets.
    
    Handles dynamic graph structures that evolve
    with market conditions.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dynamic_edges: bool = True,
        edge_threshold: float = 0.5
    ) -> None: ...
    
    def forward(
        self,
        x: Tensor,
        edge_index: Optional[Tensor] = None,
        edge_weight: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]: ...
    
    def infer_edges(
        self,
        node_features: Tensor,
        threshold: Optional[float] = None
    ) -> Tensor: ...

class TemporalGraphNetwork(_Module):
    """Temporal graph network for market evolution.
    
    Processes temporal sequences of market graphs
    to capture evolutionary patterns.
    """
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int = 2
    ) -> None: ...
    
    def forward(
        self,
        node_sequences: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        timestamps: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]: ...
    
    def encode_temporal_pattern(
        self,
        sequence: Tensor,
        time_diffs: Optional[Tensor] = None
    ) -> Tensor: ... 