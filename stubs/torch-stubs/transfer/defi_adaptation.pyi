from typing import Any, Dict, List, Optional, Tuple, Union, Callable, TypeVar
from torch import Tensor
from ..nn import _Module
from .adaptation import DomainAdapter, FeatureAlignment

class MarketRegimeAdapter(DomainAdapter):
    """Adapts models across different market regimes.
    
    Handles adaptation between different market conditions
    like high/low volatility, bull/bear markets, etc.
    """
    def __init__(
        self,
        source_model: _Module,
        regime_classifier: _Module,
        adaptation_config: Dict[str, Any]
    ) -> None: ...
    
    def detect_regime(
        self,
        market_data: Dict[str, Tensor]
    ) -> Tuple[str, Dict[str, float]]: ...
    
    def adapt_to_regime(
        self,
        target_regime: str,
        market_data: Dict[str, Tensor]
    ) -> Tuple[_Module, Dict[str, float]]: ...
    
    def compute_regime_similarity(
        self,
        source_regime: str,
        target_regime: str
    ) -> float: ...

class LiquidityPatternAdapter:
    """Adapts to different liquidity patterns.
    
    Handles adaptation between different liquidity conditions
    and pool behaviors.
    """
    def __init__(
        self,
        base_model: _Module,
        liquidity_encoder: _Module,
        pool_types: List[str]
    ) -> None: ...
    
    def encode_liquidity_pattern(
        self,
        pool_data: Dict[str, Tensor]
    ) -> Tensor: ...
    
    def adapt_to_pool(
        self,
        source_pool: str,
        target_pool: str,
        pool_data: Dict[str, Tensor]
    ) -> _Module: ...
    
    def compute_pool_similarity(
        self,
        pool1_data: Dict[str, Tensor],
        pool2_data: Dict[str, Tensor]
    ) -> float: ...

class CrossChainAdapter:
    """Adapts models across different blockchain networks.
    
    Handles adaptation between different chains with varying
    characteristics and protocols.
    """
    def __init__(
        self,
        base_model: _Module,
        chain_encoder: _Module,
        supported_chains: List[str]
    ) -> None: ...
    
    def encode_chain_features(
        self,
        chain_data: Dict[str, Any]
    ) -> Tensor: ...
    
    def adapt_to_chain(
        self,
        source_chain: str,
        target_chain: str,
        chain_data: Dict[str, Any]
    ) -> _Module: ...
    
    def compute_chain_compatibility(
        self,
        source_chain: str,
        target_chain: str
    ) -> float: ...

class VolatilityRegimeAdapter:
    """Adapts models across volatility regimes.
    
    Specializes in adaptation between periods of different
    volatility levels and patterns.
    """
    def __init__(
        self,
        base_model: _Module,
        volatility_estimator: _Module,
        regime_thresholds: Dict[str, float]
    ) -> None: ...
    
    def estimate_volatility_regime(
        self,
        price_data: Tensor,
        window_size: int = 100
    ) -> Tuple[str, Dict[str, float]]: ...
    
    def adapt_to_volatility(
        self,
        target_volatility: float,
        market_data: Dict[str, Tensor]
    ) -> _Module: ...

class ProtocolSpecificAdapter:
    """Adapts models for specific DeFi protocols.
    
    Handles adaptation between different protocol types
    and their unique characteristics.
    """
    def __init__(
        self,
        base_model: _Module,
        protocol_encoder: _Module,
        protocol_configs: Dict[str, Dict[str, Any]]
    ) -> None: ...
    
    def encode_protocol_features(
        self,
        protocol_data: Dict[str, Any]
    ) -> Tensor: ...
    
    def adapt_to_protocol(
        self,
        source_protocol: str,
        target_protocol: str,
        protocol_data: Dict[str, Any]
    ) -> _Module: ...
    
    def compute_protocol_similarity(
        self,
        protocol1: str,
        protocol2: str
    ) -> float: ...

class MultiModalAdapter:
    """Adapts models using multiple data modalities.
    
    Combines different types of market data for more
    robust domain adaptation.
    """
    def __init__(
        self,
        base_model: _Module,
        modality_encoders: Dict[str, _Module],
        fusion_method: str = 'attention'
    ) -> None: ...
    
    def encode_modalities(
        self,
        market_data: Dict[str, Dict[str, Tensor]]
    ) -> Dict[str, Tensor]: ...
    
    def fuse_modalities(
        self,
        encoded_modalities: Dict[str, Tensor]
    ) -> Tensor: ...
    
    def adapt_with_modalities(
        self,
        source_data: Dict[str, Dict[str, Tensor]],
        target_data: Dict[str, Dict[str, Tensor]]
    ) -> _Module: ... 