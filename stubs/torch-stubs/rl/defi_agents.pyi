from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from torch import Tensor
from .agents import TD3Agent, RLAgent, Experience
from ..nn import _Module

class DeFiTD3Agent(TD3Agent):
    """TD3 agent specialized for DeFi trading.
    
    Extends TD3 with DeFi-specific features like risk-aware
    action selection and multi-token portfolio management.
    """
    def __init__(
        self,
        actor: _Module,
        critic1: _Module,
        critic2: _Module,
        risk_model: _Module,
        max_position_size: Dict[str, float],
        slippage_model: Optional[_Module] = None,
        risk_weight: float = 0.5
    ) -> None: ...
    
    def select_action_with_risk(
        self,
        state: Tensor,
        market_state: Dict[str, Any],
        risk_threshold: float
    ) -> Tuple[Tensor, Dict[str, float]]: ...
    
    def estimate_execution_cost(
        self,
        action: Tensor,
        market_state: Dict[str, Any]
    ) -> Dict[str, float]: ...
    
    def compute_risk_adjusted_reward(
        self,
        reward: float,
        risk_metrics: Dict[str, float]
    ) -> float: ...
    
    def update_with_market_data(
        self,
        market_data: Dict[str, Any]
    ) -> None: ...

class RiskAwareReplayBuffer:
    """Replay buffer with risk-aware sampling.
    
    Implements prioritized experience replay with
    additional risk-based sampling strategies.
    """
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        risk_factors: List[str]
    ) -> None: ...
    
    def add_with_risk(
        self,
        experience: Experience,
        risk_metrics: Dict[str, float]
    ) -> None: ...
    
    def sample_by_risk_profile(
        self,
        batch_size: int,
        risk_weights: Dict[str, float]
    ) -> Tuple[Experience, Dict[str, Tensor]]: ...

class MarketAdaptiveNoise:
    """Adaptive noise generation for DeFi environments.
    
    Generates exploration noise based on market conditions
    and volatility.
    """
    def __init__(
        self,
        base_std: float = 0.1,
        volatility_scaling: bool = True,
        noise_decay: float = 0.995
    ) -> None: ...
    
    def generate_noise(
        self,
        action_dim: int,
        market_volatility: Dict[str, float]
    ) -> Tensor: ...
    
    def adapt_parameters(
        self,
        market_metrics: Dict[str, float]
    ) -> None: ...

class MultiPoolActionMapper:
    """Maps RL actions to multi-pool trading actions.
    
    Handles action space mapping for complex DeFi
    environments with multiple pools and routes.
    """
    def __init__(
        self,
        pools: List[str],
        tokens: List[str],
        max_route_length: int = 3
    ) -> None: ...
    
    def map_to_pool_actions(
        self,
        action: Tensor,
        pool_states: Dict[str, Any]
    ) -> Dict[str, Tensor]: ...
    
    def compute_route_probabilities(
        self,
        pool_states: Dict[str, Any],
        action_values: Tensor
    ) -> Dict[str, float]: ...

class LiquidityAwarePolicy:
    """Policy network that considers liquidity constraints.
    
    Implements a policy that explicitly accounts for
    liquidity conditions in action selection.
    """
    def __init__(
        self,
        base_policy: _Module,
        liquidity_model: _Module,
        min_liquidity_threshold: Dict[str, float]
    ) -> None: ...
    
    def forward(
        self,
        state: Tensor,
        liquidity_data: Dict[str, Tensor]
    ) -> Tuple[Tensor, Dict[str, float]]: ...
    
    def adjust_for_liquidity(
        self,
        action: Tensor,
        liquidity_metrics: Dict[str, float]
    ) -> Tensor: ...

class GasAwareTraining:
    """Training component with gas cost awareness.
    
    Modifies training process to consider gas costs
    and optimize for net profitability.
    """
    def __init__(
        self,
        base_agent: RLAgent,
        gas_model: _Module,
        min_profit_threshold: float
    ) -> None: ...
    
    def compute_gas_adjusted_reward(
        self,
        reward: float,
        gas_cost: float,
        market_price: float
    ) -> float: ...
    
    def estimate_gas_cost(
        self,
        action: Tensor,
        network_state: Dict[str, Any]
    ) -> float: ...
    
    def optimize_batch_with_gas(
        self,
        batch: Experience,
        gas_prices: Dict[str, int]
    ) -> Dict[str, float]: ... 