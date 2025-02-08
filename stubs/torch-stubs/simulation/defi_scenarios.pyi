from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from torch import Tensor
from datetime import datetime
from ..nn import _Module

class DeFiScenarioGenerator:
    """Advanced DeFi scenario generator.
    
    Generates realistic DeFi-specific market scenarios
    including MEV, flash loans, and protocol events.
    """
    def __init__(
        self,
        base_market_state: Dict[str, Tensor],
        protocol_configs: Dict[str, Any],
        chain_configs: Dict[str, Any]
    ) -> None: ...
    
    def generate_mev_scenario(
        self,
        mev_type: str,  # 'sandwich', 'frontrun', 'backrun'
        target_pools: List[str],
        impact_level: float
    ) -> Dict[str, Tensor]: ...
    
    def generate_flash_loan_scenario(
        self,
        loan_size: float,
        target_tokens: List[str],
        repayment_route: List[str]
    ) -> Dict[str, Tensor]: ...
    
    def generate_protocol_event(
        self,
        event_type: str,  # 'upgrade', 'governance', 'emergency'
        affected_protocols: List[str],
        severity: float
    ) -> Dict[str, Tensor]: ...

class LiquidityShockGenerator:
    """Liquidity shock scenario generator.
    
    Creates scenarios involving sudden liquidity changes
    and pool imbalances.
    """
    def __init__(
        self,
        pool_configs: Dict[str, Any],
        volatility_params: Dict[str, float]
    ) -> None: ...
    
    def generate_withdrawal_shock(
        self,
        pool_id: str,
        withdrawal_size: float,
        time_window: int
    ) -> Dict[str, Tensor]: ...
    
    def generate_cascade_effect(
        self,
        initial_pool: str,
        propagation_path: List[str],
        impact_decay: float
    ) -> Dict[str, Tensor]: ...
    
    def simulate_pool_migration(
        self,
        source_pool: str,
        target_pool: str,
        migration_rate: float
    ) -> Dict[str, Tensor]: ...

class CrossChainScenarios:
    """Cross-chain scenario generator.
    
    Generates scenarios involving multiple chains
    and bridge interactions.
    """
    def __init__(
        self,
        chain_states: Dict[str, Dict[str, Tensor]],
        bridge_configs: Dict[str, Any]
    ) -> None: ...
    
    def generate_bridge_event(
        self,
        event_type: str,  # 'delay', 'congestion', 'failure'
        affected_bridges: List[str],
        duration: int
    ) -> Dict[str, Dict[str, Tensor]]: ...
    
    def simulate_cross_chain_arbitrage(
        self,
        token_pair: Tuple[str, str],
        price_differential: float,
        bridge_paths: List[List[str]]
    ) -> Dict[str, Dict[str, Tensor]]: ...

class MarketManipulationScenarios:
    """Market manipulation scenario generator.
    
    Creates scenarios involving various forms of
    market manipulation and attacks.
    """
    def __init__(
        self,
        market_state: Dict[str, Tensor],
        attack_params: Dict[str, Any]
    ) -> None: ...
    
    def generate_price_manipulation(
        self,
        target_token: str,
        manipulation_size: float,
        duration: int,
        method: str = 'momentum'
    ) -> Dict[str, Tensor]: ...
    
    def generate_oracle_attack(
        self,
        target_oracle: str,
        price_deviation: float,
        attack_path: List[str]
    ) -> Dict[str, Tensor]: ...
    
    def simulate_wash_trading(
        self,
        target_pair: str,
        volume_multiplier: float,
        pattern: str = 'cyclic'
    ) -> Dict[str, Tensor]: ...

class ProtocolRiskScenarios:
    """Protocol risk scenario generator.
    
    Generates scenarios related to protocol-specific
    risks and events.
    """
    def __init__(
        self,
        protocol_states: Dict[str, Dict[str, Tensor]],
        risk_params: Dict[str, Any]
    ) -> None: ...
    
    def generate_governance_attack(
        self,
        protocol: str,
        attack_vector: str,
        success_probability: float
    ) -> Dict[str, Tensor]: ...
    
    def simulate_protocol_upgrade(
        self,
        protocol: str,
        upgrade_type: str,
        impact_distribution: Dict[str, float]
    ) -> Dict[str, Tensor]: ...
    
    def generate_economic_attack(
        self,
        target_mechanism: str,
        attack_resources: Dict[str, float],
        strategy: str
    ) -> Dict[str, Tensor]: ... 