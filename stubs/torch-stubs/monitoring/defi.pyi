from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from torch import Tensor
from datetime import datetime
from ..monitoring import MetricsLogger, PerformanceMonitor

class DeFiMetricsLogger(MetricsLogger):
    """DeFi-specific metrics logging system.
    
    Tracks metrics specific to DeFi operations including
    liquidity, slippage, gas costs, and MEV protection.
    """
    def __init__(
        self,
        chains: List[str],
        dexes: List[str],
        token_pairs: List[Tuple[str, str]],
        update_interval: str = '1m'
    ) -> None: ...
    
    def log_liquidity_metrics(
        self,
        pool_address: str,
        liquidity_data: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> None: ...
    
    def log_slippage_metrics(
        self,
        trade_data: Dict[str, Any],
        expected_price: float,
        actual_price: float
    ) -> None: ...
    
    def log_gas_metrics(
        self,
        transaction_hash: str,
        gas_used: int,
        gas_price: int,
        success: bool
    ) -> None: ...
    
    def log_mev_metrics(
        self,
        transaction_data: Dict[str, Any],
        frontrunning_detection: Dict[str, Any]
    ) -> None: ...

class LiquidityMonitor:
    """Monitors liquidity conditions across pools.
    
    Tracks and analyzes liquidity depth, concentration,
    and potential risks.
    """
    def __init__(
        self,
        pools: List[str],
        min_liquidity_threshold: Dict[str, float],
        update_frequency: str = '1m'
    ) -> None: ...
    
    def check_liquidity_depth(
        self,
        pool_address: str,
        trade_size: float
    ) -> Dict[str, float]: ...
    
    def analyze_liquidity_concentration(
        self,
        pool_address: str,
        price_range: Tuple[float, float]
    ) -> Dict[str, Any]: ...
    
    def predict_slippage(
        self,
        pool_address: str,
        trade_size: float,
        direction: str
    ) -> float: ...

class MEVProtectionMonitor:
    """Monitors and protects against MEV attacks.
    
    Implements detection and prevention of sandwich attacks,
    frontrunning, and other MEV-related risks.
    """
    def __init__(
        self,
        protection_config: Dict[str, Any],
        alert_manager: Any
    ) -> None: ...
    
    def analyze_mempool(
        self,
        pending_tx: Dict[str, Any]
    ) -> Dict[str, Any]: ...
    
    def detect_sandwich_attack(
        self,
        transaction_data: Dict[str, Any],
        mempool_state: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]: ...
    
    def compute_optimal_gas_bid(
        self,
        transaction_data: Dict[str, Any],
        current_market: Dict[str, Any]
    ) -> int: ...

class ChainStateMonitor:
    """Monitors blockchain state and network conditions.
    
    Tracks network congestion, gas prices, and other
    chain-specific metrics.
    """
    def __init__(
        self,
        chains: List[str],
        rpc_endpoints: Dict[str, str],
        cache_duration: str = '1m'
    ) -> None: ...
    
    def get_gas_estimates(
        self,
        chain_id: str,
        priority: str = 'medium'
    ) -> Dict[str, int]: ...
    
    def check_network_congestion(
        self,
        chain_id: str
    ) -> Dict[str, float]: ...
    
    def monitor_block_time(
        self,
        chain_id: str,
        window_size: int = 100
    ) -> Dict[str, float]: ...

class ArbitrageMetricsTracker:
    """Tracks arbitrage-specific performance metrics.
    
    Monitors opportunity detection, execution success,
    and profitability metrics.
    """
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        dexes: List[str],
        min_profit_threshold: float
    ) -> None: ...
    
    def log_opportunity(
        self,
        opportunity_data: Dict[str, Any],
        execution_result: Optional[Dict[str, Any]] = None
    ) -> None: ...
    
    def analyze_missed_opportunities(
        self,
        time_window: str = '1d'
    ) -> Dict[str, Any]: ...
    
    def compute_success_rate(
        self,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]: ...
    
    def analyze_profitability(
        self,
        time_window: str = '1d',
        include_gas: bool = True
    ) -> Dict[str, float]: ... 