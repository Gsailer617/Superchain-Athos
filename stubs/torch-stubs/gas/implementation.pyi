from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from torch import Tensor
import asyncio
from datetime import datetime

class GasOptimizationModel:
    """Gas optimization model.
    
    Neural network model for predicting optimal gas
    parameters based on network conditions.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        learning_rate: float = 1e-3
    ) -> None: ...
    
    async def predict(
        self,
        features: Tensor
    ) -> Dict[str, Tensor]: ...
    
    async def update(
        self,
        features: Tensor,
        targets: Tensor
    ) -> Dict[str, float]: ...
    
    async def validate(
        self,
        validation_data: Tuple[Tensor, Tensor]
    ) -> Dict[str, float]: ...

class NetworkMonitor:
    """Network monitoring system.
    
    Monitors blockchain network conditions for gas
    optimization decisions.
    """
    def __init__(
        self,
        rpc_config: Dict[str, Any],
        monitoring_interval: float = 1.0
    ) -> None: ...
    
    async def start_monitoring(self) -> None: ...
    
    async def get_network_metrics(
        self,
        metric_types: List[str]
    ) -> Dict[str, Any]: ...
    
    async def analyze_trends(
        self,
        time_window: int = 100
    ) -> Dict[str, List[float]]: ...

class TransactionManager:
    """Transaction management system.
    
    Manages transaction submission and monitoring with
    optimized gas parameters.
    """
    def __init__(
        self,
        network_interface: Any,
        retry_config: Dict[str, Any]
    ) -> None: ...
    
    async def submit_transaction(
        self,
        transaction_data: Dict[str, Any],
        gas_params: Dict[str, int]
    ) -> str: ...
    
    async def monitor_transaction(
        self,
        transaction_hash: str,
        timeout: float = 60.0
    ) -> Dict[str, Any]: ...
    
    async def estimate_confirmation_time(
        self,
        gas_params: Dict[str, int]
    ) -> float: ...

class GasCache:
    """Gas metrics caching system.
    
    Caches and manages historical gas metrics for
    efficient access and analysis.
    """
    def __init__(
        self,
        cache_size: int = 1000,
        cleanup_interval: float = 300.0
    ) -> None: ...
    
    async def add_metrics(
        self,
        block_number: int,
        metrics: Dict[str, Any]
    ) -> None: ...
    
    async def get_metrics(
        self,
        block_range: Optional[Tuple[int, int]] = None
    ) -> List[Dict[str, Any]]: ...
    
    async def cleanup_old_metrics(self) -> None: ...

class OptimizationStrategy:
    """Gas optimization strategy.
    
    Implements different strategies for gas optimization
    based on transaction requirements.
    """
    def __init__(
        self,
        strategy_type: str,
        params: Dict[str, Any]
    ) -> None: ...
    
    async def optimize(
        self,
        transaction_data: Dict[str, Any],
        network_state: Dict[str, Any]
    ) -> Dict[str, int]: ...
    
    async def evaluate_cost(
        self,
        gas_params: Dict[str, int],
        market_price: float
    ) -> float: ...
    
    async def adjust_params(
        self,
        performance_metrics: Dict[str, float]
    ) -> None: ...

class ValidationSystem:
    """Gas parameter validation system.
    
    Validates gas parameters before transaction
    submission for safety and efficiency.
    """
    def __init__(
        self,
        validation_rules: Dict[str, Callable],
        threshold_config: Dict[str, float]
    ) -> None: ...
    
    async def validate_params(
        self,
        gas_params: Dict[str, int],
        transaction_data: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]: ...
    
    async def check_safety_bounds(
        self,
        gas_params: Dict[str, int]
    ) -> bool: ...
    
    async def estimate_risk(
        self,
        gas_params: Dict[str, int],
        network_state: Dict[str, Any]
    ) -> float: ... 