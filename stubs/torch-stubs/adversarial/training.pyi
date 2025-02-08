from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from torch import Tensor
from ..nn import _Module

class AdversarialScenarioGenerator:
    """Adversarial scenario generation system.
    
    Generates challenging market scenarios to test and
    improve strategy robustness.
    """
    def __init__(
        self,
        base_market_state: Dict[str, Tensor],
        attack_types: List[str],
        perturbation_config: Dict[str, Any]
    ) -> None: ...
    
    def generate_attack_scenario(
        self,
        attack_type: str,
        severity: float,
        target_params: Dict[str, Any]
    ) -> Dict[str, Tensor]: ...
    
    def generate_flash_loan_attack(
        self,
        loan_size: float,
        attack_path: List[str],
        pool_states: Dict[str, Tensor]
    ) -> Dict[str, Tensor]: ...
    
    def generate_price_manipulation(
        self,
        target_pools: List[str],
        manipulation_params: Dict[str, float]
    ) -> Dict[str, Tensor]: ...

class AdversarialTrainer:
    """Adversarial training system.
    
    Trains strategies to be robust against various
    types of market manipulation and attacks.
    """
    def __init__(
        self,
        model: _Module,
        scenario_generator: AdversarialScenarioGenerator,
        training_config: Dict[str, Any]
    ) -> None: ...
    
    def train_step(
        self,
        batch: Dict[str, Tensor],
        adversarial_scenarios: List[Dict[str, Tensor]]
    ) -> Dict[str, float]: ...
    
    def generate_adversarial_batch(
        self,
        clean_batch: Dict[str, Tensor],
        attack_config: Dict[str, Any]
    ) -> List[Dict[str, Tensor]]: ...
    
    def evaluate_robustness(
        self,
        validation_data: Dict[str, Tensor],
        attack_types: List[str]
    ) -> Dict[str, float]: ...

class RobustnessVerifier:
    """Strategy robustness verification.
    
    Verifies strategy robustness against various
    attack scenarios and market conditions.
    """
    def __init__(
        self,
        verification_metrics: List[str],
        threshold_config: Dict[str, float]
    ) -> None: ...
    
    def verify_strategy(
        self,
        strategy: _Module,
        test_scenarios: List[Dict[str, Tensor]]
    ) -> Tuple[bool, Dict[str, float]]: ...
    
    def analyze_vulnerabilities(
        self,
        test_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]: ...
    
    def compute_robustness_score(
        self,
        performance_metrics: Dict[str, float]
    ) -> float: ...

class AttackDetector:
    """Real-time attack detection system.
    
    Detects potential market manipulation and
    adversarial activities in real-time.
    """
    def __init__(
        self,
        detection_models: Dict[str, _Module],
        alert_thresholds: Dict[str, float]
    ) -> None: ...
    
    def analyze_market_activity(
        self,
        market_state: Dict[str, Tensor],
        recent_history: Dict[str, Tensor]
    ) -> Tuple[bool, List[Dict[str, Any]]]: ...
    
    def detect_manipulation(
        self,
        price_data: Tensor,
        volume_data: Tensor,
        window_size: int = 100
    ) -> Dict[str, float]: ...
    
    def assess_risk_level(
        self,
        detection_results: Dict[str, float]
    ) -> Tuple[str, float]: ...

class SafetyController:
    """Safety control system.
    
    Implements safety measures and controls for
    strategy protection.
    """
    def __init__(
        self,
        safety_rules: Dict[str, Callable],
        emergency_actions: Dict[str, Callable]
    ) -> None: ...
    
    def validate_action(
        self,
        proposed_action: Dict[str, Tensor],
        market_state: Dict[str, Tensor],
        risk_metrics: Dict[str, float]
    ) -> Tuple[bool, Optional[Dict[str, Any]]]: ...
    
    def emergency_stop(
        self,
        reason: str,
        market_state: Dict[str, Any]
    ) -> Dict[str, Any]: ...
    
    def adjust_position_limits(
        self,
        risk_level: str,
        market_metrics: Dict[str, float]
    ) -> Dict[str, float]: ...

class DefensiveExecutor:
    """Defensive execution system.
    
    Implements defensive trade execution strategies
    against potential attacks.
    """
    def __init__(
        self,
        execution_strategies: Dict[str, Callable],
        defense_config: Dict[str, Any]
    ) -> None: ...
    
    def execute_defensively(
        self,
        trade_intent: Dict[str, Any],
        market_state: Dict[str, Tensor],
        risk_assessment: Dict[str, float]
    ) -> Tuple[Dict[str, Any], Dict[str, float]]: ...
    
    def route_defensively(
        self,
        amount: float,
        path: List[str],
        safety_params: Dict[str, float]
    ) -> Tuple[List[Dict[str, Any]], float]: ...
    
    def compute_safety_buffer(
        self,
        trade_size: float,
        market_impact: Dict[str, float]
    ) -> float: ... 