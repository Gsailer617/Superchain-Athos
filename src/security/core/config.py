"""Unified security configuration"""

from dataclasses import dataclass, field
from typing import Set, Dict, Any
from pathlib import Path

@dataclass
class SecurityConfig:
    """Unified security configuration settings"""
    
    # Runtime security settings
    max_slippage: float = 0.01  # 1%
    max_price_impact: float = 0.05  # 5%
    min_liquidity: float = 100000.0  # $100k
    max_position_size: float = 1000000.0  # $1M
    max_gas_price: int = 500  # gwei
    min_timelock: int = 86400  # 24 hours
    max_flash_loan_ratio: float = 0.75  # 75%
    
    # Contract analysis settings
    solc_version: str = "0.8.17"
    optimization_runs: int = 200
    check_reentrancy: bool = True
    check_overflow: bool = True
    check_unchecked_calls: bool = True
    max_analysis_time: int = 300  # seconds
    gas_price_percentile: int = 90
    
    # Storage analysis settings
    analyze_storage_layout: bool = True
    analyze_function_costs: bool = True
    analyze_bytecode: bool = True
    max_contract_size: int = 24576  # bytes
    
    # Validation settings
    required_audits: bool = True
    required_timelock: bool = True
    blacklisted_addresses: Set[str] = field(default_factory=set)
    
    # Tool configurations
    tool_paths: Dict[str, Path] = field(default_factory=dict)
    custom_rules: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default tool paths"""
        if not self.tool_paths:
            self.tool_paths = {
                'slither': Path('slither'),
                'mythril': Path('mythril'),
                'solc': Path('solc')
            }
        if not self.custom_rules:
            self.custom_rules = {
                'max_function_complexity': 50,
                'min_test_coverage': 85,
                'max_dependency_depth': 5
            } 