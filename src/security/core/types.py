"""Shared security types and data structures"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from enum import Enum
from datetime import datetime

class VulnerabilitySeverity(Enum):
    """Vulnerability severity levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"

@dataclass
class Vulnerability:
    """Security vulnerability information"""
    severity: VulnerabilitySeverity
    title: str
    description: str
    location: str
    line_numbers: List[int]
    impact: str
    recommendation: str
    gas_impact: Optional[int] = None
    references: List[str] = field(default_factory=list)

@dataclass
class GasProfile:
    """Gas usage profile"""
    deployment_cost: int
    function_costs: Dict[str, int]
    storage_slots: int
    bytecode_size: int
    optimization_suggestions: List[str] = field(default_factory=list)
    hot_spots: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SecurityValidation:
    """Security validation result"""
    is_valid: bool
    risk_score: float = 0.0
    reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    gas_profile: Optional[GasProfile] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ContractMetadata:
    """Smart contract metadata"""
    name: str
    version: str
    compiler: str
    optimization_enabled: bool
    optimization_runs: int
    source_hash: str
    bytecode_hash: str
    creation_date: datetime
    last_audit_date: Optional[datetime] = None
    dependencies: Set[str] = field(default_factory=set)

@dataclass
class SecurityReport:
    """Comprehensive security analysis report"""
    contract: ContractMetadata
    validation: SecurityValidation
    static_analysis: Dict[str, Any]
    dynamic_analysis: Dict[str, Any]
    gas_analysis: GasProfile
    simulation_results: Dict[str, Any]
    optimization_suggestions: List[str] = field(default_factory=list)
    audit_history: List[Dict[str, Any]] = field(default_factory=list) 