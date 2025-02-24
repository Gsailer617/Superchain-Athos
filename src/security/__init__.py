"""Security package providing comprehensive security analysis and validation"""

from .core.config import SecurityConfig
from .core.types import (
    SecurityReport,
    SecurityValidation,
    ContractMetadata,
    GasProfile,
    Vulnerability,
    VulnerabilitySeverity
)
from .service import SecurityService
from .analysis.gas import GasAnalyzer
from .analysis.vulnerabilities import VulnerabilityAnalyzer

__all__ = [
    # Main service
    'SecurityService',
    
    # Core types
    'SecurityConfig',
    'SecurityReport',
    'SecurityValidation',
    'ContractMetadata',
    'GasProfile',
    'Vulnerability',
    'VulnerabilitySeverity',
    
    # Analysis components
    'GasAnalyzer',
    'VulnerabilityAnalyzer'
]

__version__ = '2.0.0' 