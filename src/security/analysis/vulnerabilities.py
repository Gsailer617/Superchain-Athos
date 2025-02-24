"""Vulnerability analysis module"""

import logging
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import solcx
from slither.slither import Slither
from mythril.mythril import MythrilDisassembler, MythrilAnalyzer
from web3 import Web3

from ..core.config import SecurityConfig
from ..core.types import Vulnerability, VulnerabilitySeverity, SecurityValidation

logger = logging.getLogger(__name__)

class VulnerabilityAnalyzer:
    """Smart contract vulnerability analyzer"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self._init_tools()
        
    def _init_tools(self):
        """Initialize analysis tools"""
        solcx.install_solc(self.config.solc_version)
        solcx.set_solc_version(self.config.solc_version)
        self.mythril = MythrilDisassembler()
        self.mythril_analyzer = MythrilAnalyzer()
        
    async def analyze_contract(
        self,
        contract_path: Path,
        constructor_args: Optional[List] = None
    ) -> SecurityValidation:
        """Analyze contract for vulnerabilities
        
        Args:
            contract_path: Path to contract source
            constructor_args: Optional constructor arguments
            
        Returns:
            SecurityValidation with analysis results
        """
        vulnerabilities = []
        
        # Static Analysis
        static_vulns = await self._static_analysis(contract_path)
        vulnerabilities.extend(static_vulns)
        
        # Dynamic Analysis
        dynamic_vulns = await self._dynamic_analysis(
            contract_path,
            constructor_args
        )
        vulnerabilities.extend(dynamic_vulns)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(vulnerabilities)
        
        # Generate validation result
        validation = SecurityValidation(
            is_valid=not any(v.severity in [VulnerabilitySeverity.CRITICAL, VulnerabilitySeverity.HIGH]
                           for v in vulnerabilities),
            risk_score=risk_score,
            vulnerabilities=vulnerabilities
        )
        
        # Add warnings for medium/low severity issues
        for vuln in vulnerabilities:
            if vuln.severity in [VulnerabilitySeverity.MEDIUM, VulnerabilitySeverity.LOW]:
                validation.warnings.append(f"{vuln.severity.value}: {vuln.title}")
                
        return validation
        
    async def _static_analysis(self, contract_path: Path) -> List[Vulnerability]:
        """Perform static analysis using Slither"""
        try:
            vulnerabilities = []
            
            # Initialize Slither
            slither = Slither(str(contract_path))
            
            # Check for reentrancy
            if self.config.check_reentrancy:
                reentrancy = self._check_reentrancy(slither)
                vulnerabilities.extend(reentrancy)
                
            # Check for overflow
            if self.config.check_overflow:
                overflow = self._check_overflow(slither)
                vulnerabilities.extend(overflow)
                
            # Check for unchecked calls
            if self.config.check_unchecked_calls:
                unchecked = self._check_unchecked_calls(slither)
                vulnerabilities.extend(unchecked)
                
            return vulnerabilities
            
        except Exception as e:
            logger.error(f"Error in static analysis: {str(e)}")
            return []
            
    def _check_reentrancy(self, slither: Slither) -> List[Vulnerability]:
        """Check for reentrancy vulnerabilities"""
        vulnerabilities = []
        
        for contract in slither.contracts:
            for function in contract.functions:
                if function.is_protected():
                    continue
                    
                # Check for state changes after external calls
                external_calls = [c for c in function.external_calls if not c.is_protected()]
                state_vars_written = function.state_variables_written
                
                if external_calls and state_vars_written:
                    vulnerabilities.append(
                        Vulnerability(
                            severity=VulnerabilitySeverity.HIGH,
                            title="Potential Reentrancy",
                            description=(
                                f"Function {function.name} contains state changes after "
                                "external calls, potentially vulnerable to reentrancy"
                            ),
                            location=contract.name,
                            line_numbers=[function.source_mapping['start_line']],
                            impact="High - Could lead to unauthorized state changes",
                            recommendation=(
                                "Use ReentrancyGuard or ensure all state changes "
                                "happen before external calls"
                            )
                        )
                    )
                    
        return vulnerabilities
        
    def _check_overflow(self, slither: Slither) -> List[Vulnerability]:
        """Check for arithmetic overflow/underflow"""
        vulnerabilities = []
        
        for contract in slither.contracts:
            for function in contract.functions:
                for node in function.nodes:
                    if node.type in ['BINARY', 'UNARY'] and not node.is_checked():
                        vulnerabilities.append(
                            Vulnerability(
                                severity=VulnerabilitySeverity.MEDIUM,
                                title="Potential Arithmetic Overflow/Underflow",
                                description=(
                                    f"Unchecked arithmetic operation in {function.name}"
                                ),
                                location=contract.name,
                                line_numbers=[node.source_mapping['start_line']],
                                impact="Medium - Could lead to unexpected behavior",
                                recommendation="Use SafeMath or Solidity 0.8+ built-in checks"
                            )
                        )
                        
        return vulnerabilities
        
    def _check_unchecked_calls(self, slither: Slither) -> List[Vulnerability]:
        """Check for unchecked external calls"""
        vulnerabilities = []
        
        for contract in slither.contracts:
            for function in contract.functions:
                for node in function.nodes:
                    if node.type == 'CALL' and not node.is_checked():
                        vulnerabilities.append(
                            Vulnerability(
                                severity=VulnerabilitySeverity.MEDIUM,
                                title="Unchecked External Call",
                                description=(
                                    f"External call in {function.name} doesn't check return value"
                                ),
                                location=contract.name,
                                line_numbers=[node.source_mapping['start_line']],
                                impact="Medium - Could miss failed calls",
                                recommendation="Add return value checks for external calls"
                            )
                        )
                        
        return vulnerabilities
        
    async def _dynamic_analysis(
        self,
        contract_path: Path,
        constructor_args: Optional[List] = None
    ) -> List[Vulnerability]:
        """Perform dynamic analysis using Mythril"""
        try:
            vulnerabilities = []
            
            # Create Mythril analysis context
            contract = self.mythril.load_from_file(str(contract_path))
            
            # Run analysis
            report = self.mythril_analyzer.analyze(
                contract,
                timeout=self.config.max_analysis_time
            )
            
            # Convert findings to vulnerabilities
            for issue in report.issues:
                severity = self._map_mythril_severity(issue.severity)
                vulnerabilities.append(
                    Vulnerability(
                        severity=severity,
                        title=issue.title,
                        description=issue.description,
                        location=issue.contract,
                        line_numbers=issue.line_numbers,
                        impact=issue.impact,
                        recommendation=issue.recommendation
                    )
                )
                
            return vulnerabilities
            
        except Exception as e:
            logger.error(f"Error in dynamic analysis: {str(e)}")
            return []
            
    def _map_mythril_severity(self, severity: str) -> VulnerabilitySeverity:
        """Map Mythril severity to internal severity levels"""
        mapping = {
            'Critical': VulnerabilitySeverity.CRITICAL,
            'High': VulnerabilitySeverity.HIGH,
            'Medium': VulnerabilitySeverity.MEDIUM,
            'Low': VulnerabilitySeverity.LOW,
            'Info': VulnerabilitySeverity.INFO
        }
        return mapping.get(severity, VulnerabilitySeverity.INFO)
        
    def _calculate_risk_score(self, vulnerabilities: List[Vulnerability]) -> float:
        """Calculate overall risk score from vulnerabilities"""
        if not vulnerabilities:
            return 0.0
            
        # Severity weights
        weights = {
            VulnerabilitySeverity.CRITICAL: 1.0,
            VulnerabilitySeverity.HIGH: 0.8,
            VulnerabilitySeverity.MEDIUM: 0.5,
            VulnerabilitySeverity.LOW: 0.2,
            VulnerabilitySeverity.INFO: 0.1
        }
        
        # Calculate weighted score
        total_weight = sum(weights[v.severity] for v in vulnerabilities)
        max_possible = len(vulnerabilities)
        
        return min(1.0, total_weight / max_possible) 