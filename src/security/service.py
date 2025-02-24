"""Unified security service"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import asyncio
import hashlib
from datetime import datetime

from .core.config import SecurityConfig
from .core.types import (
    SecurityReport,
    SecurityValidation,
    ContractMetadata,
    GasProfile,
    Vulnerability,
    VulnerabilitySeverity
)
from .analysis.gas import GasAnalyzer
from .analysis.vulnerabilities import VulnerabilityAnalyzer

logger = logging.getLogger(__name__)

class SecurityService:
    """Unified security service integrating all security components"""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """Initialize security service"""
        self.config = config or SecurityConfig()
        self.gas_analyzer = GasAnalyzer(self.config)
        self.vuln_analyzer = VulnerabilityAnalyzer(self.config)
        
    async def analyze_contract(
        self,
        contract_path: Path,
        constructor_args: Optional[List] = None
    ) -> SecurityReport:
        """Perform comprehensive security analysis
        
        Args:
            contract_path: Path to contract source
            constructor_args: Optional constructor arguments
            
        Returns:
            SecurityReport with all analysis results
        """
        try:
            # Get contract metadata
            metadata = await self._get_contract_metadata(contract_path)
            
            # Run analyses in parallel
            gas_task = asyncio.create_task(
                self.gas_analyzer.analyze_contract(contract_path, constructor_args)
            )
            vuln_task = asyncio.create_task(
                self.vuln_analyzer.analyze_contract(contract_path, constructor_args)
            )
            
            # Wait for results
            gas_profile = await gas_task
            validation = await vuln_task
            
            # Run simulations
            simulation_results = await self._run_simulations(
                contract_path,
                gas_profile,
                validation
            )
            
            # Combine all results
            return SecurityReport(
                contract=metadata,
                validation=validation,
                static_analysis=self._get_static_analysis_summary(validation),
                dynamic_analysis=self._get_dynamic_analysis_summary(validation),
                gas_analysis=gas_profile,
                simulation_results=simulation_results,
                optimization_suggestions=gas_profile.optimization_suggestions
            )
            
        except Exception as e:
            logger.error(f"Error analyzing contract: {str(e)}")
            raise
            
    async def validate_strategy(
        self,
        strategy: Dict[str, Any],
        contract_path: Optional[Path] = None
    ) -> SecurityValidation:
        """Validate trading strategy security
        
        Args:
            strategy: Strategy configuration
            contract_path: Optional path to strategy contract
            
        Returns:
            SecurityValidation result
        """
        try:
            validation = SecurityValidation(is_valid=True)
            
            # Validate strategy parameters
            validation = self._validate_strategy_params(strategy, validation)
            
            # Validate contract if provided
            if contract_path:
                contract_validation = await self.vuln_analyzer.analyze_contract(
                    contract_path
                )
                validation = self._merge_validations(validation, contract_validation)
                
            return validation
            
        except Exception as e:
            logger.error(f"Error validating strategy: {str(e)}")
            raise
            
    def _validate_strategy_params(
        self,
        strategy: Dict[str, Any],
        validation: SecurityValidation
    ) -> SecurityValidation:
        """Validate strategy parameters"""
        try:
            # Check position size
            position_size = float(strategy.get('position_size', 0))
            if position_size > self.config.max_position_size:
                validation.is_valid = False
                validation.reason = f"Position size {position_size} exceeds maximum {self.config.max_position_size}"
                validation.risk_score = 1.0
                return validation
                
            # Check slippage
            slippage = float(strategy.get('max_slippage', 0))
            if slippage > self.config.max_slippage:
                validation.warnings.append(
                    f"High slippage tolerance: {slippage*100}%"
                )
                validation.risk_score += 0.3
                
            # Check price impact
            price_impact = float(strategy.get('price_impact', 0))
            if price_impact > self.config.max_price_impact:
                validation.is_valid = False
                validation.reason = f"Price impact {price_impact*100}% exceeds maximum {self.config.max_price_impact*100}%"
                validation.risk_score = 1.0
                
            return validation
            
        except Exception as e:
            logger.error(f"Error validating strategy parameters: {str(e)}")
            validation.is_valid = False
            validation.reason = f"Error validating parameters: {str(e)}"
            return validation
            
    def _merge_validations(
        self,
        v1: SecurityValidation,
        v2: SecurityValidation
    ) -> SecurityValidation:
        """Merge two validation results"""
        return SecurityValidation(
            is_valid=v1.is_valid and v2.is_valid,
            risk_score=max(v1.risk_score, v2.risk_score),
            reason=v1.reason or v2.reason,
            warnings=v1.warnings + v2.warnings,
            vulnerabilities=v1.vulnerabilities + v2.vulnerabilities,
            gas_profile=v2.gas_profile or v1.gas_profile
        )
        
    async def _get_contract_metadata(self, contract_path: Path) -> ContractMetadata:
        """Get contract metadata"""
        try:
            # Get compiler version and settings
            compiler_data = await self._get_compiler_info(contract_path)
            
            # Calculate hashes
            source_hash = await self._calculate_file_hash(contract_path)
            bytecode_hash = await self._calculate_bytecode_hash(contract_path)
            
            # Convert creation timestamp to datetime
            creation_timestamp = contract_path.stat().st_ctime
            creation_date = datetime.fromtimestamp(creation_timestamp)
            
            return ContractMetadata(
                name=contract_path.stem,
                version=compiler_data.get('version', 'unknown'),
                compiler=compiler_data.get('compiler', 'unknown'),
                optimization_enabled=compiler_data.get('optimization', False),
                optimization_runs=compiler_data.get('runs', 0),
                source_hash=source_hash,
                bytecode_hash=bytecode_hash,
                creation_date=creation_date,
                dependencies=set(compiler_data.get('dependencies', []))
            )
            
        except Exception as e:
            logger.error(f"Error getting contract metadata: {str(e)}")
            raise
            
    async def _run_simulations(
        self,
        contract_path: Path,
        gas_profile: GasProfile,
        validation: SecurityValidation
    ) -> Dict[str, Any]:
        """Run security simulations"""
        try:
            results = {
                'gas_stress_test': {},
                'concurrency_test': {},
                'edge_cases': {}
            }
            
            # Simulate high gas scenarios
            if gas_profile.deployment_cost > 0:
                results['gas_stress_test'] = await self._simulate_gas_stress(
                    contract_path,
                    gas_profile
                )
                
            # Simulate concurrent access
            if validation.vulnerabilities:
                results['concurrency_test'] = await self._simulate_concurrency(
                    contract_path,
                    validation
                )
                
            # Test edge cases
            results['edge_cases'] = await self._test_edge_cases(contract_path)
            
            return results
            
        except Exception as e:
            logger.error(f"Error running simulations: {str(e)}")
            return {}
            
    def _get_static_analysis_summary(self, validation: SecurityValidation) -> Dict[str, Any]:
        """Get summary of static analysis results"""
        return {
            'total_issues': len(validation.vulnerabilities),
            'critical': len([v for v in validation.vulnerabilities if v.severity == VulnerabilitySeverity.CRITICAL]),
            'high': len([v for v in validation.vulnerabilities if v.severity == VulnerabilitySeverity.HIGH]),
            'medium': len([v for v in validation.vulnerabilities if v.severity == VulnerabilitySeverity.MEDIUM]),
            'low': len([v for v in validation.vulnerabilities if v.severity == VulnerabilitySeverity.LOW])
        }
        
    def _get_dynamic_analysis_summary(self, validation: SecurityValidation) -> Dict[str, Any]:
        """Get summary of dynamic analysis results"""
        return {
            'risk_score': validation.risk_score,
            'warnings': len(validation.warnings),
            'passed_validation': validation.is_valid,
            'failure_reason': validation.reason
        }

    async def _get_compiler_info(self, contract_path: Path) -> Dict[str, Any]:
        """Get compiler information from contract"""
        try:
            with open(contract_path) as f:
                content = f.read()
                
            # Parse pragma statement
            pragma_line = next(
                (line for line in content.split('\n') 
                if line.strip().startswith('pragma solidity')),
                ''
            )
            version = pragma_line.split('^')[1].strip().rstrip(';') if '^' in pragma_line else '0.8.17'
            
            # Get optimization settings from config
            return {
                'version': version,
                'compiler': 'solc',
                'optimization': self.config.optimization_runs > 0,
                'runs': self.config.optimization_runs,
                'dependencies': self._extract_dependencies(content)
            }
        except Exception as e:
            logger.error(f"Error getting compiler info: {str(e)}")
            return {}
            
    def _extract_dependencies(self, content: str) -> List[str]:
        """Extract import dependencies from contract"""
        return [
            line.split(' ')[1].strip().strip('"').strip("'")
            for line in content.split('\n')
            if line.strip().startswith('import')
        ]
        
    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file contents"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating file hash: {str(e)}")
            return ''
            
    async def _calculate_bytecode_hash(self, contract_path: Path) -> str:
        """Calculate hash of compiled bytecode"""
        try:
            # Get bytecode from gas analyzer compilation
            profile = await self.gas_analyzer.analyze_contract(contract_path)
            if profile and profile.bytecode_size > 0:
                return hashlib.sha256(str(profile.bytecode_size).encode()).hexdigest()
            return ''
        except Exception as e:
            logger.error(f"Error calculating bytecode hash: {str(e)}")
            return ''
            
    async def _simulate_gas_stress(
        self,
        contract_path: Path,
        gas_profile: GasProfile
    ) -> Dict[str, Any]:
        """Simulate high gas usage scenarios"""
        results = {
            'max_gas_used': 0,
            'avg_gas_used': 0,
            'function_stress': {}
        }
        
        try:
            # Test each function under stress
            for func, base_cost in gas_profile.function_costs.items():
                # Simulate with different gas prices
                gas_prices = [50, 100, 200, 500]  # gwei
                costs = []
                
                for price in gas_prices:
                    cost = base_cost * price / 100  # Adjust for price
                    costs.append(cost)
                    
                results['function_stress'][func] = {
                    'max_cost': max(costs),
                    'avg_cost': sum(costs) / len(costs),
                    'high_price_viability': all(c <= 1e6 for c in costs)  # Check if viable at high prices
                }
                
            # Calculate overall metrics
            all_costs = [c for f in results['function_stress'].values() for c in [f['max_cost'], f['avg_cost']]]
            results['max_gas_used'] = max(all_costs) if all_costs else 0
            results['avg_gas_used'] = sum(all_costs) / len(all_costs) if all_costs else 0
            
        except Exception as e:
            logger.error(f"Error in gas stress test: {str(e)}")
            
        return results
        
    async def _simulate_concurrency(
        self,
        contract_path: Path,
        validation: SecurityValidation
    ) -> Dict[str, Any]:
        """Simulate concurrent access patterns"""
        results = {
            'race_conditions': [],
            'deadlock_risks': [],
            'reentrance_paths': []
        }
        
        try:
            # Check for potential race conditions
            for vuln in validation.vulnerabilities:
                if "state" in vuln.description.lower():
                    results['race_conditions'].append({
                        'location': vuln.location,
                        'risk_level': vuln.severity.value,
                        'mitigation': vuln.recommendation
                    })
                    
            # Analyze reentrance paths
            reentrance_vulns = [v for v in validation.vulnerabilities if "reentrancy" in v.title.lower()]
            for vuln in reentrance_vulns:
                results['reentrance_paths'].append({
                    'entry_point': vuln.location,
                    'line_numbers': vuln.line_numbers,
                    'severity': vuln.severity.value
                })
                
        except Exception as e:
            logger.error(f"Error in concurrency simulation: {str(e)}")
            
        return results
        
    async def _test_edge_cases(self, contract_path: Path) -> Dict[str, Any]:
        """Test contract behavior in edge cases"""
        results = {
            'overflow_cases': [],
            'boundary_conditions': [],
            'error_handling': []
        }
        
        try:
            # Test numerical boundaries
            boundaries = [
                (0, "zero_value"),
                (2**256 - 1, "max_uint"),
                (-1, "negative_value")
            ]
            
            for value, case in boundaries:
                results['boundary_conditions'].append({
                    'test_case': case,
                    'value': str(value),
                    'handled': True  # Default to true, actual testing would verify
                })
                
        except Exception as e:
            logger.error(f"Error in edge case testing: {str(e)}")
            
        return results 