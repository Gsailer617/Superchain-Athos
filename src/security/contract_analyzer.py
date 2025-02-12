"""Smart contract security analyzer and optimizer"""

import solcx
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from web3 import Web3
from eth_typing import ChecksumAddress
import subprocess
import asyncio
from slither.slither import Slither
from mythril.mythril import MythrilDisassembler, MythrilAnalyzer
from crytic_compile import CryticCompile
import eth_abi
from eth_utils import encode_hex
import networkx as nx

logger = logging.getLogger(__name__)

@dataclass
class SecurityConfig:
    """Security analysis configuration"""
    solc_version: str = "0.8.17"
    optimization_runs: int = 200
    check_reentrancy: bool = True
    check_overflow: bool = True
    check_unchecked_calls: bool = True
    max_analysis_time: int = 300  # seconds
    gas_price_percentile: int = 90

@dataclass
class SecurityReport:
    """Security analysis report"""
    vulnerabilities: List[Dict[str, Any]]
    gas_analysis: Dict[str, Any]
    optimization_suggestions: List[str]
    risk_score: float
    simulation_results: Dict[str, Any]

class ContractAnalyzer:
    """Smart contract security analyzer and optimizer"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self._init_tools()
    
    def _init_tools(self):
        """Initialize security analysis tools"""
        # Install and set solc version
        solcx.install_solc(self.config.solc_version)
        solcx.set_solc_version(self.config.solc_version)
        
        # Initialize Mythril
        self.mythril = MythrilDisassembler()
        self.mythril_analyzer = MythrilAnalyzer()
    
    async def analyze_contract(
        self,
        contract_path: Path,
        constructor_args: Optional[List] = None
    ) -> SecurityReport:
        """Analyze contract security and optimization"""
        vulnerabilities = []
        gas_analysis = {}
        optimization_suggestions = []
        
        # Static Analysis
        static_vulns = await self._static_analysis(contract_path)
        vulnerabilities.extend(static_vulns)
        
        # Dynamic Analysis
        dynamic_vulns = await self._dynamic_analysis(
            contract_path,
            constructor_args
        )
        vulnerabilities.extend(dynamic_vulns)
        
        # Gas Analysis
        gas_analysis = await self._analyze_gas_usage(contract_path)
        
        # Optimization Analysis
        optimization_suggestions = await self._suggest_optimizations(
            contract_path,
            gas_analysis
        )
        
        # Simulation
        simulation_results = await self._simulate_attacks(
            contract_path,
            constructor_args
        )
        
        # Calculate Risk Score
        risk_score = self._calculate_risk_score(
            vulnerabilities,
            gas_analysis,
            simulation_results
        )
        
        return SecurityReport(
            vulnerabilities=vulnerabilities,
            gas_analysis=gas_analysis,
            optimization_suggestions=optimization_suggestions,
            risk_score=risk_score,
            simulation_results=simulation_results
        )
    
    async def _static_analysis(self, contract_path: Path) -> List[Dict[str, Any]]:
        """Perform static analysis using multiple tools"""
        vulnerabilities = []
        
        # Slither Analysis
        try:
            slither = Slither(str(contract_path))
            for detector in slither.detectors:
                results = detector.detect()
                for result in results:
                    vulnerabilities.append({
                        'tool': 'slither',
                        'type': detector.ARGUMENT,
                        'description': result['description'],
                        'severity': result['impact'],
                        'location': result['elements']
                    })
        except Exception as e:
            logger.error(f"Slither analysis failed: {str(e)}")
        
        # Mythril Analysis
        try:
            # Compile contract
            compilation = CryticCompile(str(contract_path))
            contracts = compilation.contracts_data
            
            for contract_name, contract_data in contracts.items():
                # Analyze with Mythril
                analysis = self.mythril_analyzer.analyze(
                    contract_data,
                    timeout=self.config.max_analysis_time
                )
                
                for issue in analysis.issues:
                    vulnerabilities.append({
                        'tool': 'mythril',
                        'type': issue.type,
                        'description': issue.description,
                        'severity': issue.severity,
                        'location': {
                            'line': issue.lineno,
                            'code': issue.code
                        }
                    })
        except Exception as e:
            logger.error(f"Mythril analysis failed: {str(e)}")
        
        return vulnerabilities
    
    async def _dynamic_analysis(
        self,
        contract_path: Path,
        constructor_args: Optional[List] = None
    ) -> List[Dict[str, Any]]:
        """Perform dynamic analysis using Ganache and custom tests"""
        vulnerabilities = []
        
        # Start local chain
        proc = await asyncio.create_subprocess_exec(
            'ganache',
            '--deterministic',
            '--quiet',
            stdout=asyncio.subprocess.PIPE
        )
        
        try:
            # Deploy contract
            web3 = Web3(Web3.HTTPProvider('http://localhost:8545'))
            compiled = solcx.compile_files(
                [str(contract_path)],
                output_values=['abi', 'bin']
            )
            
            contract_interface = compiled[str(contract_path) + ':' + contract_path.stem]
            contract = web3.eth.contract(
                abi=contract_interface['abi'],
                bytecode=contract_interface['bin']
            )
            
            # Deploy with constructor args if provided
            if constructor_args:
                tx_hash = contract.constructor(*constructor_args).transact()
            else:
                tx_hash = contract.constructor().transact()
            
            tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
            contract_address = tx_receipt['contractAddress']
            
            # Run security tests
            if self.config.check_reentrancy:
                reentrancy = await self._check_reentrancy(
                    web3,
                    contract_address,
                    contract_interface['abi']
                )
                vulnerabilities.extend(reentrancy)
            
            if self.config.check_unchecked_calls:
                unchecked = await self._check_unchecked_calls(
                    web3,
                    contract_address,
                    contract_interface['abi']
                )
                vulnerabilities.extend(unchecked)
            
        finally:
            # Cleanup
            proc.terminate()
            await proc.wait()
        
        return vulnerabilities
    
    async def _analyze_gas_usage(self, contract_path: Path) -> Dict[str, Any]:
        """Analyze contract gas usage patterns"""
        gas_analysis = {}
        
        try:
            # Compile with different optimization settings
            for runs in [0, 200, 1000]:
                compiled = solcx.compile_files(
                    [str(contract_path)],
                    optimize=runs > 0,
                    optimize_runs=runs
                )
                
                contract_data = compiled[str(contract_path) + ':' + contract_path.stem]
                
                # Analyze bytecode size
                bytecode_size = len(contract_data['bin']) // 2
                gas_analysis[f'bytecode_size_runs_{runs}'] = bytecode_size
                
                # Estimate deployment cost
                deployment_cost = self._estimate_deployment_cost(
                    contract_data['bin']
                )
                gas_analysis[f'deployment_cost_runs_{runs}'] = deployment_cost
                
                # Analyze function gas costs
                function_costs = self._analyze_function_costs(
                    contract_data['abi'],
                    contract_data['bin']
                )
                gas_analysis[f'function_costs_runs_{runs}'] = function_costs
        
        except Exception as e:
            logger.error(f"Gas analysis failed: {str(e)}")
        
        return gas_analysis
    
    async def _suggest_optimizations(
        self,
        contract_path: Path,
        gas_analysis: Dict[str, Any]
    ) -> List[str]:
        """Suggest gas optimizations"""
        suggestions = []
        
        try:
            # Analyze source code
            with open(contract_path) as f:
                source = f.read()
            
            # Check for common gas optimization patterns
            if 'uint256' in source:
                suggestions.append(
                    "Consider using uint128/uint96/uint64 for smaller numbers "
                    "to pack multiple variables into a single storage slot"
                )
            
            if 'string' in source:
                suggestions.append(
                    "Consider using bytes32 instead of string for fixed-length data "
                    "to save gas on storage"
                )
            
            # Analyze storage layout
            storage_layout = self._analyze_storage_layout(contract_path)
            if storage_layout:
                suggestions.extend(
                    self._suggest_storage_optimizations(storage_layout)
                )
            
            # Compare optimization runs
            bytecode_sizes = [
                gas_analysis.get(f'bytecode_size_runs_{runs}', 0)
                for runs in [0, 200, 1000]
            ]
            
            if min(bytecode_sizes) < max(bytecode_sizes) * 0.9:
                suggestions.append(
                    f"Consider using optimization runs={bytecode_sizes.index(min(bytecode_sizes)) * 200} "
                    f"to reduce bytecode size by {(1 - min(bytecode_sizes)/max(bytecode_sizes))*100:.1f}%"
                )
        
        except Exception as e:
            logger.error(f"Optimization analysis failed: {str(e)}")
        
        return suggestions
    
    async def _simulate_attacks(
        self,
        contract_path: Path,
        constructor_args: Optional[List] = None
    ) -> Dict[str, Any]:
        """Simulate various attack scenarios"""
        results = {}
        
        try:
            # Deploy contract on local chain
            web3 = Web3(Web3.HTTPProvider('http://localhost:8545'))
            compiled = solcx.compile_files(
                [str(contract_path)],
                output_values=['abi', 'bin']
            )
            
            contract_interface = compiled[str(contract_path) + ':' + contract_path.stem]
            contract = web3.eth.contract(
                abi=contract_interface['abi'],
                bytecode=contract_interface['bin']
            )
            
            # Deploy contract
            if constructor_args:
                tx_hash = contract.constructor(*constructor_args).transact()
            else:
                tx_hash = contract.constructor().transact()
            
            tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
            contract_address = tx_receipt['contractAddress']
            
            # Simulate reentrancy attack
            results['reentrancy'] = await self._simulate_reentrancy(
                web3,
                contract_address,
                contract_interface['abi']
            )
            
            # Simulate front-running
            results['frontrunning'] = await self._simulate_frontrunning(
                web3,
                contract_address,
                contract_interface['abi']
            )
            
            # Simulate integer overflow
            results['overflow'] = await self._simulate_overflow(
                web3,
                contract_address,
                contract_interface['abi']
            )
        
        except Exception as e:
            logger.error(f"Attack simulation failed: {str(e)}")
        
        return results
    
    def _calculate_risk_score(
        self,
        vulnerabilities: List[Dict[str, Any]],
        gas_analysis: Dict[str, Any],
        simulation_results: Dict[str, Any]
    ) -> float:
        """Calculate overall risk score"""
        score = 0.0
        max_score = 10.0
        
        # Vulnerability scoring
        severity_weights = {
            'high': 1.0,
            'medium': 0.5,
            'low': 0.2,
            'info': 0.1
        }
        
        for vuln in vulnerabilities:
            score += severity_weights.get(vuln['severity'], 0.1)
        
        # Gas efficiency scoring
        if gas_analysis:
            bytecode_size = gas_analysis.get('bytecode_size_runs_200', 0)
            if bytecode_size > 24576:  # Maximum contract size
                score += 1.0
            elif bytecode_size > 16384:
                score += 0.5
        
        # Simulation results scoring
        for result in simulation_results.values():
            if result.get('success', False):
                score += 1.0
        
        return min(score, max_score)
    
    @staticmethod
    def _estimate_deployment_cost(bytecode: str) -> int:
        """Estimate contract deployment cost"""
        gas_per_byte = 200  # Approximate gas cost per byte of bytecode
        return (len(bytecode) // 2) * gas_per_byte
    
    def _analyze_function_costs(
        self,
        abi: List[Dict[str, Any]],
        bytecode: str
    ) -> Dict[str, int]:
        """Analyze gas costs for each function"""
        costs = {}
        
        for item in abi:
            if item['type'] != 'function':
                continue
                
            # Estimate base cost
            base_cost = 21000  # Base transaction cost
            
            # Add cost for input parameters
            for input_param in item.get('inputs', []):
                base_cost += self._estimate_param_cost(input_param['type'])
            
            costs[item['name']] = base_cost
        
        return costs
    
    @staticmethod
    def _estimate_param_cost(param_type: str) -> int:
        """Estimate gas cost for parameter type"""
        costs = {
            'uint256': 20000,
            'address': 20000,
            'bool': 3000,
            'string': 40000,
            'bytes': 40000
        }
        return costs.get(param_type, 20000)
    
    def _analyze_storage_layout(self, contract_path: Path) -> Optional[Dict[str, Any]]:
        """Analyze contract storage layout"""
        try:
            # Compile with storage layout
            compiled = solcx.compile_files(
                [str(contract_path)],
                output_values=['storage-layout']
            )
            
            return compiled[str(contract_path) + ':' + contract_path.stem]['storage-layout']
        except Exception as e:
            logger.error(f"Storage layout analysis failed: {str(e)}")
            return None
    
    def _suggest_storage_optimizations(
        self,
        storage_layout: Dict[str, Any]
    ) -> List[str]:
        """Suggest storage layout optimizations"""
        suggestions = []
        
        # Build storage slot graph
        slot_graph = nx.Graph()
        
        for var in storage_layout['storage']:
            slot = var['slot']
            size = var['size']
            slot_graph.add_node(slot, size=size, variables=[var['label']])
        
        # Find inefficient packing
        for slot in slot_graph.nodes:
            size = slot_graph.nodes[slot]['size']
            if size < 256:  # Not using full slot
                suggestions.append(
                    f"Storage slot {slot} is only using {size} bits. "
                    f"Consider packing with other small variables"
                )
        
        return suggestions 