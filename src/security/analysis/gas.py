"""Gas analysis and optimization module"""

import logging
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import solcx
from web3 import Web3
import networkx as nx
from collections import defaultdict

from ..core.config import SecurityConfig
from ..core.types import GasProfile, ContractMetadata

logger = logging.getLogger(__name__)

class GasAnalyzer:
    """Smart contract gas analysis and optimization"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self._init_compiler()
        
    def _init_compiler(self):
        """Initialize solidity compiler"""
        solcx.install_solc(self.config.solc_version)
        solcx.set_solc_version(self.config.solc_version)
        
    async def analyze_contract(
        self,
        contract_path: Path,
        constructor_args: Optional[List] = None
    ) -> GasProfile:
        """Analyze contract for gas optimization
        
        Args:
            contract_path: Path to contract source
            constructor_args: Optional constructor arguments
            
        Returns:
            GasProfile with analysis results
        """
        # Analyze with different optimization settings
        profiles = []
        for runs in [0, 200, 1000]:
            profile = await self._analyze_with_runs(
                contract_path,
                runs,
                constructor_args
            )
            profiles.append(profile)
            
        # Select best profile
        best_profile = min(
            profiles,
            key=lambda p: p.deployment_cost + sum(p.function_costs.values())
        )
        
        # Add optimization suggestions
        suggestions = await self._generate_suggestions(contract_path, best_profile)
        best_profile.optimization_suggestions.extend(suggestions)
        
        # Analyze hot spots
        hot_spots = await self._analyze_hot_spots(contract_path, best_profile)
        best_profile.hot_spots.update(hot_spots)
        
        return best_profile
        
    async def _analyze_with_runs(
        self,
        contract_path: Path,
        optimization_runs: int,
        constructor_args: Optional[List] = None
    ) -> GasProfile:
        """Analyze contract with specific optimization runs"""
        try:
            # Compile contract
            compiled = solcx.compile_files(
                [str(contract_path)],
                optimize=optimization_runs > 0,
                optimize_runs=optimization_runs,
                output_values=['abi', 'bin', 'storage-layout']
            )
            
            contract_id = str(contract_path) + ':' + contract_path.stem
            contract_data = compiled[contract_id]
            
            # Analyze deployment cost
            deployment_cost = self._estimate_deployment_cost(
                contract_data['bin'],
                constructor_args
            )
            
            # Analyze function costs
            function_costs = await self._analyze_function_costs(
                contract_data['abi'],
                contract_data['bin']
            )
            
            # Analyze storage
            storage_slots = len(contract_data.get('storage-layout', {}).get('storage', []))
            
            # Calculate bytecode size
            bytecode_size = len(contract_data['bin']) // 2
            
            return GasProfile(
                deployment_cost=deployment_cost,
                function_costs=function_costs,
                storage_slots=storage_slots,
                bytecode_size=bytecode_size
            )
            
        except Exception as e:
            logger.error(f"Error analyzing contract with {optimization_runs} runs: {str(e)}")
            raise
            
    def _estimate_deployment_cost(
        self,
        bytecode: str,
        constructor_args: Optional[List] = None
    ) -> int:
        """Estimate contract deployment cost"""
        try:
            # Calculate base deployment cost
            base_cost = len(bytecode) // 2 * 200  # 200 gas per byte
            
            # Add constructor args cost if provided
            if constructor_args:
                encoded_args = Web3.eth.abi.encode_abi(
                    ['bytes'],
                    [Web3.toBytes(hexstr=bytecode)]
                )
                args_cost = len(encoded_args) * 68  # 68 gas per non-zero byte
                return base_cost + args_cost
                
            return base_cost
            
        except Exception as e:
            logger.error(f"Error estimating deployment cost: {str(e)}")
            return 0
            
    async def _analyze_function_costs(
        self,
        abi: List[Dict[str, Any]],
        bytecode: str
    ) -> Dict[str, int]:
        """Analyze gas costs for contract functions"""
        try:
            costs = {}
            for item in abi:
                if item['type'] != 'function':
                    continue
                    
                name = item['name']
                inputs = item.get('inputs', [])
                
                # Estimate base cost
                base_cost = 21000  # Base transaction cost
                
                # Add cost for input parameters
                for inp in inputs:
                    base_cost += self._estimate_parameter_cost(inp['type'])
                    
                # Add execution cost estimate
                exec_cost = self._estimate_execution_cost(name, bytecode)
                
                costs[name] = base_cost + exec_cost
                
            return costs
            
        except Exception as e:
            logger.error(f"Error analyzing function costs: {str(e)}")
            return {}
            
    def _estimate_parameter_cost(self, param_type: str) -> int:
        """Estimate gas cost for parameter type"""
        # Basic cost estimates for different parameter types
        costs = {
            'uint': 3,
            'int': 3,
            'bool': 3,
            'address': 3,
            'bytes': 3,
            'string': 6,
            'array': 6
        }
        
        base_type = param_type.rstrip('0123456789[]')
        if '[]' in param_type:
            return costs.get('array', 3)
        return costs.get(base_type, 3)
        
    def _estimate_execution_cost(self, function_name: str, bytecode: str) -> int:
        """Estimate function execution cost"""
        try:
            # Simple heuristic based on bytecode analysis
            function_sig = Web3.keccak(text=function_name)[:4].hex()
            
            # Find function entry point
            entry_point = bytecode.find(function_sig)
            if entry_point == -1:
                return 1000  # Default cost
                
            # Count operations after entry point
            op_count = bytecode[entry_point:].count('5b')  # JUMPDEST opcode
            return op_count * 100  # Rough estimate
            
        except Exception as e:
            logger.error(f"Error estimating execution cost: {str(e)}")
            return 1000
            
    async def _generate_suggestions(
        self,
        contract_path: Path,
        profile: GasProfile
    ) -> List[str]:
        """Generate gas optimization suggestions"""
        suggestions = []
        
        # Check contract size
        if profile.bytecode_size > self.config.max_contract_size:
            suggestions.append(
                f"Contract size ({profile.bytecode_size} bytes) exceeds limit "
                f"({self.config.max_contract_size} bytes). Consider splitting into multiple contracts."
            )
            
        # Check storage slots
        if profile.storage_slots > 50:  # Arbitrary threshold
            suggestions.append(
                f"High number of storage slots ({profile.storage_slots}). "
                "Consider using packed storage or reducing state variables."
            )
            
        # Check function costs
        expensive_funcs = [
            (name, cost) for name, cost in profile.function_costs.items()
            if cost > 100000  # Arbitrary threshold
        ]
        if expensive_funcs:
            for name, cost in expensive_funcs:
                suggestions.append(
                    f"Function '{name}' has high gas cost ({cost} gas). "
                    "Consider optimizing or splitting functionality."
                )
                
        return suggestions
        
    async def _analyze_hot_spots(
        self,
        contract_path: Path,
        profile: GasProfile
    ) -> Dict[str, Any]:
        """Analyze gas usage hot spots"""
        try:
            hot_spots = {
                'storage_access': [],
                'loops': [],
                'complex_operations': []
            }
            
            # Analyze source code
            with open(contract_path) as f:
                source = f.read()
                
            # Find storage access patterns
            storage_pattern = r'(\w+)\s*=\s*'
            matches = defaultdict(int)
            for match in re.finditer(storage_pattern, source):
                var_name = match.group(1)
                matches[var_name] += 1
                
            # Add frequently accessed variables
            for var, count in matches.items():
                if count > 5:  # Arbitrary threshold
                    hot_spots['storage_access'].append({
                        'variable': var,
                        'access_count': count
                    })
                    
            # Find loops
            loop_pattern = r'(for|while)\s*\('
            for match in re.finditer(loop_pattern, source):
                line_no = source[:match.start()].count('\n') + 1
                hot_spots['loops'].append({
                    'type': match.group(1),
                    'line': line_no
                })
                
            # Find complex operations
            complex_ops = [
                (r'\*\*', 'exponentiation'),
                (r'keccak256', 'hashing'),
                (r'abi\.encode', 'abi encoding')
            ]
            for pattern, op_type in complex_ops:
                for match in re.finditer(pattern, source):
                    line_no = source[:match.start()].count('\n') + 1
                    hot_spots['complex_operations'].append({
                        'type': op_type,
                        'line': line_no
                    })
                    
            return hot_spots
            
        except Exception as e:
            logger.error(f"Error analyzing hot spots: {str(e)}")
            return {} 