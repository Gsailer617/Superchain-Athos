"""Gas optimization manager for smart contracts"""

from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from web3 import Web3
from eth_typing import ChecksumAddress
import solcx
from solcx import compile_source, compile_files
import networkx as nx
from collections import defaultdict
import re

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Gas optimization configuration"""
    solc_version: str = "0.8.17"
    optimization_runs: int = 200
    analyze_storage_layout: bool = True
    analyze_function_costs: bool = True
    analyze_bytecode: bool = True
    max_contract_size: int = 24576  # bytes

@dataclass
class GasProfile:
    """Gas usage profile"""
    deployment_cost: int
    function_costs: Dict[str, int]
    storage_slots: int
    bytecode_size: int
    optimization_suggestions: List[str]

class GasOptimizer:
    """Smart contract gas optimizer"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self._init_compiler()
    
    def _init_compiler(self):
        """Initialize solidity compiler"""
        solcx.install_solc(self.config.solc_version)
        solcx.set_solc_version(self.config.solc_version)
    
    def analyze_contract(
        self,
        contract_path: Path,
        constructor_args: Optional[List] = None
    ) -> GasProfile:
        """Analyze contract for gas optimization"""
        
        # Compile contract with different optimization settings
        profiles = []
        for runs in [0, 200, 1000]:
            profile = self._analyze_with_runs(
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
        best_profile.optimization_suggestions.extend(
            self._suggest_optimizations(contract_path, best_profile)
        )
        
        return best_profile
    
    def _analyze_with_runs(
        self,
        contract_path: Path,
        optimization_runs: int,
        constructor_args: Optional[List] = None
    ) -> GasProfile:
        """Analyze contract with specific optimization runs"""
        
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
        function_costs = self._analyze_function_costs(
            contract_data['abi'],
            contract_data['bin']
        )
        
        # Analyze storage layout
        storage_slots = self._analyze_storage_layout(
            contract_data.get('storage-layout', {})
        )
        
        # Analyze bytecode
        bytecode_size = len(contract_data['bin']) // 2
        
        return GasProfile(
            deployment_cost=deployment_cost,
            function_costs=function_costs,
            storage_slots=storage_slots,
            bytecode_size=bytecode_size,
            optimization_suggestions=[]
        )
    
    def _estimate_deployment_cost(
        self,
        bytecode: str,
        constructor_args: Optional[List] = None
    ) -> int:
        """Estimate contract deployment cost"""
        base_cost = 21000  # Base transaction cost
        
        # Cost for bytecode
        bytecode_cost = 200 * (len(bytecode) // 2)  # 200 gas per byte
        
        # Cost for constructor args
        args_cost = 0
        if constructor_args:
            for arg in constructor_args:
                if isinstance(arg, str):
                    args_cost += 68 * len(arg)  # String storage cost
                else:
                    args_cost += 20000  # Default storage cost
        
        return base_cost + bytecode_cost + args_cost
    
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
            
            # Calculate base cost
            base_cost = 21000  # Base transaction cost
            
            # Add cost for input parameters
            for input_param in item.get('inputs', []):
                base_cost += self._estimate_param_cost(input_param['type'])
            
            # Add cost for output parameters
            for output_param in item.get('outputs', []):
                base_cost += self._estimate_param_cost(output_param['type'])
            
            # Add storage operation costs
            if item.get('stateMutability') == 'view':
                base_cost += 100  # View function overhead
            else:
                base_cost += 5000  # State-changing function overhead
            
            costs[item['name']] = base_cost
        
        return costs
    
    def _analyze_storage_layout(
        self,
        storage_layout: Dict[str, Any]
    ) -> int:
        """Analyze contract storage layout"""
        if not storage_layout:
            return 0
        
        # Build storage slot graph
        slot_graph = nx.Graph()
        
        for var in storage_layout.get('storage', []):
            slot = var['slot']
            size = var['size']
            slot_graph.add_node(
                slot,
                size=size,
                variables=[var['label']]
            )
        
        return len(slot_graph.nodes)
    
    def _suggest_optimizations(
        self,
        contract_path: Path,
        profile: GasProfile
    ) -> List[str]:
        """Suggest gas optimizations"""
        suggestions = []
        
        # Read contract source
        with open(contract_path) as f:
            source = f.read()
        
        # Check contract size
        if profile.bytecode_size > self.config.max_contract_size:
            suggestions.append(
                f"Contract size ({profile.bytecode_size} bytes) exceeds maximum "
                f"({self.config.max_contract_size} bytes). Consider splitting into multiple contracts."
            )
        
        # Check storage patterns
        if 'mapping' in source:
            suggestions.append(
                "Consider using nested mappings instead of mapping to structs "
                "to save gas on storage operations."
            )
        
        if 'string' in source:
            suggestions.append(
                "Consider using bytes32 instead of string for fixed-length data "
                "to save gas on storage."
            )
        
        # Check function patterns
        if 'view' not in source and 'pure' not in source:
            suggestions.append(
                "Consider marking constant functions as view/pure "
                "to save gas on execution."
            )
        
        # Check variable patterns
        if 'uint256' in source:
            suggestions.append(
                "Consider using uint128/uint96/uint64 for smaller numbers "
                "to pack multiple variables into a single storage slot."
            )
        
        # Check loop patterns
        if 'for' in source:
            suggestions.append(
                "Consider caching array length outside loops "
                "to save gas on each iteration."
            )
        
        # Check event patterns
        if 'event' in source and 'indexed' not in source:
            suggestions.append(
                "Consider adding indexed parameters to events "
                "to optimize gas cost for event emission."
            )
        
        return suggestions
    
    @staticmethod
    def _estimate_param_cost(param_type: str) -> int:
        """Estimate gas cost for parameter type"""
        costs = {
            'uint256': 20000,
            'uint128': 10000,
            'uint96': 8000,
            'uint64': 6000,
            'uint32': 4000,
            'uint16': 3000,
            'uint8': 2000,
            'bool': 3000,
            'address': 20000,
            'string': 40000,
            'bytes': 40000,
            'bytes32': 20000
        }
        
        # Handle arrays
        if '[' in param_type:
            base_type = param_type[:param_type.index('[')]
            base_cost = costs.get(base_type, 20000)
            return base_cost * 2  # Double cost for arrays
        
        return costs.get(param_type, 20000)
    
    def optimize_contract(
        self,
        contract_path: Path,
        output_path: Optional[Path] = None
    ) -> Tuple[str, GasProfile]:
        """Optimize contract and return optimized source"""
        # Analyze current gas usage
        original_profile = self.analyze_contract(contract_path)
        
        # Read contract source
        with open(contract_path) as f:
            source = f.read()
        
        # Apply optimizations
        optimized_source = self._apply_optimizations(source)
        
        # Save optimized contract
        if output_path:
            with open(output_path, 'w') as f:
                f.write(optimized_source)
        
        # Analyze optimized gas usage
        optimized_profile = self._analyze_source(
            optimized_source,
            contract_path.stem
        )
        
        return optimized_source, optimized_profile
    
    def _apply_optimizations(self, source: str) -> str:
        """Apply gas optimizations to contract source"""
        # Replace uint256 with smaller types where possible
        source = self._optimize_uint_types(source)
        
        # Replace strings with bytes32 where possible
        source = self._optimize_strings(source)
        
        # Optimize storage layout
        source = self._optimize_storage_layout(source)
        
        # Optimize loops
        source = self._optimize_loops(source)
        
        # Add view/pure modifiers
        source = self._optimize_function_modifiers(source)
        
        return source
    
    def _optimize_uint_types(self, source: str) -> str:
        """Optimize uint type usage"""
        # Find uint256 variables that can be smaller
        pattern = r'uint256\s+(\w+)\s*;'
        matches = re.finditer(pattern, source)
        
        for match in matches:
            var_name = match.group(1)
            # Check variable usage to determine if smaller type is possible
            if self._can_use_smaller_uint(source, var_name):
                source = source.replace(
                    f'uint256 {var_name}',
                    f'uint64 {var_name}'  # Use uint64 as safe default
                )
        
        return source
    
    def _optimize_strings(self, source: str) -> str:
        """Optimize string usage"""
        # Find string variables that can be bytes32
        pattern = r'string\s+(\w+)\s*;'
        matches = re.finditer(pattern, source)
        
        for match in matches:
            var_name = match.group(1)
            # Check if string is used with fixed length
            if self._is_fixed_length_string(source, var_name):
                source = source.replace(
                    f'string {var_name}',
                    f'bytes32 {var_name}'
                )
        
        return source
    
    def _optimize_storage_layout(self, source: str) -> str:
        """Optimize storage variable layout"""
        # Parse storage variables
        pattern = r'(uint\d+|bool|address|bytes\d*)\s+(\w+)\s*;'
        variables = re.finditer(pattern, source)
        
        # Group variables by size
        size_groups = defaultdict(list)
        for match in variables:
            var_type = match.group(1)
            var_name = match.group(2)
            size = self._get_type_size(var_type)
            size_groups[size].append((var_type, var_name))
        
        # Reorder variables for optimal packing
        ordered_vars = []
        for size in sorted(size_groups.keys(), reverse=True):
            ordered_vars.extend(size_groups[size])
        
        # Replace original declarations
        result = source
        for var_type, var_name in ordered_vars:
            pattern = f'{var_type}\\s+{var_name}\\s*;'
            result = re.sub(
                pattern,
                f'{var_type} {var_name};',
                result,
                count=1
            )
        
        return result
    
    def _optimize_loops(self, source: str) -> str:
        """Optimize loop patterns"""
        # Cache array lengths
        pattern = r'for\s*\(\s*\w+\s+\w+\s*=\s*0\s*;\s*\w+\s*<\s*(\w+)\.length\s*;'
        matches = re.finditer(pattern, source)
        
        result = source
        for match in matches:
            array_name = match.group(1)
            length_var = f'{array_name}Length'
            
            # Add length caching
            result = result.replace(
                match.group(0),
                f'uint256 {length_var} = {array_name}.length;\n'
                f'for (uint256 i = 0; i < {length_var};'
            )
        
        return result
    
    def _optimize_function_modifiers(self, source: str) -> str:
        """Optimize function modifiers"""
        # Find functions without modifiers that don't modify state
        pattern = r'function\s+(\w+)\s*\([^)]*\)(?!\s*view|\s*pure)[^{]*{'
        matches = re.finditer(pattern, source)
        
        result = source
        for match in matches:
            func_name = match.group(1)
            if self._is_view_function(source, func_name):
                result = result.replace(
                    match.group(0),
                    match.group(0).replace(
                        f'function {func_name}',
                        f'function {func_name} view'
                    )
                )
        
        return result
    
    @staticmethod
    def _can_use_smaller_uint(source: str, var_name: str) -> bool:
        """Check if variable can use smaller uint type"""
        # Simple heuristic: check if variable is used in large number operations
        large_number_pattern = r'[0-9]{10,}|10\*\*\d\d+'
        var_usage = re.findall(
            f'{var_name}\\s*[=<>+\\-*/]\\s*({large_number_pattern})',
            source
        )
        return not bool(var_usage)
    
    @staticmethod
    def _is_fixed_length_string(source: str, var_name: str) -> bool:
        """Check if string variable has fixed length usage"""
        # Check if string is only assigned string literals
        assignments = re.findall(
            f'{var_name}\\s*=\\s*"[^"]*"',
            source
        )
        return bool(assignments) and all(
            len(a.split('"')[1]) <= 32
            for a in assignments
        )
    
    @staticmethod
    def _get_type_size(var_type: str) -> int:
        """Get size in bits for variable type"""
        if var_type.startswith('uint'):
            return int(var_type[4:]) if len(var_type) > 4 else 256
        elif var_type == 'bool':
            return 8
        elif var_type == 'address':
            return 160
        elif var_type.startswith('bytes'):
            return int(var_type[5:]) * 8 if len(var_type) > 5 else 256
        return 256
    
    @staticmethod
    def _is_view_function(source: str, func_name: str) -> bool:
        """Check if function can be marked as view"""
        # Find function body
        pattern = f'function\\s+{func_name}[^{{]*{{([^}}]*)}}'
        match = re.search(pattern, source)
        if not match:
            return False
        
        body = match.group(1)
        
        # Check for state-changing operations
        state_changing = [
            'storage',
            'emit',
            'selfdestruct',
            'transfer',
            'send',
            'call',
            'delegatecall',
            'staticcall'
        ]
        
        return not any(op in body for op in state_changing) 