"""Integration module for connecting gas optimization with execution components"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from web3 import Web3

from src.gas.gas_manager import GasManager
from src.gas.enhanced_optimizer import EnhancedGasOptimizer
from src.execution import ExecutionEngine, TransactionBuilder, ExecutionManager

logger = logging.getLogger(__name__)

class GasExecutionIntegrator:
    """Integrates gas optimization with execution components"""
    
    def __init__(
        self,
        web3: Web3,
        config: Optional[Dict[str, Any]] = None,
        execution_manager: Optional[ExecutionManager] = None,
        gas_optimizer: Optional[EnhancedGasOptimizer] = None
    ):
        """Initialize the gas execution integrator
        
        Args:
            web3: Web3 instance for blockchain interaction
            config: Configuration dictionary
            execution_manager: Optional ExecutionManager instance
            gas_optimizer: Optional EnhancedGasOptimizer instance
        """
        self.web3 = web3
        self.config = config or {}
        
        # Create or use provided gas optimizer
        if gas_optimizer:
            self.gas_optimizer = gas_optimizer
        else:
            gas_manager = GasManager(web3, config)
            self.gas_optimizer = EnhancedGasOptimizer(web3, config, gas_manager)
        
        # Create or use provided execution manager
        if execution_manager:
            self.execution_manager = execution_manager
        else:
            self.execution_manager = ExecutionManager(web3, config)
            
        # Connect gas optimizer to execution components
        self._connect_components()
        
    def _connect_components(self):
        """Connect gas optimization to execution components"""
        # Get execution engine and transaction builder from manager
        execution_engine = self.execution_manager.execution_engine
        transaction_builder = self.execution_manager.transaction_builder
        
        # Set gas optimizer in execution engine
        if hasattr(execution_engine, 'set_gas_optimizer'):
            execution_engine.set_gas_optimizer(self.gas_optimizer)
        else:
            logger.warning("ExecutionEngine does not support setting gas optimizer")
            
        # Set gas optimizer in transaction builder
        if hasattr(transaction_builder, 'set_gas_optimizer'):
            transaction_builder.set_gas_optimizer(self.gas_optimizer)
        else:
            logger.warning("TransactionBuilder does not support setting gas optimizer")
    
    async def execute_with_optimized_gas(
        self,
        tx_params: Dict[str, Any],
        optimization_mode: str = 'normal',
        wait_for_receipt: bool = True,
        timeout: int = 120
    ) -> Dict[str, Any]:
        """Execute a transaction with optimized gas settings
        
        Args:
            tx_params: Transaction parameters
            optimization_mode: Gas optimization mode ('economy', 'normal', 'performance', 'urgent')
            wait_for_receipt: Whether to wait for transaction receipt
            timeout: Timeout in seconds for waiting for receipt
            
        Returns:
            Dictionary with transaction result
        """
        try:
            # Optimize gas settings
            optimized_params = await self.gas_optimizer.optimize_gas_settings(tx_params)
            
            # Merge optimized gas settings with original params
            tx_params.update(optimized_params)
            
            # Execute transaction
            result = await self.execution_manager.execute_transaction(
                tx_params,
                wait_for_receipt=wait_for_receipt,
                timeout=timeout
            )
            
            # Add gas optimization metrics to result
            result['gas_optimization'] = {
                'mode': optimization_mode,
                'estimated_savings': await self._calculate_savings(tx_params, result)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing transaction with optimized gas: {str(e)}")
            raise
    
    async def _calculate_savings(
        self,
        tx_params: Dict[str, Any],
        result: Dict[str, Any]
    ) -> float:
        """Calculate gas savings from optimization
        
        Args:
            tx_params: Transaction parameters with optimized gas
            result: Transaction execution result
            
        Returns:
            Estimated savings percentage
        """
        try:
            # Get original gas price if available
            original_gas_price = tx_params.get('original_gas_price', 0)
            if original_gas_price == 0:
                # Try to get network average if original not provided
                network_stats = await self.gas_optimizer.get_optimization_stats()
                original_gas_price = network_stats.get('avg_gas_price', 0)
                
            # Get actual gas price used
            receipt = result.get('receipt', {})
            effective_gas_price = receipt.get('effectiveGasPrice', 0)
            
            if original_gas_price > 0 and effective_gas_price > 0:
                return (original_gas_price - effective_gas_price) / original_gas_price
                
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating gas savings: {str(e)}")
            return 0.0
    
    async def estimate_transaction_cost(
        self,
        tx_params: Dict[str, Any],
        optimization_mode: str = 'normal'
    ) -> Dict[str, Any]:
        """Estimate transaction cost with optimized gas
        
        Args:
            tx_params: Transaction parameters
            optimization_mode: Gas optimization mode
            
        Returns:
            Dictionary with cost estimation details
        """
        try:
            # Set optimization mode
            self.gas_optimizer.mode = optimization_mode
            
            # Get network congestion
            congestion = await self.gas_optimizer.get_network_congestion()
            
            # Optimize gas settings
            optimized_params = await self.gas_optimizer.optimize_gas_settings(tx_params)
            
            # Estimate gas cost
            gas_cost = await self.gas_optimizer.estimate_gas_cost({**tx_params, **optimized_params})
            
            # Estimate wait time
            wait_time = self.gas_optimizer._estimate_wait_time(congestion, optimization_mode)
            
            return {
                'estimated_cost_wei': gas_cost,
                'estimated_cost_eth': gas_cost / 1e18 if gas_cost > 0 else 0,
                'estimated_wait_time': wait_time,
                'network_congestion': congestion,
                'optimization_mode': optimization_mode,
                'gas_params': optimized_params
            }
            
        except Exception as e:
            logger.error(f"Error estimating transaction cost: {str(e)}")
            return {
                'error': str(e),
                'estimated_cost_wei': 0,
                'estimated_cost_eth': 0,
                'estimated_wait_time': 0,
                'network_congestion': 0,
                'optimization_mode': optimization_mode
            }
    
    async def cleanup(self):
        """Clean up resources"""
        await self.gas_optimizer.cleanup()
        await self.execution_manager.cleanup() 