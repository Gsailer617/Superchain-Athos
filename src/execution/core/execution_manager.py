"""
Execution Manager - Unified interface for transaction building and execution
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Union, Tuple
from web3 import Web3
from web3.types import TxParams, TxReceipt

from src.core.types import (
    ExecutionResult, 
    ExecutionStatus, 
    OpportunityType, 
    FlashLoanOpportunityType
)
from src.execution.execution_engine import ExecutionEngine
from src.execution.transaction_builder import TransactionBuilder, CrossChainTransactionBuilder
from src.gas.gas_manager import GasManager
from src.validation.market_validator import MarketValidator

logger = logging.getLogger(__name__)

class ExecutionManager:
    """
    Manages the execution of transactions by coordinating between
    transaction builders and the execution engine.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the execution manager with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize Web3 connection
        alchemy_key = config.get('alchemy_key')
        if not alchemy_key:
            raise ValueError("Alchemy API key not provided in config")
            
        self.web3 = Web3(Web3.HTTPProvider(
            f"https://base-mainnet.g.alchemy.com/v2/{alchemy_key}",
            request_kwargs={
                'timeout': 30,
                'headers': {'User-Agent': 'FlashingBase/1.0.0'}
            }
        ))
        
        if not self.web3.is_connected():
            raise ValueError("Failed to connect to Base mainnet via Alchemy")
        
        # Initialize components
        self.gas_manager = GasManager(self.web3, config)
        self.market_validator = MarketValidator(self.web3, config)
        
        self.transaction_builder = TransactionBuilder(
            config, 
            gas_manager=self.gas_manager,
            market_validator=self.market_validator
        )
        
        self.cross_chain_builder = CrossChainTransactionBuilder(config)
        
        self.execution_engine = ExecutionEngine(
            self.web3,
            self.gas_manager,
            self.market_validator,
            config
        )
        
        # Performance tracking
        self.execution_history: List[ExecutionResult] = []
        self.max_history_size = config.get('max_history_size', 100)
        
    async def execute_opportunity(
        self, 
        opportunity: Union[OpportunityType, FlashLoanOpportunityType],
        use_eip1559: bool = True,
        simulate_first: bool = True
    ) -> ExecutionResult:
        """
        Build and execute a transaction for an opportunity
        
        Args:
            opportunity: The opportunity to execute
            use_eip1559: Whether to use EIP-1559 transaction format
            simulate_first: Whether to simulate the transaction before execution
            
        Returns:
            ExecutionResult with the result of the execution
        """
        try:
            # Build transaction
            tx_params = await self.transaction_builder.build_transaction(
                opportunity,
                use_eip1559=use_eip1559
            )
            
            if not tx_params:
                return ExecutionResult(
                    status=ExecutionStatus.INVALID_OPPORTUNITY,
                    success=False,
                    error_message="Failed to build transaction"
                )
            
            # Sign transaction
            signed_tx = await self.transaction_builder.sign_transaction(tx_params)
            if not signed_tx:
                return ExecutionResult(
                    status=ExecutionStatus.EXECUTION_ERROR,
                    success=False,
                    error_message="Failed to sign transaction"
                )
            
            # Execute transaction
            result = await self.execution_engine.execute_transaction(
                tx_params,
                simulate_first=simulate_first
            )
            
            # Track execution history
            self._track_execution(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing opportunity: {str(e)}")
            return ExecutionResult(
                status=ExecutionStatus.EXECUTION_ERROR,
                success=False,
                error_message=str(e)
            )
    
    async def execute_cross_chain_opportunity(
        self,
        opportunity: Dict[str, Any],
        simulate_first: bool = True
    ) -> Tuple[ExecutionResult, Optional[ExecutionResult]]:
        """
        Execute a cross-chain opportunity
        
        Args:
            opportunity: Cross-chain opportunity details
            simulate_first: Whether to simulate transactions before execution
            
        Returns:
            Tuple of (source_result, target_result)
        """
        try:
            # Build cross-chain transactions
            tx_result = await self.cross_chain_builder.build_cross_chain_transaction(opportunity)
            
            if not tx_result.success or not tx_result.source_tx:
                return (
                    ExecutionResult(
                        status=ExecutionStatus.INVALID_OPPORTUNITY,
                        success=False,
                        error_message=tx_result.error or "Failed to build cross-chain transaction"
                    ),
                    None
                )
            
            # Get source chain Web3 connection
            source_web3 = self.cross_chain_builder.web3_connections.get(opportunity['source_chain'])
            if not source_web3:
                return (
                    ExecutionResult(
                        status=ExecutionStatus.NETWORK_ERROR,
                        success=False,
                        error_message=f"No Web3 connection for {opportunity['source_chain']}"
                    ),
                    None
                )
            
            # Create execution engine for source chain
            source_engine = ExecutionEngine(
                source_web3,
                self.gas_manager,
                self.market_validator,
                self.config
            )
            
            # Execute source transaction
            source_result = await source_engine.execute_transaction(
                tx_result.source_tx,
                simulate_first=simulate_first
            )
            
            # Track execution
            self._track_execution(source_result)
            
            # If source transaction failed or no target transaction, return early
            if not source_result.success or not tx_result.target_tx:
                return (source_result, None)
            
            # Get target chain Web3 connection
            target_web3 = self.cross_chain_builder.web3_connections.get(opportunity['target_chain'])
            if not target_web3:
                return (
                    source_result,
                    ExecutionResult(
                        status=ExecutionStatus.NETWORK_ERROR,
                        success=False,
                        error_message=f"No Web3 connection for {opportunity['target_chain']}"
                    )
                )
            
            # Create execution engine for target chain
            target_engine = ExecutionEngine(
                target_web3,
                self.gas_manager,
                self.market_validator,
                self.config
            )
            
            # Execute target transaction
            target_result = await target_engine.execute_transaction(
                tx_result.target_tx,
                simulate_first=simulate_first
            )
            
            # Track execution
            self._track_execution(target_result)
            
            return (source_result, target_result)
            
        except Exception as e:
            logger.error(f"Error executing cross-chain opportunity: {str(e)}")
            return (
                ExecutionResult(
                    status=ExecutionStatus.EXECUTION_ERROR,
                    success=False,
                    error_message=str(e)
                ),
                None
            )
    
    async def execute_batch(
        self,
        opportunities: List[Union[OpportunityType, FlashLoanOpportunityType]],
        use_eip1559: bool = True,
        simulate_first: bool = True
    ) -> List[ExecutionResult]:
        """
        Execute multiple opportunities in parallel
        
        Args:
            opportunities: List of opportunities to execute
            use_eip1559: Whether to use EIP-1559 transaction format
            simulate_first: Whether to simulate transactions before execution
            
        Returns:
            List of execution results
        """
        tasks = [
            self.execute_opportunity(opp, use_eip1559, simulate_first)
            for opp in opportunities
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to ExecutionResult
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(
                    ExecutionResult(
                        status=ExecutionStatus.EXECUTION_ERROR,
                        success=False,
                        error_message=str(result)
                    )
                )
            else:
                processed_results.append(result)
                
        return processed_results
    
    def _track_execution(self, result: ExecutionResult) -> None:
        """Track execution history"""
        self.execution_history.append(result)
        
        # Limit history size
        if len(self.execution_history) > self.max_history_size:
            self.execution_history = self.execution_history[-self.max_history_size:]
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {
                "total_executions": 0,
                "success_rate": 0,
                "avg_gas_used": 0,
                "avg_execution_time": 0
            }
        
        total = len(self.execution_history)
        successful = sum(1 for r in self.execution_history if r.success)
        
        # Calculate averages for successful transactions
        successful_txs = [r for r in self.execution_history if r.success]
        avg_gas = sum(r.gas_used or 0 for r in successful_txs) / max(len(successful_txs), 1)
        avg_time = sum(r.execution_time or 0 for r in successful_txs) / max(len(successful_txs), 1)
        
        return {
            "total_executions": total,
            "success_rate": successful / total if total > 0 else 0,
            "avg_gas_used": avg_gas,
            "avg_execution_time": avg_time,
            "engine_metrics": self.execution_engine.get_performance_metrics()
        }
    
    async def cleanup(self) -> None:
        """Clean up resources"""
        await self.execution_engine.cleanup()
        await self.transaction_builder.cleanup()
        await self.cross_chain_builder.cleanup() 