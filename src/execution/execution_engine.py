import logging
from typing import Dict, Any, Optional, List, Tuple
from web3 import Web3
from web3.types import TxParams, TxReceipt
import time
import asyncio
from dataclasses import dataclass

from src.core.types import ExecutionResult, ExecutionStatus
from src.gas.gas_manager import GasManager
from src.validation.market_validator import MarketValidator

logger = logging.getLogger(__name__)

@dataclass
class TransactionMetrics:
    """Metrics for transaction execution performance tracking"""
    execution_time: float
    gas_used: int
    gas_price: int
    block_number: int
    confirmation_time: float
    retry_count: int = 0

class ExecutionEngine:
    """Handles transaction execution with gas management and market validation"""

    def __init__(
        self,
        web3: Web3,
        gas_manager: GasManager,
        market_validator: MarketValidator,
        config: Optional[Dict[str, Any]] = None
    ):
        self.web3 = web3
        self.gas_manager = gas_manager
        self.market_validator = market_validator
        self.config = config or {}
        self.max_retries = self.config.get('execution', {}).get('max_retries', 3)
        self.confirmation_blocks = self.config.get('execution', {}).get('confirmation_blocks', 2)
        self.metrics: List[TransactionMetrics] = []
        self.pending_transactions: Dict[str, float] = {}  # tx_hash -> submission_time

    async def execute_transaction(
        self,
        tx_params: TxParams,
        retry_count: int = 0,
        simulate_first: bool = True
    ) -> ExecutionResult:
        """Execute a transaction with retry logic and gas optimization"""
        start_time = time.time()
        
        try:
            # Validate market conditions before execution
            if not await self.market_validator.validate_conditions():
                return ExecutionResult(
                    status=ExecutionStatus.MARKET_CONDITIONS_CHANGED,
                    success=False,
                    error_message="Market conditions no longer favorable"
                )

            # Simulate transaction first if requested
            if simulate_first:
                simulation_result = await self.simulate_transaction(tx_params)
                if simulation_result.status != ExecutionStatus.SUCCESS:
                    return ExecutionResult(
                        status=ExecutionStatus.SIMULATION_FAILED,
                        success=False,
                        error_message=f"Transaction simulation failed: {simulation_result.error_message}"
                    )

            # Optimize gas settings
            optimized_gas = await self.gas_manager.optimize_gas_settings(tx_params)
            tx_params.update(optimized_gas)

            # Send transaction
            tx_hash = await self.web3.eth.send_transaction(tx_params)
            self.pending_transactions[tx_hash.hex()] = start_time
            
            # Wait for transaction receipt
            receipt = await self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            # Calculate metrics
            execution_time = time.time() - start_time
            
            if receipt.status == 1:
                # Track metrics for successful transactions
                self._track_metrics(TransactionMetrics(
                    execution_time=execution_time,
                    gas_used=receipt.gasUsed,
                    gas_price=tx_params.get('gasPrice', 0) or tx_params.get('maxFeePerGas', 0),
                    block_number=receipt.blockNumber,
                    confirmation_time=execution_time,
                    retry_count=retry_count
                ))
                
                # Remove from pending transactions
                if tx_hash.hex() in self.pending_transactions:
                    del self.pending_transactions[tx_hash.hex()]
                
                return ExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    success=True,
                    gas_used=receipt.gasUsed,
                    tx_hash=receipt.transactionHash.hex(),
                    execution_time=execution_time,
                    block_number=receipt.blockNumber
                )
            else:
                if retry_count < self.max_retries:
                    logger.warning(f"Transaction failed, retrying ({retry_count + 1}/{self.max_retries})")
                    return await self.execute_transaction(tx_params, retry_count + 1, simulate_first)
                else:
                    return ExecutionResult(
                        status=ExecutionStatus.EXECUTION_ERROR,
                        success=False,
                        error_message="Transaction reverted"
                    )

        except Exception as e:
            error_msg = f"Error executing transaction: {str(e)}"
            logger.error(error_msg)
            
            if retry_count < self.max_retries:
                # Adjust gas price for next retry if it's a gas-related issue
                if "underpriced" in str(e).lower() or "gas" in str(e).lower():
                    tx_params = await self._bump_gas_price(tx_params)
                
                logger.warning(f"Retrying transaction ({retry_count + 1}/{self.max_retries})")
                return await self.execute_transaction(tx_params, retry_count + 1, simulate_first)
            
            return ExecutionResult(
                status=ExecutionStatus.NETWORK_ERROR,
                success=False,
                error_message=error_msg
            )

    async def execute_batch(self, transactions: List[TxParams]) -> List[ExecutionResult]:
        """Execute multiple transactions concurrently"""
        tasks = [self.execute_transaction(tx) for tx in transactions]
        return await asyncio.gather(*tasks)

    async def simulate_transaction(self, tx_params: TxParams) -> ExecutionResult:
        """Simulate a transaction before execution to check for potential issues"""
        try:
            # Create a copy of tx_params for simulation
            sim_params = dict(tx_params)
            
            # Remove nonce for simulation if present
            if 'nonce' in sim_params:
                del sim_params['nonce']
            
            # Call the contract to simulate the transaction
            await self.web3.eth.call(sim_params)
            
            # Estimate gas to ensure it's executable
            gas_estimate = await self.estimate_gas(sim_params)
            
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                success=True,
                gas_used=gas_estimate
            )
        except Exception as e:
            logger.error(f"Transaction simulation failed: {str(e)}")
            return ExecutionResult(
                status=ExecutionStatus.SIMULATION_FAILED,
                success=False,
                error_message=str(e)
            )

    async def wait_for_transaction(self, tx_hash: str, timeout: int = 300) -> ExecutionResult:
        """Wait for a transaction to be confirmed and return its receipt"""
        try:
            start_time = time.time()
            receipt = await self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout)
            confirmation_time = time.time() - start_time
            
            if receipt.status == 1:
                return ExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    success=True,
                    gas_used=receipt.gasUsed,
                    tx_hash=receipt.transactionHash.hex(),
                    execution_time=confirmation_time,
                    block_number=receipt.blockNumber
                )
            else:
                return ExecutionResult(
                    status=ExecutionStatus.EXECUTION_ERROR,
                    success=False,
                    error_message="Transaction reverted"
                )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.NETWORK_ERROR,
                success=False,
                error_message=str(e)
            )

    async def estimate_gas(self, tx_params: TxParams) -> int:
        """Estimate gas for a transaction"""
        try:
            return await self.web3.eth.estimate_gas(tx_params)
        except Exception as e:
            logger.error(f"Error estimating gas: {str(e)}")
            raise

    async def _bump_gas_price(self, tx_params: TxParams, multiplier: float = 1.2) -> TxParams:
        """Bump gas price for a transaction that might be stuck"""
        new_params = dict(tx_params)
        
        # Handle both legacy and EIP-1559 transactions
        if 'gasPrice' in new_params:
            new_params['gasPrice'] = int(new_params['gasPrice'] * multiplier)
        elif 'maxFeePerGas' in new_params:
            new_params['maxFeePerGas'] = int(new_params['maxFeePerGas'] * multiplier)
            new_params['maxPriorityFeePerGas'] = int(new_params['maxPriorityFeePerGas'] * multiplier)
        
        return new_params

    def _track_metrics(self, metrics: TransactionMetrics) -> None:
        """Track transaction execution metrics for performance analysis"""
        self.metrics.append(metrics)
        
        # Keep only the last 100 metrics to avoid memory issues
        if len(self.metrics) > 100:
            self.metrics = self.metrics[-100:]

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get aggregated performance metrics for executed transactions"""
        if not self.metrics:
            return {
                "avg_execution_time": 0,
                "avg_gas_used": 0,
                "avg_confirmation_time": 0,
                "retry_rate": 0,
                "success_rate": 1.0,
                "total_transactions": 0
            }
        
        total = len(self.metrics)
        retried = sum(1 for m in self.metrics if m.retry_count > 0)
        
        return {
            "avg_execution_time": sum(m.execution_time for m in self.metrics) / total,
            "avg_gas_used": sum(m.gas_used for m in self.metrics) / total,
            "avg_confirmation_time": sum(m.confirmation_time for m in self.metrics) / total,
            "retry_rate": retried / total if total > 0 else 0,
            "success_rate": 1.0,  # All tracked metrics are from successful txs
            "total_transactions": total
        }

    async def cleanup(self) -> None:
        """Clean up resources and check for any pending transactions"""
        if self.pending_transactions:
            logger.warning(f"There are {len(self.pending_transactions)} pending transactions during cleanup")
            
            # Check status of pending transactions
            for tx_hash, submission_time in list(self.pending_transactions.items()):
                try:
                    receipt = await self.web3.eth.get_transaction_receipt(tx_hash)
                    if receipt:
                        logger.info(f"Transaction {tx_hash} was mined during cleanup")
                        del self.pending_transactions[tx_hash]
                except Exception as e:
                    logger.error(f"Error checking transaction {tx_hash} during cleanup: {str(e)}") 