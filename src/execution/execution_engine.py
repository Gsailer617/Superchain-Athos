import logging
from typing import Dict, Any, Optional
from web3 import Web3
from web3.types import TxParams, TxReceipt

from src.core.types import ExecutionResult, ExecutionStatus
from src.gas.gas_manager import GasManager
from src.validation.market_validator import MarketValidator

logger = logging.getLogger(__name__)

class ExecutionEngine:
    """Handles transaction execution with gas management and market validation"""

    def __init__(
        self,
        web3: Web3,
        gas_manager: GasManager,
        market_validator: MarketValidator
    ):
        self.web3 = web3
        self.gas_manager = gas_manager
        self.market_validator = market_validator
        self.max_retries = 3

    async def execute_transaction(
        self,
        tx_params: TxParams,
        retry_count: int = 0
    ) -> ExecutionResult:
        """Execute a transaction with retry logic and gas optimization"""
        try:
            # Validate market conditions before execution
            if not await self.market_validator.validate_conditions():
                return ExecutionResult(
                    status=ExecutionStatus.MARKET_CONDITIONS_CHANGED,
                    success=False,
                    error_message="Market conditions no longer favorable"
                )

            # Optimize gas settings
            optimized_gas = await self.gas_manager.optimize_gas_settings(tx_params)
            tx_params.update(optimized_gas)

            # Send transaction
            tx_hash = await self.web3.eth.send_transaction(tx_params)
            receipt = await self.web3.eth.wait_for_transaction_receipt(tx_hash)

            if receipt.status == 1:
                return ExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    success=True,
                    gas_used=receipt.gasUsed,
                    tx_hash=receipt.transactionHash.hex()
                )
            else:
                if retry_count < self.max_retries:
                    logger.warning(f"Transaction failed, retrying ({retry_count + 1}/{self.max_retries})")
                    return await self.execute_transaction(tx_params, retry_count + 1)
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
                logger.warning(f"Retrying transaction ({retry_count + 1}/{self.max_retries})")
                return await self.execute_transaction(tx_params, retry_count + 1)
            
            return ExecutionResult(
                status=ExecutionStatus.NETWORK_ERROR,
                success=False,
                error_message=error_msg
            )

    async def estimate_gas(self, tx_params: TxParams) -> int:
        """Estimate gas for a transaction"""
        try:
            return await self.web3.eth.estimate_gas(tx_params)
        except Exception as e:
            logger.error(f"Error estimating gas: {str(e)}")
            raise 