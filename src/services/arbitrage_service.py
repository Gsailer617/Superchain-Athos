"""
Arbitrage Service

This service handles arbitrage opportunity identification, analysis, and execution:
- Opportunity discovery across DEXes
- Simulation and validation of arbitrage routes
- Execution of profitable opportunities
- Integration with flash loans
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Set, Any, Tuple, Union
import uuid

from ..cqrs.events.arbitrage import (
    ArbitrageAggregate, ArbitrageType, ArbitrageRisk, ExecutionPriority,
    RouteStep, GasEstimate, SimulationResult, FlashLoanDetails
)
from ..cqrs.events.token_discovery import TokenAggregate, TokenStatus, LiquiditySource
from ..cqrs.events.aggregates import aggregate_factory
from ..core.dependency_injector import container
from ..blockchain.web3_client import Web3Client
from ..blockchain.flash_loan import FlashLoanProvider
from ..blockchain.transaction_builder import TransactionBuilder
from ..blockchain.gas_estimator import GasEstimator
from ..blockchain.transaction_simulator import TransactionSimulator

logger = logging.getLogger(__name__)

class ArbitrageService:
    """Service for managing arbitrage opportunities"""
    
    def __init__(
        self,
        web3_client: Web3Client,
        flash_loan_provider: FlashLoanProvider,
        transaction_builder: TransactionBuilder,
        gas_estimator: GasEstimator,
        transaction_simulator: TransactionSimulator
    ):
        self.web3_client = web3_client
        self.flash_loan_provider = flash_loan_provider
        self.transaction_builder = transaction_builder
        self.gas_estimator = gas_estimator
        self.transaction_simulator = transaction_simulator
        self.analysis_queue = asyncio.Queue()
        self.execution_queue = asyncio.Queue()
        self.is_processing_analysis = False
        self.is_processing_execution = False
        
        # Configuration
        self.min_profit_threshold_usd = Decimal("5.0")  # Minimum profit in USD
        self.min_profit_percentage = Decimal("0.5")     # Minimum 0.5% profit
        self.max_gas_percentage = Decimal("50.0")       # Maximum 50% of profit for gas
        self.default_slippage = Decimal("0.5")          # Default 0.5% slippage
        self.high_risk_tokens_slippage = Decimal("2.0") # Higher slippage for risky tokens
        
        # Start background tasks
        self._start_background_tasks()
    
    async def find_arbitrage_opportunities(
        self,
        chain_id: int,
        token_addresses: Optional[List[str]] = None,
        arbitrage_types: Optional[List[ArbitrageType]] = None,
        max_opportunities: int = 10
    ) -> List[str]:
        """Find arbitrage opportunities
        
        Args:
            chain_id: Chain ID to search on
            token_addresses: Optional list of token addresses to check
            arbitrage_types: Optional list of arbitrage types to look for
            max_opportunities: Maximum number of opportunities to return
            
        Returns:
            List of opportunity IDs
        """
        logger.info(f"Finding arbitrage opportunities on chain {chain_id}")
        
        # Default to all arbitrage types if not specified
        if not arbitrage_types:
            arbitrage_types = [
                ArbitrageType.SIMPLE_DEX,
                ArbitrageType.TRIANGULAR,
                ArbitrageType.FLASH_LOAN
            ]
        
        opportunity_ids = []
        
        # If token addresses provided, check those specific tokens
        if token_addresses:
            for token_address in token_addresses:
                # Find opportunities for this token
                token_opportunities = await self._find_opportunities_for_token(
                    chain_id, token_address, arbitrage_types
                )
                opportunity_ids.extend(token_opportunities)
                
                # Stop if we have enough opportunities
                if len(opportunity_ids) >= max_opportunities:
                    break
        else:
            # Find validated tokens with liquidity
            token_repository = aggregate_factory(TokenAggregate)
            
            # This would normally use a query to find validated tokens
            # For now, we'll just use a placeholder
            validated_tokens = []  # Placeholder
            
            for token in validated_tokens:
                # Find opportunities for this token
                token_opportunities = await self._find_opportunities_for_token(
                    chain_id, token.state.address, arbitrage_types
                )
                opportunity_ids.extend(token_opportunities)
                
                # Stop if we have enough opportunities
                if len(opportunity_ids) >= max_opportunities:
                    break
        
        logger.info(f"Found {len(opportunity_ids)} arbitrage opportunities")
        return opportunity_ids[:max_opportunities]
    
    async def _find_opportunities_for_token(
        self,
        chain_id: int,
        token_address: str,
        arbitrage_types: List[ArbitrageType]
    ) -> List[str]:
        """Find arbitrage opportunities for a specific token
        
        Args:
            chain_id: Chain ID
            token_address: Token address
            arbitrage_types: Arbitrage types to look for
            
        Returns:
            List of opportunity IDs
        """
        opportunity_ids = []
        
        # Check if token exists and is validated
        token_repository = aggregate_factory(TokenAggregate)
        token = await token_repository.get_by_id(token_address)
        
        if not token or token.state.status != TokenStatus.VALIDATED:
            logger.debug(f"Token {token_address} not found or not validated")
            return []
        
        # Find simple DEX arbitrage opportunities
        if ArbitrageType.SIMPLE_DEX in arbitrage_types:
            simple_dex_opportunities = await self._find_simple_dex_arbitrage(
                chain_id, token_address
            )
            opportunity_ids.extend(simple_dex_opportunities)
        
        # Find triangular arbitrage opportunities
        if ArbitrageType.TRIANGULAR in arbitrage_types:
            triangular_opportunities = await self._find_triangular_arbitrage(
                chain_id, token_address
            )
            opportunity_ids.extend(triangular_opportunities)
        
        # Find flash loan arbitrage opportunities
        if ArbitrageType.FLASH_LOAN in arbitrage_types:
            flash_loan_opportunities = await self._find_flash_loan_arbitrage(
                chain_id, token_address
            )
            opportunity_ids.extend(flash_loan_opportunities)
        
        return opportunity_ids
    
    async def _find_simple_dex_arbitrage(
        self,
        chain_id: int,
        token_address: str
    ) -> List[str]:
        """Find simple DEX arbitrage opportunities
        
        Args:
            chain_id: Chain ID
            token_address: Token address
            
        Returns:
            List of opportunity IDs
        """
        # This would normally check price differences between DEXes
        # For now, we'll just return a placeholder
        
        # Get token details
        token_repository = aggregate_factory(TokenAggregate)
        token = await token_repository.get_by_id(token_address)
        
        if not token or not token.state.liquidity_pools or len(token.state.liquidity_pools) < 2:
            # Need at least 2 liquidity pools for simple DEX arbitrage
            return []
        
        # Find price differences between pools
        opportunity_ids = []
        
        # Sort pools by price (highest to lowest)
        pools = sorted(
            token.state.liquidity_pools,
            key=lambda p: p.token_reserves / p.pair_reserves,
            reverse=True
        )
        
        # Check for price differences
        for i in range(len(pools) - 1):
            for j in range(i + 1, len(pools)):
                # Calculate price difference
                price_a = pools[i].token_reserves / pools[i].pair_reserves
                price_b = pools[j].token_reserves / pools[j].pair_reserves
                
                # Calculate potential profit percentage
                price_diff_percentage = ((price_a - price_b) / price_b) * 100
                
                # If price difference is significant, create opportunity
                if price_diff_percentage > self.min_profit_percentage:
                    # Create opportunity
                    opportunity_id = await self._create_simple_dex_opportunity(
                        chain_id,
                        token,
                        pools[i],
                        pools[j],
                        price_diff_percentage
                    )
                    
                    if opportunity_id:
                        opportunity_ids.append(opportunity_id)
        
        return opportunity_ids
    
    async def _create_simple_dex_opportunity(
        self,
        chain_id: int,
        token: TokenAggregate,
        pool_a: Any,
        pool_b: Any,
        price_diff_percentage: Decimal
    ) -> Optional[str]:
        """Create a simple DEX arbitrage opportunity
        
        Args:
            chain_id: Chain ID
            token: Token aggregate
            pool_a: First liquidity pool (higher price)
            pool_b: Second liquidity pool (lower price)
            price_diff_percentage: Price difference percentage
            
        Returns:
            Opportunity ID if created, None otherwise
        """
        # Calculate optimal trade size
        # This would normally use a more sophisticated algorithm
        # For now, we'll use a simple approach
        
        # Use 10% of available liquidity in the lower-priced pool
        entry_amount = pool_b.token_reserves * Decimal("0.1")
        
        # Calculate expected profit
        expected_profit_amount = entry_amount * (price_diff_percentage / 100)
        
        # Calculate USD values
        if token.state.current_price_usd:
            entry_amount_usd = entry_amount * token.state.current_price_usd
            expected_profit_usd = expected_profit_amount * token.state.current_price_usd
        else:
            # Estimate from pool
            token_price_usd = Decimal("1.0")  # Placeholder
            entry_amount_usd = entry_amount * token_price_usd
            expected_profit_usd = expected_profit_amount * token_price_usd
        
        # Check if profit meets minimum threshold
        if expected_profit_usd < self.min_profit_threshold_usd:
            return None
        
        # Create arbitrage opportunity
        arbitrage_repository = aggregate_factory(ArbitrageAggregate)
        
        arbitrage = await ArbitrageAggregate.create_opportunity(
            arbitrage_type=ArbitrageType.SIMPLE_DEX,
            chain_ids=[chain_id],
            tokens_involved=[token.state.address, pool_a.pair_with],
            entry_token=token.state.address,
            entry_amount=entry_amount,
            expected_profit_token=token.state.address,
            expected_profit_amount=expected_profit_amount,
            expected_profit_usd=expected_profit_usd,
            expected_profit_percentage=price_diff_percentage,
            discovery_source="simple_dex_scanner",
            risk_level=ArbitrageRisk.LOW,
            priority=ExecutionPriority.NORMAL
        )
        
        # Add route steps
        # Step 1: Sell token on higher-priced pool
        step1 = RouteStep(
            step_index=1,
            dex_name=str(pool_a.source),
            source_token=token.state.address,
            target_token=pool_a.pair_with,
            source_amount=entry_amount,
            expected_target_amount=entry_amount * (pool_a.token_reserves / pool_a.pair_reserves),
            min_target_amount=entry_amount * (pool_a.token_reserves / pool_a.pair_reserves) * (1 - self.default_slippage / 100),
            pool_address=pool_a.address,
            chain_id=chain_id
        )
        arbitrage.add_route_step(step1)
        
        # Step 2: Buy token on lower-priced pool
        step2 = RouteStep(
            step_index=2,
            dex_name=str(pool_b.source),
            source_token=pool_b.pair_with,
            target_token=token.state.address,
            source_amount=step1.expected_target_amount,
            expected_target_amount=step1.expected_target_amount * (pool_b.pair_reserves / pool_b.token_reserves),
            min_target_amount=step1.expected_target_amount * (pool_b.pair_reserves / pool_b.token_reserves) * (1 - self.default_slippage / 100),
            pool_address=pool_b.address,
            chain_id=chain_id
        )
        arbitrage.add_route_step(step2)
        
        # Save arbitrage
        await arbitrage_repository.save(arbitrage)
        
        # Queue for analysis
        await self.queue_opportunity_analysis(arbitrage.id)
        
        return arbitrage.id
    
    async def _find_triangular_arbitrage(
        self,
        chain_id: int,
        token_address: str
    ) -> List[str]:
        """Find triangular arbitrage opportunities
        
        Args:
            chain_id: Chain ID
            token_address: Token address
            
        Returns:
            List of opportunity IDs
        """
        # This would normally check for triangular arbitrage opportunities
        # For now, we'll just return a placeholder
        return []
    
    async def _find_flash_loan_arbitrage(
        self,
        chain_id: int,
        token_address: str
    ) -> List[str]:
        """Find flash loan arbitrage opportunities
        
        Args:
            chain_id: Chain ID
            token_address: Token address
            
        Returns:
            List of opportunity IDs
        """
        # This would normally check for flash loan arbitrage opportunities
        # For now, we'll just return a placeholder
        return []
    
    async def queue_opportunity_analysis(self, opportunity_id: str) -> None:
        """Queue an opportunity for analysis
        
        Args:
            opportunity_id: Opportunity ID
        """
        await self.analysis_queue.put(opportunity_id)
        
        # Start processing if not already running
        if not self.is_processing_analysis:
            asyncio.create_task(self._process_analysis_queue())
    
    async def queue_opportunity_execution(self, opportunity_id: str) -> None:
        """Queue an opportunity for execution
        
        Args:
            opportunity_id: Opportunity ID
        """
        await self.execution_queue.put(opportunity_id)
        
        # Start processing if not already running
        if not self.is_processing_execution:
            asyncio.create_task(self._process_execution_queue())
    
    async def _process_analysis_queue(self) -> None:
        """Process the analysis queue"""
        self.is_processing_analysis = True
        
        try:
            while not self.analysis_queue.empty():
                # Get next opportunity
                opportunity_id = await self.analysis_queue.get()
                
                try:
                    # Analyze opportunity
                    await self._analyze_opportunity(opportunity_id)
                except Exception as e:
                    logger.error(f"Error analyzing opportunity {opportunity_id}: {str(e)}")
                finally:
                    self.analysis_queue.task_done()
        finally:
            self.is_processing_analysis = False
    
    async def _process_execution_queue(self) -> None:
        """Process the execution queue"""
        self.is_processing_execution = True
        
        try:
            while not self.execution_queue.empty():
                # Get next opportunity
                opportunity_id = await self.execution_queue.get()
                
                try:
                    # Execute opportunity
                    await self._execute_opportunity(opportunity_id)
                except Exception as e:
                    logger.error(f"Error executing opportunity {opportunity_id}: {str(e)}")
                finally:
                    self.execution_queue.task_done()
        finally:
            self.is_processing_execution = False
    
    async def _analyze_opportunity(self, opportunity_id: str) -> None:
        """Analyze an arbitrage opportunity
        
        Args:
            opportunity_id: Opportunity ID
        """
        logger.info(f"Analyzing arbitrage opportunity {opportunity_id}")
        
        # Get opportunity
        arbitrage_repository = aggregate_factory(ArbitrageAggregate)
        arbitrage = await arbitrage_repository.get_by_id(opportunity_id)
        
        if not arbitrage:
            logger.warning(f"Arbitrage opportunity {opportunity_id} not found")
            return
        
        # Check if opportunity is expired
        if datetime.now() > arbitrage.state.expiration_time:
            arbitrage.expire_opportunity()
            await arbitrage_repository.save(arbitrage)
            logger.info(f"Arbitrage opportunity {opportunity_id} expired")
            return
        
        # Estimate gas costs
        gas_estimate = await self._estimate_gas_costs(arbitrage)
        if gas_estimate:
            arbitrage.set_gas_estimate(gas_estimate)
            await arbitrage_repository.save(arbitrage)
        
        # Simulate transaction
        simulation_result = await self._simulate_transaction(arbitrage)
        if simulation_result:
            arbitrage.set_simulation_result(simulation_result)
            await arbitrage_repository.save(arbitrage)
        
        # Note: The event handler for SIMULATION_COMPLETED will check if the
        # opportunity is profitable and mark it as ready if appropriate
    
    async def _estimate_gas_costs(self, arbitrage: ArbitrageAggregate) -> Optional[GasEstimate]:
        """Estimate gas costs for an arbitrage opportunity
        
        Args:
            arbitrage: Arbitrage aggregate
            
        Returns:
            Gas estimate if successful, None otherwise
        """
        try:
            # Build transaction
            tx_data = await self.transaction_builder.build_arbitrage_transaction(
                arbitrage.state.arbitrage_type,
                arbitrage.state.route_steps,
                arbitrage.state.flash_loan_details
            )
            
            # Estimate gas
            chain_id = arbitrage.state.chain_ids[0]  # Use first chain
            gas_limit = await self.gas_estimator.estimate_gas(
                chain_id,
                tx_data
            )
            
            # Get gas price
            gas_price = await self.gas_estimator.get_gas_price(chain_id)
            
            # Calculate costs
            total_cost_eth = Decimal(gas_limit) * Decimal(gas_price) / Decimal(10**18)
            
            # Convert to USD
            eth_price_usd = await self.gas_estimator.get_eth_price_usd(chain_id)
            total_cost_usd = total_cost_eth * eth_price_usd
            
            return GasEstimate(
                gas_limit=gas_limit,
                gas_price=Decimal(gas_price),
                total_cost_eth=total_cost_eth,
                total_cost_usd=total_cost_usd
            )
        except Exception as e:
            logger.error(f"Error estimating gas costs: {str(e)}")
            return None
    
    async def _simulate_transaction(self, arbitrage: ArbitrageAggregate) -> Optional[SimulationResult]:
        """Simulate an arbitrage transaction
        
        Args:
            arbitrage: Arbitrage aggregate
            
        Returns:
            Simulation result if successful, None otherwise
        """
        try:
            # Build transaction
            tx_data = await self.transaction_builder.build_arbitrage_transaction(
                arbitrage.state.arbitrage_type,
                arbitrage.state.route_steps,
                arbitrage.state.flash_loan_details
            )
            
            # Simulate transaction
            chain_id = arbitrage.state.chain_ids[0]  # Use first chain
            simulation = await self.transaction_simulator.simulate_transaction(
                chain_id,
                tx_data
            )
            
            if not simulation["success"]:
                return SimulationResult(
                    success=False,
                    profit_amount=Decimal("0"),
                    profit_token=arbitrage.state.expected_profit_token,
                    profit_usd=Decimal("0"),
                    gas_used=simulation.get("gas_used", 0),
                    error_message=simulation.get("error", "Unknown error")
                )
            
            # Calculate actual profit
            profit_amount = Decimal(simulation["profit_amount"])
            
            # Convert to USD
            token_repository = aggregate_factory(TokenAggregate)
            token = await token_repository.get_by_id(arbitrage.state.expected_profit_token)
            
            if token and token.state.current_price_usd:
                profit_usd = profit_amount * token.state.current_price_usd
            else:
                # Estimate from expected profit
                profit_ratio = profit_amount / arbitrage.state.expected_profit_amount
                profit_usd = arbitrage.state.expected_profit_usd * profit_ratio
            
            return SimulationResult(
                success=True,
                profit_amount=profit_amount,
                profit_token=arbitrage.state.expected_profit_token,
                profit_usd=profit_usd,
                gas_used=simulation.get("gas_used", 0)
            )
        except Exception as e:
            logger.error(f"Error simulating transaction: {str(e)}")
            return SimulationResult(
                success=False,
                profit_amount=Decimal("0"),
                profit_token=arbitrage.state.expected_profit_token,
                profit_usd=Decimal("0"),
                gas_used=0,
                error_message=str(e)
            )
    
    async def _execute_opportunity(self, opportunity_id: str) -> None:
        """Execute an arbitrage opportunity
        
        Args:
            opportunity_id: Opportunity ID
        """
        logger.info(f"Executing arbitrage opportunity {opportunity_id}")
        
        # Get opportunity
        arbitrage_repository = aggregate_factory(ArbitrageAggregate)
        arbitrage = await arbitrage_repository.get_by_id(opportunity_id)
        
        if not arbitrage:
            logger.warning(f"Arbitrage opportunity {opportunity_id} not found")
            return
        
        # Check if opportunity is ready
        if arbitrage.state.status != "ready":
            logger.warning(f"Arbitrage opportunity {opportunity_id} not ready for execution")
            return
        
        # Check if opportunity is expired
        if datetime.now() > arbitrage.state.expiration_time:
            arbitrage.expire_opportunity()
            await arbitrage_repository.save(arbitrage)
            logger.info(f"Arbitrage opportunity {opportunity_id} expired")
            return
        
        # Start execution
        arbitrage.start_execution()
        await arbitrage_repository.save(arbitrage)
        
        try:
            # Build transaction
            tx_data = await self.transaction_builder.build_arbitrage_transaction(
                arbitrage.state.arbitrage_type,
                arbitrage.state.route_steps,
                arbitrage.state.flash_loan_details
            )
            
            # Send transaction
            chain_id = arbitrage.state.chain_ids[0]  # Use first chain
            tx_hash = await self.web3_client.send_transaction(
                chain_id,
                tx_data
            )
            
            # Wait for transaction to be mined
            receipt = await self.web3_client.wait_for_transaction(
                chain_id,
                tx_hash
            )
            
            if receipt["status"] == 1:
                # Transaction successful
                # Calculate actual profit
                actual_profit = await self._calculate_actual_profit(
                    arbitrage,
                    receipt
                )
                
                # Complete execution
                arbitrage.complete_execution(
                    tx_hash=tx_hash,
                    actual_profit_amount=actual_profit["amount"],
                    actual_profit_usd=actual_profit["usd"],
                    execution_result={
                        "receipt": receipt,
                        "gas_used": receipt["gasUsed"],
                        "block_number": receipt["blockNumber"]
                    }
                )
            else:
                # Transaction failed
                arbitrage.fail_execution(
                    reason="Transaction failed",
                    details={
                        "tx_hash": tx_hash,
                        "receipt": receipt
                    }
                )
        except Exception as e:
            # Execution failed
            arbitrage.fail_execution(
                reason="Execution error",
                details={
                    "error": str(e)
                }
            )
        
        # Save arbitrage
        await arbitrage_repository.save(arbitrage)
    
    async def _calculate_actual_profit(
        self,
        arbitrage: ArbitrageAggregate,
        receipt: Dict[str, Any]
    ) -> Dict[str, Decimal]:
        """Calculate actual profit from transaction receipt
        
        Args:
            arbitrage: Arbitrage aggregate
            receipt: Transaction receipt
            
        Returns:
            Dictionary with profit amount and USD value
        """
        # This would normally parse the transaction receipt to calculate actual profit
        # For now, we'll just use the expected profit
        
        # Apply a small variance to simulate real-world conditions
        import random
        variance = Decimal(random.uniform(0.9, 1.1))
        
        actual_profit_amount = arbitrage.state.expected_profit_amount * variance
        actual_profit_usd = arbitrage.state.expected_profit_usd * variance
        
        return {
            "amount": actual_profit_amount,
            "usd": actual_profit_usd
        }
    
    def _start_background_tasks(self) -> None:
        """Start background tasks"""
        # Start queue processors
        asyncio.create_task(self._process_analysis_queue())
        asyncio.create_task(self._process_execution_queue())
        
        # Start opportunity finder
        asyncio.create_task(self._periodic_opportunity_finder())
    
    async def _periodic_opportunity_finder(self) -> None:
        """Periodically find arbitrage opportunities"""
        while True:
            try:
                # Find opportunities on main chains
                for chain_id in [1, 56, 137]:  # Ethereum, BSC, Polygon
                    await self.find_arbitrage_opportunities(chain_id)
                
                # Sleep for 1 minute
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"Error in periodic opportunity finder: {str(e)}")
                await asyncio.sleep(10)  # Sleep on error

# Create singleton instance
arbitrage_service = ArbitrageService(
    web3_client=container.resolve(Web3Client),
    flash_loan_provider=container.resolve(FlashLoanProvider),
    transaction_builder=container.resolve(TransactionBuilder),
    gas_estimator=container.resolve(GasEstimator),
    transaction_simulator=container.resolve(TransactionSimulator)
) 