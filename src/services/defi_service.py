"""
DeFi Service

This service provides a unified interface for DeFi operations:
- Arbitrage opportunity discovery and execution
- Yield farming position management
- Portfolio optimization
- Risk management
- Performance tracking
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Set, Any, Tuple, Union
import uuid

from ..cqrs.events.token_discovery import TokenAggregate, TokenStatus
from ..cqrs.events.arbitrage import ArbitrageAggregate, ArbitrageType, ArbitrageStatus
from ..cqrs.events.aggregates import aggregate_factory
from ..core.dependency_injector import container
from .arbitrage_service import arbitrage_service
from .yield_farming_service import yield_farming_service, YieldOpportunity, YieldPosition

logger = logging.getLogger(__name__)

class PortfolioAllocation:
    """Portfolio allocation recommendation"""
    
    def __init__(
        self,
        wallet_address: str,
        total_value_usd: Decimal,
        arbitrage_allocation_usd: Decimal,
        yield_allocation_usd: Decimal,
        reserve_allocation_usd: Decimal,
        arbitrage_opportunities: List[Dict[str, Any]],
        yield_opportunities: List[Dict[str, Any]],
        risk_score: int,
        expected_monthly_return: Decimal,
        timestamp: datetime = None
    ):
        self.wallet_address = wallet_address
        self.total_value_usd = total_value_usd
        self.arbitrage_allocation_usd = arbitrage_allocation_usd
        self.yield_allocation_usd = yield_allocation_usd
        self.reserve_allocation_usd = reserve_allocation_usd
        self.arbitrage_opportunities = arbitrage_opportunities
        self.yield_opportunities = yield_opportunities
        self.risk_score = risk_score
        self.expected_monthly_return = expected_monthly_return
        self.timestamp = timestamp or datetime.now()

class PerformanceMetrics:
    """Performance metrics for a wallet"""
    
    def __init__(
        self,
        wallet_address: str,
        start_date: datetime,
        end_date: datetime,
        starting_balance_usd: Decimal,
        ending_balance_usd: Decimal,
        total_profit_usd: Decimal,
        total_profit_percentage: Decimal,
        annualized_return: Decimal,
        arbitrage_profit_usd: Decimal,
        yield_farming_profit_usd: Decimal,
        successful_arbitrages: int,
        failed_arbitrages: int,
        active_yield_positions: int,
        closed_yield_positions: int,
        best_performing_strategy: str,
        worst_performing_strategy: str,
        risk_adjusted_return: Decimal
    ):
        self.wallet_address = wallet_address
        self.start_date = start_date
        self.end_date = end_date
        self.starting_balance_usd = starting_balance_usd
        self.ending_balance_usd = ending_balance_usd
        self.total_profit_usd = total_profit_usd
        self.total_profit_percentage = total_profit_percentage
        self.annualized_return = annualized_return
        self.arbitrage_profit_usd = arbitrage_profit_usd
        self.yield_farming_profit_usd = yield_farming_profit_usd
        self.successful_arbitrages = successful_arbitrages
        self.failed_arbitrages = failed_arbitrages
        self.active_yield_positions = active_yield_positions
        self.closed_yield_positions = closed_yield_positions
        self.best_performing_strategy = best_performing_strategy
        self.worst_performing_strategy = worst_performing_strategy
        self.risk_adjusted_return = risk_adjusted_return

class DeFiService:
    """Unified service for DeFi operations"""
    
    def __init__(self):
        self.arbitrage_service = arbitrage_service
        self.yield_farming_service = yield_farming_service
        
        # In-memory storage (would be replaced with a database in production)
        self.portfolio_allocations: Dict[str, PortfolioAllocation] = {}
        self.performance_metrics: Dict[str, PerformanceMetrics] = {}
        
        # Configuration
        self.default_risk_profile = {
            "conservative": {
                "arbitrage_allocation": Decimal("0.1"),  # 10%
                "yield_allocation": Decimal("0.6"),      # 60%
                "reserve_allocation": Decimal("0.3"),    # 30%
                "max_risk_level": "low"
            },
            "moderate": {
                "arbitrage_allocation": Decimal("0.2"),  # 20%
                "yield_allocation": Decimal("0.6"),      # 60%
                "reserve_allocation": Decimal("0.2"),    # 20%
                "max_risk_level": "medium"
            },
            "aggressive": {
                "arbitrage_allocation": Decimal("0.4"),  # 40%
                "yield_allocation": Decimal("0.5"),      # 50%
                "reserve_allocation": Decimal("0.1"),    # 10%
                "max_risk_level": "high"
            }
        }
        
        # Start background tasks
        self._start_background_tasks()
    
    async def get_defi_opportunities(
        self,
        wallet_address: str,
        chain_ids: List[int],
        risk_profile: str = "moderate",
        max_opportunities: int = 10,
        include_arbitrage: bool = True,
        include_yield: bool = True
    ) -> Dict[str, Any]:
        """Get DeFi opportunities for a wallet
        
        Args:
            wallet_address: Wallet address
            chain_ids: Chain IDs to search on
            risk_profile: Risk profile (conservative, moderate, aggressive)
            max_opportunities: Maximum number of opportunities to return
            include_arbitrage: Whether to include arbitrage opportunities
            include_yield: Whether to include yield farming opportunities
            
        Returns:
            Dictionary with arbitrage and yield opportunities
        """
        logger.info(f"Getting DeFi opportunities for wallet {wallet_address}")
        
        # Get risk profile configuration
        risk_config = self.default_risk_profile.get(risk_profile, self.default_risk_profile["moderate"])
        max_risk_level = risk_config["max_risk_level"]
        
        opportunities = {
            "arbitrage": [],
            "yield_farming": [],
            "risk_profile": risk_profile,
            "max_risk_level": max_risk_level
        }
        
        # Get arbitrage opportunities
        if include_arbitrage:
            for chain_id in chain_ids:
                # Find arbitrage opportunities
                opportunity_ids = await self.arbitrage_service.find_arbitrage_opportunities(
                    chain_id=chain_id,
                    max_opportunities=max_opportunities
                )
                
                # Get opportunity details
                for opportunity_id in opportunity_ids:
                    # This would normally get the opportunity details from the repository
                    # For now, we'll just use a placeholder
                    opportunities["arbitrage"].append({
                        "id": opportunity_id,
                        "chain_id": chain_id,
                        "type": "simple_dex",
                        "expected_profit_usd": "10.0",
                        "risk_level": "medium"
                    })
        
        # Get yield farming opportunities
        if include_yield:
            for chain_id in chain_ids:
                # Find yield opportunities
                yield_opps = await self.yield_farming_service.discover_opportunities(
                    chain_id=chain_id,
                    max_risk_level=max_risk_level,
                    max_results=max_opportunities
                )
                
                # Add to list
                for opp in yield_opps:
                    opportunities["yield_farming"].append({
                        "id": opp.id,
                        "chain_id": opp.chain_id,
                        "protocol": opp.protocol,
                        "pool_name": opp.pool_name,
                        "apy": str(opp.total_apy),
                        "tvl_usd": str(opp.tvl_usd),
                        "risk_level": opp.risk_level,
                        "tokens": opp.token_symbols
                    })
        
        logger.info(f"Found {len(opportunities['arbitrage'])} arbitrage and {len(opportunities['yield_farming'])} yield opportunities")
        return opportunities
    
    async def get_portfolio_allocation(
        self,
        wallet_address: str,
        total_value_usd: Decimal,
        risk_profile: str = "moderate",
        chain_ids: Optional[List[int]] = None
    ) -> PortfolioAllocation:
        """Get portfolio allocation recommendation
        
        Args:
            wallet_address: Wallet address
            total_value_usd: Total portfolio value in USD
            risk_profile: Risk profile (conservative, moderate, aggressive)
            chain_ids: Chain IDs to consider
            
        Returns:
            Portfolio allocation recommendation
        """
        logger.info(f"Getting portfolio allocation for wallet {wallet_address}")
        
        # Default chain IDs if not specified
        if not chain_ids:
            chain_ids = [1, 56, 137]  # Ethereum, BSC, Polygon
        
        # Get risk profile configuration
        risk_config = self.default_risk_profile.get(risk_profile, self.default_risk_profile["moderate"])
        
        # Calculate allocations
        arbitrage_allocation_usd = total_value_usd * risk_config["arbitrage_allocation"]
        yield_allocation_usd = total_value_usd * risk_config["yield_allocation"]
        reserve_allocation_usd = total_value_usd * risk_config["reserve_allocation"]
        
        # Get opportunities
        opportunities = await self.get_defi_opportunities(
            wallet_address=wallet_address,
            chain_ids=chain_ids,
            risk_profile=risk_profile
        )
        
        # Calculate expected return
        arbitrage_return = Decimal("0")
        for opp in opportunities["arbitrage"]:
            arbitrage_return += Decimal(opp["expected_profit_usd"])
        
        yield_return = Decimal("0")
        for opp in opportunities["yield_farming"]:
            # Convert APY to monthly return
            apy = Decimal(opp["apy"])
            monthly_return = apy / 12
            
            # Calculate allocation for this opportunity
            # This is a simplified approach - in a real implementation, you would
            # use a portfolio optimization algorithm
            allocation = yield_allocation_usd / len(opportunities["yield_farming"])
            
            # Calculate expected return
            yield_return += allocation * (monthly_return / 100)
        
        # Calculate total expected monthly return
        expected_monthly_return = arbitrage_return + yield_return
        
        # Create portfolio allocation
        allocation = PortfolioAllocation(
            wallet_address=wallet_address,
            total_value_usd=total_value_usd,
            arbitrage_allocation_usd=arbitrage_allocation_usd,
            yield_allocation_usd=yield_allocation_usd,
            reserve_allocation_usd=reserve_allocation_usd,
            arbitrage_opportunities=opportunities["arbitrage"],
            yield_opportunities=opportunities["yield_farming"],
            risk_score=self._risk_profile_score(risk_profile),
            expected_monthly_return=expected_monthly_return
        )
        
        # Store allocation
        self.portfolio_allocations[wallet_address] = allocation
        
        logger.info(f"Created portfolio allocation for wallet {wallet_address}")
        return allocation
    
    async def get_performance_metrics(
        self,
        wallet_address: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> PerformanceMetrics:
        """Get performance metrics for a wallet
        
        Args:
            wallet_address: Wallet address
            start_date: Start date for metrics (default: 30 days ago)
            end_date: End date for metrics (default: now)
            
        Returns:
            Performance metrics
        """
        logger.info(f"Getting performance metrics for wallet {wallet_address}")
        
        # Set default dates if not specified
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        # Get arbitrage performance
        arbitrage_profit_usd = Decimal("0")
        successful_arbitrages = 0
        failed_arbitrages = 0
        
        # This would normally query the arbitrage repository
        # For now, we'll just use placeholders
        arbitrage_profit_usd = Decimal("100.0")
        successful_arbitrages = 5
        failed_arbitrages = 2
        
        # Get yield farming performance
        yield_farming_profit_usd = Decimal("0")
        active_yield_positions = 0
        closed_yield_positions = 0
        
        # Get active positions
        active_positions = await self.yield_farming_service.get_positions_by_wallet(
            wallet_address=wallet_address,
            status="active"
        )
        
        active_yield_positions = len(active_positions)
        
        # Calculate yield farming profit
        for position in active_positions:
            yield_farming_profit_usd += position.profit_usd
        
        # Get closed positions
        closed_positions = await self.yield_farming_service.get_positions_by_wallet(
            wallet_address=wallet_address,
            status="closed"
        )
        
        closed_yield_positions = len(closed_positions)
        
        # Calculate yield farming profit from closed positions
        for position in closed_positions:
            if position.entry_timestamp >= start_date and position.metadata.get("exit_timestamp"):
                exit_time = datetime.fromisoformat(position.metadata["exit_timestamp"])
                if exit_time <= end_date:
                    yield_farming_profit_usd += position.profit_usd
        
        # Calculate total profit
        total_profit_usd = arbitrage_profit_usd + yield_farming_profit_usd
        
        # Calculate other metrics
        starting_balance_usd = Decimal("1000.0")  # Placeholder
        ending_balance_usd = starting_balance_usd + total_profit_usd
        
        # Calculate profit percentage
        total_profit_percentage = (total_profit_usd / starting_balance_usd) * 100
        
        # Calculate annualized return
        days = (end_date - start_date).days
        if days > 0:
            annualized_return = total_profit_percentage * (365 / days)
        else:
            annualized_return = Decimal("0")
        
        # Determine best and worst performing strategies
        if arbitrage_profit_usd > yield_farming_profit_usd:
            best_performing_strategy = "arbitrage"
            worst_performing_strategy = "yield_farming"
        else:
            best_performing_strategy = "yield_farming"
            worst_performing_strategy = "arbitrage"
        
        # Calculate risk-adjusted return (simplified Sharpe ratio)
        # In a real implementation, you would calculate the standard deviation of returns
        risk_adjusted_return = annualized_return / 10  # Placeholder
        
        # Create performance metrics
        metrics = PerformanceMetrics(
            wallet_address=wallet_address,
            start_date=start_date,
            end_date=end_date,
            starting_balance_usd=starting_balance_usd,
            ending_balance_usd=ending_balance_usd,
            total_profit_usd=total_profit_usd,
            total_profit_percentage=total_profit_percentage,
            annualized_return=annualized_return,
            arbitrage_profit_usd=arbitrage_profit_usd,
            yield_farming_profit_usd=yield_farming_profit_usd,
            successful_arbitrages=successful_arbitrages,
            failed_arbitrages=failed_arbitrages,
            active_yield_positions=active_yield_positions,
            closed_yield_positions=closed_yield_positions,
            best_performing_strategy=best_performing_strategy,
            worst_performing_strategy=worst_performing_strategy,
            risk_adjusted_return=risk_adjusted_return
        )
        
        # Store metrics
        self.performance_metrics[wallet_address] = metrics
        
        logger.info(f"Generated performance metrics for wallet {wallet_address}")
        return metrics
    
    async def execute_arbitrage_opportunity(
        self,
        opportunity_id: str,
        wallet_address: str,
        gas_price_gwei: Optional[Decimal] = None
    ) -> bool:
        """Execute an arbitrage opportunity
        
        Args:
            opportunity_id: Opportunity ID
            wallet_address: Wallet address
            gas_price_gwei: Optional gas price in Gwei
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Executing arbitrage opportunity {opportunity_id} for wallet {wallet_address}")
        
        # Queue opportunity for execution
        await self.arbitrage_service.queue_opportunity_execution(opportunity_id)
        
        # In a real implementation, you would wait for the execution to complete
        # and return the result
        return True
    
    async def create_yield_position(
        self,
        opportunity_id: str,
        wallet_address: str,
        deposit_amounts: Dict[str, Decimal],
        gas_price_gwei: Optional[Decimal] = None
    ) -> Optional[str]:
        """Create a yield farming position
        
        Args:
            opportunity_id: Opportunity ID
            wallet_address: Wallet address
            deposit_amounts: Token amounts to deposit
            gas_price_gwei: Optional gas price in Gwei
            
        Returns:
            Position ID if successful, None otherwise
        """
        logger.info(f"Creating yield position for opportunity {opportunity_id} and wallet {wallet_address}")
        
        # Create position
        position_id = await self.yield_farming_service.create_position(
            opportunity_id=opportunity_id,
            wallet_address=wallet_address,
            deposit_amounts=deposit_amounts,
            gas_price_gwei=gas_price_gwei
        )
        
        if position_id:
            logger.info(f"Created yield position {position_id}")
        else:
            logger.error(f"Failed to create yield position")
        
        return position_id
    
    async def harvest_yield_position(
        self,
        position_id: str,
        auto_compound: bool = True,
        gas_price_gwei: Optional[Decimal] = None
    ) -> bool:
        """Harvest rewards for a yield position
        
        Args:
            position_id: Position ID
            auto_compound: Whether to auto-compound rewards
            gas_price_gwei: Optional gas price in Gwei
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Harvesting yield position {position_id}")
        
        # Harvest rewards
        success = await self.yield_farming_service.harvest_rewards(
            position_id=position_id,
            auto_compound=auto_compound,
            gas_price_gwei=gas_price_gwei
        )
        
        if success:
            logger.info(f"Successfully harvested yield position {position_id}")
        else:
            logger.error(f"Failed to harvest yield position {position_id}")
        
        return success
    
    async def exit_yield_position(
        self,
        position_id: str,
        gas_price_gwei: Optional[Decimal] = None
    ) -> bool:
        """Exit a yield farming position
        
        Args:
            position_id: Position ID
            gas_price_gwei: Optional gas price in Gwei
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Exiting yield position {position_id}")
        
        # Exit position
        success = await self.yield_farming_service.exit_position(
            position_id=position_id,
            gas_price_gwei=gas_price_gwei
        )
        
        if success:
            logger.info(f"Successfully exited yield position {position_id}")
        else:
            logger.error(f"Failed to exit yield position {position_id}")
        
        return success
    
    async def optimize_portfolio(
        self,
        wallet_address: str,
        total_value_usd: Decimal,
        risk_profile: str = "moderate",
        chain_ids: Optional[List[int]] = None,
        rebalance: bool = False
    ) -> Dict[str, Any]:
        """Optimize portfolio allocation
        
        Args:
            wallet_address: Wallet address
            total_value_usd: Total portfolio value in USD
            risk_profile: Risk profile (conservative, moderate, aggressive)
            chain_ids: Chain IDs to consider
            rebalance: Whether to rebalance existing positions
            
        Returns:
            Optimization result
        """
        logger.info(f"Optimizing portfolio for wallet {wallet_address}")
        
        # Get portfolio allocation
        allocation = await self.get_portfolio_allocation(
            wallet_address=wallet_address,
            total_value_usd=total_value_usd,
            risk_profile=risk_profile,
            chain_ids=chain_ids
        )
        
        # Get current positions
        current_yield_positions = await self.yield_farming_service.get_positions_by_wallet(
            wallet_address=wallet_address,
            status="active"
        )
        
        # Calculate current allocation
        current_yield_allocation = sum(position.current_value_usd for position in current_yield_positions)
        
        # Calculate target allocation
        target_yield_allocation = allocation.yield_allocation_usd
        
        # Determine if rebalancing is needed
        rebalance_needed = abs(current_yield_allocation - target_yield_allocation) > (target_yield_allocation * Decimal("0.1"))
        
        # Rebalance if needed and requested
        if rebalance and rebalance_needed:
            # This would normally rebalance the portfolio
            # For now, we'll just log the action
            logger.info(f"Rebalancing portfolio for wallet {wallet_address}")
            
            # If current allocation is too high, exit some positions
            if current_yield_allocation > target_yield_allocation:
                excess = current_yield_allocation - target_yield_allocation
                
                # Sort positions by APY (lowest first)
                positions_to_exit = sorted(
                    current_yield_positions,
                    key=lambda p: p.apy_realized
                )
                
                # Exit positions until we reach target allocation
                for position in positions_to_exit:
                    if excess <= 0:
                        break
                    
                    # Exit position
                    await self.exit_yield_position(position.id)
                    
                    # Update excess
                    excess -= position.current_value_usd
            
            # If current allocation is too low, create new positions
            elif current_yield_allocation < target_yield_allocation:
                deficit = target_yield_allocation - current_yield_allocation
                
                # Get best opportunities
                best_opportunities = await self.yield_farming_service.get_best_opportunities(
                    risk_level=risk_profile,
                    limit=3
                )
                
                # Create positions for best opportunities
                for opportunity in best_opportunities:
                    if deficit <= 0:
                        break
                    
                    # Calculate deposit amount
                    deposit_amount = min(deficit, Decimal("1000.0"))  # Placeholder
                    
                    # Create position
                    # This is a simplified approach - in a real implementation, you would
                    # determine the actual tokens and amounts to deposit
                    await self.create_yield_position(
                        opportunity_id=opportunity.id,
                        wallet_address=wallet_address,
                        deposit_amounts={"0x...": deposit_amount}  # Placeholder
                    )
                    
                    # Update deficit
                    deficit -= deposit_amount
        
        # Return optimization result
        return {
            "wallet_address": wallet_address,
            "risk_profile": risk_profile,
            "total_value_usd": str(total_value_usd),
            "target_yield_allocation": str(target_yield_allocation),
            "current_yield_allocation": str(current_yield_allocation),
            "rebalance_needed": rebalance_needed,
            "rebalance_performed": rebalance and rebalance_needed,
            "expected_monthly_return": str(allocation.expected_monthly_return),
            "arbitrage_opportunities": len(allocation.arbitrage_opportunities),
            "yield_opportunities": len(allocation.yield_opportunities)
        }
    
    def _risk_profile_score(self, risk_profile: str) -> int:
        """Convert risk profile to numeric score
        
        Args:
            risk_profile: Risk profile string
            
        Returns:
            Risk score (1-5, higher = riskier)
        """
        risk_scores = {
            "conservative": 1,
            "moderate": 3,
            "aggressive": 5
        }
        
        return risk_scores.get(risk_profile, 3)  # Default to moderate
    
    def _start_background_tasks(self) -> None:
        """Start background tasks"""
        # Start portfolio optimizer
        asyncio.create_task(self._optimize_portfolios_periodically())
    
    async def _optimize_portfolios_periodically(self) -> None:
        """Periodically optimize portfolios"""
        while True:
            try:
                # Optimize portfolios for all wallets
                for wallet_address in self.portfolio_allocations:
                    allocation = self.portfolio_allocations[wallet_address]
                    
                    # Optimize portfolio
                    await self.optimize_portfolio(
                        wallet_address=wallet_address,
                        total_value_usd=allocation.total_value_usd,
                        risk_profile=self._risk_profile_name(allocation.risk_score),
                        rebalance=True
                    )
                
                # Sleep for 24 hours
                await asyncio.sleep(24 * 3600)
            except Exception as e:
                logger.error(f"Error optimizing portfolios: {str(e)}")
                await asyncio.sleep(60)  # Sleep on error
    
    def _risk_profile_name(self, risk_score: int) -> str:
        """Convert risk score to profile name
        
        Args:
            risk_score: Risk score (1-5)
            
        Returns:
            Risk profile name
        """
        if risk_score <= 2:
            return "conservative"
        elif risk_score <= 4:
            return "moderate"
        else:
            return "aggressive"

# Create singleton instance
defi_service = DeFiService() 