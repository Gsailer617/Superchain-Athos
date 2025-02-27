"""
Yield Farming Service

This service handles DeFi yield farming opportunities:
- Discovery of yield farming opportunities across protocols
- Analysis and comparison of APY/APR
- Automated position management
- Compounding and harvesting rewards
- Risk assessment and optimization
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Set, Any, Tuple, Union
import uuid

from ..cqrs.events.token_discovery import TokenAggregate, TokenStatus, LiquiditySource
from ..cqrs.events.aggregates import aggregate_factory
from ..core.dependency_injector import container
from ..blockchain.web3_client import Web3Client
from ..blockchain.transaction_builder import TransactionBuilder
from ..blockchain.gas_estimator import GasEstimator
from ..blockchain.transaction_simulator import TransactionSimulator

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Domain Models
# -----------------------------------------------------------------------------

class YieldProtocol:
    """Supported yield farming protocols"""
    AAVE = "aave"
    COMPOUND = "compound"
    CURVE = "curve"
    CONVEX = "convex"
    YEARN = "yearn"
    SUSHISWAP = "sushiswap"
    UNISWAP_V3 = "uniswap_v3"
    BALANCER = "balancer"
    PANCAKESWAP = "pancakeswap"
    TRADER_JOE = "trader_joe"
    BEEFY = "beefy"
    CUSTOM = "custom"

class YieldType:
    """Types of yield"""
    LENDING = "lending"
    LIQUIDITY_PROVIDING = "liquidity_providing"
    STAKING = "staking"
    FARMING = "farming"
    LEVERAGED = "leveraged"

class RiskLevel:
    """Risk levels for yield opportunities"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class YieldOpportunity:
    """Yield farming opportunity"""
    
    def __init__(
        self,
        id: str,
        chain_id: int,
        protocol: str,
        yield_type: str,
        pool_id: str,
        pool_name: str,
        tokens: List[str],
        token_symbols: List[str],
        apy: Decimal,
        tvl_usd: Decimal,
        risk_level: str,
        requires_active_management: bool = False,
        min_deposit_usd: Optional[Decimal] = None,
        max_deposit_usd: Optional[Decimal] = None,
        rewards_tokens: Optional[List[str]] = None,
        rewards_apy: Optional[Decimal] = None,
        total_apy: Optional[Decimal] = None,
        updated_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.id = id
        self.chain_id = chain_id
        self.protocol = protocol
        self.yield_type = yield_type
        self.pool_id = pool_id
        self.pool_name = pool_name
        self.tokens = tokens
        self.token_symbols = token_symbols
        self.apy = apy
        self.tvl_usd = tvl_usd
        self.risk_level = risk_level
        self.requires_active_management = requires_active_management
        self.min_deposit_usd = min_deposit_usd
        self.max_deposit_usd = max_deposit_usd
        self.rewards_tokens = rewards_tokens or []
        self.rewards_apy = rewards_apy or Decimal("0")
        self.total_apy = total_apy or apy
        self.updated_at = updated_at or datetime.now()
        self.metadata = metadata or {}

class YieldPosition:
    """User position in a yield farming opportunity"""
    
    def __init__(
        self,
        id: str,
        opportunity_id: str,
        wallet_address: str,
        deposit_amounts: Dict[str, Decimal],  # token_address -> amount
        deposit_usd_value: Decimal,
        entry_timestamp: datetime,
        last_harvest_timestamp: Optional[datetime] = None,
        harvested_rewards: Optional[Dict[str, Decimal]] = None,
        current_value_usd: Optional[Decimal] = None,
        profit_usd: Optional[Decimal] = None,
        apy_realized: Optional[Decimal] = None,
        status: str = "active",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.id = id
        self.opportunity_id = opportunity_id
        self.wallet_address = wallet_address
        self.deposit_amounts = deposit_amounts
        self.deposit_usd_value = deposit_usd_value
        self.entry_timestamp = entry_timestamp
        self.last_harvest_timestamp = last_harvest_timestamp
        self.harvested_rewards = harvested_rewards or {}
        self.current_value_usd = current_value_usd or deposit_usd_value
        self.profit_usd = profit_usd or Decimal("0")
        self.apy_realized = apy_realized or Decimal("0")
        self.status = status
        self.metadata = metadata or {}

class YieldFarmingService:
    """Service for managing yield farming opportunities"""
    
    def __init__(
        self,
        web3_client: Web3Client,
        transaction_builder: TransactionBuilder,
        gas_estimator: GasEstimator,
        transaction_simulator: TransactionSimulator
    ):
        self.web3_client = web3_client
        self.transaction_builder = transaction_builder
        self.gas_estimator = gas_estimator
        self.transaction_simulator = transaction_simulator
        
        # In-memory storage (would be replaced with a database in production)
        self.opportunities: Dict[str, YieldOpportunity] = {}
        self.positions: Dict[str, YieldPosition] = {}
        
        # Protocol adapters
        self.protocol_adapters = {}
        
        # Configuration
        self.min_apy_threshold = Decimal("5.0")  # Minimum 5% APY
        self.max_risk_level = RiskLevel.HIGH     # Maximum risk level to consider
        self.auto_compound_threshold = Decimal("50.0")  # Auto-compound if rewards > $50
        self.harvest_interval_hours = 24         # Harvest every 24 hours
        
        # Start background tasks
        self._start_background_tasks()
    
    async def discover_opportunities(
        self,
        chain_id: int,
        protocols: Optional[List[str]] = None,
        yield_types: Optional[List[str]] = None,
        min_apy: Optional[Decimal] = None,
        max_risk_level: Optional[str] = None,
        tokens: Optional[List[str]] = None,
        max_results: int = 100
    ) -> List[YieldOpportunity]:
        """Discover yield farming opportunities
        
        Args:
            chain_id: Chain ID to search on
            protocols: Optional list of protocols to check
            yield_types: Optional list of yield types to look for
            min_apy: Minimum APY to consider
            max_risk_level: Maximum risk level to consider
            tokens: Optional list of tokens to filter by
            max_results: Maximum number of results to return
            
        Returns:
            List of yield opportunities
        """
        logger.info(f"Discovering yield opportunities on chain {chain_id}")
        
        # Use default values if not specified
        min_apy = min_apy or self.min_apy_threshold
        max_risk_level = max_risk_level or self.max_risk_level
        
        # Default to all protocols if not specified
        if not protocols:
            protocols = [
                YieldProtocol.AAVE,
                YieldProtocol.COMPOUND,
                YieldProtocol.CURVE,
                YieldProtocol.YEARN,
                YieldProtocol.SUSHISWAP,
                YieldProtocol.UNISWAP_V3
            ]
        
        # Default to all yield types if not specified
        if not yield_types:
            yield_types = [
                YieldType.LENDING,
                YieldType.LIQUIDITY_PROVIDING,
                YieldType.STAKING,
                YieldType.FARMING
            ]
        
        # Collect opportunities from all protocols
        all_opportunities = []
        
        for protocol in protocols:
            # Get protocol adapter
            adapter = self._get_protocol_adapter(protocol, chain_id)
            if not adapter:
                logger.warning(f"No adapter found for protocol {protocol} on chain {chain_id}")
                continue
            
            # Get opportunities from protocol
            try:
                protocol_opportunities = await adapter.get_opportunities(yield_types)
                all_opportunities.extend(protocol_opportunities)
            except Exception as e:
                logger.error(f"Error getting opportunities from {protocol}: {str(e)}")
        
        # Filter opportunities
        filtered_opportunities = []
        
        for opportunity in all_opportunities:
            # Check APY
            if opportunity.total_apy < min_apy:
                continue
            
            # Check risk level
            if self._risk_level_value(opportunity.risk_level) > self._risk_level_value(max_risk_level):
                continue
            
            # Check tokens if specified
            if tokens and not any(token in opportunity.tokens for token in tokens):
                continue
            
            # Add to filtered list
            filtered_opportunities.append(opportunity)
            
            # Store opportunity
            self.opportunities[opportunity.id] = opportunity
        
        # Sort by APY (highest first)
        filtered_opportunities.sort(key=lambda o: o.total_apy, reverse=True)
        
        logger.info(f"Found {len(filtered_opportunities)} yield opportunities")
        return filtered_opportunities[:max_results]
    
    async def get_opportunity(self, opportunity_id: str) -> Optional[YieldOpportunity]:
        """Get a yield opportunity by ID
        
        Args:
            opportunity_id: Opportunity ID
            
        Returns:
            Yield opportunity if found, None otherwise
        """
        # Check in-memory cache
        if opportunity_id in self.opportunities:
            return self.opportunities[opportunity_id]
        
        # If not in cache, try to fetch from protocol
        parts = opportunity_id.split(":")
        if len(parts) >= 3:
            protocol = parts[0]
            chain_id = int(parts[1])
            pool_id = parts[2]
            
            # Get protocol adapter
            adapter = self._get_protocol_adapter(protocol, chain_id)
            if adapter:
                try:
                    opportunity = await adapter.get_opportunity(pool_id)
                    if opportunity:
                        # Store in cache
                        self.opportunities[opportunity.id] = opportunity
                        return opportunity
                except Exception as e:
                    logger.error(f"Error getting opportunity {opportunity_id}: {str(e)}")
        
        return None
    
    async def create_position(
        self,
        opportunity_id: str,
        wallet_address: str,
        deposit_amounts: Dict[str, Decimal],
        gas_price_gwei: Optional[Decimal] = None
    ) -> Optional[str]:
        """Create a new yield farming position
        
        Args:
            opportunity_id: Opportunity ID
            wallet_address: User's wallet address
            deposit_amounts: Token amounts to deposit
            gas_price_gwei: Optional gas price in Gwei
            
        Returns:
            Position ID if successful, None otherwise
        """
        # Get opportunity
        opportunity = await self.get_opportunity(opportunity_id)
        if not opportunity:
            logger.error(f"Opportunity {opportunity_id} not found")
            return None
        
        # Validate deposit amounts
        for token_address in deposit_amounts:
            if token_address not in opportunity.tokens:
                logger.error(f"Token {token_address} not accepted for opportunity {opportunity_id}")
                return None
        
        # Calculate USD value of deposit
        deposit_usd_value = await self._calculate_deposit_value(
            opportunity.chain_id,
            deposit_amounts
        )
        
        # Check minimum deposit
        if opportunity.min_deposit_usd and deposit_usd_value < opportunity.min_deposit_usd:
            logger.error(f"Deposit value ${deposit_usd_value} below minimum ${opportunity.min_deposit_usd}")
            return None
        
        # Check maximum deposit
        if opportunity.max_deposit_usd and deposit_usd_value > opportunity.max_deposit_usd:
            logger.error(f"Deposit value ${deposit_usd_value} above maximum ${opportunity.max_deposit_usd}")
            return None
        
        # Get protocol adapter
        protocol = opportunity.protocol
        chain_id = opportunity.chain_id
        adapter = self._get_protocol_adapter(protocol, chain_id)
        
        if not adapter:
            logger.error(f"No adapter found for protocol {protocol} on chain {chain_id}")
            return None
        
        # Build transaction
        try:
            tx_data = await adapter.build_deposit_transaction(
                opportunity.pool_id,
                wallet_address,
                deposit_amounts
            )
        except Exception as e:
            logger.error(f"Error building deposit transaction: {str(e)}")
            return None
        
        # Set gas price if specified
        if gas_price_gwei:
            tx_data["gasPrice"] = int(gas_price_gwei * 10**9)
        
        # Send transaction
        try:
            tx_hash = await self.web3_client.send_transaction(
                chain_id,
                tx_data,
                wallet_address
            )
            
            # Wait for transaction to be mined
            receipt = await self.web3_client.wait_for_transaction(
                chain_id,
                tx_hash
            )
            
            if receipt["status"] != 1:
                logger.error(f"Deposit transaction failed: {tx_hash}")
                return None
            
            # Create position
            position_id = f"{opportunity_id}:{wallet_address}:{str(uuid.uuid4())[:8]}"
            
            position = YieldPosition(
                id=position_id,
                opportunity_id=opportunity_id,
                wallet_address=wallet_address,
                deposit_amounts=deposit_amounts,
                deposit_usd_value=deposit_usd_value,
                entry_timestamp=datetime.now(),
                metadata={
                    "tx_hash": tx_hash,
                    "block_number": receipt["blockNumber"]
                }
            )
            
            # Store position
            self.positions[position_id] = position
            
            logger.info(f"Created position {position_id} for opportunity {opportunity_id}")
            return position_id
            
        except Exception as e:
            logger.error(f"Error sending deposit transaction: {str(e)}")
            return None
    
    async def get_position(self, position_id: str) -> Optional[YieldPosition]:
        """Get a yield position by ID
        
        Args:
            position_id: Position ID
            
        Returns:
            Yield position if found, None otherwise
        """
        # Check in-memory cache
        if position_id in self.positions:
            position = self.positions[position_id]
            
            # Update position data
            await self._update_position_data(position)
            
            return position
        
        return None
    
    async def get_positions_by_wallet(
        self,
        wallet_address: str,
        chain_id: Optional[int] = None,
        status: Optional[str] = None
    ) -> List[YieldPosition]:
        """Get all positions for a wallet
        
        Args:
            wallet_address: Wallet address
            chain_id: Optional chain ID to filter by
            status: Optional status to filter by
            
        Returns:
            List of yield positions
        """
        positions = []
        
        for position in self.positions.values():
            if position.wallet_address.lower() == wallet_address.lower():
                # Apply filters
                if chain_id:
                    opportunity = await self.get_opportunity(position.opportunity_id)
                    if not opportunity or opportunity.chain_id != chain_id:
                        continue
                
                if status and position.status != status:
                    continue
                
                # Update position data
                await self._update_position_data(position)
                
                positions.append(position)
        
        return positions
    
    async def harvest_rewards(
        self,
        position_id: str,
        auto_compound: bool = False,
        gas_price_gwei: Optional[Decimal] = None
    ) -> bool:
        """Harvest rewards for a position
        
        Args:
            position_id: Position ID
            auto_compound: Whether to auto-compound rewards
            gas_price_gwei: Optional gas price in Gwei
            
        Returns:
            True if successful, False otherwise
        """
        # Get position
        position = await self.get_position(position_id)
        if not position:
            logger.error(f"Position {position_id} not found")
            return False
        
        # Get opportunity
        opportunity = await self.get_opportunity(position.opportunity_id)
        if not opportunity:
            logger.error(f"Opportunity {position.opportunity_id} not found")
            return False
        
        # Get protocol adapter
        protocol = opportunity.protocol
        chain_id = opportunity.chain_id
        adapter = self._get_protocol_adapter(protocol, chain_id)
        
        if not adapter:
            logger.error(f"No adapter found for protocol {protocol} on chain {chain_id}")
            return False
        
        # Check if harvesting is supported
        if not hasattr(adapter, "build_harvest_transaction"):
            logger.error(f"Harvesting not supported for protocol {protocol}")
            return False
        
        # Build transaction
        try:
            tx_data = await adapter.build_harvest_transaction(
                opportunity.pool_id,
                position.wallet_address,
                auto_compound
            )
        except Exception as e:
            logger.error(f"Error building harvest transaction: {str(e)}")
            return False
        
        # Set gas price if specified
        if gas_price_gwei:
            tx_data["gasPrice"] = int(gas_price_gwei * 10**9)
        
        # Send transaction
        try:
            tx_hash = await self.web3_client.send_transaction(
                chain_id,
                tx_data,
                position.wallet_address
            )
            
            # Wait for transaction to be mined
            receipt = await self.web3_client.wait_for_transaction(
                chain_id,
                tx_hash
            )
            
            if receipt["status"] != 1:
                logger.error(f"Harvest transaction failed: {tx_hash}")
                return False
            
            # Update position
            position.last_harvest_timestamp = datetime.now()
            
            # Get harvested rewards
            harvested_rewards = await adapter.get_harvested_rewards(
                opportunity.pool_id,
                position.wallet_address,
                receipt
            )
            
            # Update harvested rewards
            for token, amount in harvested_rewards.items():
                if token in position.harvested_rewards:
                    position.harvested_rewards[token] += amount
                else:
                    position.harvested_rewards[token] = amount
            
            # Update position data
            await self._update_position_data(position)
            
            logger.info(f"Harvested rewards for position {position_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending harvest transaction: {str(e)}")
            return False
    
    async def exit_position(
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
        # Get position
        position = await self.get_position(position_id)
        if not position:
            logger.error(f"Position {position_id} not found")
            return False
        
        # Get opportunity
        opportunity = await self.get_opportunity(position.opportunity_id)
        if not opportunity:
            logger.error(f"Opportunity {position.opportunity_id} not found")
            return False
        
        # Get protocol adapter
        protocol = opportunity.protocol
        chain_id = opportunity.chain_id
        adapter = self._get_protocol_adapter(protocol, chain_id)
        
        if not adapter:
            logger.error(f"No adapter found for protocol {protocol} on chain {chain_id}")
            return False
        
        # Build transaction
        try:
            tx_data = await adapter.build_withdraw_transaction(
                opportunity.pool_id,
                position.wallet_address,
                None  # Withdraw all
            )
        except Exception as e:
            logger.error(f"Error building withdraw transaction: {str(e)}")
            return False
        
        # Set gas price if specified
        if gas_price_gwei:
            tx_data["gasPrice"] = int(gas_price_gwei * 10**9)
        
        # Send transaction
        try:
            tx_hash = await self.web3_client.send_transaction(
                chain_id,
                tx_data,
                position.wallet_address
            )
            
            # Wait for transaction to be mined
            receipt = await self.web3_client.wait_for_transaction(
                chain_id,
                tx_hash
            )
            
            if receipt["status"] != 1:
                logger.error(f"Withdraw transaction failed: {tx_hash}")
                return False
            
            # Update position
            position.status = "closed"
            position.metadata["exit_tx_hash"] = tx_hash
            position.metadata["exit_block_number"] = receipt["blockNumber"]
            position.metadata["exit_timestamp"] = datetime.now().isoformat()
            
            # Update position data one last time
            await self._update_position_data(position)
            
            logger.info(f"Exited position {position_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending withdraw transaction: {str(e)}")
            return False
    
    async def get_best_opportunities(
        self,
        chain_id: Optional[int] = None,
        risk_level: Optional[str] = None,
        yield_type: Optional[str] = None,
        min_tvl_usd: Optional[Decimal] = None,
        token_address: Optional[str] = None,
        limit: int = 10
    ) -> List[YieldOpportunity]:
        """Get the best yield opportunities based on criteria
        
        Args:
            chain_id: Optional chain ID to filter by
            risk_level: Optional maximum risk level
            yield_type: Optional yield type to filter by
            min_tvl_usd: Optional minimum TVL in USD
            token_address: Optional token address to filter by
            limit: Maximum number of results
            
        Returns:
            List of yield opportunities
        """
        # Filter opportunities
        filtered = []
        
        for opportunity in self.opportunities.values():
            # Apply filters
            if chain_id and opportunity.chain_id != chain_id:
                continue
            
            if risk_level and self._risk_level_value(opportunity.risk_level) > self._risk_level_value(risk_level):
                continue
            
            if yield_type and opportunity.yield_type != yield_type:
                continue
            
            if min_tvl_usd and opportunity.tvl_usd < min_tvl_usd:
                continue
            
            if token_address and token_address not in opportunity.tokens:
                continue
            
            filtered.append(opportunity)
        
        # Sort by APY (highest first)
        filtered.sort(key=lambda o: o.total_apy, reverse=True)
        
        return filtered[:limit]
    
    async def _calculate_deposit_value(
        self,
        chain_id: int,
        deposit_amounts: Dict[str, Decimal]
    ) -> Decimal:
        """Calculate the USD value of a deposit
        
        Args:
            chain_id: Chain ID
            deposit_amounts: Token amounts to deposit
            
        Returns:
            USD value of deposit
        """
        total_value = Decimal("0")
        
        for token_address, amount in deposit_amounts.items():
            # Get token price
            token_price = await self._get_token_price_usd(chain_id, token_address)
            
            # Add to total
            total_value += amount * token_price
        
        return total_value
    
    async def _get_token_price_usd(self, chain_id: int, token_address: str) -> Decimal:
        """Get the USD price of a token
        
        Args:
            chain_id: Chain ID
            token_address: Token address
            
        Returns:
            USD price of token
        """
        # Try to get from token aggregate
        token_repository = aggregate_factory(TokenAggregate)
        token = await token_repository.get_by_id(token_address)
        
        if token and token.state.current_price_usd:
            return token.state.current_price_usd
        
        # If not available, use a price oracle
        # This is a placeholder - in a real implementation, you would use a price oracle
        return Decimal("1.0")
    
    async def _update_position_data(self, position: YieldPosition) -> None:
        """Update position data with latest values
        
        Args:
            position: Position to update
        """
        # Get opportunity
        opportunity = await self.get_opportunity(position.opportunity_id)
        if not opportunity:
            logger.warning(f"Opportunity {position.opportunity_id} not found")
            return
        
        # Get protocol adapter
        protocol = opportunity.protocol
        chain_id = opportunity.chain_id
        adapter = self._get_protocol_adapter(protocol, chain_id)
        
        if not adapter:
            logger.warning(f"No adapter found for protocol {protocol} on chain {chain_id}")
            return
        
        try:
            # Get current position value
            position_data = await adapter.get_position_data(
                opportunity.pool_id,
                position.wallet_address
            )
            
            if position_data:
                # Update current value
                position.current_value_usd = position_data.get("value_usd", position.deposit_usd_value)
                
                # Calculate profit
                position.profit_usd = position.current_value_usd - position.deposit_usd_value
                
                # Calculate realized APY
                days_since_entry = (datetime.now() - position.entry_timestamp).total_seconds() / (24 * 3600)
                if days_since_entry > 0:
                    position.apy_realized = (position.profit_usd / position.deposit_usd_value) * (365 / days_since_entry) * 100
        except Exception as e:
            logger.error(f"Error updating position data: {str(e)}")
    
    def _get_protocol_adapter(self, protocol: str, chain_id: int):
        """Get a protocol adapter
        
        Args:
            protocol: Protocol name
            chain_id: Chain ID
            
        Returns:
            Protocol adapter if available, None otherwise
        """
        adapter_key = f"{protocol}:{chain_id}"
        
        if adapter_key in self.protocol_adapters:
            return self.protocol_adapters[adapter_key]
        
        # This would normally create and return a protocol adapter
        # For now, we'll just return None
        return None
    
    def _risk_level_value(self, risk_level: str) -> int:
        """Convert risk level to numeric value for comparison
        
        Args:
            risk_level: Risk level string
            
        Returns:
            Numeric value (higher = riskier)
        """
        risk_values = {
            RiskLevel.VERY_LOW: 1,
            RiskLevel.LOW: 2,
            RiskLevel.MEDIUM: 3,
            RiskLevel.HIGH: 4,
            RiskLevel.VERY_HIGH: 5
        }
        
        return risk_values.get(risk_level, 3)  # Default to medium
    
    def _start_background_tasks(self) -> None:
        """Start background tasks"""
        # Start opportunity updater
        asyncio.create_task(self._update_opportunities_periodically())
        
        # Start position manager
        asyncio.create_task(self._manage_positions_periodically())
    
    async def _update_opportunities_periodically(self) -> None:
        """Periodically update yield opportunities"""
        while True:
            try:
                # Update opportunities on main chains
                for chain_id in [1, 56, 137]:  # Ethereum, BSC, Polygon
                    await self.discover_opportunities(chain_id)
                
                # Sleep for 1 hour
                await asyncio.sleep(3600)
            except Exception as e:
                logger.error(f"Error updating opportunities: {str(e)}")
                await asyncio.sleep(60)  # Sleep on error
    
    async def _manage_positions_periodically(self) -> None:
        """Periodically manage yield positions"""
        while True:
            try:
                # Check all active positions
                for position_id, position in list(self.positions.items()):
                    if position.status != "active":
                        continue
                    
                    # Update position data
                    await self._update_position_data(position)
                    
                    # Check if harvesting is needed
                    if position.last_harvest_timestamp:
                        hours_since_harvest = (datetime.now() - position.last_harvest_timestamp).total_seconds() / 3600
                        if hours_since_harvest >= self.harvest_interval_hours:
                            # Auto-harvest if configured
                            opportunity = await self.get_opportunity(position.opportunity_id)
                            if opportunity and opportunity.requires_active_management:
                                await self.harvest_rewards(position_id, auto_compound=True)
                
                # Sleep for 15 minutes
                await asyncio.sleep(900)
            except Exception as e:
                logger.error(f"Error managing positions: {str(e)}")
                await asyncio.sleep(60)  # Sleep on error

# Create singleton instance
yield_farming_service = YieldFarmingService(
    web3_client=container.resolve(Web3Client),
    transaction_builder=container.resolve(TransactionBuilder),
    gas_estimator=container.resolve(GasEstimator),
    transaction_simulator=container.resolve(TransactionSimulator)
) 