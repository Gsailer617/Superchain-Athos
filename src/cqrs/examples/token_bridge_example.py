"""
Token Bridge CQRS Example

This module provides a comprehensive example of using the CQRS pattern for a token bridge:
- Commands and queries for token bridging
- Command and query handlers
- Event handling
- Aggregate for token bridge state
"""

import asyncio
import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any

from pydantic import BaseModel, Field

from ..commands.base import Command, CommandHandler
from ..queries.base import Query, QueryHandler
from ..events.base import Event, EventHandler
from ..decorators import handles_command, handles_query, handles_event, command, query
from ..events.aggregates import Aggregate, AggregateState
from ..events.event_bus import EventBus
from ..handlers.dispatcher import CommandDispatcher, QueryDispatcher
from ...core.dependency_injector import container

# -----------------------------------------------------------------------------
# Domain Models
# -----------------------------------------------------------------------------

class BridgeNetwork(str, Enum):
    """Supported bridge networks"""
    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    BINANCE = "binance"

class TokenType(str, Enum):
    """Token types"""
    ERC20 = "erc20"
    ERC721 = "erc721"
    ERC1155 = "erc1155"

class BridgeStatus(str, Enum):
    """Bridge transaction status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class Token(BaseModel):
    """Token model"""
    address: str
    symbol: str
    decimals: int
    token_type: TokenType = TokenType.ERC20
    
    def __str__(self) -> str:
        return f"{self.symbol} ({self.address[:8]}...{self.address[-6:]})"

class BridgeTransaction(BaseModel):
    """Bridge transaction model"""
    id: str
    user_address: str
    source_network: BridgeNetwork
    target_network: BridgeNetwork
    token: Token
    amount: Decimal
    status: BridgeStatus = BridgeStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    hash_source: Optional[str] = None
    hash_target: Optional[str] = None
    error_message: Optional[str] = None

# -----------------------------------------------------------------------------
# Aggregate
# -----------------------------------------------------------------------------

class BridgeState(AggregateState):
    """State for bridge aggregate"""
    user_address: str
    source_network: BridgeNetwork
    target_network: BridgeNetwork
    token: Token
    amount: Decimal
    status: BridgeStatus = BridgeStatus.PENDING
    hash_source: Optional[str] = None
    hash_target: Optional[str] = None
    error_message: Optional[str] = None

class BridgeAggregate(Aggregate[BridgeState]):
    """Aggregate for managing bridge transactions"""
    
    @classmethod
    async def create_bridge(
        cls,
        user_address: str,
        source_network: BridgeNetwork,
        target_network: BridgeNetwork,
        token: Token,
        amount: Decimal
    ) -> "BridgeAggregate":
        """Create a new bridge transaction
        
        Args:
            user_address: User's wallet address
            source_network: Source network
            target_network: Target network
            token: Token to bridge
            amount: Amount to bridge
            
        Returns:
            Bridge aggregate
        """
        # Create aggregate
        aggregate = await cls.create(
            user_address=user_address,
            source_network=source_network,
            target_network=target_network,
            token=token,
            amount=amount,
            status=BridgeStatus.PENDING
        )
        
        # Record creation event
        aggregate.record_event(
            event_type="BRIDGE_CREATED",
            payload={
                "user_address": user_address,
                "source_network": source_network,
                "target_network": target_network,
                "token": token.dict(),
                "amount": str(amount)
            }
        )
        
        return aggregate
    
    def start_bridging(self) -> None:
        """Start the bridging process"""
        if self.state.status != BridgeStatus.PENDING:
            raise ValueError(f"Cannot start bridging from status {self.state.status}")
        
        self.record_event(
            event_type="BRIDGE_STARTED",
            payload={
                "id": self.state.id
            }
        )
    
    def set_source_hash(self, hash_source: str) -> None:
        """Set source transaction hash
        
        Args:
            hash_source: Source transaction hash
        """
        self.record_event(
            event_type="SOURCE_HASH_SET",
            payload={
                "hash_source": hash_source
            }
        )
    
    def set_target_hash(self, hash_target: str) -> None:
        """Set target transaction hash
        
        Args:
            hash_target: Target transaction hash
        """
        self.record_event(
            event_type="TARGET_HASH_SET",
            payload={
                "hash_target": hash_target
            }
        )
    
    def complete_bridging(self) -> None:
        """Mark bridging as completed"""
        if self.state.status == BridgeStatus.COMPLETED:
            return
        
        self.record_event(
            event_type="BRIDGE_COMPLETED",
            payload={
                "id": self.state.id
            }
        )
    
    def fail_bridging(self, error_message: str) -> None:
        """Mark bridging as failed
        
        Args:
            error_message: Error message
        """
        if self.state.status == BridgeStatus.FAILED:
            return
        
        self.record_event(
            event_type="BRIDGE_FAILED",
            payload={
                "id": self.state.id,
                "error_message": error_message
            }
        )
    
    # Event handlers
    
    def apply_bridge_created(self, event: Event) -> None:
        """Apply BRIDGE_CREATED event
        
        Args:
            event: Event
        """
        # State is already initialized in constructor
        pass
    
    def apply_bridge_started(self, event: Event) -> None:
        """Apply BRIDGE_STARTED event
        
        Args:
            event: Event
        """
        self.state.status = BridgeStatus.IN_PROGRESS
        self.state.updated_at = datetime.now()
    
    def apply_source_hash_set(self, event: Event) -> None:
        """Apply SOURCE_HASH_SET event
        
        Args:
            event: Event
        """
        self.state.hash_source = event.payload["hash_source"]
        self.state.updated_at = datetime.now()
    
    def apply_target_hash_set(self, event: Event) -> None:
        """Apply TARGET_HASH_SET event
        
        Args:
            event: Event
        """
        self.state.hash_target = event.payload["hash_target"]
        self.state.updated_at = datetime.now()
    
    def apply_bridge_completed(self, event: Event) -> None:
        """Apply BRIDGE_COMPLETED event
        
        Args:
            event: Event
        """
        self.state.status = BridgeStatus.COMPLETED
        self.state.updated_at = datetime.now()
    
    def apply_bridge_failed(self, event: Event) -> None:
        """Apply BRIDGE_FAILED event
        
        Args:
            event: Event
        """
        self.state.status = BridgeStatus.FAILED
        self.state.error_message = event.payload["error_message"]
        self.state.updated_at = datetime.now()

# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------

class InitiateBridgeCommand(Command):
    """Command to initiate a bridge transaction"""
    user_address: str
    source_network: BridgeNetwork
    target_network: BridgeNetwork
    token_address: str
    token_symbol: str
    token_decimals: int
    token_type: TokenType = TokenType.ERC20
    amount: Decimal

class SetSourceHashCommand(Command):
    """Command to set source transaction hash"""
    bridge_id: str
    hash_source: str

class SetTargetHashCommand(Command):
    """Command to set target transaction hash"""
    bridge_id: str
    hash_target: str

class CompleteBridgeCommand(Command):
    """Command to mark bridge as completed"""
    bridge_id: str

class FailBridgeCommand(Command):
    """Command to mark bridge as failed"""
    bridge_id: str
    error_message: str

# -----------------------------------------------------------------------------
# Command Handlers
# -----------------------------------------------------------------------------

@handles_command(InitiateBridgeCommand)
class InitiateBridgeHandler(CommandHandler[InitiateBridgeCommand]):
    """Handler for InitiateBridgeCommand"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
    
    async def handle(self, command: InitiateBridgeCommand) -> None:
        """Handle command
        
        Args:
            command: Command
        """
        # Create token
        token = Token(
            address=command.token_address,
            symbol=command.token_symbol,
            decimals=command.token_decimals,
            token_type=command.token_type
        )
        
        # Create bridge aggregate
        bridge = await BridgeAggregate.create_bridge(
            user_address=command.user_address,
            source_network=command.source_network,
            target_network=command.target_network,
            token=token,
            amount=command.amount
        )
        
        # Start bridging
        bridge.start_bridging()
        
        # Get repository
        from ..events.aggregates import aggregate_factory
        repository = aggregate_factory(BridgeAggregate)
        
        # Save aggregate
        await repository.save(bridge)

@handles_command(SetSourceHashCommand)
class SetSourceHashHandler(CommandHandler[SetSourceHashCommand]):
    """Handler for SetSourceHashCommand"""
    
    async def handle(self, command: SetSourceHashCommand) -> None:
        """Handle command
        
        Args:
            command: Command
        """
        # Get repository
        from ..events.aggregates import aggregate_factory
        repository = aggregate_factory(BridgeAggregate)
        
        # Get aggregate
        bridge = await repository.get_by_id(command.bridge_id)
        if not bridge:
            raise ValueError(f"Bridge {command.bridge_id} not found")
        
        # Set source hash
        bridge.set_source_hash(command.hash_source)
        
        # Save aggregate
        await repository.save(bridge)

@handles_command(SetTargetHashCommand)
class SetTargetHashHandler(CommandHandler[SetTargetHashCommand]):
    """Handler for SetTargetHashCommand"""
    
    async def handle(self, command: SetTargetHashCommand) -> None:
        """Handle command
        
        Args:
            command: Command
        """
        # Get repository
        from ..events.aggregates import aggregate_factory
        repository = aggregate_factory(BridgeAggregate)
        
        # Get aggregate
        bridge = await repository.get_by_id(command.bridge_id)
        if not bridge:
            raise ValueError(f"Bridge {command.bridge_id} not found")
        
        # Set target hash
        bridge.set_target_hash(command.hash_target)
        
        # Save aggregate
        await repository.save(bridge)

@handles_command(CompleteBridgeCommand)
class CompleteBridgeHandler(CommandHandler[CompleteBridgeCommand]):
    """Handler for CompleteBridgeCommand"""
    
    async def handle(self, command: CompleteBridgeCommand) -> None:
        """Handle command
        
        Args:
            command: Command
        """
        # Get repository
        from ..events.aggregates import aggregate_factory
        repository = aggregate_factory(BridgeAggregate)
        
        # Get aggregate
        bridge = await repository.get_by_id(command.bridge_id)
        if not bridge:
            raise ValueError(f"Bridge {command.bridge_id} not found")
        
        # Complete bridging
        bridge.complete_bridging()
        
        # Save aggregate
        await repository.save(bridge)

@handles_command(FailBridgeCommand)
class FailBridgeHandler(CommandHandler[FailBridgeCommand]):
    """Handler for FailBridgeCommand"""
    
    async def handle(self, command: FailBridgeCommand) -> None:
        """Handle command
        
        Args:
            command: Command
        """
        # Get repository
        from ..events.aggregates import aggregate_factory
        repository = aggregate_factory(BridgeAggregate)
        
        # Get aggregate
        bridge = await repository.get_by_id(command.bridge_id)
        if not bridge:
            raise ValueError(f"Bridge {command.bridge_id} not found")
        
        # Fail bridging
        bridge.fail_bridging(command.error_message)
        
        # Save aggregate
        await repository.save(bridge)

# -----------------------------------------------------------------------------
# Queries
# -----------------------------------------------------------------------------

class GetBridgeQuery(Query):
    """Query to get bridge by ID"""
    bridge_id: str

class GetBridgesByUserQuery(Query):
    """Query to get bridges by user"""
    user_address: str
    status: Optional[BridgeStatus] = None
    limit: int = 10
    offset: int = 0

class GetBridgeStatsQuery(Query):
    """Query to get bridge statistics"""
    period: str = "day"  # day, week, month, all

# -----------------------------------------------------------------------------
# Query Results
# -----------------------------------------------------------------------------

class BridgeResult(BaseModel):
    """Result for GetBridgeQuery"""
    id: str
    user_address: str
    source_network: BridgeNetwork
    target_network: BridgeNetwork
    token: Token
    amount: Decimal
    status: BridgeStatus
    created_at: datetime
    updated_at: datetime
    hash_source: Optional[str] = None
    hash_target: Optional[str] = None
    error_message: Optional[str] = None

class BridgeListResult(BaseModel):
    """Result for GetBridgesByUserQuery"""
    bridges: List[BridgeResult]
    total: int

class BridgeStatItem(BaseModel):
    """Bridge statistics item"""
    date: str
    count: int
    volume: Decimal
    success_rate: float

class BridgeStatsResult(BaseModel):
    """Result for GetBridgeStatsQuery"""
    period: str
    total_count: int
    total_volume: Decimal
    average_success_rate: float
    by_network: Dict[BridgeNetwork, BridgeStatItem]
    by_date: List[BridgeStatItem]

# -----------------------------------------------------------------------------
# Query Handlers
# -----------------------------------------------------------------------------

@handles_query(GetBridgeQuery)
class GetBridgeHandler(QueryHandler[GetBridgeQuery, BridgeResult]):
    """Handler for GetBridgeQuery"""
    
    async def handle(self, query: GetBridgeQuery) -> BridgeResult:
        """Handle query
        
        Args:
            query: Query
            
        Returns:
            Bridge information
        """
        # Get repository
        from ..events.aggregates import aggregate_factory
        repository = aggregate_factory(BridgeAggregate)
        
        # Get aggregate
        bridge = await repository.get_by_id(query.bridge_id)
        if not bridge:
            raise ValueError(f"Bridge {query.bridge_id} not found")
        
        # Return result
        return BridgeResult(
            id=bridge.id,
            user_address=bridge.state.user_address,
            source_network=bridge.state.source_network,
            target_network=bridge.state.target_network,
            token=bridge.state.token,
            amount=bridge.state.amount,
            status=bridge.state.status,
            created_at=bridge.state.created_at,
            updated_at=bridge.state.updated_at,
            hash_source=bridge.state.hash_source,
            hash_target=bridge.state.hash_target,
            error_message=bridge.state.error_message
        )

@handles_query(GetBridgesByUserQuery)
class GetBridgesByUserHandler(QueryHandler[GetBridgesByUserQuery, BridgeListResult]):
    """Handler for GetBridgesByUserQuery"""
    
    async def handle(self, query: GetBridgesByUserQuery) -> BridgeListResult:
        """Handle query
        
        Args:
            query: Query
            
        Returns:
            List of bridges
        """
        # This is a simplified implementation that would normally query a read model database
        # In a real application, you would use a dedicated read model optimized for queries
        
        # For demonstration purposes, we'll simulate a list of bridges
        bridges = []
        
        # Get event store
        from ..events.base import EventStore
        event_store = container.resolve(EventStore)
        
        # Get events of type BRIDGE_CREATED
        events = await event_store.get_events_by_type("BRIDGE_CREATED")
        
        # Filter events by user address
        user_events = [e for e in events if e.payload.get("user_address") == query.user_address]
        
        # Get aggregates for each bridge
        from ..events.aggregates import aggregate_factory
        repository = aggregate_factory(BridgeAggregate)
        
        for event in user_events:
            # Get aggregate
            bridge = await repository.get_by_id(event.aggregate_id)
            if not bridge:
                continue
            
            # Filter by status if specified
            if query.status and bridge.state.status != query.status:
                continue
            
            # Add to list
            bridges.append(BridgeResult(
                id=bridge.id,
                user_address=bridge.state.user_address,
                source_network=bridge.state.source_network,
                target_network=bridge.state.target_network,
                token=bridge.state.token,
                amount=bridge.state.amount,
                status=bridge.state.status,
                created_at=bridge.state.created_at,
                updated_at=bridge.state.updated_at,
                hash_source=bridge.state.hash_source,
                hash_target=bridge.state.hash_target,
                error_message=bridge.state.error_message
            ))
        
        # Sort by created_at (newest first)
        bridges.sort(key=lambda b: b.created_at, reverse=True)
        
        # Apply pagination
        total = len(bridges)
        bridges = bridges[query.offset:query.offset + query.limit]
        
        return BridgeListResult(
            bridges=bridges,
            total=total
        )

@handles_query(GetBridgeStatsQuery)
class GetBridgeStatsHandler(QueryHandler[GetBridgeStatsQuery, BridgeStatsResult]):
    """Handler for GetBridgeStatsQuery"""
    
    async def handle(self, query: GetBridgeStatsQuery) -> BridgeStatsResult:
        """Handle query
        
        Args:
            query: Query
            
        Returns:
            Bridge statistics
        """
        # This is a simplified implementation that would normally query a read model database
        # In a real application, you would use a dedicated read model optimized for statistics
        
        # For demonstration purposes, we'll return sample statistics
        from datetime import timedelta
        import random
        
        # Generate dates based on period
        if query.period == "day":
            dates = [(datetime.now() - timedelta(hours=i)).strftime("%H:00") for i in range(24)]
        elif query.period == "week":
            dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
        elif query.period == "month":
            dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(30)]
        else:  # all
            dates = [(datetime.now() - timedelta(days=i*30)).strftime("%Y-%m") for i in range(6)]
        
        # Generate statistics by date
        by_date = []
        total_count = 0
        total_volume = Decimal(0)
        total_success = 0
        
        for date in dates:
            count = random.randint(10, 100)
            volume = Decimal(random.randint(5000, 50000))
            success_rate = random.uniform(0.9, 1.0)
            
            by_date.append(BridgeStatItem(
                date=date,
                count=count,
                volume=volume,
                success_rate=success_rate
            ))
            
            total_count += count
            total_volume += volume
            total_success += int(count * success_rate)
        
        # Generate statistics by network
        by_network = {}
        for network in BridgeNetwork:
            count = random.randint(50, 500)
            volume = Decimal(random.randint(10000, 100000))
            success_rate = random.uniform(0.9, 1.0)
            
            by_network[network] = BridgeStatItem(
                date=query.period,
                count=count,
                volume=volume,
                success_rate=success_rate
            )
        
        # Calculate average success rate
        average_success_rate = total_success / total_count if total_count > 0 else 1.0
        
        return BridgeStatsResult(
            period=query.period,
            total_count=total_count,
            total_volume=total_volume,
            average_success_rate=average_success_rate,
            by_network=by_network,
            by_date=by_date
        )

# -----------------------------------------------------------------------------
# Event Handlers
# -----------------------------------------------------------------------------

@handles_event("BRIDGE_CREATED")
class BridgeCreatedHandler(EventHandler):
    """Handler for BRIDGE_CREATED event"""
    
    async def handle(self, event: Event) -> None:
        """Handle event
        
        Args:
            event: Event
        """
        print(f"Bridge created: {event.aggregate_id}")
        
        # This would typically update a read model or trigger side effects
        # For example, sending notifications, updating statistics, etc.

@handles_event("BRIDGE_STARTED")
class BridgeStartedHandler(EventHandler):
    """Handler for BRIDGE_STARTED event"""
    
    async def handle(self, event: Event) -> None:
        """Handle event
        
        Args:
            event: Event
        """
        print(f"Bridge started: {event.aggregate_id}")
        
        # In a real application, this would initiate the actual blockchain transaction
        # For demonstration purposes, we'll simulate setting the source hash after a delay
        async def simulate_source_tx():
            # Wait for transaction
            await asyncio.sleep(2)
            
            # Get command dispatcher
            command_dispatcher = container.resolve(CommandDispatcher)
            
            # Set source hash
            await command_dispatcher.dispatch(SetSourceHashCommand(
                id=str(uuid.uuid4()),
                bridge_id=event.aggregate_id,
                hash_source=f"0x{uuid.uuid4().hex}"
            ))
        
        # Start simulation
        asyncio.create_task(simulate_source_tx())

@handles_event("SOURCE_HASH_SET")
class SourceHashSetHandler(EventHandler):
    """Handler for SOURCE_HASH_SET event"""
    
    async def handle(self, event: Event) -> None:
        """Handle event
        
        Args:
            event: Event
        """
        print(f"Source hash set: {event.payload['hash_source']}")
        
        # In a real application, this would monitor the transaction and wait for confirmation
        # For demonstration purposes, we'll simulate setting the target hash after a delay
        async def simulate_target_tx():
            # Wait for transaction
            await asyncio.sleep(3)
            
            # Get command dispatcher
            command_dispatcher = container.resolve(CommandDispatcher)
            
            # Set target hash
            await command_dispatcher.dispatch(SetTargetHashCommand(
                id=str(uuid.uuid4()),
                bridge_id=event.aggregate_id,
                hash_target=f"0x{uuid.uuid4().hex}"
            ))
        
        # Start simulation
        asyncio.create_task(simulate_target_tx())

@handles_event("TARGET_HASH_SET")
class TargetHashSetHandler(EventHandler):
    """Handler for TARGET_HASH_SET event"""
    
    async def handle(self, event: Event) -> None:
        """Handle event
        
        Args:
            event: Event
        """
        print(f"Target hash set: {event.payload['hash_target']}")
        
        # In a real application, this would monitor the transaction and wait for confirmation
        # For demonstration purposes, we'll simulate completing the bridge after a delay
        async def simulate_completion():
            # Wait for transaction
            await asyncio.sleep(1)
            
            # Get command dispatcher
            command_dispatcher = container.resolve(CommandDispatcher)
            
            # Complete bridge
            await command_dispatcher.dispatch(CompleteBridgeCommand(
                id=str(uuid.uuid4()),
                bridge_id=event.aggregate_id
            ))
        
        # Start simulation
        asyncio.create_task(simulate_completion())

@handles_event("BRIDGE_COMPLETED")
class BridgeCompletedHandler(EventHandler):
    """Handler for BRIDGE_COMPLETED event"""
    
    async def handle(self, event: Event) -> None:
        """Handle event
        
        Args:
            event: Event
        """
        print(f"Bridge completed: {event.aggregate_id}")
        
        # This would typically update a read model or trigger side effects
        # For example, sending notifications, updating statistics, etc.

@handles_event("BRIDGE_FAILED")
class BridgeFailedHandler(EventHandler):
    """Handler for BRIDGE_FAILED event"""
    
    async def handle(self, event: Event) -> None:
        """Handle event
        
        Args:
            event: Event
        """
        print(f"Bridge failed: {event.aggregate_id} - {event.payload['error_message']}")
        
        # This would typically update a read model or trigger side effects
        # For example, sending notifications, updating statistics, etc.

# -----------------------------------------------------------------------------
# Helper Functions with Decorators
# -----------------------------------------------------------------------------

@command()
async def initiate_bridge(
    user_address: str,
    source_network: BridgeNetwork,
    target_network: BridgeNetwork,
    token_address: str,
    token_symbol: str,
    token_decimals: int,
    token_type: TokenType,
    amount: Decimal
) -> str:
    """Initiate a bridge transaction
    
    This function is automatically converted to a command and dispatched.
    
    Args:
        user_address: User's wallet address
        source_network: Source network
        target_network: Target network
        token_address: Token address
        token_symbol: Token symbol
        token_decimals: Token decimals
        token_type: Token type
        amount: Amount to bridge
        
    Returns:
        Bridge ID
    """
    # This function body is never executed
    # The decorator converts it to a command and dispatches it
    pass

@query()
async def get_bridge(bridge_id: str) -> BridgeResult:
    """Get bridge by ID
    
    This function is automatically converted to a query and dispatched.
    
    Args:
        bridge_id: Bridge ID
        
    Returns:
        Bridge information
    """
    # This function body is never executed
    # The decorator converts it to a query and dispatches it
    pass

@query()
async def get_bridges_by_user(
    user_address: str,
    status: Optional[BridgeStatus] = None,
    limit: int = 10,
    offset: int = 0
) -> BridgeListResult:
    """Get bridges by user
    
    This function is automatically converted to a query and dispatched.
    
    Args:
        user_address: User's wallet address
        status: Filter by status
        limit: Maximum number of results
        offset: Offset for pagination
        
    Returns:
        List of bridges
    """
    # This function body is never executed
    # The decorator converts it to a query and dispatches it
    pass

# -----------------------------------------------------------------------------
# Demo
# -----------------------------------------------------------------------------

async def run_demo():
    """Run a demo of the token bridge CQRS example"""
    # Get command dispatcher
    command_dispatcher = container.resolve(CommandDispatcher)
    
    # Get query dispatcher
    query_dispatcher = container.resolve(QueryDispatcher)
    
    # Initiate a bridge
    bridge_id = str(uuid.uuid4())
    
    print("\n--- Initiating bridge ---")
    await command_dispatcher.dispatch(InitiateBridgeCommand(
        id=bridge_id,
        user_address="0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
        source_network=BridgeNetwork.ETHEREUM,
        target_network=BridgeNetwork.POLYGON,
        token_address="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        token_symbol="WETH",
        token_decimals=18,
        token_type=TokenType.ERC20,
        amount=Decimal("1.5")
    ))
    
    # Wait for transaction to complete
    print("\n--- Waiting for transaction to complete ---")
    
    # Poll bridge status
    completed = False
    for _ in range(10):
        await asyncio.sleep(1)
        
        try:
            # Get bridge
            bridge = await query_dispatcher.dispatch(GetBridgeQuery(
                id=str(uuid.uuid4()),
                bridge_id=bridge_id
            ))
            
            print(f"Bridge status: {bridge.status}")
            
            if bridge.status == BridgeStatus.COMPLETED:
                completed = True
                break
            elif bridge.status == BridgeStatus.FAILED:
                print(f"Bridge failed: {bridge.error_message}")
                break
        except Exception as e:
            print(f"Error getting bridge: {e}")
    
    if not completed:
        print("Transaction did not complete in time")
    
    # Get bridges by user
    print("\n--- Getting bridges by user ---")
    bridges = await query_dispatcher.dispatch(GetBridgesByUserQuery(
        id=str(uuid.uuid4()),
        user_address="0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
    ))
    
    print(f"Found {bridges.total} bridges:")
    for bridge in bridges.bridges:
        print(f"  {bridge.id}: {bridge.source_network} -> {bridge.target_network}, {bridge.amount} {bridge.token.symbol}, status: {bridge.status}")
    
    # Get bridge statistics
    print("\n--- Getting bridge statistics ---")
    stats = await query_dispatcher.dispatch(GetBridgeStatsQuery(
        id=str(uuid.uuid4()),
        period="week"
    ))
    
    print(f"Statistics for {stats.period}:")
    print(f"  Total count: {stats.total_count}")
    print(f"  Total volume: {stats.total_volume}")
    print(f"  Average success rate: {stats.average_success_rate:.2%}")
    
    print("\n--- Demo completed ---")

if __name__ == "__main__":
    # Run demo
    asyncio.run(run_demo()) 