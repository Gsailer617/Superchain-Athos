"""
Aggregate System for Event Sourcing

This module provides an enhanced aggregate system for event sourcing:
- Base Aggregate class with version tracking
- Event application and recording
- Snapshot support
- Repository pattern for loading/saving aggregates
"""

import abc
import asyncio
import inspect
import json
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Type, TypeVar, get_type_hints, cast

import structlog
from pydantic import BaseModel, Field, root_validator

from .base import Event, EventStore, AggregateRoot
from .event_bus import EventBus
from ..core.dependency_injector import container

logger = structlog.get_logger(__name__)

T = TypeVar('T', bound='Aggregate')
E = TypeVar('E', bound=Event)

class AggregateState(BaseModel):
    """Base class for aggregate state
    
    Attributes:
        id: Unique identifier for the aggregate
        version: Current version of the aggregate
        created_at: Timestamp when the aggregate was created
        updated_at: Timestamp when the aggregate was last updated
    """
    id: str
    version: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class Aggregate(AggregateRoot, Generic[T]):
    """Base class for all aggregates
    
    Attributes:
        state: Current state of the aggregate
        uncommitted_events: Events that have been applied but not yet committed
    """
    
    def __init__(self, state: AggregateState):
        self.state = state
        self.uncommitted_events: List[Event] = []
        self._event_handlers: Dict[str, Callable[[Event], None]] = {}
        self._setup_event_handlers()
    
    def _setup_event_handlers(self) -> None:
        """Set up event handlers based on method annotations"""
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if name.startswith("apply_"):
                event_type = name[len("apply_"):].upper()
                self._event_handlers[event_type] = method
    
    @classmethod
    async def create(cls: Type[T], aggregate_id: Optional[str] = None, **kwargs) -> T:
        """Create a new aggregate
        
        Args:
            aggregate_id: Optional aggregate ID (generated if not provided)
            **kwargs: Initial state parameters
            
        Returns:
            New aggregate instance
        """
        # Get state class from type hints
        state_class = get_type_hints(cls)["state"].__args__[0]
        
        # Create state
        state_data = kwargs.copy()
        state_data["id"] = aggregate_id or str(uuid.uuid4())
        state = state_class(**state_data)
        
        # Create aggregate
        return cls(state)
    
    def apply_event(self, event: Event) -> None:
        """Apply an event to the aggregate
        
        Args:
            event: Event to apply
        """
        # Get event handler
        handler = self._event_handlers.get(event.event_type)
        if not handler:
            logger.warning(f"No handler for event type {event.event_type} in {self.__class__.__name__}")
            return
        
        # Apply event
        handler(event)
        
        # Update state
        self.state.version += 1
        self.state.updated_at = datetime.now()
    
    def record_event(self, event_type: str, payload: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Event:
        """Record a new event
        
        Args:
            event_type: Type of event
            payload: Event payload
            metadata: Optional event metadata
            
        Returns:
            Recorded event
        """
        # Create event
        event = Event(
            id=str(uuid.uuid4()),
            aggregate_id=self.state.id,
            event_type=event_type,
            version=self.state.version + 1,
            payload=payload,
            metadata=metadata or {}
        )
        
        # Apply event
        self.apply_event(event)
        
        # Add to uncommitted events
        self.uncommitted_events.append(event)
        
        return event
    
    @property
    def id(self) -> str:
        """Get aggregate ID"""
        return self.state.id
    
    @property
    def version(self) -> int:
        """Get aggregate version"""
        return self.state.version
    
    def has_uncommitted_events(self) -> bool:
        """Check if the aggregate has uncommitted events"""
        return len(self.uncommitted_events) > 0
    
    def mark_events_as_committed(self) -> None:
        """Mark all uncommitted events as committed"""
        self.uncommitted_events.clear()

class Snapshot(BaseModel):
    """Snapshot of an aggregate state
    
    Attributes:
        aggregate_id: ID of the aggregate
        aggregate_type: Type of the aggregate
        version: Version of the aggregate
        state: Serialized state
        created_at: Timestamp when the snapshot was created
    """
    aggregate_id: str
    aggregate_type: str
    version: int
    state: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.now)

class SnapshotStore(abc.ABC):
    """Interface for snapshot storage"""
    
    @abc.abstractmethod
    async def save_snapshot(self, snapshot: Snapshot) -> None:
        """Save a snapshot
        
        Args:
            snapshot: Snapshot to save
        """
        pass
    
    @abc.abstractmethod
    async def get_latest_snapshot(self, aggregate_id: str, aggregate_type: str) -> Optional[Snapshot]:
        """Get the latest snapshot for an aggregate
        
        Args:
            aggregate_id: ID of the aggregate
            aggregate_type: Type of the aggregate
            
        Returns:
            Latest snapshot, if any
        """
        pass

class AggregateRepository(Generic[T]):
    """Repository for loading and saving aggregates
    
    Attributes:
        aggregate_type: Type of aggregate this repository handles
        event_store: Store for aggregate events
        snapshot_store: Store for aggregate snapshots
        event_bus: Bus for publishing events
    """
    
    def __init__(self, aggregate_type: Type[T]):
        self.aggregate_type = aggregate_type
        self.event_store: Optional[EventStore] = None
        self.snapshot_store: Optional[SnapshotStore] = None
        self.event_bus: Optional[EventBus] = None
        self.snapshot_frequency = 10  # Create snapshot every N events
    
    def _resolve_dependencies(self) -> None:
        """Resolve dependencies"""
        try:
            if not self.event_store:
                self.event_store = container.resolve(EventStore)
            if not self.snapshot_store:
                self.snapshot_store = container.resolve(SnapshotStore)
            if not self.event_bus:
                self.event_bus = container.resolve(EventBus)
        except Exception as e:
            logger.error(f"Error resolving dependencies: {str(e)}")
            raise
    
    async def get_by_id(self, aggregate_id: str) -> Optional[T]:
        """Get an aggregate by ID
        
        Args:
            aggregate_id: ID of the aggregate
            
        Returns:
            Aggregate instance, if found
        """
        self._resolve_dependencies()
        
        # Try to load from snapshot first
        if self.snapshot_store:
            snapshot = await self.snapshot_store.get_latest_snapshot(
                aggregate_id, 
                self.aggregate_type.__name__
            )
            
            if snapshot:
                # Create aggregate from snapshot
                state_class = get_type_hints(self.aggregate_type)["state"].__args__[0]
                state = state_class(**snapshot.state)
                aggregate = self.aggregate_type(state)
                
                # Load events after snapshot
                events = await self.event_store.get_events_by_aggregate_id(
                    aggregate_id, 
                    snapshot.version + 1
                )
                
                # Apply events
                for event in events:
                    aggregate.apply_event(event)
                
                return aggregate
        
        # Load from events
        events = await self.event_store.get_events_by_aggregate_id(aggregate_id)
        if not events:
            return None
        
        # Create aggregate
        aggregate = await self.aggregate_type.create(aggregate_id=aggregate_id)
        
        # Apply events
        for event in events:
            aggregate.apply_event(event)
        
        return aggregate
    
    async def save(self, aggregate: T) -> None:
        """Save an aggregate
        
        Args:
            aggregate: Aggregate to save
        """
        self._resolve_dependencies()
        
        # Check if there are uncommitted events
        if not aggregate.has_uncommitted_events():
            return
        
        # Save events
        for event in aggregate.uncommitted_events:
            await self.event_store.save_event(event)
            
            # Publish event
            if self.event_bus:
                asyncio.create_task(self.event_bus.publish(event))
        
        # Create snapshot if needed
        if self.snapshot_store and aggregate.version % self.snapshot_frequency == 0:
            snapshot = Snapshot(
                aggregate_id=aggregate.id,
                aggregate_type=self.aggregate_type.__name__,
                version=aggregate.version,
                state=aggregate.state.dict()
            )
            await self.snapshot_store.save_snapshot(snapshot)
        
        # Mark events as committed
        aggregate.mark_events_as_committed()

class EventSourcingConfig(BaseModel):
    """Configuration for event sourcing
    
    Attributes:
        snapshot_frequency: How often to create snapshots (every N events)
        snapshot_store_enabled: Whether to use snapshot store
        replay_batch_size: Batch size when replaying events
    """
    snapshot_frequency: int = 10
    snapshot_store_enabled: bool = True
    replay_batch_size: int = 100

def aggregate_factory(aggregate_type: Type[T]) -> AggregateRepository[T]:
    """Create a repository for an aggregate type
    
    Args:
        aggregate_type: Type of aggregate
        
    Returns:
        Repository for the aggregate type
    """
    try:
        # Get configuration
        event_sourcing_config = container.resolve(EventSourcingConfig)
        
        # Create repository
        repository = AggregateRepository(aggregate_type)
        repository.snapshot_frequency = event_sourcing_config.snapshot_frequency
        
        # Resolve dependencies
        repository.event_store = container.resolve(EventStore)
        if event_sourcing_config.snapshot_store_enabled:
            repository.snapshot_store = container.resolve(SnapshotStore)
        repository.event_bus = container.resolve(EventBus)
        
        return repository
    except Exception as e:
        logger.error(f"Error creating aggregate repository: {str(e)}")
        # Return repository without resolved dependencies
        return AggregateRepository(aggregate_type)

def apply_event(event_type: str):
    """Decorator to mark a method as an event handler
    
    Args:
        event_type: Type of event
        
    Example:
        @apply_event("USER_CREATED")
        def apply_user_created(self, event):
            self.state.name = event.payload["name"]
    """
    def decorator(func):
        # Add attribute to function
        func.__event_type__ = event_type
        return func
    return decorator

def register_aggregate_repositories() -> None:
    """Register all aggregate repositories in the container"""
    for subclass in Aggregate.__subclasses__():
        repository = aggregate_factory(subclass)
        container.register_instance(
            f"{subclass.__name__}Repository", 
            repository
        )

class FileSnapshotStore(SnapshotStore):
    """File-based implementation of snapshot store
    
    Attributes:
        directory: Directory for storing snapshots
    """
    
    def __init__(self, directory: str = "./snapshots"):
        import os
        self.directory = directory
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
    
    def _get_snapshot_path(self, aggregate_id: str, aggregate_type: str) -> str:
        """Get path for a snapshot file
        
        Args:
            aggregate_id: ID of the aggregate
            aggregate_type: Type of the aggregate
            
        Returns:
            Path to snapshot file
        """
        import os
        return os.path.join(self.directory, f"{aggregate_type}_{aggregate_id}.json")
    
    async def save_snapshot(self, snapshot: Snapshot) -> None:
        """Save a snapshot
        
        Args:
            snapshot: Snapshot to save
        """
        import aiofiles
        import json
        
        path = self._get_snapshot_path(snapshot.aggregate_id, snapshot.aggregate_type)
        
        # Convert to dict
        snapshot_dict = snapshot.dict()
        
        # Convert datetime to string
        snapshot_dict["created_at"] = snapshot_dict["created_at"].isoformat()
        
        async with aiofiles.open(path, "w") as f:
            await f.write(json.dumps(snapshot_dict, indent=2))
    
    async def get_latest_snapshot(self, aggregate_id: str, aggregate_type: str) -> Optional[Snapshot]:
        """Get the latest snapshot for an aggregate
        
        Args:
            aggregate_id: ID of the aggregate
            aggregate_type: Type of the aggregate
            
        Returns:
            Latest snapshot, if any
        """
        import aiofiles
        import json
        import os
        
        path = self._get_snapshot_path(aggregate_id, aggregate_type)
        
        # Check if file exists
        if not os.path.exists(path):
            return None
        
        try:
            async with aiofiles.open(path, "r") as f:
                content = await f.read()
                
            # Parse JSON
            snapshot_dict = json.loads(content)
            
            # Convert string to datetime
            snapshot_dict["created_at"] = datetime.fromisoformat(snapshot_dict["created_at"])
            
            return Snapshot(**snapshot_dict)
        except Exception as e:
            logger.error(f"Error loading snapshot: {str(e)}")
            return None

class RedisSnapshotStore(SnapshotStore):
    """Redis-based implementation of snapshot store
    
    Attributes:
        redis: Redis client
        prefix: Key prefix for snapshots
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", prefix: str = "snapshot:"):
        import redis.asyncio as redis
        self.redis = redis.from_url(redis_url)
        self.prefix = prefix
    
    def _get_snapshot_key(self, aggregate_id: str, aggregate_type: str) -> str:
        """Get key for a snapshot
        
        Args:
            aggregate_id: ID of the aggregate
            aggregate_type: Type of the aggregate
            
        Returns:
            Redis key
        """
        return f"{self.prefix}{aggregate_type}:{aggregate_id}"
    
    async def save_snapshot(self, snapshot: Snapshot) -> None:
        """Save a snapshot
        
        Args:
            snapshot: Snapshot to save
        """
        import json
        
        key = self._get_snapshot_key(snapshot.aggregate_id, snapshot.aggregate_type)
        
        # Convert to dict
        snapshot_dict = snapshot.dict()
        
        # Convert datetime to string
        snapshot_dict["created_at"] = snapshot_dict["created_at"].isoformat()
        
        await self.redis.set(key, json.dumps(snapshot_dict))
    
    async def get_latest_snapshot(self, aggregate_id: str, aggregate_type: str) -> Optional[Snapshot]:
        """Get the latest snapshot for an aggregate
        
        Args:
            aggregate_id: ID of the aggregate
            aggregate_type: Type of the aggregate
            
        Returns:
            Latest snapshot, if any
        """
        import json
        
        key = self._get_snapshot_key(aggregate_id, aggregate_type)
        
        # Get snapshot
        snapshot_json = await self.redis.get(key)
        if not snapshot_json:
            return None
        
        try:
            # Parse JSON
            snapshot_dict = json.loads(snapshot_json)
            
            # Convert string to datetime
            snapshot_dict["created_at"] = datetime.fromisoformat(snapshot_dict["created_at"])
            
            return Snapshot(**snapshot_dict)
        except Exception as e:
            logger.error(f"Error loading snapshot: {str(e)}")
            return None 