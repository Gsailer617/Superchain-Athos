from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar
from uuid import UUID

T = TypeVar('T')

@dataclass
class Event:
    """Base event class"""
    id: str
    aggregate_id: str
    event_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)

class EventStore(ABC):
    """Base event store interface"""
    
    @abstractmethod
    async def save_event(self, event: Event) -> None:
        """Save an event to the store"""
        pass
    
    @abstractmethod
    async def get_events(
        self,
        aggregate_id: str,
        since_version: Optional[int] = None
    ) -> List[Event]:
        """Get events for an aggregate"""
        pass
    
    @abstractmethod
    async def get_latest_version(self, aggregate_id: str) -> int:
        """Get latest version for an aggregate"""
        pass

class EventHandler(ABC, Generic[T]):
    """Base event handler interface"""
    
    @abstractmethod
    async def handle(self, event: T) -> None:
        """Handle an event"""
        pass

class AggregateRoot(ABC):
    """Base aggregate root class for event sourcing"""
    
    def __init__(self, id: str):
        self.id = id
        self._version = 0
        self._changes: List[Event] = []
    
    @property
    def version(self) -> int:
        return self._version
    
    def get_uncommitted_changes(self) -> List[Event]:
        return self._changes
    
    def mark_changes_as_committed(self) -> None:
        self._changes.clear()
    
    def load_from_history(self, events: List[Event]) -> None:
        for event in events:
            self._apply_change(event, is_new=False)
    
    def apply_change(self, event: Event) -> None:
        self._apply_change(event, is_new=True)
    
    def _apply_change(self, event: Event, is_new: bool = True) -> None:
        self.apply(event)
        if is_new:
            self._changes.append(event)
        self._version = event.version
    
    @abstractmethod
    def apply(self, event: Event) -> None:
        """Apply an event to the aggregate"""
        pass 