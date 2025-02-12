from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generic, TypeVar

T = TypeVar('T')

@dataclass
class Command:
    """Base command class"""
    id: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class CommandHandler(ABC, Generic[T]):
    """Base command handler interface"""
    
    @abstractmethod
    async def handle(self, command: T) -> None:
        """Handle a command"""
        pass

    @abstractmethod
    async def validate(self, command: T) -> bool:
        """Validate a command before handling"""
        pass 