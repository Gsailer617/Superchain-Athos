from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generic, TypeVar

T = TypeVar('T')
R = TypeVar('R')

@dataclass
class Query:
    """Base query class"""
    id: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class QueryHandler(ABC, Generic[T, R]):
    """Base query handler interface"""
    
    @abstractmethod
    async def handle(self, query: T) -> R:
        """Handle a query and return result"""
        pass

    @abstractmethod
    async def validate(self, query: T) -> bool:
        """Validate a query before handling"""
        pass 