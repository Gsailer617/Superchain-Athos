from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from .base import Query, QueryHandler
from ..events.base import EventStore
import structlog
from datetime import datetime

logger = structlog.get_logger(__name__)

@dataclass
class GetTokenDetailsQuery(Query):
    """Query to get token details"""
    id: str
    chain_id: int
    token_address: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GetTokenValidationStatusQuery(Query):
    """Query to get token validation status"""
    id: str
    chain_id: int
    token_address: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TokenDetails:
    """Token details response"""
    chain_id: int
    token_address: str
    source: str
    discovery_time: str
    validation_status: str
    validation_data: Optional[Dict[str, Any]] = None

class TokenDetailsQueryHandler(QueryHandler[GetTokenDetailsQuery, TokenDetails]):
    """Handles token details queries"""
    
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
    
    async def validate(self, query: GetTokenDetailsQuery) -> bool:
        """Validate token details query"""
        if not query.token_address.startswith('0x'):
            return False
        if not isinstance(query.chain_id, int):
            return False
        return True
    
    async def handle(self, query: GetTokenDetailsQuery) -> TokenDetails:
        """Handle token details query"""
        try:
            # Get all events for the token
            events = await self.event_store.get_events(query.token_address)
            
            if not events:
                raise ValueError(f"Token {query.token_address} not found")
            
            # Find discovery and validation events
            discovery_event = next(
                (e for e in events if e.event_type == "TOKEN_DISCOVERED"),
                None
            )
            validation_event = next(
                (e for e in events if e.event_type == "TOKEN_VALIDATED"),
                None
            )
            
            if not discovery_event:
                raise ValueError(f"No discovery event found for {query.token_address}")
            
            return TokenDetails(
                chain_id=query.chain_id,
                token_address=query.token_address,
                source=discovery_event.payload["source"],
                discovery_time=discovery_event.timestamp.isoformat(),
                validation_status="VALIDATED" if validation_event else "PENDING",
                validation_data=validation_event.payload if validation_event else None
            )
            
        except Exception as e:
            logger.error(
                "Error handling token details query",
                error=str(e),
                token_address=query.token_address
            )
            raise

class TokenValidationStatusQueryHandler(QueryHandler[GetTokenValidationStatusQuery, str]):
    """Handles token validation status queries"""
    
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
    
    async def validate(self, query: GetTokenValidationStatusQuery) -> bool:
        """Validate token validation status query"""
        if not query.token_address.startswith('0x'):
            return False
        if not isinstance(query.chain_id, int):
            return False
        return True
    
    async def handle(self, query: GetTokenValidationStatusQuery) -> str:
        """Handle token validation status query"""
        try:
            # Get all events for the token
            events = await self.event_store.get_events(query.token_address)
            
            # Find validation event
            validation_event = next(
                (e for e in events if e.event_type == "TOKEN_VALIDATED"),
                None
            )
            
            return "VALIDATED" if validation_event else "PENDING"
            
        except Exception as e:
            logger.error(
                "Error handling validation status query",
                error=str(e),
                token_address=query.token_address
            )
            raise 