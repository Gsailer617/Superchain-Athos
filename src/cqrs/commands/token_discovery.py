from dataclasses import dataclass, field
from typing import Dict, Any, List
from uuid import uuid4
from .base import Command, CommandHandler
from ..events.base import Event, EventStore
from ...utils.rate_limiter import rate_limiter
import structlog
from datetime import datetime

logger = structlog.get_logger(__name__)

@dataclass
class DiscoverTokenCommand(Command):
    """Command to discover new tokens"""
    id: str
    chain_id: int
    token_address: str
    source: str
    validation_data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidateTokenCommand(Command):
    """Command to validate a token"""
    id: str
    chain_id: int
    token_address: str
    validation_rules: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class TokenDiscoveryCommandHandler(CommandHandler[DiscoverTokenCommand]):
    """Handles token discovery commands"""
    
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
    
    async def validate(self, command: DiscoverTokenCommand) -> bool:
        """Validate token discovery command"""
        if not command.token_address.startswith('0x'):
            return False
        if not isinstance(command.chain_id, int):
            return False
        return True
    
    async def handle(self, command: DiscoverTokenCommand) -> None:
        """Handle token discovery command"""
        try:
            # Rate limit the discovery process
            await rate_limiter.acquire(f"discovery_{command.source}")
            
            # Create token discovered event
            event = Event(
                id=str(uuid4()),
                aggregate_id=command.token_address,
                event_type="TOKEN_DISCOVERED",
                payload={
                    "chain_id": command.chain_id,
                    "token_address": command.token_address,
                    "source": command.source,
                    "validation_data": command.validation_data
                }
            )
            
            # Save event
            await self.event_store.save_event(event)
            logger.info(
                "Token discovery event saved",
                token_address=command.token_address,
                chain_id=command.chain_id
            )
            
        except Exception as e:
            logger.error(
                "Error handling token discovery command",
                error=str(e),
                token_address=command.token_address
            )
            raise

class TokenValidationCommandHandler(CommandHandler[ValidateTokenCommand]):
    """Handles token validation commands"""
    
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
    
    async def validate(self, command: ValidateTokenCommand) -> bool:
        """Validate token validation command"""
        if not command.token_address.startswith('0x'):
            return False
        if not isinstance(command.chain_id, int):
            return False
        if not command.validation_rules:
            return False
        return True
    
    async def handle(self, command: ValidateTokenCommand) -> None:
        """Handle token validation command"""
        try:
            # Rate limit validation
            await rate_limiter.acquire("token_validation")
            
            # Create token validated event
            event = Event(
                id=str(uuid4()),
                aggregate_id=command.token_address,
                event_type="TOKEN_VALIDATED",
                payload={
                    "chain_id": command.chain_id,
                    "token_address": command.token_address,
                    "validation_rules": command.validation_rules,
                    "validation_timestamp": command.timestamp.isoformat()
                }
            )
            
            # Save event
            await self.event_store.save_event(event)
            logger.info(
                "Token validation event saved",
                token_address=command.token_address,
                chain_id=command.chain_id
            )
            
        except Exception as e:
            logger.error(
                "Error handling token validation command",
                error=str(e),
                token_address=command.token_address
            )
            raise 