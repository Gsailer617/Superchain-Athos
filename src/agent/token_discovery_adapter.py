from typing import Dict, Any, List, Optional
from uuid import uuid4
from datetime import datetime
import structlog
from .token_discovery import TokenDiscovery, TokenData, ValidationResult
from ..cqrs.commands.token_discovery import (
    DiscoverTokenCommand,
    ValidateTokenCommand,
    TokenDiscoveryCommandHandler,
    TokenValidationCommandHandler
)
from ..cqrs.queries.token_discovery import (
    GetTokenDetailsQuery,
    GetTokenValidationStatusQuery,
    TokenDetailsQueryHandler,
    TokenValidationStatusQueryHandler,
    TokenDetails
)
from ..cqrs.events.redis_store import RedisEventStore
from ..utils.bulkhead.base import BulkheadRegistry

logger = structlog.get_logger(__name__)

class TokenDiscoveryAdapter:
    """
    Adapter that integrates the existing TokenDiscovery class with CQRS and Event Sourcing.
    
    This adapter:
    1. Uses the existing TokenDiscovery for core functionality
    2. Implements CQRS pattern for command/query separation
    3. Uses Event Sourcing for state changes
    4. Implements Bulkhead pattern for isolation
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Initialize core components
        self.token_discovery = TokenDiscovery(config)
        self.event_store = RedisEventStore(
            redis_url=config.get('redis_url', 'redis://localhost'),
            namespace="token_discovery"
        )
        
        # Initialize CQRS handlers
        self.discovery_handler = TokenDiscoveryCommandHandler(self.event_store)
        self.validation_handler = TokenValidationCommandHandler(self.event_store)
        self.details_handler = TokenDetailsQueryHandler(self.event_store)
        self.status_handler = TokenValidationStatusQueryHandler(self.event_store)
        
        # Initialize bulkheads
        self.bulkheads = BulkheadRegistry()
        self._setup_bulkheads()
    
    def _setup_bulkheads(self) -> None:
        """Setup bulkheads for component isolation"""
        # Discovery bulkhead
        self.bulkheads.register(
            "discovery",
            max_concurrent_calls=5,
            max_queue_size=100,
            timeout_seconds=30.0
        )
        
        # Validation bulkhead
        self.bulkheads.register(
            "validation",
            max_concurrent_calls=10,
            max_queue_size=50,
            timeout_seconds=15.0
        )
        
        # Query bulkhead
        self.bulkheads.register(
            "query",
            max_concurrent_calls=20,
            max_queue_size=200,
            timeout_seconds=5.0
        )
    
    async def discover_token(
        self,
        chain_id: int,
        token_address: str,
        source: str,
        validation_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Discover a new token using CQRS pattern"""
        try:
            # Create discovery command
            command = DiscoverTokenCommand(
                id=str(uuid4()),
                chain_id=chain_id,
                token_address=token_address,
                source=source,
                validation_data=validation_data or {}
            )
            
            # Execute with bulkhead protection
            discovery_bulkhead = self.bulkheads.get("discovery")
            async with discovery_bulkhead:
                # Validate and handle command
                if await self.discovery_handler.validate(command):
                    await self.discovery_handler.handle(command)
                    return True
                return False
                
        except Exception as e:
            logger.error(
                "Error discovering token",
                error=str(e),
                token_address=token_address
            )
            return False
    
    async def validate_token(
        self,
        chain_id: int,
        token_address: str,
        validation_rules: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Validate a token using CQRS pattern"""
        try:
            # Create validation command
            command = ValidateTokenCommand(
                id=str(uuid4()),
                chain_id=chain_id,
                token_address=token_address,
                validation_rules=validation_rules or {}
            )
            
            # Execute with bulkhead protection
            validation_bulkhead = self.bulkheads.get("validation")
            async with validation_bulkhead:
                # Validate and handle command
                if await self.validation_handler.validate(command):
                    await self.validation_handler.handle(command)
                    return True
                return False
                
        except Exception as e:
            logger.error(
                "Error validating token",
                error=str(e),
                token_address=token_address
            )
            return False
    
    async def get_token_details(
        self,
        chain_id: int,
        token_address: str
    ) -> Optional[TokenDetails]:
        """Get token details using CQRS pattern"""
        try:
            # Create query
            query = GetTokenDetailsQuery(
                id=str(uuid4()),
                chain_id=chain_id,
                token_address=token_address
            )
            
            # Execute with bulkhead protection
            query_bulkhead = self.bulkheads.get("query")
            async with query_bulkhead:
                # Validate and handle query
                if await self.details_handler.validate(query):
                    return await self.details_handler.handle(query)
                return None
                
        except Exception as e:
            logger.error(
                "Error getting token details",
                error=str(e),
                token_address=token_address
            )
            return None
    
    async def get_validation_status(
        self,
        chain_id: int,
        token_address: str
    ) -> str:
        """Get token validation status using CQRS pattern"""
        try:
            # Create query
            query = GetTokenValidationStatusQuery(
                id=str(uuid4()),
                chain_id=chain_id,
                token_address=token_address
            )
            
            # Execute with bulkhead protection
            query_bulkhead = self.bulkheads.get("query")
            async with query_bulkhead:
                # Validate and handle query
                if await self.status_handler.validate(query):
                    return await self.status_handler.handle(query)
                return "UNKNOWN"
                
        except Exception as e:
            logger.error(
                "Error getting validation status",
                error=str(e),
                token_address=token_address
            )
            return "ERROR" 