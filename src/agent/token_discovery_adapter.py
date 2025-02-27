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
from ..core.web3_config import get_web3
import asyncio
from functools import lru_cache

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
        self.config = config
        self.web3 = get_web3()
        self.token_discovery = TokenDiscovery(config, self.web3)
        self.event_store = RedisEventStore(
            redis_url=config.get('redis_url', 'redis://localhost'),
            namespace="token_discovery"
        )
        
        # Initialize CQRS handlers
        self.discovery_handler = TokenDiscoveryCommandHandler(self.event_store)
        self.validation_handler = TokenValidationCommandHandler(self.event_store)
        self.details_handler = TokenDetailsQueryHandler(self.event_store)
        self.status_handler = TokenValidationStatusQueryHandler(self.event_store)
        
        # Initialize bulkheads with improved configurations
        self.bulkheads = BulkheadRegistry()
        self._setup_bulkheads()
        
        # Initialize token discover task manager
        self.discovery_tasks = {}
        self.validation_tasks = {}
    
    def _setup_bulkheads(self) -> None:
        """Setup bulkheads for component isolation with optimized settings"""
        # Discovery bulkhead - increased capacity
        self.bulkheads.register(
            "discovery",
            max_concurrent_calls=self.config.get('discovery_concurrent', 10),
            max_queue_size=self.config.get('discovery_queue_size', 200),
            timeout_seconds=self.config.get('discovery_timeout', 45.0)
        )
        
        # Validation bulkhead - increased throughput
        self.bulkheads.register(
            "validation",
            max_concurrent_calls=self.config.get('validation_concurrent', 20),
            max_queue_size=self.config.get('validation_queue_size', 100),
            timeout_seconds=self.config.get('validation_timeout', 20.0)
        )
        
        # Query bulkhead - prioritize fast responses
        self.bulkheads.register(
            "query",
            max_concurrent_calls=self.config.get('query_concurrent', 30),
            max_queue_size=self.config.get('query_queue_size', 300),
            timeout_seconds=self.config.get('query_timeout', 5.0)
        )
    
    async def discover_token(
        self,
        chain_id: int,
        token_address: str,
        source: str,
        validation_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Discover a new token using CQRS pattern with enhanced reliability"""
        try:
            # Create command ID with structured format for tracing
            command_id = f"discover_{chain_id}_{token_address[:8]}_{uuid4().hex[:8]}"
            
            # Create discovery command
            command = DiscoverTokenCommand(
                id=command_id,
                chain_id=chain_id,
                token_address=token_address,
                source=source,
                validation_data=validation_data or {}
            )
            
            # Check if already processing this token
            task_key = f"{chain_id}_{token_address}"
            if task_key in self.discovery_tasks and not self.discovery_tasks[task_key].done():
                logger.debug(
                    "Token discovery already in progress",
                    token_address=token_address,
                    chain_id=chain_id
                )
                return True
                
            # Execute with bulkhead protection and concurrency management
            discovery_bulkhead = self.bulkheads.get("discovery")
            
            # Use the fast bloom filter check from token_discovery
            if token_address in self.token_discovery.token_filter:
                logger.debug(
                    "Token already in bloom filter",
                    token_address=token_address
                )
                return True
                
            async with discovery_bulkhead:
                # Validate and handle command
                if await self.discovery_handler.validate(command):
                    # Create task and track it
                    task = asyncio.create_task(self.discovery_handler.handle(command))
                    self.discovery_tasks[task_key] = task
                    
                    # Add cleanup callback
                    task.add_done_callback(
                        lambda t, tk=task_key: self.discovery_tasks.pop(tk, None)
                    )
                    
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
        """Validate a token using CQRS pattern with progressive validation"""
        try:
            # Create command ID with structured format for tracing
            command_id = f"validate_{chain_id}_{token_address[:8]}_{uuid4().hex[:8]}"
            
            # Create validation command
            command = ValidateTokenCommand(
                id=command_id,
                chain_id=chain_id,
                token_address=token_address,
                validation_rules=validation_rules or {}
            )
            
            # Check if already validating this token
            task_key = f"{chain_id}_{token_address}"
            if task_key in self.validation_tasks and not self.validation_tasks[task_key].done():
                logger.debug(
                    "Token validation already in progress",
                    token_address=token_address,
                    chain_id=chain_id
                )
                return True
            
            # Execute with bulkhead protection
            validation_bulkhead = self.bulkheads.get("validation")
            async with validation_bulkhead:
                # Validate and handle command
                if await self.validation_handler.validate(command):
                    # Create task and track it
                    task = asyncio.create_task(self.validation_handler.handle(command))
                    self.validation_tasks[task_key] = task
                    
                    # Add cleanup callback
                    task.add_done_callback(
                        lambda t, tk=task_key: self.validation_tasks.pop(tk, None)
                    )
                    
                    return True
                return False
                
        except Exception as e:
            logger.error(
                "Error validating token",
                error=str(e),
                token_address=token_address
            )
            return False
    
    @lru_cache(maxsize=1000)
    async def get_token_details(
        self,
        chain_id: int,
        token_address: str
    ) -> Optional[TokenDetails]:
        """Get token details using CQRS pattern with LRU caching for repeated calls"""
        try:
            # Create query ID with structured format for tracing
            query_id = f"details_{chain_id}_{token_address[:8]}_{uuid4().hex[:8]}"
            
            # Create query
            query = GetTokenDetailsQuery(
                id=query_id,
                chain_id=chain_id,
                token_address=token_address
            )
            
            # Execute with bulkhead protection
            query_bulkhead = self.bulkheads.get("query")
            async with query_bulkhead:
                # Check in bloom filter first (fastest check)
                if not token_address in self.token_discovery.token_filter:
                    logger.debug(
                        "Token not in bloom filter",
                        token_address=token_address
                    )
                    return None
                
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
        """Get token validation status using CQRS pattern with optimized performance"""
        try:
            # Create query ID with structured format for tracing
            query_id = f"status_{chain_id}_{token_address[:8]}_{uuid4().hex[:8]}"
            
            # Create query
            query = GetTokenValidationStatusQuery(
                id=query_id,
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
    
    async def discover_new_tokens(self, chain_id: int = 1) -> List[Dict[str, Any]]:
        """Discover new tokens using the enhanced token discovery engine"""
        try:
            # Use the improved discovery system with chain_id support
            discovery_bulkhead = self.bulkheads.get("discovery")
            async with discovery_bulkhead:
                discovered_tokens = await self.token_discovery.discover_new_tokens(chain_id)
                
                # Process discovered tokens through the CQRS system
                for token in discovered_tokens:
                    await self.discover_token(
                        chain_id=chain_id,
                        token_address=token.get('address'),
                        source=token.get('source', 'automatic'),
                        validation_data=token.get('validation', {})
                    )
                
                return discovered_tokens
        except Exception as e:
            logger.error(
                "Error in bulk token discovery",
                error=str(e),
                chain_id=chain_id
            )
            return [] 