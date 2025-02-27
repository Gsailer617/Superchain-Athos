"""
Command and Query Dispatcher

This module provides centralized dispatchers for routing commands and queries to their handlers.
Features:
- Automatic handler registration and discovery
- Middleware support for cross-cutting concerns
- Integration with the bulkhead pattern for isolation
- Metrics collection
"""

from typing import Dict, Type, Any, Optional, List, Callable, TypeVar, Generic, cast
import importlib
import inspect
import structlog
import asyncio
from ..commands.base import Command, CommandHandler
from ..queries.base import Query, QueryHandler
from ..bulkhead.base import Bulkhead, BulkheadRegistry
from ..events.base import EventStore
from ...core.dependency_injector import container

logger = structlog.get_logger(__name__)

T = TypeVar('T')
R = TypeVar('R')

class CommandMiddleware:
    """Base class for command middleware"""
    
    async def before_execution(self, command: Command) -> Command:
        """Called before command execution"""
        return command
    
    async def after_execution(self, command: Command, result: Any) -> Any:
        """Called after command execution"""
        return result
    
    async def on_error(self, command: Command, error: Exception) -> None:
        """Called when command execution fails"""
        pass

class QueryMiddleware:
    """Base class for query middleware"""
    
    async def before_execution(self, query: Query) -> Query:
        """Called before query execution"""
        return query
    
    async def after_execution(self, query: Query, result: Any) -> Any:
        """Called after query execution"""
        return result
    
    async def on_error(self, query: Query, error: Exception) -> None:
        """Called when query execution fails"""
        pass

class CommandDispatcher:
    """Dispatches commands to registered handlers"""
    
    def __init__(self, bulkhead_registry: Optional[BulkheadRegistry] = None):
        self.handlers: Dict[Type[Command], CommandHandler] = {}
        self.middlewares: List[CommandMiddleware] = []
        self.bulkhead_registry = bulkhead_registry
        self.metrics: Dict[str, int] = {
            "total_commands": 0,
            "successful_commands": 0,
            "failed_commands": 0,
        }
    
    def register_handler(self, command_type: Type[T], handler: CommandHandler[T]) -> None:
        """Register a command handler
        
        Args:
            command_type: Type of command to handle
            handler: Handler for the command
        """
        if command_type in self.handlers:
            raise ValueError(f"Handler already registered for {command_type.__name__}")
        
        self.handlers[command_type] = handler
        logger.info(f"Registered handler for {command_type.__name__}")
    
    def register_middleware(self, middleware: CommandMiddleware) -> None:
        """Register command middleware
        
        Args:
            middleware: Middleware instance to add
        """
        self.middlewares.append(middleware)
        logger.info(f"Registered command middleware {middleware.__class__.__name__}")
    
    async def dispatch(self, command: Command, use_bulkhead: bool = True) -> None:
        """Dispatch a command to its handler
        
        Args:
            command: Command to dispatch
            use_bulkhead: Whether to use bulkhead for isolation
            
        Raises:
            ValueError: If no handler is registered for the command
        """
        command_type = type(command)
        self.metrics["total_commands"] += 1
        
        # Find handler
        handler = self.handlers.get(command_type)
        if not handler:
            error_msg = f"No handler registered for {command_type.__name__}"
            logger.error(error_msg)
            self.metrics["failed_commands"] += 1
            raise ValueError(error_msg)
        
        # Apply bulkhead if available and requested
        if use_bulkhead and self.bulkhead_registry:
            bulkhead_name = f"command_{command_type.__name__}"
            bulkhead = self.bulkhead_registry.get(bulkhead_name)
            
            if bulkhead:
                return await bulkhead.execute(self._handle_command, command, handler)
        
        # Handle directly if no bulkhead
        return await self._handle_command(command, handler)
    
    async def _handle_command(self, command: Command, handler: CommandHandler) -> None:
        """Internal method to handle command with middleware
        
        Args:
            command: Command to handle
            handler: Handler for the command
        """
        try:
            # Apply before middleware
            processed_command = command
            for middleware in self.middlewares:
                processed_command = await middleware.before_execution(processed_command)
            
            # Validate command
            if not await handler.validate(processed_command):
                raise ValueError(f"Command validation failed for {type(command).__name__}")
            
            # Handle command
            result = await handler.handle(processed_command)
            
            # Apply after middleware
            for middleware in reversed(self.middlewares):
                result = await middleware.after_execution(processed_command, result)
            
            self.metrics["successful_commands"] += 1
            
        except Exception as e:
            self.metrics["failed_commands"] += 1
            
            # Apply error middleware
            for middleware in self.middlewares:
                await middleware.on_error(command, e)
            
            raise

class QueryDispatcher:
    """Dispatches queries to registered handlers"""
    
    def __init__(self, bulkhead_registry: Optional[BulkheadRegistry] = None):
        self.handlers: Dict[Type[Query], QueryHandler] = {}
        self.middlewares: List[QueryMiddleware] = []
        self.bulkhead_registry = bulkhead_registry
        self.metrics: Dict[str, int] = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
        }
    
    def register_handler(self, query_type: Type[T], handler: QueryHandler[T, R]) -> None:
        """Register a query handler
        
        Args:
            query_type: Type of query to handle
            handler: Handler for the query
        """
        if query_type in self.handlers:
            raise ValueError(f"Handler already registered for {query_type.__name__}")
        
        self.handlers[query_type] = handler
        logger.info(f"Registered handler for {query_type.__name__}")
    
    def register_middleware(self, middleware: QueryMiddleware) -> None:
        """Register query middleware
        
        Args:
            middleware: Middleware instance to add
        """
        self.middlewares.append(middleware)
        logger.info(f"Registered query middleware {middleware.__class__.__name__}")
    
    async def dispatch(self, query: Query, use_bulkhead: bool = True) -> Any:
        """Dispatch a query to its handler
        
        Args:
            query: Query to dispatch
            use_bulkhead: Whether to use bulkhead for isolation
            
        Returns:
            Query result
            
        Raises:
            ValueError: If no handler is registered for the query
        """
        query_type = type(query)
        self.metrics["total_queries"] += 1
        
        # Find handler
        handler = self.handlers.get(query_type)
        if not handler:
            error_msg = f"No handler registered for {query_type.__name__}"
            logger.error(error_msg)
            self.metrics["failed_queries"] += 1
            raise ValueError(error_msg)
        
        # Apply bulkhead if available and requested
        if use_bulkhead and self.bulkhead_registry:
            bulkhead_name = f"query_{query_type.__name__}"
            bulkhead = self.bulkhead_registry.get(bulkhead_name)
            
            if bulkhead:
                return await bulkhead.execute(self._handle_query, query, handler)
        
        # Handle directly if no bulkhead
        return await self._handle_query(query, handler)
    
    async def _handle_query(self, query: Query, handler: QueryHandler) -> Any:
        """Internal method to handle query with middleware
        
        Args:
            query: Query to handle
            handler: Handler for the query
            
        Returns:
            Query result
        """
        try:
            # Apply before middleware
            processed_query = query
            for middleware in self.middlewares:
                processed_query = await middleware.before_execution(processed_query)
            
            # Validate query
            if not await handler.validate(processed_query):
                raise ValueError(f"Query validation failed for {type(query).__name__}")
            
            # Handle query
            result = await handler.handle(processed_query)
            
            # Apply after middleware
            for middleware in reversed(self.middlewares):
                result = await middleware.after_execution(processed_query, result)
            
            self.metrics["successful_queries"] += 1
            return result
            
        except Exception as e:
            self.metrics["failed_queries"] += 1
            
            # Apply error middleware
            for middleware in self.middlewares:
                await middleware.on_error(query, e)
            
            raise

# Create singleton instances
command_dispatcher = CommandDispatcher()
query_dispatcher = QueryDispatcher()

def register_with_container():
    """Register dispatchers with dependency container"""
    bulkhead_registry = container.resolve(BulkheadRegistry)
    
    # Create and register dispatchers
    command_dispatcher = CommandDispatcher(bulkhead_registry)
    query_dispatcher = QueryDispatcher(bulkhead_registry)
    
    container.register_instance(CommandDispatcher, command_dispatcher)
    container.register_instance(QueryDispatcher, query_dispatcher) 