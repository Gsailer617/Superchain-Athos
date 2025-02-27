"""
CQRS Initializer Module

This module initializes all CQRS components during application startup:
- Registers command and query handlers
- Sets up event handlers and subscriptions 
- Configures middleware
- Initializes bulkheads
- Connects components with the dependency injection system
"""

import asyncio
import structlog
import inspect
from typing import Dict, List, Any, Type, Optional, Set
import importlib
import pkgutil
import sys
import os
from pathlib import Path

from .commands.base import Command, CommandHandler
from .queries.base import Query, QueryHandler
from .events.base import Event, EventHandler
from .handlers.dispatcher import CommandDispatcher, QueryDispatcher
from .handlers.middleware import (
    create_standard_command_middleware,
    create_standard_query_middleware
)
from .events.event_bus import EventBus, RetryPolicy
from .bulkhead.base import BulkheadRegistry

from ..core.dependency_injector import container
from ..core.error_handling import ErrorHandler
from ..core.health_monitor import HealthMonitor

logger = structlog.get_logger(__name__)

class CqrsInitializer:
    """Initializes all CQRS components"""
    
    def __init__(self):
        self.command_handlers: Dict[Type[Command], Type[CommandHandler]] = {}
        self.query_handlers: Dict[Type[Query], Type[QueryHandler]] = {}
        self.event_handlers: Dict[str, List[Type[EventHandler]]] = {}
        self.bulkhead_registry: Optional[BulkheadRegistry] = None
        self.command_dispatcher: Optional[CommandDispatcher] = None
        self.query_dispatcher: Optional[QueryDispatcher] = None
        self.event_bus: Optional[EventBus] = None
        self.error_handler: Optional[ErrorHandler] = None
        self.health_monitor: Optional[HealthMonitor] = None
    
    async def initialize(self) -> None:
        """Initialize all CQRS components"""
        logger.info("Initializing CQRS components")
        
        # Resolve dependencies
        self._resolve_dependencies()
        
        # Register bulkheads
        self._register_bulkheads()
        
        # Discover handlers
        self._discover_handlers()
        
        # Register handlers
        self._register_handlers()
        
        # Configure middleware
        self._configure_middleware()
        
        logger.info("CQRS components initialized")
    
    def _resolve_dependencies(self) -> None:
        """Resolve dependencies from container"""
        try:
            self.bulkhead_registry = container.resolve(BulkheadRegistry)
            self.command_dispatcher = container.resolve(CommandDispatcher)
            self.query_dispatcher = container.resolve(QueryDispatcher)
            self.event_bus = container.resolve(EventBus)
            self.error_handler = container.resolve(ErrorHandler)
            
            try:
                self.health_monitor = container.resolve(HealthMonitor)
            except Exception as e:
                logger.warning(f"Health monitor not available: {str(e)}")
                self.health_monitor = None
                
        except Exception as e:
            logger.error(f"Error resolving dependencies: {str(e)}")
            raise
    
    def _register_bulkheads(self) -> None:
        """Register bulkheads for CQRS components"""
        if not self.bulkhead_registry:
            logger.warning("Bulkhead registry not available")
            return
            
        # Command bulkheads
        self.bulkhead_registry.register(
            "command_default",
            max_concurrent_calls=20,
            max_queue_size=50,
            timeout_seconds=60.0
        )
        
        # Query bulkheads
        self.bulkhead_registry.register(
            "query_default",
            max_concurrent_calls=30,
            max_queue_size=100,
            timeout_seconds=30.0
        )
        
        # Event bulkheads
        self.bulkhead_registry.register(
            "event_default",
            max_concurrent_calls=50,
            max_queue_size=200,
            timeout_seconds=120.0
        )
        
        logger.info("Registered bulkheads for CQRS components")
    
    def _discover_handlers(self) -> None:
        """Discover command, query, and event handlers"""
        # Base path for discovery
        cqrs_path = Path(__file__).parent
        
        # Discover command handlers
        command_handlers_path = cqrs_path / "commands"
        self._discover_command_handlers(command_handlers_path)
        
        # Discover query handlers
        query_handlers_path = cqrs_path / "queries"
        self._discover_query_handlers(query_handlers_path)
        
        # Discover event handlers
        event_handlers_path = cqrs_path / "events"
        self._discover_event_handlers(event_handlers_path)
        
        logger.info(
            "Discovered handlers",
            command_handlers=len(self.command_handlers),
            query_handlers=len(self.query_handlers),
            event_handlers=sum(len(handlers) for handlers in self.event_handlers.values())
        )
    
    def _discover_command_handlers(self, path: Path) -> None:
        """Discover command handlers in path"""
        if not path.exists() or not path.is_dir():
            logger.warning(f"Command handlers path not found: {path}")
            return
            
        # Import all modules in the commands directory
        for item in path.iterdir():
            if item.is_file() and item.suffix == '.py' and item.name != '__init__.py':
                module_name = f"src.cqrs.commands.{item.stem}"
                try:
                    module = importlib.import_module(module_name)
                    
                    # Find command handlers
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, CommandHandler) and 
                            obj != CommandHandler):
                            
                            # Find the command type
                            command_type = self._find_handler_type(obj, Command)
                            if command_type:
                                self.command_handlers[command_type] = obj
                                
                except Exception as e:
                    logger.error(
                        f"Error importing command module {module_name}: {str(e)}"
                    )
    
    def _discover_query_handlers(self, path: Path) -> None:
        """Discover query handlers in path"""
        if not path.exists() or not path.is_dir():
            logger.warning(f"Query handlers path not found: {path}")
            return
            
        # Import all modules in the queries directory
        for item in path.iterdir():
            if item.is_file() and item.suffix == '.py' and item.name != '__init__.py':
                module_name = f"src.cqrs.queries.{item.stem}"
                try:
                    module = importlib.import_module(module_name)
                    
                    # Find query handlers
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, QueryHandler) and 
                            obj != QueryHandler):
                            
                            # Find the query type
                            query_type = self._find_handler_type(obj, Query)
                            if query_type:
                                self.query_handlers[query_type] = obj
                                
                except Exception as e:
                    logger.error(
                        f"Error importing query module {module_name}: {str(e)}"
                    )
    
    def _discover_event_handlers(self, path: Path) -> None:
        """Discover event handlers in path"""
        if not path.exists() or not path.is_dir():
            logger.warning(f"Event handlers path not found: {path}")
            return
            
        # Import all modules in the events directory
        for item in path.iterdir():
            if item.is_file() and item.suffix == '.py' and item.name != '__init__.py':
                module_name = f"src.cqrs.events.{item.stem}"
                try:
                    module = importlib.import_module(module_name)
                    
                    # Find event handlers
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, EventHandler) and 
                            obj != EventHandler):
                            
                            # Try to find event type from handle method annotation
                            event_types = self._find_event_types(obj)
                            for event_type in event_types:
                                if event_type not in self.event_handlers:
                                    self.event_handlers[event_type] = []
                                self.event_handlers[event_type].append(obj)
                                
                except Exception as e:
                    logger.error(
                        f"Error importing event module {module_name}: {str(e)}"
                    )
    
    def _find_handler_type(
        self,
        handler_class: Type,
        base_class: Type
    ) -> Optional[Type]:
        """Find the handled type from handler class generic parameters
        
        Args:
            handler_class: Handler class to inspect
            base_class: Base class that defines the generic parameter
            
        Returns:
            Handled type if found, None otherwise
        """
        # Try to find type from class annotations
        class_annotations = getattr(handler_class, '__annotations__', {})
        for name, annotation in class_annotations.items():
            if getattr(annotation, '__origin__', None) == base_class:
                args = getattr(annotation, '__args__', [])
                if args:
                    return args[0]
        
        # Try to find type from handle method annotations
        handle_method = getattr(handler_class, 'handle', None)
        if handle_method:
            method_annotations = getattr(handle_method, '__annotations__', {})
            for name, annotation in method_annotations.items():
                if (name != 'return' and 
                    isinstance(annotation, type) and 
                    issubclass(annotation, base_class)):
                    return annotation
        
        return None
    
    def _find_event_types(self, handler_class: Type) -> Set[str]:
        """Find event types from handler class
        
        Args:
            handler_class: Handler class to inspect
            
        Returns:
            Set of event types this handler can handle
        """
        # Look for @handles decorator or HANDLED_EVENTS class attribute
        handled_events = getattr(handler_class, 'HANDLED_EVENTS', None)
        if handled_events:
            return set(handled_events)
            
        # Default to class name without "Handler" suffix
        class_name = handler_class.__name__
        if class_name.endswith('Handler'):
            event_type = class_name[:-7].upper()
            return {event_type}
            
        return {'EVENT_DEFAULT'}
    
    def _register_handlers(self) -> None:
        """Register discovered handlers with dispatchers"""
        if not self.command_dispatcher or not self.query_dispatcher or not self.event_bus:
            logger.error("Dispatchers not available")
            return
            
        # Register command handlers
        for command_type, handler_class in self.command_handlers.items():
            try:
                # Create handler instance
                handler = self._create_handler_instance(handler_class)
                
                # Register with dispatcher
                self.command_dispatcher.register_handler(command_type, handler)
                
                logger.info(
                    "Registered command handler",
                    command=command_type.__name__,
                    handler=handler_class.__name__
                )
                
            except Exception as e:
                logger.error(
                    f"Error registering command handler: {str(e)}",
                    command=command_type.__name__,
                    handler=handler_class.__name__
                )
        
        # Register query handlers
        for query_type, handler_class in self.query_handlers.items():
            try:
                # Create handler instance
                handler = self._create_handler_instance(handler_class)
                
                # Register with dispatcher
                self.query_dispatcher.register_handler(query_type, handler)
                
                logger.info(
                    "Registered query handler",
                    query=query_type.__name__,
                    handler=handler_class.__name__
                )
                
            except Exception as e:
                logger.error(
                    f"Error registering query handler: {str(e)}",
                    query=query_type.__name__,
                    handler=handler_class.__name__
                )
        
        # Register event handlers
        for event_type, handler_classes in self.event_handlers.items():
            for handler_class in handler_classes:
                try:
                    # Create handler instance
                    handler = self._create_handler_instance(handler_class)
                    
                    # Create retry policy
                    retry_policy = RetryPolicy(
                        max_retries=3,
                        retry_delay_seconds=5,
                        exponential_backoff=True
                    )
                    
                    # Register with event bus
                    self.event_bus.subscribe(event_type, handler, retry_policy)
                    
                    logger.info(
                        "Registered event handler",
                        event_type=event_type,
                        handler=handler_class.__name__
                    )
                    
                except Exception as e:
                    logger.error(
                        f"Error registering event handler: {str(e)}",
                        event_type=event_type,
                        handler=handler_class.__name__
                    )
    
    def _create_handler_instance(self, handler_class: Type) -> Any:
        """Create handler instance resolving dependencies"""
        # Get constructor signature
        signature = inspect.signature(handler_class.__init__)
        parameters = {}
        
        # Resolve dependencies for parameters
        for name, param in signature.parameters.items():
            if name == 'self':
                continue
                
            param_type = param.annotation
            if param_type != inspect.Parameter.empty:
                try:
                    parameters[name] = container.resolve(param_type)
                except Exception as e:
                    logger.warning(
                        f"Could not resolve parameter {name} of type {param_type}: {str(e)}"
                    )
                    # If parameter has default, we can continue
                    if param.default != inspect.Parameter.empty:
                        parameters[name] = param.default
                    else:
                        raise
        
        # Create instance
        return handler_class(**parameters)
    
    def _configure_middleware(self) -> None:
        """Configure middleware for dispatchers"""
        if not self.command_dispatcher or not self.query_dispatcher:
            logger.error("Dispatchers not available")
            return
            
        if not self.error_handler:
            logger.warning("Error handler not available, skipping error handling middleware")
            return
            
        # Create command middleware
        command_middleware = create_standard_command_middleware(
            self.error_handler,
            self.health_monitor
        )
        
        # Create query middleware
        query_middleware = create_standard_query_middleware(
            self.error_handler,
            self.health_monitor,
            enable_caching=True
        )
        
        # Register middleware
        for middleware in command_middleware:
            self.command_dispatcher.register_middleware(middleware)
            
        for middleware in query_middleware:
            self.query_dispatcher.register_middleware(middleware)
            
        logger.info(
            "Configured middleware",
            command_middleware=len(command_middleware),
            query_middleware=len(query_middleware)
        )

# Create singleton initializer
cqrs_initializer = CqrsInitializer()

async def initialize_cqrs() -> None:
    """Initialize CQRS components"""
    await cqrs_initializer.initialize() 