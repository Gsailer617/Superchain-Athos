"""
Handler Registry

This module provides a registry for CQRS handlers:
- Automatic discovery of command, query, and event handlers
- Handler registration with dispatchers
- Handler instance creation with dependency injection
"""

import asyncio
import importlib
import inspect
import os
import sys
import pkgutil
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, get_type_hints, cast, Union

import structlog

from ..commands.base import Command, CommandHandler
from ..queries.base import Query, QueryHandler
from ..events.base import Event, EventHandler
from ..handlers.dispatcher import CommandDispatcher, QueryDispatcher
from ..events.event_bus import EventBus
from ...core.dependency_injector import container

logger = structlog.get_logger(__name__)

T = TypeVar('T')

class HandlerRegistry:
    """Registry for CQRS handlers
    
    Attributes:
        command_handlers: Registered command handlers
        query_handlers: Registered query handlers
        event_handlers: Registered event handlers
    """
    
    def __init__(self):
        self.command_handlers: Dict[Type[Command], List[Type[CommandHandler]]] = {}
        self.query_handlers: Dict[Type[Query], Type[QueryHandler]] = {}
        self.event_handlers: Dict[str, List[Type[EventHandler]]] = {}
        
    def register_command_handler(self, command_type: Type[Command], handler_type: Type[CommandHandler]) -> None:
        """Register a command handler
        
        Args:
            command_type: Type of command
            handler_type: Type of handler
        """
        if command_type not in self.command_handlers:
            self.command_handlers[command_type] = []
            
        if handler_type not in self.command_handlers[command_type]:
            self.command_handlers[command_type].append(handler_type)
            logger.debug(f"Registered command handler {handler_type.__name__} for {command_type.__name__}")
        
    def register_query_handler(self, query_type: Type[Query], handler_type: Type[QueryHandler]) -> None:
        """Register a query handler
        
        Args:
            query_type: Type of query
            handler_type: Type of handler
        """
        if query_type in self.query_handlers:
            logger.warning(f"Query handler for {query_type.__name__} already registered, overriding")
            
        self.query_handlers[query_type] = handler_type
        logger.debug(f"Registered query handler {handler_type.__name__} for {query_type.__name__}")
        
    def register_event_handler(self, event_type: str, handler_type: Type[EventHandler]) -> None:
        """Register an event handler
        
        Args:
            event_type: Type of event
            handler_type: Type of handler
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
            
        if handler_type not in self.event_handlers[event_type]:
            self.event_handlers[event_type].append(handler_type)
            logger.debug(f"Registered event handler {handler_type.__name__} for {event_type}")
    
    def get_command_handlers(self, command_type: Type[Command]) -> List[Type[CommandHandler]]:
        """Get handlers for a command type
        
        Args:
            command_type: Type of command
            
        Returns:
            List of handler types
        """
        return self.command_handlers.get(command_type, [])
    
    def get_query_handler(self, query_type: Type[Query]) -> Optional[Type[QueryHandler]]:
        """Get handler for a query type
        
        Args:
            query_type: Type of query
            
        Returns:
            Handler type, if any
        """
        return self.query_handlers.get(query_type)
    
    def get_event_handlers(self, event_type: str) -> List[Type[EventHandler]]:
        """Get handlers for an event type
        
        Args:
            event_type: Type of event
            
        Returns:
            List of handler types
        """
        return self.event_handlers.get(event_type, [])
    
    def has_command_handler(self, command_type: Type[Command]) -> bool:
        """Check if a command type has a handler
        
        Args:
            command_type: Type of command
            
        Returns:
            True if handler exists
        """
        return command_type in self.command_handlers and len(self.command_handlers[command_type]) > 0
    
    def has_query_handler(self, query_type: Type[Query]) -> bool:
        """Check if a query type has a handler
        
        Args:
            query_type: Type of query
            
        Returns:
            True if handler exists
        """
        return query_type in self.query_handlers
    
    def has_event_handler(self, event_type: str) -> bool:
        """Check if an event type has a handler
        
        Args:
            event_type: Type of event
            
        Returns:
            True if handler exists
        """
        return event_type in self.event_handlers and len(self.event_handlers[event_type]) > 0

class HandlerDiscovery:
    """Discover handlers in packages
    
    Attributes:
        registry: Handler registry
    """
    
    def __init__(self, registry: HandlerRegistry):
        self.registry = registry
        
    def discover_in_directory(self, directory: str, package_prefix: str = "") -> None:
        """Discover handlers in a directory
        
        Args:
            directory: Directory to search
            package_prefix: Package prefix
        """
        logger.info(f"Discovering handlers in directory {directory}")
        
        # Check if directory exists
        if not os.path.exists(directory) or not os.path.isdir(directory):
            logger.warning(f"Directory {directory} does not exist")
            return
        
        # Add to path if not already
        if directory not in sys.path:
            sys.path.append(directory)
            
        # Find all modules
        for _, name, is_pkg in pkgutil.iter_modules([directory]):
            if is_pkg:
                # Recursively discover in subpackage
                sub_directory = os.path.join(directory, name)
                sub_package = f"{package_prefix}.{name}" if package_prefix else name
                self.discover_in_directory(sub_directory, sub_package)
            else:
                # Import module
                module_name = f"{package_prefix}.{name}" if package_prefix else name
                try:
                    module = importlib.import_module(module_name)
                    self.discover_in_module(module)
                except ImportError:
                    logger.exception(f"Error importing module {module_name}")
    
    def discover_in_package(self, package_name: str) -> None:
        """Discover handlers in a package
        
        Args:
            package_name: Package name
        """
        logger.info(f"Discovering handlers in package {package_name}")
        
        try:
            # Import package
            package = importlib.import_module(package_name)
            
            # Get package path
            if hasattr(package, '__path__'):
                for _, name, is_pkg in pkgutil.iter_modules(package.__path__):
                    # Import module or subpackage
                    module_name = f"{package_name}.{name}"
                    try:
                        module = importlib.import_module(module_name)
                        if is_pkg:
                            self.discover_in_package(module_name)
                        else:
                            self.discover_in_module(module)
                    except ImportError:
                        logger.exception(f"Error importing module {module_name}")
        except ImportError:
            logger.exception(f"Error importing package {package_name}")
    
    def discover_in_module(self, module) -> None:
        """Discover handlers in a module
        
        Args:
            module: Module to search
        """
        # Find all classes in module
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Skip if not defined in this module
            if obj.__module__ != module.__name__:
                continue
                
            # Check if command handler
            if issubclass(obj, CommandHandler) and obj != CommandHandler:
                self._register_command_handler(obj)
                
            # Check if query handler
            if issubclass(obj, QueryHandler) and obj != QueryHandler:
                self._register_query_handler(obj)
                
            # Check if event handler
            if issubclass(obj, EventHandler) and obj != EventHandler:
                self._register_event_handler(obj)
    
    def _register_command_handler(self, handler_type: Type[CommandHandler]) -> None:
        """Register a command handler
        
        Args:
            handler_type: Type of handler
        """
        # Get command type from type hints or annotations
        command_type = None
        
        # Try to get from generic type parameters
        try:
            if hasattr(handler_type, '__orig_bases__'):
                for base in handler_type.__orig_bases__:
                    if hasattr(base, '__origin__') and base.__origin__ is CommandHandler:
                        command_type = base.__args__[0]
                        break
        except (AttributeError, IndexError):
            pass
            
        # Try to get from annotations
        if not command_type and hasattr(handler_type, '__annotations__'):
            command_type = handler_type.__annotations__.get('_command_type')
            
        # Skip if no command type
        if not command_type:
            logger.warning(f"Could not determine command type for {handler_type.__name__}")
            return
            
        # Register handler
        self.registry.register_command_handler(command_type, handler_type)
    
    def _register_query_handler(self, handler_type: Type[QueryHandler]) -> None:
        """Register a query handler
        
        Args:
            handler_type: Type of handler
        """
        # Get query type from type hints or annotations
        query_type = None
        
        # Try to get from generic type parameters
        try:
            if hasattr(handler_type, '__orig_bases__'):
                for base in handler_type.__orig_bases__:
                    if hasattr(base, '__origin__') and base.__origin__ is QueryHandler:
                        query_type = base.__args__[0]
                        break
        except (AttributeError, IndexError):
            pass
            
        # Try to get from annotations
        if not query_type and hasattr(handler_type, '__annotations__'):
            query_type = handler_type.__annotations__.get('_query_type')
            
        # Skip if no query type
        if not query_type:
            logger.warning(f"Could not determine query type for {handler_type.__name__}")
            return
            
        # Register handler
        self.registry.register_query_handler(query_type, handler_type)
    
    def _register_event_handler(self, handler_type: Type[EventHandler]) -> None:
        """Register an event handler
        
        Args:
            handler_type: Type of handler
        """
        # Get event types from handler
        event_types = getattr(handler_type, 'HANDLED_EVENTS', [])
        
        # Skip if no event types
        if not event_types:
            logger.warning(f"No event types for {handler_type.__name__}")
            return
            
        # Register handlers
        for event_type in event_types:
            self.registry.register_event_handler(event_type, handler_type)

class HandlerFactory:
    """Factory for creating handler instances
    
    This class creates handler instances with dependencies injected.
    """
    
    @staticmethod
    def create_instance(handler_type: Type[T], **kwargs) -> T:
        """Create a handler instance
        
        Args:
            handler_type: Type of handler
            **kwargs: Additional constructor arguments
            
        Returns:
            Handler instance
        """
        # Get constructor parameters
        signature = inspect.signature(handler_type.__init__)
        parameters = {}
        
        # Add provided arguments
        parameters.update(kwargs)
        
        # Resolve constructor dependencies
        for name, param in signature.parameters.items():
            # Skip self parameter
            if name == 'self':
                continue
                
            # Skip if already provided
            if name in parameters:
                continue
                
            # Get parameter type
            param_type = param.annotation
            if param_type is inspect.Parameter.empty:
                if param.default is not inspect.Parameter.empty:
                    # Use default value
                    parameters[name] = param.default
                else:
                    logger.warning(f"Parameter {name} of {handler_type.__name__} has no type annotation or default")
                continue
                
            # Resolve dependency
            try:
                parameters[name] = container.resolve(param_type)
            except Exception as e:
                logger.warning(f"Could not resolve parameter {name} of type {param_type} for {handler_type.__name__}: {e}")
                if param.default is not inspect.Parameter.empty:
                    # Use default value
                    parameters[name] = param.default
        
        # Create instance
        return handler_type(**parameters)

class HandlerRegistrar:
    """Register handlers with dispatchers
    
    Attributes:
        registry: Handler registry
    """
    
    def __init__(self, registry: HandlerRegistry):
        self.registry = registry
        
    async def register_with_dispatchers(self) -> None:
        """Register handlers with dispatchers"""
        await self._register_command_handlers()
        await self._register_query_handlers()
        await self._register_event_handlers()
    
    async def _register_command_handlers(self) -> None:
        """Register command handlers"""
        try:
            # Get command dispatcher
            command_dispatcher = container.resolve(CommandDispatcher)
            
            # Register handlers
            for command_type, handler_types in self.registry.command_handlers.items():
                for handler_type in handler_types:
                    # Create handler instance
                    handler = HandlerFactory.create_instance(handler_type)
                    
                    # Register handler
                    command_dispatcher.register_handler(command_type, handler)
                    
            logger.info(f"Registered {len(command_dispatcher.handlers)} command handlers")
        except Exception as e:
            logger.error(f"Error registering command handlers: {e}")
    
    async def _register_query_handlers(self) -> None:
        """Register query handlers"""
        try:
            # Get query dispatcher
            query_dispatcher = container.resolve(QueryDispatcher)
            
            # Register handlers
            for query_type, handler_type in self.registry.query_handlers.items():
                # Create handler instance
                handler = HandlerFactory.create_instance(handler_type)
                
                # Register handler
                query_dispatcher.register_handler(query_type, handler)
                
            logger.info(f"Registered {len(query_dispatcher.handlers)} query handlers")
        except Exception as e:
            logger.error(f"Error registering query handlers: {e}")
    
    async def _register_event_handlers(self) -> None:
        """Register event handlers"""
        try:
            # Get event bus
            event_bus = container.resolve(EventBus)
            
            # Register handlers
            registered_count = 0
            for event_type, handler_types in self.registry.event_handlers.items():
                for handler_type in handler_types:
                    # Create handler instance
                    handler = HandlerFactory.create_instance(handler_type)
                    
                    # Register handler
                    event_bus.subscribe(event_type, handler)
                    registered_count += 1
                    
            logger.info(f"Registered {registered_count} event handlers")
        except Exception as e:
            logger.error(f"Error registering event handlers: {e}")

# Create singleton instance
registry = HandlerRegistry()

async def discover_and_register_handlers(
    command_handlers_path: Optional[str] = None,
    query_handlers_path: Optional[str] = None,
    event_handlers_path: Optional[str] = None,
    packages: Optional[List[str]] = None
) -> None:
    """Discover and register handlers
    
    Args:
        command_handlers_path: Path to command handlers directory
        query_handlers_path: Path to query handlers directory
        event_handlers_path: Path to event handlers directory
        packages: Packages to search
    """
    # Create discovery
    discovery = HandlerDiscovery(registry)
    
    # Discover in directories
    if command_handlers_path:
        discovery.discover_in_directory(command_handlers_path)
    
    if query_handlers_path:
        discovery.discover_in_directory(query_handlers_path)
    
    if event_handlers_path:
        discovery.discover_in_directory(event_handlers_path)
    
    # Discover in packages
    if packages:
        for package in packages:
            discovery.discover_in_package(package)
    
    # Register with dispatchers
    registrar = HandlerRegistrar(registry)
    await registrar.register_with_dispatchers()
    
    logger.info("Handler discovery and registration complete")

def register_handler_manually(
    handler_type: Union[Type[CommandHandler], Type[QueryHandler], Type[EventHandler]],
    command_type: Optional[Type[Command]] = None,
    query_type: Optional[Type[Query]] = None,
    event_types: Optional[List[str]] = None
) -> None:
    """Register a handler manually
    
    Args:
        handler_type: Type of handler
        command_type: Type of command (for command handlers)
        query_type: Type of query (for query handlers)
        event_types: Types of events (for event handlers)
    """
    if issubclass(handler_type, CommandHandler) and command_type:
        registry.register_command_handler(command_type, handler_type)
    
    if issubclass(handler_type, QueryHandler) and query_type:
        registry.register_query_handler(query_type, handler_type)
    
    if issubclass(handler_type, EventHandler) and event_types:
        for event_type in event_types:
            registry.register_event_handler(event_type, handler_type) 