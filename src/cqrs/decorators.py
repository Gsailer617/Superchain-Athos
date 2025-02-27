"""
CQRS Decorators and Utilities

This module provides decorators and utilities to make it easy to use CQRS:
- Decorators for registering handlers
- Decorators for automatically generating command and query objects
- Utilities for working with events
"""

import functools
import inspect
import uuid
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, get_type_hints
from datetime import datetime
import asyncio
import structlog
from pydantic import BaseModel, create_model

from .commands.base import Command, CommandHandler
from .queries.base import Query, QueryHandler
from .events.base import Event, EventHandler
from .handlers.dispatcher import CommandDispatcher, QueryDispatcher
from .events.event_bus import EventBus, PublishOptions
from ..core.dependency_injector import container

logger = structlog.get_logger(__name__)

T = TypeVar('T')
R = TypeVar('R')

def handles_command(command_type: Type[Command]):
    """Decorator to mark a class as a command handler
    
    Args:
        command_type: Type of command this handler handles
        
    Example:
        @handles_command(MyCommand)
        class MyCommandHandler(CommandHandler[MyCommand]):
            ...
    """
    def decorator(cls):
        # Add annotation to class
        if not hasattr(cls, '__annotations__'):
            cls.__annotations__ = {}
        cls.__annotations__['_command_type'] = command_type
        
        return cls
    return decorator

def handles_query(query_type: Type[Query]):
    """Decorator to mark a class as a query handler
    
    Args:
        query_type: Type of query this handler handles
        
    Example:
        @handles_query(MyQuery)
        class MyQueryHandler(QueryHandler[MyQuery, MyResult]):
            ...
    """
    def decorator(cls):
        # Add annotation to class
        if not hasattr(cls, '__annotations__'):
            cls.__annotations__ = {}
        cls.__annotations__['_query_type'] = query_type
        
        return cls
    return decorator

def handles_event(event_type: str):
    """Decorator to mark a class as an event handler
    
    Args:
        event_type: Type of event this handler handles
        
    Example:
        @handles_event("USER_CREATED")
        class UserCreatedHandler(EventHandler[Event]):
            ...
    """
    def decorator(cls):
        # Add attribute to class
        if not hasattr(cls, 'HANDLED_EVENTS'):
            cls.HANDLED_EVENTS = [event_type]
        else:
            cls.HANDLED_EVENTS.append(event_type)
        
        return cls
    return decorator

def command(name: Optional[str] = None):
    """Decorator to convert a function into a command dispatcher
    
    Args:
        name: Optional name for the command (defaults to function name)
        
    Example:
        @command()
        async def create_user(name: str, email: str) -> None:
            # Automatically converted to a command object and dispatched
            pass
    """
    def decorator(func):
        # Get function signature
        signature = inspect.signature(func)
        func_name = name or func.__name__
        
        # Create command class dynamically
        command_name = f"{func_name.title().replace('_', '')}Command"
        command_attrs = {
            'id': (str, ...),
            'timestamp': (datetime, Field(default_factory=datetime.now)),
            'metadata': (Dict[str, Any], Field(default_factory=dict))
        }
        
        # Add function parameters to command attributes
        for param_name, param in signature.parameters.items():
            if param.default is inspect.Parameter.empty:
                command_attrs[param_name] = (param.annotation, ...)
            else:
                command_attrs[param_name] = (param.annotation, param.default)
        
        # Create command class
        command_class = create_model(
            command_name,
            __base__=Command,
            **command_attrs
        )
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Create command instance
            command_id = str(uuid.uuid4())
            command_data = {'id': command_id}
            
            # Add function arguments
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            command_data.update(bound.arguments)
            
            # Create command
            command = command_class(**command_data)
            
            # Get command dispatcher
            try:
                command_dispatcher = container.resolve(CommandDispatcher)
            except Exception as e:
                logger.error(f"Error resolving command dispatcher: {str(e)}")
                raise
            
            # Dispatch command
            await command_dispatcher.dispatch(command)
            
            # Return command ID for reference
            return command_id
            
        return wrapper
    return decorator

def query(name: Optional[str] = None):
    """Decorator to convert a function into a query dispatcher
    
    Args:
        name: Optional name for the query (defaults to function name)
        
    Example:
        @query()
        async def get_user(user_id: str) -> User:
            # Automatically converted to a query object and dispatched
            pass
    """
    def decorator(func):
        # Get function signature
        signature = inspect.signature(func)
        return_type = signature.return_annotation
        func_name = name or func.__name__
        
        # Create query class dynamically
        query_name = f"{func_name.title().replace('_', '')}Query"
        query_attrs = {
            'id': (str, ...),
            'timestamp': (datetime, Field(default_factory=datetime.now)),
            'metadata': (Dict[str, Any], Field(default_factory=dict))
        }
        
        # Add function parameters to query attributes
        for param_name, param in signature.parameters.items():
            if param.default is inspect.Parameter.empty:
                query_attrs[param_name] = (param.annotation, ...)
            else:
                query_attrs[param_name] = (param.annotation, param.default)
        
        # Create query class
        query_class = create_model(
            query_name,
            __base__=Query,
            **query_attrs
        )
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Create query instance
            query_id = str(uuid.uuid4())
            query_data = {'id': query_id}
            
            # Add function arguments
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            query_data.update(bound.arguments)
            
            # Create query
            query = query_class(**query_data)
            
            # Get query dispatcher
            try:
                query_dispatcher = container.resolve(QueryDispatcher)
            except Exception as e:
                logger.error(f"Error resolving query dispatcher: {str(e)}")
                raise
            
            # Dispatch query
            result = await query_dispatcher.dispatch(query)
            
            # Return result
            return result
            
        return wrapper
    return decorator

def publish_event(
    event_type: str,
    aggregate_id: str,
    payload: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Utility function to publish an event
    
    Args:
        event_type: Type of event
        aggregate_id: ID of the aggregate
        payload: Event payload
        metadata: Optional event metadata
        
    Returns:
        Event ID
    """
    try:
        # Create event
        event_id = str(uuid.uuid4())
        event = Event(
            id=event_id,
            aggregate_id=aggregate_id,
            event_type=event_type,
            payload=payload,
            metadata=metadata or {}
        )
        
        # Get event bus
        try:
            event_bus = container.resolve(EventBus)
        except Exception as e:
            logger.error(f"Error resolving event bus: {str(e)}")
            raise
        
        # Publish event
        asyncio.create_task(event_bus.publish(event))
        
        return event_id
        
    except Exception as e:
        logger.error(f"Error publishing event: {str(e)}")
        raise

def with_bulkhead(name: str):
    """Decorator to run a function within a bulkhead
    
    Args:
        name: Name of the bulkhead
        
    Example:
        @with_bulkhead("my_bulkhead")
        async def my_function():
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            from .bulkhead.base import BulkheadRegistry
            
            # Get bulkhead registry
            try:
                bulkhead_registry = container.resolve(BulkheadRegistry)
            except Exception as e:
                logger.error(f"Error resolving bulkhead registry: {str(e)}")
                return await func(*args, **kwargs)
            
            # Get bulkhead
            bulkhead = bulkhead_registry.get(name)
            if not bulkhead:
                logger.warning(f"Bulkhead {name} not found")
                return await func(*args, **kwargs)
            
            # Execute within bulkhead
            return await bulkhead.execute(func, *args, **kwargs)
            
        return wrapper
    return decorator

class CommandBuilder:
    """Builder pattern for creating commands
    
    Example:
        command = (
            CommandBuilder("CreateUser")
            .with_param("name", "John")
            .with_param("email", "john@example.com")
            .build()
        )
    """
    
    def __init__(self, command_type: Union[str, Type[Command]]):
        self.command_type = command_type
        self.params = {}
        self.metadata = {}
        self.command_id = str(uuid.uuid4())
    
    def with_id(self, command_id: str) -> 'CommandBuilder':
        """Set command ID"""
        self.command_id = command_id
        return self
    
    def with_param(self, name: str, value: Any) -> 'CommandBuilder':
        """Add a parameter"""
        self.params[name] = value
        return self
    
    def with_metadata(self, key: str, value: Any) -> 'CommandBuilder':
        """Add metadata"""
        self.metadata[key] = value
        return self
    
    def build(self) -> Command:
        """Build the command"""
        if isinstance(self.command_type, str):
            # Dynamically create command type
            command_name = f"{self.command_type}Command"
            
            # Look for existing command type
            for subclass in Command.__subclasses__():
                if subclass.__name__ == command_name:
                    command_class = subclass
                    break
            else:
                # Create command class
                command_attrs = {
                    'id': (str, ...),
                    'timestamp': (datetime, Field(default_factory=datetime.now)),
                    'metadata': (Dict[str, Any], Field(default_factory=dict))
                }
                
                # Add parameters
                for name, value in self.params.items():
                    command_attrs[name] = (type(value), ...)
                
                command_class = create_model(
                    command_name,
                    __base__=Command,
                    **command_attrs
                )
        else:
            command_class = self.command_type
        
        # Create command
        command_data = {
            'id': self.command_id,
            'metadata': self.metadata
        }
        command_data.update(self.params)
        
        return command_class(**command_data)
    
    async def dispatch(self) -> str:
        """Build and dispatch the command"""
        command = self.build()
        
        # Get command dispatcher
        try:
            command_dispatcher = container.resolve(CommandDispatcher)
        except Exception as e:
            logger.error(f"Error resolving command dispatcher: {str(e)}")
            raise
        
        # Dispatch command
        await command_dispatcher.dispatch(command)
        
        return command.id

class QueryBuilder:
    """Builder pattern for creating queries
    
    Example:
        result = await (
            QueryBuilder("GetUser")
            .with_param("user_id", "123")
            .dispatch()
        )
    """
    
    def __init__(self, query_type: Union[str, Type[Query]]):
        self.query_type = query_type
        self.params = {}
        self.metadata = {}
        self.query_id = str(uuid.uuid4())
    
    def with_id(self, query_id: str) -> 'QueryBuilder':
        """Set query ID"""
        self.query_id = query_id
        return self
    
    def with_param(self, name: str, value: Any) -> 'QueryBuilder':
        """Add a parameter"""
        self.params[name] = value
        return self
    
    def with_metadata(self, key: str, value: Any) -> 'QueryBuilder':
        """Add metadata"""
        self.metadata[key] = value
        return self
    
    def build(self) -> Query:
        """Build the query"""
        if isinstance(self.query_type, str):
            # Dynamically create query type
            query_name = f"{self.query_type}Query"
            
            # Look for existing query type
            for subclass in Query.__subclasses__():
                if subclass.__name__ == query_name:
                    query_class = subclass
                    break
            else:
                # Create query class
                query_attrs = {
                    'id': (str, ...),
                    'timestamp': (datetime, Field(default_factory=datetime.now)),
                    'metadata': (Dict[str, Any], Field(default_factory=dict))
                }
                
                # Add parameters
                for name, value in self.params.items():
                    query_attrs[name] = (type(value), ...)
                
                query_class = create_model(
                    query_name,
                    __base__=Query,
                    **query_attrs
                )
        else:
            query_class = self.query_type
        
        # Create query
        query_data = {
            'id': self.query_id,
            'metadata': self.metadata
        }
        query_data.update(self.params)
        
        return query_class(**query_data)
    
    async def dispatch(self) -> Any:
        """Build and dispatch the query"""
        query = self.build()
        
        # Get query dispatcher
        try:
            query_dispatcher = container.resolve(QueryDispatcher)
        except Exception as e:
            logger.error(f"Error resolving query dispatcher: {str(e)}")
            raise
        
        # Dispatch query
        return await query_dispatcher.dispatch(query) 