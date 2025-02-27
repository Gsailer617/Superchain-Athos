"""
Middleware Components for CQRS

This module provides middleware implementations for commands and queries:
- Logging middleware
- Metrics middleware 
- Validation middleware
- Error handling middleware
- Caching middleware for queries
"""

import time
import structlog
from datetime import datetime
from typing import Any, Dict, Optional, Set, List, TypeVar, cast
import asyncio
import json
import functools
import hashlib

from ..commands.base import Command
from ..queries.base import Query
from .dispatcher import CommandMiddleware, QueryMiddleware
from ...core.error_handling import ErrorHandler, ErrorSeverity
from ...core.health_monitor import HealthMonitor

logger = structlog.get_logger(__name__)
T = TypeVar('T')

class LoggingMiddleware(CommandMiddleware, QueryMiddleware):
    """Middleware for logging commands and queries"""
    
    def __init__(self, log_level: str = "INFO"):
        self.log_level = log_level
        self.log_func = getattr(logger, log_level.lower())
    
    async def before_execution(self, message: Any) -> Any:
        """Log before execution"""
        message_type = type(message).__name__
        self.log_func(
            f"Processing {message_type}",
            message_id=getattr(message, 'id', None),
            message_type=message_type,
            timestamp=datetime.now().isoformat()
        )
        return message
    
    async def after_execution(self, message: Any, result: Any) -> Any:
        """Log after execution"""
        message_type = type(message).__name__
        self.log_func(
            f"Completed {message_type}",
            message_id=getattr(message, 'id', None),
            message_type=message_type,
            timestamp=datetime.now().isoformat(),
            success=True
        )
        return result
    
    async def on_error(self, message: Any, error: Exception) -> None:
        """Log on error"""
        message_type = type(message).__name__
        logger.error(
            f"Error processing {message_type}",
            message_id=getattr(message, 'id', None),
            message_type=message_type,
            error=str(error),
            error_type=type(error).__name__,
            timestamp=datetime.now().isoformat()
        )

class MetricsMiddleware(CommandMiddleware, QueryMiddleware):
    """Middleware for collecting metrics on commands and queries"""
    
    def __init__(self, health_monitor: Optional[HealthMonitor] = None):
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.health_monitor = health_monitor
        self.execution_times: Dict[str, List[float]] = {}
    
    async def before_execution(self, message: Any) -> Any:
        """Record start time"""
        message_type = type(message).__name__
        
        # Initialize metrics for message type if needed
        if message_type not in self.metrics:
            self.metrics[message_type] = {
                "count": 0,
                "success_count": 0,
                "error_count": 0,
                "avg_execution_time": 0.0,
                "last_execution_time": None
            }
            self.execution_times[message_type] = []
        
        # Increment count
        self.metrics[message_type]["count"] += 1
        
        # Store start time in message metadata
        message.metadata["start_time"] = time.time()
        return message
    
    async def after_execution(self, message: Any, result: Any) -> Any:
        """Record metrics after execution"""
        message_type = type(message).__name__
        start_time = message.metadata.get("start_time")
        
        if start_time:
            execution_time = time.time() - start_time
            
            # Update metrics
            self.metrics[message_type]["success_count"] += 1
            self.metrics[message_type]["last_execution_time"] = execution_time
            
            # Update average execution time
            self.execution_times[message_type].append(execution_time)
            # Keep only last 100 execution times
            if len(self.execution_times[message_type]) > 100:
                self.execution_times[message_type] = self.execution_times[message_type][-100:]
            
            self.metrics[message_type]["avg_execution_time"] = (
                sum(self.execution_times[message_type]) / 
                len(self.execution_times[message_type])
            )
            
            # Add execution time to result metadata if it's a dict
            if isinstance(result, dict) and "metadata" not in result:
                result["metadata"] = {}
                result["metadata"]["execution_time"] = execution_time
            
        return result
    
    async def on_error(self, message: Any, error: Exception) -> None:
        """Record error metrics"""
        message_type = type(message).__name__
        
        if message_type in self.metrics:
            self.metrics[message_type]["error_count"] += 1
    
    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get collected metrics"""
        return self.metrics

class ErrorHandlingMiddleware(CommandMiddleware, QueryMiddleware):
    """Middleware for handling errors in commands and queries"""
    
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
    
    async def on_error(self, message: Any, error: Exception) -> None:
        """Handle error with error handler"""
        message_type = type(message).__name__
        component = f"cqrs.{message_type.lower()}"
        
        # Determine severity based on error type
        severity = ErrorSeverity.ERROR
        if isinstance(error, ValueError):
            severity = ErrorSeverity.WARNING
        elif isinstance(error, RuntimeError):
            severity = ErrorSeverity.ERROR
        
        # Record error
        self.error_handler.record_error(
            component=component,
            error=error,
            severity=severity,
            context={
                "message_id": getattr(message, 'id', None),
                "message_type": message_type,
                "timestamp": datetime.now().isoformat()
            }
        )

class ValidationMiddleware(CommandMiddleware, QueryMiddleware):
    """Middleware for additional validation of commands and queries"""
    
    def __init__(self):
        self.validators: Dict[str, List[callable]] = {}
    
    def add_validator(self, message_type: str, validator: callable) -> None:
        """Add a validator for a message type"""
        if message_type not in self.validators:
            self.validators[message_type] = []
        
        self.validators[message_type].append(validator)
    
    async def before_execution(self, message: Any) -> Any:
        """Run validators before execution"""
        message_type = type(message).__name__
        
        # Run registered validators
        if message_type in self.validators:
            for validator in self.validators[message_type]:
                result = validator(message)
                
                # Handle async validators
                if asyncio.iscoroutine(result):
                    result = await result
                
                if not result:
                    raise ValueError(f"Validation failed for {message_type}")
        
        return message

class QueryCachingMiddleware(QueryMiddleware):
    """Middleware for caching query results"""
    
    def __init__(self, ttl_seconds: int = 60, max_cache_size: int = 100):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl_seconds = ttl_seconds
        self.max_cache_size = max_cache_size
        self.metrics = {
            "hits": 0,
            "misses": 0,
            "cache_size": 0
        }
    
    def _get_cache_key(self, query: Query) -> str:
        """Generate a cache key for a query"""
        # Create a deterministic representation of the query
        query_dict = {
            "type": type(query).__name__,
            "id": query.id
        }
        
        # Add all fields from the query
        for key, value in query.__dict__.items():
            if key not in ["id", "timestamp", "metadata"]:
                query_dict[key] = value
        
        # Generate a hash
        query_json = json.dumps(query_dict, sort_keys=True)
        return hashlib.md5(query_json.encode()).hexdigest()
    
    async def before_execution(self, query: Query) -> Query:
        """Check cache before execution"""
        # Skip caching if requested in metadata
        if query.metadata.get("skip_cache", False):
            return query
        
        cache_key = self._get_cache_key(query)
        
        # Check if cached result exists and is valid
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            
            # Check if entry is still valid
            if time.time() - cache_entry["timestamp"] < self.ttl_seconds:
                # Add cache metadata
                query.metadata["cache_hit"] = True
                query.metadata["cached_at"] = cache_entry["timestamp"]
                
                self.metrics["hits"] += 1
                
                # Return cached result by storing it in metadata
                query.metadata["cached_result"] = cache_entry["result"]
                return query
        
        self.metrics["misses"] += 1
        return query
    
    async def after_execution(self, query: Query, result: Any) -> Any:
        """Cache result after execution"""
        # Skip caching if requested or if we had a cache hit
        if (query.metadata.get("skip_cache", False) or 
            query.metadata.get("cache_hit", False)):
            return result
        
        cache_key = self._get_cache_key(query)
        
        # Cache the result
        self.cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }
        
        self.metrics["cache_size"] = len(self.cache)
        
        # Clean up cache if it's too large
        if len(self.cache) > self.max_cache_size:
            # Remove oldest entries
            oldest_keys = sorted(
                self.cache.keys(),
                key=lambda k: self.cache[k]["timestamp"]
            )[:len(self.cache) - self.max_cache_size]
            
            for key in oldest_keys:
                del self.cache[key]
            
            self.metrics["cache_size"] = len(self.cache)
        
        return result

# Create factory functions for middleware
def create_standard_command_middleware(
    error_handler: ErrorHandler,
    health_monitor: Optional[HealthMonitor] = None
) -> List[CommandMiddleware]:
    """Create standard middleware for commands"""
    return [
        LoggingMiddleware(),
        MetricsMiddleware(health_monitor),
        ErrorHandlingMiddleware(error_handler),
        ValidationMiddleware()
    ]

def create_standard_query_middleware(
    error_handler: ErrorHandler,
    health_monitor: Optional[HealthMonitor] = None,
    enable_caching: bool = True
) -> List[QueryMiddleware]:
    """Create standard middleware for queries"""
    middleware = [
        LoggingMiddleware(),
        MetricsMiddleware(health_monitor),
        ErrorHandlingMiddleware(error_handler),
        ValidationMiddleware()
    ]
    
    if enable_caching:
        middleware.append(QueryCachingMiddleware())
    
    return middleware 