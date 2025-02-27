"""
Event Bus Module

This module provides an event bus for publishing and subscribing to events:
- Support for both synchronous and asynchronous event processing
- Event subscription based on event types
- Integration with message brokers for distributed processing
- Retry policies for failed event handlers
"""

import asyncio
import structlog
from typing import Dict, List, Any, Callable, Optional, Set, Type, Union, TypeVar
from dataclasses import dataclass, field
import time
import json
import traceback
from datetime import datetime

from .base import Event, EventHandler
from ..bulkhead.base import Bulkhead, BulkheadRegistry
from ...core.error_handling import ErrorHandler, ErrorSeverity

logger = structlog.get_logger(__name__)
T = TypeVar('T')

@dataclass
class RetryPolicy:
    """Retry policy for failed event handlers"""
    max_retries: int = 3
    retry_delay_seconds: int = 5
    exponential_backoff: bool = True  # Increases delay with each retry
    discard_after_max_retries: bool = False  # If False, will log but not retry

@dataclass
class PublishOptions:
    """Options for publishing an event"""
    retry_policy: Optional[RetryPolicy] = None
    use_bulkhead: bool = True
    save_to_store: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

class EventBus:
    """Event bus for publishing and subscribing to events"""
    
    def __init__(
        self,
        error_handler: Optional[ErrorHandler] = None,
        bulkhead_registry: Optional[BulkheadRegistry] = None
    ):
        self.subscribers: Dict[str, List[EventHandler]] = {}
        self.error_handler = error_handler
        self.bulkhead_registry = bulkhead_registry
        self.event_types: Set[str] = set()
        self.retry_policies: Dict[str, RetryPolicy] = {}
        self.processing_queue = asyncio.Queue()
        self.is_processing = False
        self.metrics: Dict[str, Dict[str, Any]] = {}
    
    def subscribe(
        self,
        event_type: str,
        handler: EventHandler,
        retry_policy: Optional[RetryPolicy] = None
    ) -> None:
        """Subscribe a handler to an event type
        
        Args:
            event_type: Type of event to subscribe to
            handler: Handler for the event
            retry_policy: Optional retry policy for the handler
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        self.subscribers[event_type].append(handler)
        self.event_types.add(event_type)
        
        if retry_policy:
            handler_key = f"{event_type}:{handler.__class__.__name__}"
            self.retry_policies[handler_key] = retry_policy
        
        logger.info(
            f"Subscribed handler to event type",
            event_type=event_type,
            handler=handler.__class__.__name__
        )
    
    async def publish(
        self,
        event: Event,
        options: Optional[PublishOptions] = None
    ) -> None:
        """Publish an event to subscribers
        
        Args:
            event: Event to publish
            options: Optional publishing options
        """
        options = options or PublishOptions()
        
        # Initialize metrics for event type if needed
        if event.event_type not in self.metrics:
            self.metrics[event.event_type] = {
                "published_count": 0,
                "processed_count": 0,
                "error_count": 0,
                "last_event_time": None
            }
        
        self.metrics[event.event_type]["published_count"] += 1
        self.metrics[event.event_type]["last_event_time"] = time.time()
        
        # Add optional metadata
        if options.metadata:
            for key, value in options.metadata.items():
                if key not in event.metadata:
                    event.metadata[key] = value
        
        # Add publish time
        event.metadata["published_at"] = datetime.now().isoformat()
        
        # Queue the event for processing
        await self.processing_queue.put((event, options))
        
        # Start processing if not already running
        if not self.is_processing:
            asyncio.create_task(self._process_events())
            self.is_processing = True
        
        logger.info(
            "Event published",
            event_id=event.id,
            event_type=event.event_type,
            aggregate_id=event.aggregate_id
        )
    
    async def _process_events(self) -> None:
        """Process events from the queue"""
        try:
            while not self.processing_queue.empty():
                event, options = await self.processing_queue.get()
                
                try:
                    # Process the event
                    await self._process_event(event, options)
                finally:
                    self.processing_queue.task_done()
                    
            self.is_processing = False
            
        except Exception as e:
            logger.error(
                "Error processing events",
                error=str(e),
                traceback=traceback.format_exc()
            )
            self.is_processing = False
    
    async def _process_event(self, event: Event, options: PublishOptions) -> None:
        """Process a single event
        
        Args:
            event: Event to process
            options: Publishing options
        """
        event_type = event.event_type
        
        # Check if we have subscribers for this event
        if event_type not in self.subscribers:
            logger.warning(
                "No subscribers for event type",
                event_type=event_type,
                event_id=event.id
            )
            return
        
        # Process with each subscriber
        for handler in self.subscribers[event_type]:
            handler_name = handler.__class__.__name__
            
            try:
                # Apply bulkhead if requested and available
                if options.use_bulkhead and self.bulkhead_registry:
                    bulkhead_name = f"event_{event_type}"
                    bulkhead = self.bulkhead_registry.get(bulkhead_name)
                    
                    if bulkhead:
                        await bulkhead.execute(handler.handle, event)
                    else:
                        await handler.handle(event)
                else:
                    await handler.handle(event)
                
                self.metrics[event_type]["processed_count"] += 1
                
                logger.debug(
                    "Event handled",
                    event_id=event.id,
                    event_type=event_type,
                    handler=handler_name
                )
                
            except Exception as e:
                error_message = str(e)
                self.metrics[event_type]["error_count"] += 1
                
                logger.error(
                    "Error handling event",
                    event_id=event.id,
                    event_type=event_type,
                    handler=handler_name,
                    error=error_message,
                    traceback=traceback.format_exc()
                )
                
                # Record error if handler available
                if self.error_handler:
                    self.error_handler.record_error(
                        component=f"cqrs.event.{handler_name}",
                        error=e,
                        severity=ErrorSeverity.ERROR,
                        context={
                            "event_id": event.id,
                            "event_type": event_type,
                            "aggregate_id": event.aggregate_id
                        }
                    )
                
                # Check if we should retry
                handler_key = f"{event_type}:{handler_name}"
                retry_policy = self.retry_policies.get(
                    handler_key,
                    options.retry_policy
                )
                
                if retry_policy and self._should_retry(event, handler, retry_policy):
                    asyncio.create_task(
                        self._retry_handler(event, handler, retry_policy)
                    )
    
    def _should_retry(
        self,
        event: Event,
        handler: EventHandler,
        retry_policy: RetryPolicy
    ) -> bool:
        """Determine if we should retry a failed handler
        
        Args:
            event: Event that failed
            handler: Handler that failed
            retry_policy: Retry policy to apply
            
        Returns:
            True if we should retry, False otherwise
        """
        # Get current retry count
        retry_count = event.metadata.get(f"retry_count_{handler.__class__.__name__}", 0)
        
        # Check if we've reached max retries
        if retry_count >= retry_policy.max_retries:
            if not retry_policy.discard_after_max_retries:
                logger.warning(
                    "Max retries reached for handler",
                    event_id=event.id,
                    event_type=event.event_type,
                    handler=handler.__class__.__name__,
                    retry_count=retry_count
                )
            return False
        
        return True
    
    async def _retry_handler(
        self,
        event: Event,
        handler: EventHandler,
        retry_policy: RetryPolicy
    ) -> None:
        """Retry a failed handler with exponential backoff
        
        Args:
            event: Event to retry
            handler: Handler to retry
            retry_policy: Retry policy to apply
        """
        handler_name = handler.__class__.__name__
        
        # Get and increment retry count
        retry_count = event.metadata.get(f"retry_count_{handler_name}", 0)
        retry_count += 1
        event.metadata[f"retry_count_{handler_name}"] = retry_count
        
        # Calculate delay with exponential backoff if enabled
        delay = retry_policy.retry_delay_seconds
        if retry_policy.exponential_backoff:
            delay = delay * (2 ** (retry_count - 1))
        
        logger.info(
            "Scheduling retry for handler",
            event_id=event.id,
            event_type=event.event_type,
            handler=handler_name,
            retry_count=retry_count,
            delay=delay
        )
        
        # Wait for delay
        await asyncio.sleep(delay)
        
        # Try again
        try:
            await handler.handle(event)
            
            logger.info(
                "Retry succeeded for handler",
                event_id=event.id,
                event_type=event.event_type,
                handler=handler_name,
                retry_count=retry_count
            )
            
        except Exception as e:
            logger.error(
                "Retry failed for handler",
                event_id=event.id,
                event_type=event.event_type,
                handler=handler_name,
                retry_count=retry_count,
                error=str(e)
            )
            
            # Record error if handler available
            if self.error_handler:
                self.error_handler.record_error(
                    component=f"cqrs.event.{handler_name}.retry",
                    error=e,
                    severity=ErrorSeverity.ERROR,
                    context={
                        "event_id": event.id,
                        "event_type": event.event_type,
                        "aggregate_id": event.aggregate_id,
                        "retry_count": retry_count
                    }
                )
            
            # Try again if we haven't reached max retries
            if self._should_retry(event, handler, retry_policy):
                asyncio.create_task(
                    self._retry_handler(event, handler, retry_policy)
                )
    
    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all event types"""
        return self.metrics

class ExternalEventPublisher:
    """Publishes events to external message brokers"""
    
    def __init__(self, name: str, connection_config: Dict[str, Any]):
        self.name = name
        self.connection_config = connection_config
        self.is_connected = False
        self.metrics = {
            "published_count": 0,
            "error_count": 0,
            "last_publish_time": None
        }
    
    async def connect(self) -> None:
        """Connect to the external broker"""
        # Implementation depends on the specific broker
        self.is_connected = True
        logger.info(f"Connected to external broker: {self.name}")
    
    async def publish(self, event: Event) -> None:
        """Publish an event to the external broker
        
        Args:
            event: Event to publish
        """
        try:
            if not self.is_connected:
                await self.connect()
            
            # Prepare event for external publishing
            event_data = {
                "id": event.id,
                "aggregate_id": event.aggregate_id,
                "event_type": event.event_type,
                "timestamp": event.timestamp.isoformat(),
                "version": event.version,
                "metadata": event.metadata,
                "payload": event.payload
            }
            
            # Implementation depends on the specific broker
            # This is a placeholder for the implementation
            logger.info(
                f"Publishing event to external broker: {self.name}",
                event_id=event.id,
                event_type=event.event_type
            )
            
            self.metrics["published_count"] += 1
            self.metrics["last_publish_time"] = time.time()
            
        except Exception as e:
            self.metrics["error_count"] += 1
            logger.error(
                f"Error publishing to external broker: {self.name}",
                event_id=event.id,
                error=str(e)
            )
            raise

class EventPublisherRegistry:
    """Registry for external event publishers"""
    
    def __init__(self):
        self.publishers: Dict[str, ExternalEventPublisher] = {}
    
    def register_publisher(
        self,
        name: str,
        publisher: ExternalEventPublisher
    ) -> None:
        """Register an external publisher
        
        Args:
            name: Name of the publisher
            publisher: Publisher instance
        """
        self.publishers[name] = publisher
        logger.info(f"Registered external publisher: {name}")
    
    async def publish_to_all(self, event: Event) -> None:
        """Publish an event to all registered publishers
        
        Args:
            event: Event to publish
        """
        for name, publisher in self.publishers.items():
            try:
                await publisher.publish(event)
            except Exception as e:
                logger.error(
                    f"Error publishing to {name}",
                    event_id=event.id,
                    error=str(e)
                )
    
    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all publishers"""
        return {
            name: publisher.metrics
            for name, publisher in self.publishers.items()
        }

# Create singleton instance
event_bus = EventBus()

def register_with_container():
    """Register event bus with dependency container"""
    from ...core.dependency_injector import container
    from ...core.error_handling import ErrorHandler
    
    error_handler = container.resolve(ErrorHandler)
    
    # Create and register event bus
    event_bus = EventBus(error_handler)
    container.register_instance(EventBus, event_bus) 