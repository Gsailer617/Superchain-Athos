"""
Real-Time Data Component Module

This module provides real-time data handling capabilities for the visualization dashboard,
including WebSocket connections, live updates, and event processing.
"""

import asyncio
import json
import logging
import websockets
from typing import Dict, List, Optional, Union, Any, Callable, Set
from datetime import datetime, timedelta
import structlog
from dataclasses import dataclass
from prometheus_client import Counter, Histogram, Gauge
from collections import deque
import aiohttp
from functools import partial

logger = structlog.get_logger(__name__)

# Metrics for monitoring real-time performance
METRICS = {
    'websocket_latency': Histogram(
        'realtime_websocket_latency_seconds',
        'WebSocket message latency',
        buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    ),
    'message_count': Counter(
        'realtime_messages_total',
        'Total number of real-time messages',
        ['type']
    ),
    'connection_errors': Counter(
        'realtime_connection_errors_total',
        'Total number of connection errors',
        ['type']
    ),
    'active_connections': Gauge(
        'realtime_active_connections',
        'Number of active WebSocket connections'
    )
}

@dataclass
class RealtimeConfig:
    """Configuration for real-time data handling"""
    websocket_url: str
    reconnect_interval: float = 1.0
    max_reconnect_attempts: int = 5
    message_buffer_size: int = 1000
    batch_size: int = 10
    update_interval: float = 0.1  # seconds
    heartbeat_interval: float = 30.0  # seconds

class RealtimeDataManager:
    """
    Manages real-time data connections and updates.
    
    Features:
    - WebSocket connection management
    - Automatic reconnection
    - Message buffering and batching
    - Event subscription system
    - Performance monitoring
    """
    
    def __init__(self, config: RealtimeConfig):
        """Initialize real-time data manager"""
        self.config = config
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False
        self.running = False
        self.message_buffer: deque = deque(maxlen=config.message_buffer_size)
        self.subscribers: Dict[str, Set[Callable]] = {}
        self.tasks: List[asyncio.Task] = []
        self._setup_metrics()
    
    def _setup_metrics(self) -> None:
        """Initialize metrics tracking"""
        METRICS['active_connections'].set(0)
        for msg_type in ['trade', 'gas', 'network', 'opportunity']:
            METRICS['message_count'].labels(type=msg_type)
            METRICS['connection_errors'].labels(type=msg_type)
    
    async def start(self) -> None:
        """Start real-time data processing"""
        if self.running:
            return
        
        self.running = True
        self.tasks = [
            asyncio.create_task(self._maintain_connection()),
            asyncio.create_task(self._process_messages()),
            asyncio.create_task(self._send_heartbeat())
        ]
        
        logger.info("Real-time data processing started")
    
    async def stop(self) -> None:
        """Stop real-time data processing"""
        self.running = False
        
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        for task in self.tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self.tasks = []
        logger.info("Real-time data processing stopped")
    
    def subscribe(self, event_type: str, callback: Callable) -> None:
        """Subscribe to real-time events"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = set()
        self.subscribers[event_type].add(callback)
    
    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """Unsubscribe from real-time events"""
        if event_type in self.subscribers:
            self.subscribers[event_type].discard(callback)
    
    async def _maintain_connection(self) -> None:
        """Maintain WebSocket connection with automatic reconnection"""
        reconnect_attempts = 0
        
        while self.running:
            try:
                if not self.connected:
                    async with websockets.connect(self.config.websocket_url) as ws:
                        self.websocket = ws
                        self.connected = True
                        METRICS['active_connections'].inc()
                        reconnect_attempts = 0
                        logger.info("WebSocket connection established")
                        
                        while self.running:
                            try:
                                message = await ws.recv()
                                await self._handle_message(message)
                            except websockets.ConnectionClosed:
                                break
                    
                    self.connected = False
                    METRICS['active_connections'].dec()
                    logger.warning("WebSocket connection closed")
            
            except Exception as e:
                self.connected = False
                reconnect_attempts += 1
                METRICS['connection_errors'].labels(type='websocket').inc()
                
                if reconnect_attempts >= self.config.max_reconnect_attempts:
                    logger.error("Max reconnection attempts reached", error=str(e))
                    break
                
                logger.warning(
                    "Connection error, attempting reconnect",
                    attempt=reconnect_attempts,
                    error=str(e)
                )
                await asyncio.sleep(self.config.reconnect_interval)
    
    async def _handle_message(self, message: str) -> None:
        """Handle incoming WebSocket message"""
        try:
            with METRICS['websocket_latency'].time():
                data = json.loads(message)
                message_type = data.get('type', 'unknown')
                
                # Update metrics
                METRICS['message_count'].labels(type=message_type).inc()
                
                # Add to buffer
                self.message_buffer.append(data)
                
                # Process high-priority messages immediately
                if data.get('priority') == 'high':
                    await self._process_message(data)
                
        except json.JSONDecodeError as e:
            logger.error("Error decoding message", error=str(e))
        except Exception as e:
            logger.error("Error handling message", error=str(e))
    
    async def _process_messages(self) -> None:
        """Process buffered messages in batches"""
        while self.running:
            try:
                if len(self.message_buffer) >= self.config.batch_size:
                    messages = []
                    for _ in range(self.config.batch_size):
                        if self.message_buffer:
                            messages.append(self.message_buffer.popleft())
                    
                    # Group messages by type
                    grouped_messages: Dict[str, List[Dict]] = {}
                    for msg in messages:
                        msg_type = msg.get('type', 'unknown')
                        if msg_type not in grouped_messages:
                            grouped_messages[msg_type] = []
                        grouped_messages[msg_type].append(msg)
                    
                    # Process each group
                    for msg_type, msgs in grouped_messages.items():
                        await self._process_message_group(msg_type, msgs)
                
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                logger.error("Error processing messages", error=str(e))
    
    async def _process_message_group(
        self,
        message_type: str,
        messages: List[Dict]
    ) -> None:
        """Process a group of messages of the same type"""
        if message_type in self.subscribers:
            for subscriber in self.subscribers[message_type]:
                try:
                    await asyncio.create_task(subscriber(messages))
                except Exception as e:
                    logger.error(
                        "Error in subscriber callback",
                        subscriber=subscriber.__name__,
                        error=str(e)
                    )
    
    async def _process_message(self, message: Dict) -> None:
        """Process a single message"""
        message_type = message.get('type', 'unknown')
        if message_type in self.subscribers:
            for subscriber in self.subscribers[message_type]:
                try:
                    await asyncio.create_task(subscriber([message]))
                except Exception as e:
                    logger.error(
                        "Error in subscriber callback",
                        subscriber=subscriber.__name__,
                        error=str(e)
                    )
    
    async def _send_heartbeat(self) -> None:
        """Send periodic heartbeat to keep connection alive"""
        while self.running:
            try:
                if self.websocket and self.connected:
                    await self.websocket.send(json.dumps({
                        'type': 'heartbeat',
                        'timestamp': datetime.now().isoformat()
                    }))
                await asyncio.sleep(self.config.heartbeat_interval)
            except Exception as e:
                logger.error("Error sending heartbeat", error=str(e))
                self.connected = False

class DashboardUpdater:
    """
    Handles real-time updates to the dashboard.
    
    Features:
    - Automatic data updates
    - Chart refresh management
    - Performance optimization
    """
    
    def __init__(
        self,
        realtime_manager: RealtimeDataManager,
        update_callbacks: Dict[str, Callable]
    ):
        """Initialize dashboard updater"""
        self.realtime_manager = realtime_manager
        self.update_callbacks = update_callbacks
        self.last_updates: Dict[str, datetime] = {}
        self._setup_subscriptions()
    
    def _setup_subscriptions(self) -> None:
        """Setup subscriptions for different data types"""
        for event_type in self.update_callbacks:
            self.realtime_manager.subscribe(
                event_type,
                partial(self._handle_update, event_type)
            )
    
    async def _handle_update(
        self,
        event_type: str,
        messages: List[Dict]
    ) -> None:
        """Handle updates for specific data types"""
        try:
            # Check if update is needed
            now = datetime.now()
            if event_type in self.last_updates:
                time_since_last = (now - self.last_updates[event_type]).total_seconds()
                if time_since_last < 1.0:  # Minimum 1 second between updates
                    return
            
            # Call update callback
            await self.update_callbacks[event_type](messages)
            self.last_updates[event_type] = now
            
        except Exception as e:
            logger.error(
                "Error handling dashboard update",
                event_type=event_type,
                error=str(e)
            )

class EventBuffer:
    """
    Buffers and aggregates real-time events.
    
    Features:
    - Event buffering with TTL
    - Automatic cleanup
    - Aggregation functions
    """
    
    def __init__(self, ttl: float = 60.0):
        """Initialize event buffer"""
        self.buffer: Dict[str, deque] = {}
        self.ttl = ttl
        self.last_cleanup = datetime.now()
    
    def add_event(self, event_type: str, event_data: Dict) -> None:
        """Add event to buffer"""
        if event_type not in self.buffer:
            self.buffer[event_type] = deque(maxlen=1000)
        
        self.buffer[event_type].append({
            **event_data,
            '_timestamp': datetime.now()
        })
        
        # Periodic cleanup
        if (datetime.now() - self.last_cleanup).seconds > 60:
            self._cleanup()
    
    def get_events(
        self,
        event_type: str,
        time_window: Optional[float] = None
    ) -> List[Dict]:
        """Get events from buffer"""
        if event_type not in self.buffer:
            return []
        
        events = list(self.buffer[event_type])
        if time_window is not None:
            cutoff = datetime.now() - timedelta(seconds=time_window)
            events = [
                e for e in events
                if e['_timestamp'] > cutoff
            ]
        
        return events
    
    def _cleanup(self) -> None:
        """Clean up expired events"""
        cutoff = datetime.now() - timedelta(seconds=self.ttl)
        for event_type in self.buffer:
            self.buffer[event_type] = deque(
                [e for e in self.buffer[event_type] if e['_timestamp'] > cutoff],
                maxlen=1000
            )
        self.last_cleanup = datetime.now() 