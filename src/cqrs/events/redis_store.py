from typing import Dict, List, Optional, Any
from datetime import datetime
import aioredis
import json
from .base import Event, EventStore
import structlog

logger = structlog.get_logger(__name__)

class RedisEventStore(EventStore):
    """Redis-based event store implementation"""
    
    def __init__(self, redis_url: str, namespace: str = "events"):
        self.redis_url = redis_url
        self.namespace = namespace
        self.redis: Optional[aioredis.Redis] = None
    
    async def init(self):
        """Initialize Redis connection"""
        if not self.redis:
            self.redis = await aioredis.from_url(
                self.redis_url,
                encoding='utf-8',
                decode_responses=True
            )
    
    async def save_event(self, event: Event) -> None:
        """Save an event to Redis"""
        if not self.redis:
            await self.init()
        
        try:
            # Create event data
            event_data = {
                "id": event.id,
                "aggregate_id": event.aggregate_id,
                "event_type": event.event_type,
                "timestamp": event.timestamp.isoformat(),
                "version": event.version,
                "metadata": event.metadata,
                "payload": event.payload
            }
            
            # Save event in a list for the aggregate
            key = f"{self.namespace}:events:{event.aggregate_id}"
            await self.redis.rpush(key, json.dumps(event_data))
            
            # Update latest version
            version_key = f"{self.namespace}:version:{event.aggregate_id}"
            await self.redis.set(version_key, event.version)
            
            logger.info(
                "Event saved",
                event_id=event.id,
                aggregate_id=event.aggregate_id,
                event_type=event.event_type
            )
            
        except Exception as e:
            logger.error(
                "Error saving event",
                error=str(e),
                event_id=event.id
            )
            raise
    
    async def get_events(
        self,
        aggregate_id: str,
        since_version: Optional[int] = None
    ) -> List[Event]:
        """Get events for an aggregate from Redis"""
        if not self.redis:
            await self.init()
        
        try:
            # Get all events for the aggregate
            key = f"{self.namespace}:events:{aggregate_id}"
            events_data = await self.redis.lrange(key, 0, -1)
            
            # Parse events
            events = []
            for event_json in events_data:
                event_data = json.loads(event_json)
                
                # Skip events before since_version
                if since_version and event_data["version"] <= since_version:
                    continue
                
                event = Event(
                    id=event_data["id"],
                    aggregate_id=event_data["aggregate_id"],
                    event_type=event_data["event_type"],
                    timestamp=datetime.fromisoformat(event_data["timestamp"]),
                    version=event_data["version"],
                    metadata=event_data["metadata"],
                    payload=event_data["payload"]
                )
                events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(
                "Error getting events",
                error=str(e),
                aggregate_id=aggregate_id
            )
            raise
    
    async def get_latest_version(self, aggregate_id: str) -> int:
        """Get latest version for an aggregate from Redis"""
        if not self.redis:
            await self.init()
        
        try:
            # Get latest version
            version_key = f"{self.namespace}:version:{aggregate_id}"
            version = await self.redis.get(version_key)
            
            return int(version) if version else 0
            
        except Exception as e:
            logger.error(
                "Error getting latest version",
                error=str(e),
                aggregate_id=aggregate_id
            )
            raise 