"""
Cache Module

This module provides Redis-based caching functionality with configurable expiration policies
and automatic refresh mechanisms.
"""

from redis.asyncio import Redis
from typing import Optional, Dict, Any
from dataclasses import dataclass
import json
import time
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Configuration for caching"""
    duration: int = 1800  # 30 minutes default
    refresh_threshold: int = 1500  # 25 minutes
    max_size: int = 10000  # Maximum cache entries

class AsyncCache:
    """
    Asynchronous caching implementation using Redis
    """
    
    def __init__(self, redis_url: str, config: CacheConfig = CacheConfig()):
        """
        Initialize the cache with Redis connection and configuration
        
        Args:
            redis_url: Redis connection URL
            config: Cache configuration parameters
        """
        self.redis: Redis = Redis.from_url(redis_url)
        self.config = config
        
    async def get(self, key: str) -> Optional[Dict]:
        """
        Get a value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if exists and not expired, None otherwise
        """
        try:
            data = await self.redis.get(key)
            if not data:
                return None
                
            result = json.loads(data)
                
            # Check if refresh is needed
            if self._needs_refresh(result):
                return result
                
            return result
            except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            return None
            
    async def set(self, key: str, value: Dict) -> None:
        """
        Set a value in cache
        
        Args:
            key: Cache key
            value: Value to cache
        """
        try:
            # Add timestamp if not present
            if 'timestamp' not in value:
                value['timestamp'] = time.time()
                
            await self.redis.setex(
                key,
                self.config.duration,
                json.dumps(value)
            )
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            
    def _needs_refresh(self, data: Dict) -> bool:
        """
        Check if cached data needs refresh
        
        Args:
            data: Cached data with timestamp
            
        Returns:
            True if data should be refreshed, False otherwise
        """
        if 'timestamp' not in data:
            return True
            
        age = time.time() - data['timestamp']
        return age > self.config.refresh_threshold
        
    async def delete(self, key: str) -> None:
        """
        Delete a key from cache
        
        Args:
            key: Cache key to delete
        """
        try:
            await self.redis.delete(key)
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")

    async def clear(self) -> None:
        """Clear all cached data"""
        try:
            await self.redis.flushdb()
        except Exception as e:
            logger.error(f"Error clearing cache: {e}") 