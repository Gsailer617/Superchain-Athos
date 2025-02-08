"""
Caching Mechanism Module

This module provides thread-safe and async caching capabilities with TTL management,
performance monitoring, and automatic cleanup of expired entries.
"""

import asyncio
import time
from typing import Dict, Any, Optional, TypeVar, Generic, Union, Callable
from datetime import datetime, timedelta
import threading
from dataclasses import dataclass
import structlog
from prometheus_client import Counter, Histogram, Gauge
from functools import wraps
import json

logger = structlog.get_logger(__name__)

# Type variables for generic caching
KT = TypeVar('KT')  # Key type
VT = TypeVar('VT')  # Value type

# Metrics for cache performance monitoring
METRICS = {
    'cache_hits': Counter(
        'cache_hits_total',
        'Total number of cache hits',
        ['cache_type']
    ),
    'cache_misses': Counter(
        'cache_misses_total',
        'Total number of cache misses',
        ['cache_type']
    ),
    'cache_size': Gauge(
        'cache_size_bytes',
        'Current size of cache in bytes',
        ['cache_type']
    ),
    'cache_latency': Histogram(
        'cache_operation_seconds',
        'Time spent on cache operations',
        ['operation'],
        buckets=[0.0001, 0.001, 0.01, 0.1, 1.0]
    ),
    'cache_evictions': Counter(
        'cache_evictions_total',
        'Total number of cache evictions',
        ['reason']
    )
}

@dataclass
class CacheConfig:
    """Configuration for cache behavior"""
    ttl: float = 300.0  # Default TTL of 5 minutes
    max_size: int = 10000  # Maximum number of items
    cleanup_interval: float = 60.0  # Cleanup every minute
    enable_metrics: bool = True
    serializer: Optional[Callable[[Any], str]] = None
    deserializer: Optional[Callable[[str], Any]] = None

@dataclass
class CacheEntry(Generic[VT]):
    """Cache entry with value and metadata"""
    value: VT
    expires_at: float
    size: int = 0
    access_count: int = 0
    last_access: float = 0.0

class Cache(Generic[KT, VT]):
    """
    Thread-safe cache implementation with TTL support.
    
    Features:
    - Thread-safe operations
    - TTL management
    - Size-based eviction
    - Performance monitoring
    - Automatic cleanup
    """
    
    def __init__(self, config: CacheConfig):
        """Initialize cache with configuration"""
        self.config = config
        self._cache: Dict[KT, CacheEntry[VT]] = {}
        self._lock = threading.RLock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._setup_metrics()
    
    def _setup_metrics(self) -> None:
        """Initialize cache metrics"""
        if self.config.enable_metrics:
            METRICS['cache_size'].labels(cache_type='memory').set(0)
            for metric in ['hits', 'misses']:
                METRICS[f'cache_{metric}'].labels(cache_type='memory')
    
    def get(self, key: KT) -> Optional[VT]:
        """Get value from cache with TTL check"""
        with self._lock:
            try:
                with METRICS['cache_latency'].labels(operation='get').time():
                    if key in self._cache:
                        entry = self._cache[key]
                        current_time = time.time()
                        
                        # Check if entry is expired
                        if current_time > entry.expires_at:
                            del self._cache[key]
                            if self.config.enable_metrics:
                                METRICS['cache_evictions'].labels(reason='expired').inc()
                                METRICS['cache_misses'].labels(cache_type='memory').inc()
                            return None
                        
                        # Update access metadata
                        entry.access_count += 1
                        entry.last_access = current_time
                        
                        if self.config.enable_metrics:
                            METRICS['cache_hits'].labels(cache_type='memory').inc()
                        
                        return entry.value
                    
                    if self.config.enable_metrics:
                        METRICS['cache_misses'].labels(cache_type='memory').inc()
                    return None
                    
            except Exception as e:
                logger.error("Error getting from cache", error=str(e))
                return None
    
    def set(self, key: KT, value: VT, ttl: Optional[float] = None) -> bool:
        """Set value in cache with optional TTL override"""
        with self._lock:
            try:
                with METRICS['cache_latency'].labels(operation='set').time():
                    # Check cache size limit
                    if len(self._cache) >= self.config.max_size:
                        self._evict_entries()
                    
                    # Calculate entry size
                    size = self._calculate_size(value)
                    
                    # Create cache entry
                    expires_at = time.time() + (ttl or self.config.ttl)
                    entry = CacheEntry(
                        value=value,
                        expires_at=expires_at,
                        size=size,
                        access_count=0,
                        last_access=time.time()
                    )
                    
                    self._cache[key] = entry
                    
                    if self.config.enable_metrics:
                        METRICS['cache_size'].labels(cache_type='memory').inc(size)
                    
                    return True
                    
            except Exception as e:
                logger.error("Error setting cache value", error=str(e))
                return False
    
    def delete(self, key: KT) -> bool:
        """Delete entry from cache"""
        with self._lock:
            try:
                with METRICS['cache_latency'].labels(operation='delete').time():
                    if key in self._cache:
                        entry = self._cache[key]
                        del self._cache[key]
                        
                        if self.config.enable_metrics:
                            METRICS['cache_size'].labels(cache_type='memory').dec(entry.size)
                            METRICS['cache_evictions'].labels(reason='explicit').inc()
                        
                        return True
                    return False
                    
            except Exception as e:
                logger.error("Error deleting from cache", error=str(e))
                return False
    
    def clear(self) -> None:
        """Clear all entries from cache"""
        with self._lock:
            try:
                with METRICS['cache_latency'].labels(operation='clear').time():
                    self._cache.clear()
                    if self.config.enable_metrics:
                        METRICS['cache_size'].labels(cache_type='memory').set(0)
                        METRICS['cache_evictions'].labels(reason='clear').inc()
                    
            except Exception as e:
                logger.error("Error clearing cache", error=str(e))
    
    def _evict_entries(self) -> None:
        """Evict entries based on access patterns"""
        try:
            # Sort entries by access count and last access time
            entries = sorted(
                self._cache.items(),
                key=lambda x: (x[1].access_count, x[1].last_access)
            )
            
            # Remove 10% of least accessed entries
            num_to_remove = max(1, len(entries) // 10)
            for key, entry in entries[:num_to_remove]:
                del self._cache[key]
                if self.config.enable_metrics:
                    METRICS['cache_size'].labels(cache_type='memory').dec(entry.size)
                    METRICS['cache_evictions'].labels(reason='size').inc()
                
        except Exception as e:
            logger.error("Error evicting cache entries", error=str(e))
    
    def _calculate_size(self, value: VT) -> int:
        """Calculate approximate size of cached value"""
        try:
            if self.config.serializer:
                serialized = self.config.serializer(value)
                return len(serialized)
            return len(str(value))
        except Exception:
            return 1
    
    async def start_cleanup(self) -> None:
        """Start periodic cleanup task"""
        if self._cleanup_task is not None:
            return
        
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Cache cleanup task started")
    
    async def stop_cleanup(self) -> None:
        """Stop periodic cleanup task"""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("Cache cleanup task stopped")
    
    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of expired entries"""
        while True:
            try:
                with METRICS['cache_latency'].labels(operation='cleanup').time():
                    current_time = time.time()
                    with self._lock:
                        # Find expired entries
                        expired_keys = [
                            key for key, entry in self._cache.items()
                            if current_time > entry.expires_at
                        ]
                        
                        # Remove expired entries
                        for key in expired_keys:
                            entry = self._cache[key]
                            del self._cache[key]
                            if self.config.enable_metrics:
                                METRICS['cache_size'].labels(cache_type='memory').dec(entry.size)
                                METRICS['cache_evictions'].labels(reason='expired').inc()
                
                await asyncio.sleep(self.config.cleanup_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in cache cleanup", error=str(e))
                await asyncio.sleep(self.config.cleanup_interval)

class AsyncCache(Generic[KT, VT]):
    """
    Asynchronous cache implementation.
    
    Features:
    - Async operations
    - TTL management
    - Size-based eviction
    - Performance monitoring
    """
    
    def __init__(self, config: CacheConfig):
        """Initialize async cache"""
        self.config = config
        self._cache: Dict[KT, CacheEntry[VT]] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._setup_metrics()
    
    def _setup_metrics(self) -> None:
        """Initialize cache metrics"""
        if self.config.enable_metrics:
            METRICS['cache_size'].labels(cache_type='async').set(0)
            for metric in ['hits', 'misses']:
                METRICS[f'cache_{metric}'].labels(cache_type='async')
    
    async def get(self, key: KT) -> Optional[VT]:
        """Get value from cache asynchronously"""
        async with self._lock:
            try:
                with METRICS['cache_latency'].labels(operation='async_get').time():
                    if key in self._cache:
                        entry = self._cache[key]
                        current_time = time.time()
                        
                        # Check if entry is expired
                        if current_time > entry.expires_at:
                            del self._cache[key]
                            if self.config.enable_metrics:
                                METRICS['cache_evictions'].labels(reason='expired').inc()
                                METRICS['cache_misses'].labels(cache_type='async').inc()
                            return None
                        
                        # Update access metadata
                        entry.access_count += 1
                        entry.last_access = current_time
                        
                        if self.config.enable_metrics:
                            METRICS['cache_hits'].labels(cache_type='async').inc()
                        
                        return entry.value
                    
                    if self.config.enable_metrics:
                        METRICS['cache_misses'].labels(cache_type='async').inc()
                    return None
                    
            except Exception as e:
                logger.error("Error getting from async cache", error=str(e))
                return None
    
    async def set(self, key: KT, value: VT, ttl: Optional[float] = None) -> bool:
        """Set value in cache asynchronously"""
        async with self._lock:
            try:
                with METRICS['cache_latency'].labels(operation='async_set').time():
                    # Check cache size limit
                    if len(self._cache) >= self.config.max_size:
                        await self._evict_entries()
                    
                    # Calculate entry size
                    size = self._calculate_size(value)
                    
                    # Create cache entry
                    expires_at = time.time() + (ttl or self.config.ttl)
                    entry = CacheEntry(
                        value=value,
                        expires_at=expires_at,
                        size=size,
                        access_count=0,
                        last_access=time.time()
                    )
                    
                    self._cache[key] = entry
                    
                    if self.config.enable_metrics:
                        METRICS['cache_size'].labels(cache_type='async').inc(size)
                    
                    return True
                    
            except Exception as e:
                logger.error("Error setting async cache value", error=str(e))
                return False
    
    async def delete(self, key: KT) -> bool:
        """Delete entry from cache asynchronously"""
        async with self._lock:
            try:
                with METRICS['cache_latency'].labels(operation='async_delete').time():
                    if key in self._cache:
                        entry = self._cache[key]
                        del self._cache[key]
                        
                        if self.config.enable_metrics:
                            METRICS['cache_size'].labels(cache_type='async').dec(entry.size)
                            METRICS['cache_evictions'].labels(reason='explicit').inc()
                        
                        return True
                    return False
                    
            except Exception as e:
                logger.error("Error deleting from async cache", error=str(e))
                return False
    
    async def clear(self) -> None:
        """Clear all entries from cache asynchronously"""
        async with self._lock:
            try:
                with METRICS['cache_latency'].labels(operation='async_clear').time():
                    self._cache.clear()
                    if self.config.enable_metrics:
                        METRICS['cache_size'].labels(cache_type='async').set(0)
                        METRICS['cache_evictions'].labels(reason='clear').inc()
                    
            except Exception as e:
                logger.error("Error clearing async cache", error=str(e))
    
    async def _evict_entries(self) -> None:
        """Evict entries based on access patterns"""
        try:
            # Sort entries by access count and last access time
            entries = sorted(
                self._cache.items(),
                key=lambda x: (x[1].access_count, x[1].last_access)
            )
            
            # Remove 10% of least accessed entries
            num_to_remove = max(1, len(entries) // 10)
            for key, entry in entries[:num_to_remove]:
                del self._cache[key]
                if self.config.enable_metrics:
                    METRICS['cache_size'].labels(cache_type='async').dec(entry.size)
                    METRICS['cache_evictions'].labels(reason='size').inc()
                
        except Exception as e:
            logger.error("Error evicting async cache entries", error=str(e))
    
    def _calculate_size(self, value: VT) -> int:
        """Calculate approximate size of cached value"""
        try:
            if self.config.serializer:
                serialized = self.config.serializer(value)
                return len(serialized)
            return len(str(value))
        except Exception:
            return 1 