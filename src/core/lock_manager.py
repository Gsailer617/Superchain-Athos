"""
Distributed Lock Manager Module

This module provides distributed locking capabilities:
- Redis-based distributed locks
- Lock acquisition with timeout
- Automatic lock release
- Lock health monitoring
- Deadlock prevention
"""

import asyncio
from typing import Optional, Dict, Any, Set, List
from datetime import datetime, timedelta
import structlog
from redis.asyncio import Redis
from prometheus_client import Counter, Gauge, Histogram
import uuid
import json
from dataclasses import dataclass
from contextlib import asynccontextmanager

logger = structlog.get_logger(__name__)

@dataclass
class LockConfig:
    """Lock configuration"""
    ttl: int = 30  # Lock timeout in seconds
    retry_interval: float = 0.1  # Seconds between retries
    retry_timeout: float = 10.0  # Maximum time to retry
    extend_interval: float = 10.0  # Interval to extend lock
    extend_threshold: float = 0.5  # Threshold to start extending (% of TTL)

class LockAcquisitionError(Exception):
    """Error acquiring lock"""
    pass

class LockReleaseError(Exception):
    """Error releasing lock"""
    pass

class DistributedLockManager:
    """Distributed lock manager implementation"""
    
    def __init__(self, redis_url: str):
        self.redis: Optional[Redis] = None
        self.redis_url = redis_url
        self._active_locks: Dict[str, str] = {}  # resource -> lock_id
        self._lock_extensions: Dict[str, asyncio.Task] = {}
        self._setup_metrics()

    def _setup_metrics(self):
        """Initialize Prometheus metrics"""
        self._lock_acquisitions = Counter(
            'lock_acquisitions_total',
            'Number of lock acquisitions',
            ['resource', 'status']
        )
        self._lock_releases = Counter(
            'lock_releases_total',
            'Number of lock releases',
            ['resource', 'status']
        )
        self._active_locks_gauge = Gauge(
            'active_locks',
            'Number of active locks',
            ['resource']
        )
        self._lock_wait_time = Histogram(
            'lock_wait_seconds',
            'Time spent waiting for locks',
            ['resource']
        )
        self._lock_hold_time = Histogram(
            'lock_hold_seconds',
            'Time locks are held',
            ['resource']
        )

    async def init(self):
        """Initialize Redis connection"""
        if not self.redis:
            self.redis = Redis.from_url(self.redis_url, decode_responses=True)
            # Test connection
            await self.redis.ping()

    async def close(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()
            self.redis = None

    @asynccontextmanager
    async def lock(self,
                  resource: str,
                  owner: str = None,
                  config: Optional[LockConfig] = None) -> str:
        """Acquire and automatically release a lock"""
        if not config:
            config = LockConfig()
            
        lock_id = None
        try:
            lock_id = await self.acquire_lock(resource, owner, config)
            yield lock_id
        finally:
            if lock_id:
                await self.release_lock(resource, lock_id)

    async def acquire_lock(self,
                          resource: str,
                          owner: str = None,
                          config: Optional[LockConfig] = None) -> str:
        """Acquire a distributed lock"""
        if not self.redis:
            await self.init()
            
        if not config:
            config = LockConfig()
            
        owner = owner or str(uuid.uuid4())
        lock_id = f"{owner}:{uuid.uuid4()}"
        start_time = asyncio.get_event_loop().time()
        
        try:
            deadline = start_time + config.retry_timeout
            while True:
                # Try to acquire lock
                acquired = await self.redis.set(
                    f"lock:{resource}",
                    lock_id,
                    ex=config.ttl,
                    nx=True
                )
                
                if acquired:
                    # Lock acquired successfully
                    self._active_locks[resource] = lock_id
                    self._lock_acquisitions.labels(
                        resource=resource,
                        status="success"
                    ).inc()
                    self._active_locks_gauge.labels(resource=resource).inc()
                    
                    # Start lock extension task
                    self._start_lock_extension(resource, lock_id, config)
                    return lock_id
                
                # Check timeout
                if asyncio.get_event_loop().time() >= deadline:
                    raise LockAcquisitionError(
                        f"Timeout acquiring lock for {resource}"
                    )
                
                # Wait before retry
                await asyncio.sleep(config.retry_interval)
                
        except Exception as e:
            self._lock_acquisitions.labels(
                resource=resource,
                status="error"
            ).inc()
            raise LockAcquisitionError(f"Error acquiring lock: {str(e)}")
        finally:
            duration = asyncio.get_event_loop().time() - start_time
            self._lock_wait_time.labels(resource=resource).observe(duration)

    async def release_lock(self, resource: str, lock_id: str):
        """Release a distributed lock"""
        if not self.redis:
            await self.init()
            
        start_time = asyncio.get_event_loop().time()
        try:
            # Verify we still own the lock
            current_lock = await self.redis.get(f"lock:{resource}")
            if current_lock != lock_id:
                logger.warning(
                    "Lock already released or taken by another owner",
                    resource=resource,
                    lock_id=lock_id,
                    current_lock=current_lock
                )
                return
            
            # Release the lock
            await self.redis.delete(f"lock:{resource}")
            self._active_locks.pop(resource, None)
            self._lock_releases.labels(
                resource=resource,
                status="success"
            ).inc()
            self._active_locks_gauge.labels(resource=resource).dec()
            
            # Stop lock extension
            if resource in self._lock_extensions:
                self._lock_extensions[resource].cancel()
                del self._lock_extensions[resource]
            
        except Exception as e:
            self._lock_releases.labels(
                resource=resource,
                status="error"
            ).inc()
            raise LockReleaseError(f"Error releasing lock: {str(e)}")
        finally:
            duration = asyncio.get_event_loop().time() - start_time
            self._lock_hold_time.labels(resource=resource).observe(duration)

    def _start_lock_extension(self,
                            resource: str,
                            lock_id: str,
                            config: LockConfig):
        """Start background task to extend lock"""
        async def extend_lock():
            try:
                while True:
                    await asyncio.sleep(config.extend_interval)
                    
                    # Verify we still own the lock
                    current_lock = await self.redis.get(f"lock:{resource}")
                    if current_lock != lock_id:
                        logger.warning(
                            "Lock lost while extending",
                            resource=resource,
                            lock_id=lock_id
                        )
                        break
                    
                    # Extend the lock
                    await self.redis.expire(
                        f"lock:{resource}",
                        config.ttl
                    )
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(
                    "Error extending lock",
                    resource=resource,
                    lock_id=lock_id,
                    error=str(e)
                )
        
        # Start extension task
        self._lock_extensions[resource] = asyncio.create_task(extend_lock())

    async def get_lock_info(self, resource: str) -> Optional[Dict[str, Any]]:
        """Get information about a lock"""
        if not self.redis:
            await self.init()
            
        try:
            lock_id = await self.redis.get(f"lock:{resource}")
            if not lock_id:
                return None
                
            ttl = await self.redis.ttl(f"lock:{resource}")
            owner = lock_id.split(':')[0] if ':' in lock_id else None
            
            return {
                'resource': resource,
                'lock_id': lock_id,
                'owner': owner,
                'ttl': ttl,
                'is_active': resource in self._active_locks
            }
        except Exception as e:
            logger.error(f"Error getting lock info: {str(e)}")
            return None

    async def get_all_locks(self) -> List[Dict[str, Any]]:
        """Get information about all active locks"""
        if not self.redis:
            await self.init()
            
        try:
            locks = []
            async for key in self.redis.scan_iter("lock:*"):
                resource = key.split(':', 1)[1]
                lock_info = await self.get_lock_info(resource)
                if lock_info:
                    locks.append(lock_info)
            return locks
        except Exception as e:
            logger.error(f"Error getting all locks: {str(e)}")
            return []

    async def clear_all_locks(self):
        """Clear all locks (use with caution)"""
        if not self.redis:
            await self.init()
            
        try:
            async for key in self.redis.scan_iter("lock:*"):
                await self.redis.delete(key)
            
            self._active_locks.clear()
            for task in self._lock_extensions.values():
                task.cancel()
            self._lock_extensions.clear()
            
        except Exception as e:
            logger.error(f"Error clearing locks: {str(e)}")

# Global lock manager instance (initialize with Redis URL)
lock_manager: Optional[DistributedLockManager] = None 