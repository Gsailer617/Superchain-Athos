"""
Rate Limiter Module

This module provides asynchronous rate limiting functionality using semaphores
for controlling API request rates.
"""

import asyncio
from typing import Dict, Optional
import logging
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    max_requests: int = 1  # Maximum concurrent requests
    requests_per_second: float = 1.0  # Rate limit in requests per second
    burst_size: Optional[int] = None  # Maximum burst size (if None, uses max_requests)

class AsyncRateLimiter:
    """
    Asynchronous rate limiter using semaphores and token bucket algorithm
    """
    
    def __init__(self, name: str, config: RateLimitConfig):
        """
        Initialize rate limiter
        
        Args:
            name: Name of the rate limited resource
            config: Rate limit configuration
        """
        self.name = name
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_requests)
        self.last_request_time = 0.0
        self.tokens = config.burst_size or config.max_requests
        self._lock = asyncio.Lock()
        
    async def acquire(self):
        """
        Acquire rate limit permit
        
        Raises:
            asyncio.TimeoutError: If permit cannot be acquired within timeout
        """
        async with self._lock:
            # Update token bucket
            now = time.time()
            time_passed = now - self.last_request_time
            self.tokens = min(
                self.config.burst_size or self.config.max_requests,
                self.tokens + time_passed * self.config.requests_per_second
            )
            
            if self.tokens < 1:
                # Calculate wait time for next token
                wait_time = (1 - self.tokens) / self.config.requests_per_second
                logger.debug(
                    f"Rate limit reached for {self.name}, "
                    f"waiting {wait_time:.2f} seconds"
                )
                await asyncio.sleep(wait_time)
                self.tokens = 1
            
            self.tokens -= 1
            self.last_request_time = now
            
        await self.semaphore.acquire()
        
    def release(self):
        """Release rate limit permit"""
        self.semaphore.release()
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.acquire()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        self.release()

class RateLimiterRegistry:
    """
    Registry for managing multiple rate limiters
    """
    
    def __init__(self):
        """Initialize empty registry"""
        self._limiters: Dict[str, AsyncRateLimiter] = {}
        
    def get_limiter(
        self,
        name: str,
        max_requests: int = 1,
        requests_per_second: float = 1.0,
        burst_size: Optional[int] = None
    ) -> AsyncRateLimiter:
        """
        Get or create rate limiter
        
        Args:
            name: Name of the rate limited resource
            max_requests: Maximum concurrent requests
            requests_per_second: Rate limit in requests per second
            burst_size: Maximum burst size
            
        Returns:
            Rate limiter instance
        """
        if name not in self._limiters:
            config = RateLimitConfig(
                max_requests=max_requests,
                requests_per_second=requests_per_second,
                burst_size=burst_size
            )
            self._limiters[name] = AsyncRateLimiter(name, config)
        return self._limiters[name]
        
    async def cleanup(self):
        """Clean up all rate limiters"""
        self._limiters.clear()

# Global registry instance
registry = RateLimiterRegistry() 