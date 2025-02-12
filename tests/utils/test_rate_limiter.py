"""Unit tests for the rate limiter module"""

import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock
from src.utils.rate_limiter import (
    AsyncRateLimiter,
    RateLimiterRegistry,
    RateLimitConfig
)

@pytest.fixture
def rate_limiter():
    """Create a rate limiter instance"""
    config = RateLimitConfig(
        max_requests=2,
        requests_per_second=1.0,
        burst_size=3
    )
    return AsyncRateLimiter('test', config)

@pytest.fixture
def registry():
    """Create a rate limiter registry instance"""
    return RateLimiterRegistry()

@pytest.mark.asyncio
async def test_rate_limiter_basic_acquire_release(rate_limiter):
    """Test basic acquire and release functionality"""
    # Should acquire without waiting
    await rate_limiter.acquire()
    assert rate_limiter.tokens < rate_limiter.config.max_requests
    
    # Release should work
    rate_limiter.release()
    assert rate_limiter.semaphore._value == rate_limiter.config.max_requests

@pytest.mark.asyncio
async def test_rate_limiter_max_concurrent(rate_limiter):
    """Test maximum concurrent requests limit"""
    # Acquire up to max_requests
    for _ in range(rate_limiter.config.max_requests):
        await rate_limiter.acquire()
    
    # Next acquire should block
    with pytest.raises(asyncio.TimeoutError):
        async with asyncio.timeout(0.1):
            await rate_limiter.acquire()
    
    # Release one, should be able to acquire again
    rate_limiter.release()
    await rate_limiter.acquire()

@pytest.mark.asyncio
async def test_rate_limiter_requests_per_second():
    """Test requests per second limiting"""
    config = RateLimitConfig(
        max_requests=5,
        requests_per_second=2.0  # 2 requests per second
    )
    limiter = AsyncRateLimiter('test', config)
    
    start_time = time.time()
    
    # Make 4 requests (should take ~2 seconds)
    for _ in range(4):
        await limiter.acquire()
        limiter.release()
    
    duration = time.time() - start_time
    assert duration >= 1.5  # Allow some margin for timing

@pytest.mark.asyncio
async def test_rate_limiter_burst():
    """Test burst handling"""
    config = RateLimitConfig(
        max_requests=3,
        requests_per_second=1.0,
        burst_size=3
    )
    limiter = AsyncRateLimiter('test', config)
    
    # Should handle burst of 3 requests immediately
    for _ in range(3):
        await limiter.acquire()
        limiter.release()
    
    # Next request should be rate limited
    start_time = time.time()
    await limiter.acquire()
    duration = time.time() - start_time
    assert duration >= 0.9  # Should wait ~1 second

@pytest.mark.asyncio
async def test_rate_limiter_context_manager(rate_limiter):
    """Test async context manager interface"""
    async with rate_limiter:
        assert rate_limiter.semaphore._value < rate_limiter.config.max_requests
    
    assert rate_limiter.semaphore._value == rate_limiter.config.max_requests

def test_registry_get_limiter(registry):
    """Test getting/creating limiters from registry"""
    # Get new limiter
    limiter1 = registry.get_limiter('test1', max_requests=2)
    assert isinstance(limiter1, AsyncRateLimiter)
    assert limiter1.config.max_requests == 2
    
    # Get existing limiter
    limiter2 = registry.get_limiter('test1')
    assert limiter1 is limiter2  # Should return same instance
    
    # Get limiter with different config
    limiter3 = registry.get_limiter('test2', max_requests=3)
    assert limiter3 is not limiter1
    assert limiter3.config.max_requests == 3

@pytest.mark.asyncio
async def test_registry_cleanup(registry):
    """Test registry cleanup"""
    # Create some limiters
    registry.get_limiter('test1')
    registry.get_limiter('test2')
    
    assert len(registry._limiters) == 2
    
    # Cleanup
    await registry.cleanup()
    assert len(registry._limiters) == 0

@pytest.mark.asyncio
async def test_multiple_limiters(registry):
    """Test multiple limiters working independently"""
    limiter1 = registry.get_limiter('test1', max_requests=1)
    limiter2 = registry.get_limiter('test2', max_requests=1)
    
    # Acquire first limiter
    await limiter1.acquire()
    
    # Should still be able to acquire second limiter
    await limiter2.acquire()
    
    # Release both
    limiter1.release()
    limiter2.release()

@pytest.mark.asyncio
async def test_rate_limiter_error_handling():
    """Test error handling in rate limiter"""
    config = RateLimitConfig(max_requests=1)
    limiter = AsyncRateLimiter('test', config)
    
    # Simulate error during execution
    with pytest.raises(ValueError):
        async with limiter:
            raise ValueError("Test error")
    
    # Semaphore should be released despite error
    assert limiter.semaphore._value == 1

@pytest.mark.asyncio
async def test_custom_rate_limit_config():
    """Test rate limiter with custom configuration"""
    config = RateLimitConfig(
        max_requests=5,
        requests_per_second=10.0,
        burst_size=8
    )
    limiter = AsyncRateLimiter('test', config)
    
    assert limiter.config.max_requests == 5
    assert limiter.config.requests_per_second == 10.0
    assert limiter.config.burst_size == 8
    assert limiter.tokens == 8  # Initial tokens should equal burst_size 