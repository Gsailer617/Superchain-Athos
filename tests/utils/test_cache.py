"""Unit tests for the cache module"""

import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from src.utils.cache import AsyncCache, CacheConfig

@pytest.fixture
def mock_redis():
    """Create a mock Redis instance"""
    redis_mock = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.setex = AsyncMock()
    redis_mock.delete = AsyncMock()
    redis_mock.flushdb = AsyncMock()
    return redis_mock

@pytest.fixture
async def cache(mock_redis):
    """Create a cache instance with mocked Redis"""
    with patch('src.utils.cache.Redis.from_url', return_value=mock_redis):
        cache = AsyncCache('redis://dummy', CacheConfig())
        yield cache

@pytest.mark.asyncio
async def test_get_missing_key(cache, mock_redis):
    """Test getting a non-existent key"""
    mock_redis.get.return_value = None
    result = await cache.get('missing_key')
    assert result is None
    mock_redis.get.assert_called_once_with('missing_key')

@pytest.mark.asyncio
async def test_get_existing_key(cache, mock_redis):
    """Test getting an existing key"""
    mock_data = {'value': 42, 'timestamp': time.time()}
    mock_redis.get.return_value = '{"value": 42, "timestamp": 123456789.0}'
    
    result = await cache.get('existing_key')
    assert result is not None
    assert 'value' in result
    assert result['value'] == 42
    mock_redis.get.assert_called_once_with('existing_key')

@pytest.mark.asyncio
async def test_set_new_value(cache, mock_redis):
    """Test setting a new value"""
    value = {'test': 'data'}
    await cache.set('new_key', value)
    
    # Verify Redis setex was called with correct arguments
    mock_redis.setex.assert_called_once()
    args = mock_redis.setex.call_args[0]
    assert args[0] == 'new_key'
    assert args[1] == cache.config.duration
    assert 'test' in args[2]
    assert 'timestamp' in args[2]

@pytest.mark.asyncio
async def test_delete_key(cache, mock_redis):
    """Test deleting a key"""
    await cache.delete('key_to_delete')
    mock_redis.delete.assert_called_once_with('key_to_delete')

@pytest.mark.asyncio
async def test_clear_cache(cache, mock_redis):
    """Test clearing the entire cache"""
    await cache.clear()
    mock_redis.flushdb.assert_called_once()

@pytest.mark.asyncio
async def test_needs_refresh(cache):
    """Test refresh check logic"""
    current_time = time.time()
    
    # Test data that needs refresh
    old_data = {'timestamp': current_time - cache.config.refresh_threshold - 100}
    assert cache._needs_refresh(old_data) is True
    
    # Test data that doesn't need refresh
    new_data = {'timestamp': current_time}
    assert cache._needs_refresh(new_data) is False
    
    # Test data without timestamp
    invalid_data = {'value': 42}
    assert cache._needs_refresh(invalid_data) is True

@pytest.mark.asyncio
async def test_redis_connection_error(mock_redis):
    """Test handling Redis connection error"""
    mock_redis.get.side_effect = Exception("Connection error")
    
    with patch('src.utils.cache.Redis.from_url', return_value=mock_redis):
        cache = AsyncCache('redis://dummy', CacheConfig())
        result = await cache.get('any_key')
        assert result is None

@pytest.mark.asyncio
async def test_custom_config(mock_redis):
    """Test cache with custom configuration"""
    custom_config = CacheConfig(
        duration=60,
        refresh_threshold=30,
        max_size=100
    )
    
    with patch('src.utils.cache.Redis.from_url', return_value=mock_redis):
        cache = AsyncCache('redis://dummy', custom_config)
        value = {'test': 'data'}
        await cache.set('test_key', value)
        
        mock_redis.setex.assert_called_once()
        args = mock_redis.setex.call_args[0]
        assert args[1] == 60  # Custom duration 