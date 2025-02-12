"""Unit tests for the token discovery module"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json
import time
from web3 import Web3
from src.agent.token_discovery import (
    TokenDiscovery,
    TokenData,
    ValidationResult,
    SentimentScore,
    DexLiquidity,
    HolderDistribution
)

@pytest.fixture
def mock_web3():
    """Create a mock Web3 instance"""
    mock = MagicMock()
    mock.eth.get_code = AsyncMock(return_value=bytes([1, 2, 3]))
    mock.eth.get_logs = AsyncMock(return_value=[])
    mock.eth.get_block_number = AsyncMock(return_value=1000000)
    mock.to_checksum_address = Web3.to_checksum_address
    return mock

@pytest.fixture
def mock_cache():
    """Create a mock cache instance"""
    mock = AsyncMock()
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock()
    return mock

@pytest.fixture
def mock_metrics():
    """Create a mock metrics manager"""
    mock = MagicMock()
    mock.record_discovery = MagicMock()
    mock.record_validation = MagicMock()
    mock.record_api_error = MagicMock()
    mock.record_sentiment_score = MagicMock()
    return mock

@pytest.fixture
def mock_rate_limiters():
    """Create mock rate limiters"""
    mock = MagicMock()
    mock.get_limiter = MagicMock(return_value=AsyncMock())
    return mock

@pytest.fixture
def mock_sentiment():
    """Create a mock sentiment analyzer"""
    mock = AsyncMock()
    mock.get_token_sentiment = AsyncMock(
        return_value=SentimentScore(0.8, 0.9, {'telegram': 0.8})
    )
    return mock

@pytest.fixture
def token_discovery(
    mock_web3,
    mock_cache,
    mock_metrics,
    mock_rate_limiters,
    mock_sentiment
):
    """Create a TokenDiscovery instance with mocked dependencies"""
    config = {
        'redis_url': 'redis://localhost',
        'etherscan_api_key': 'test_key',
        'defillama_api': 'http://api.defillama.com',
        'dexscreener_api': 'http://api.dexscreener.com',
        'pinksale_api': 'http://api.pinksale.com',
        'dxsale_api': 'http://api.dxsale.com'
    }
    
    with patch('src.agent.token_discovery.get_web3', return_value=mock_web3), \
         patch('src.agent.token_discovery.get_async_web3', return_value=mock_web3), \
         patch('src.agent.token_discovery.AsyncCache', return_value=mock_cache), \
         patch('src.agent.token_discovery.MetricsManager', return_value=mock_metrics), \
         patch('src.agent.token_discovery.RateLimiterRegistry', return_value=mock_rate_limiters), \
         patch('src.agent.token_discovery.SentimentAnalyzer', return_value=mock_sentiment):
        
        discovery = TokenDiscovery(config)
        return discovery

@pytest.mark.asyncio
async def test_discover_new_tokens(token_discovery, mock_web3):
    """Test discovering new tokens"""
    # Mock some token events
    mock_web3.eth.get_logs.return_value = [
        {
            'address': '0x123...',
            'blockNumber': 1000000,
            'transactionHash': '0xabc...',
            'topics': ['0x...']
        }
    ]
    
    tokens = await token_discovery.discover_new_tokens()
    assert len(tokens) > 0
    assert isinstance(tokens[0], dict)
    assert 'address' in tokens[0]
    assert 'source' in tokens[0]

@pytest.mark.asyncio
async def test_validate_token(token_discovery, mock_cache, mock_metrics):
    """Test token validation"""
    token_address = '0x123...'
    
    # Test cache miss
    mock_cache.get.return_value = None
    result = await token_discovery.validate_token(token_address)
    assert isinstance(result, bool)
    mock_metrics.record_cache_miss.assert_called_once()
    
    # Test cache hit
    mock_cache.get.return_value = {'is_valid': True}
    result = await token_discovery.validate_token(token_address)
    assert result is True
    mock_metrics.record_cache_hit.assert_called_once()

@pytest.mark.asyncio
async def test_scan_dex_listings(token_discovery):
    """Test scanning DEX listings"""
    with patch('aiohttp.ClientSession') as mock_session:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            'tokens': [
                {'address': '0x123...', 'symbol': 'TEST'}
            ]
        })
        mock_session.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )
        
        tokens = await token_discovery._scan_dex_listings()
        assert len(tokens) > 0
        assert 'address' in tokens[0]
        assert 'source' in tokens[0]
        assert 'timestamp' in tokens[0]

@pytest.mark.asyncio
async def test_scan_token_events(token_discovery, mock_web3):
    """Test scanning token events"""
    # Mock token creation events
    mock_web3.eth.get_logs.return_value = [
        {
            'address': '0x123...',
            'blockNumber': 1000000,
            'topics': [
                '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef',
                '0x0000000000000000000000000000000000000000000000000000000000000000'
            ]
        }
    ]
    
    tokens = await token_discovery._scan_token_events()
    assert len(tokens) > 0
    assert isinstance(tokens[0], TokenData)
    assert tokens[0].source == 'event'

@pytest.mark.asyncio
async def test_scan_social_media(token_discovery):
    """Test scanning social media"""
    tokens = await token_discovery._scan_social_media()
    assert isinstance(tokens, list)

@pytest.mark.asyncio
async def test_get_token_sentiment(token_discovery, mock_sentiment):
    """Test getting token sentiment"""
    token_address = '0x123...'
    messages = {
        'telegram': ['Great project!'],
        'discord': ['Amazing team!']
    }
    
    result = await token_discovery.get_social_sentiment(token_address)
    assert isinstance(result, SentimentScore)
    assert -1.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0

@pytest.mark.asyncio
async def test_error_handling(token_discovery, mock_metrics):
    """Test error handling"""
    # Test with invalid token address
    result = await token_discovery.validate_token('invalid_address')
    assert result is False
    mock_metrics.record_api_error.assert_called()
    
    # Test with network error
    with patch('aiohttp.ClientSession') as mock_session:
        mock_session.return_value.__aenter__.return_value.get = AsyncMock(
            side_effect=Exception("Network error")
        )
        tokens = await token_discovery._scan_dex_listings()
        assert len(tokens) == 0
        mock_metrics.record_api_error.assert_called()

@pytest.mark.asyncio
async def test_rate_limiting(token_discovery, mock_rate_limiters):
    """Test rate limiting"""
    # Get a rate limiter
    limiter = mock_rate_limiters.get_limiter('test')
    assert limiter is not None
    
    # Test rate limited operation
    async with limiter:
        # Should not raise any errors
        pass

@pytest.mark.asyncio
async def test_metrics_recording(token_discovery, mock_metrics):
    """Test metrics recording"""
    # Record discovery
    await token_discovery.discover_new_tokens()
    mock_metrics.record_discovery.assert_called()
    
    # Record validation
    await token_discovery.validate_token('0x123...')
    mock_metrics.record_validation.assert_called()
    
    # Record sentiment
    await token_discovery.get_social_sentiment('0x123...')
    mock_metrics.record_sentiment_score.assert_called()

@pytest.mark.asyncio
async def test_cache_operations(token_discovery, mock_cache):
    """Test cache operations"""
    token_address = '0x123...'
    
    # Test cache miss
    mock_cache.get.return_value = None
    await token_discovery.validate_token(token_address)
    mock_cache.get.assert_called_once()
    
    # Test cache set
    mock_cache.set.assert_called_once()
    args = mock_cache.set.call_args[0]
    assert args[0] == f"validation:{token_address}"
    assert isinstance(args[1], dict)

@pytest.mark.asyncio
async def test_holder_distribution(token_discovery):
    """Test holder distribution analysis"""
    distribution = HolderDistribution.default()
    assert distribution.top_10_percent == 1.0
    assert distribution.top_50_percent == 1.0
    assert distribution.gini == 1.0

@pytest.mark.asyncio
async def test_dex_liquidity(token_discovery):
    """Test DEX liquidity data structure"""
    liquidity = DexLiquidity(
        liquidity=1000000.0,
        locked_liquidity=800000.0,
        pairs=2
    )
    assert liquidity.liquidity == 1000000.0
    assert liquidity.locked_liquidity == 800000.0
    assert liquidity.pairs == 2 