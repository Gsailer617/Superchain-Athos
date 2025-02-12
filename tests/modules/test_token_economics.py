"""Integration tests for the token economics module"""

import pytest
import asyncio
from web3 import Web3
from src.modules.token_economics import token_economics, TokenMetrics

@pytest.fixture
def web3():
    """Web3 fixture for testing"""
    return Web3(Web3.HTTPProvider('http://localhost:8545'))

@pytest.fixture
def token_address():
    """Sample token address for testing"""
    return "0x6B175474E89094C44Da98b954EedeAC495271d0F"  # DAI

@pytest.mark.asyncio
async def test_token_metrics_retrieval(token_address):
    """Test basic token metrics retrieval"""
    metrics = await token_economics.get_token_metrics(token_address)
    
    assert isinstance(metrics, TokenMetrics)
    assert metrics.market_cap >= 0
    assert metrics.total_supply > 0
    assert metrics.circulating_supply > 0
    assert metrics.holders_count > 0
    assert metrics.liquidity_usd > 0
    assert metrics.volume_24h >= 0
    assert metrics.price_usd > 0

@pytest.mark.asyncio
async def test_token_validation(token_address):
    """Test token validation"""
    is_valid, reason = await token_economics.validate_token(token_address)
    
    assert isinstance(is_valid, bool)
    assert isinstance(reason, str)
    
    # DAI should pass validation
    assert is_valid is True
    
    # Test invalid token
    invalid_result, invalid_reason = await token_economics.validate_token(
        "0x0000000000000000000000000000000000000000"
    )
    assert invalid_result is False
    assert len(invalid_reason) > 0

@pytest.mark.asyncio
async def test_holder_distribution(token_address):
    """Test holder distribution analysis"""
    distribution = await token_economics.analyze_holder_distribution(token_address)
    
    assert isinstance(distribution, dict)
    assert len(distribution) > 0
    
    # Verify percentages
    total_percentage = sum(distribution.values())
    assert 0 <= total_percentage <= 100
    
    # Top holders should have higher percentages
    sorted_percentages = sorted(distribution.values(), reverse=True)
    assert sorted_percentages == list(distribution.values())

@pytest.mark.asyncio
async def test_price_impact(token_address):
    """Test price impact estimation"""
    # Test with small amount
    small_impact = await token_economics.estimate_price_impact(
        token_address,
        amount_usd=10000
    )
    
    # Test with large amount
    large_impact = await token_economics.estimate_price_impact(
        token_address,
        amount_usd=10000000
    )
    
    assert 0 <= small_impact <= 100
    assert 0 <= large_impact <= 100
    assert large_impact > small_impact

@pytest.mark.asyncio
async def test_metrics_collection(token_address):
    """Test metrics collection"""
    # Get metrics to trigger collection
    await token_economics.get_token_metrics(token_address)
    
    # Check token metrics gauge
    for metric in token_economics._token_metrics.collect():
        for sample in metric.samples:
            if sample.labels['token_address'] == token_address:
                assert sample.value >= 0
    
    # Check analysis time histogram
    for metric in token_economics._token_analysis_time.collect():
        for sample in metric.samples:
            assert sample.value > 0

@pytest.mark.asyncio
async def test_concurrent_analysis(token_address):
    """Test concurrent token analysis"""
    # Test multiple concurrent analyses
    async def analyze():
        return await token_economics.get_token_metrics(token_address)
    
    results = await asyncio.gather(*[analyze() for _ in range(3)])
    
    assert len(results) == 3
    assert all(isinstance(r, TokenMetrics) for r in results)
    # All results should be the same due to caching
    assert all(r == results[0] for r in results)

@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling"""
    # Test with invalid token address
    with pytest.raises(Exception):
        await token_economics.get_token_metrics("invalid_address")
    
    # Test with invalid chain ID
    with pytest.raises(Exception):
        await token_economics.get_token_metrics(
            "0x0000000000000000000000000000000000000000",
            chain_id=999999
        )

@pytest.mark.asyncio
async def test_caching(token_address):
    """Test caching behavior"""
    # First call should hit the API
    start_time = asyncio.get_event_loop().time()
    first_result = await token_economics.get_token_metrics(token_address)
    first_duration = asyncio.get_event_loop().time() - start_time
    
    # Second call should be cached and faster
    start_time = asyncio.get_event_loop().time()
    second_result = await token_economics.get_token_metrics(token_address)
    second_duration = asyncio.get_event_loop().time() - start_time
    
    assert second_duration < first_duration
    assert first_result == second_result

@pytest.mark.asyncio
async def test_rate_limiting(token_address):
    """Test rate limiting behavior"""
    # Make multiple rapid requests
    async def make_request():
        return await token_economics.get_token_metrics(token_address)
    
    start_time = asyncio.get_event_loop().time()
    results = await asyncio.gather(*[make_request() for _ in range(5)])
    duration = asyncio.get_event_loop().time() - start_time
    
    # Should take some time due to rate limiting
    assert duration >= 0.5  # At least 500ms
    assert all(isinstance(r, TokenMetrics) for r in results) 