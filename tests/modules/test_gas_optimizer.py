"""Integration tests for the gas optimization module"""

import pytest
import asyncio
from web3 import Web3
from src.modules.gas_optimizer import GasOptimizer
from datetime import datetime, timedelta

@pytest.fixture
def web3():
    """Web3 fixture for testing"""
    return Web3(Web3.HTTPProvider('http://localhost:8545'))

@pytest.fixture
def gas_optimizer(web3):
    """Gas optimizer fixture"""
    return GasOptimizer(web3)

@pytest.fixture
def sample_transaction():
    """Sample transaction for testing"""
    return {
        'to': '0x742d35Cc6634C0532925a3b844Bc454e4438f44e',
        'value': 1000000000000000000,  # 1 ETH
        'from': '0x742d35Cc6634C0532925a3b844Bc454e4438f44e'
    }

@pytest.mark.asyncio
async def test_current_gas_price(gas_optimizer):
    """Test current gas price retrieval"""
    gas_price = await gas_optimizer.get_current_gas_price()
    assert isinstance(gas_price, int)
    assert gas_price > 0
    
    # Test caching
    cached_price = await gas_optimizer.get_current_gas_price()
    assert cached_price == gas_price

@pytest.mark.asyncio
async def test_gas_cost_estimation(gas_optimizer, sample_transaction):
    """Test gas cost estimation"""
    gas_estimate, gas_price = await gas_optimizer.estimate_gas_cost(sample_transaction)
    
    assert isinstance(gas_estimate, int)
    assert isinstance(gas_price, int)
    assert gas_estimate > 0
    assert gas_price > 0

@pytest.mark.asyncio
async def test_historical_gas_prices(gas_optimizer):
    """Test historical gas price analysis"""
    prices = await gas_optimizer.get_historical_gas_prices(hours=1)
    
    assert isinstance(prices, list)
    assert len(prices) > 0
    
    for timestamp, price in prices:
        assert isinstance(timestamp, datetime)
        assert isinstance(price, int)
        assert price >= 0

@pytest.mark.asyncio
async def test_gas_timing_optimization(gas_optimizer):
    """Test gas timing optimization"""
    optimal_hour = await gas_optimizer.optimize_gas_timing(
        target_gas_price=50000000000,  # 50 Gwei
        max_wait_time=3600
    )
    
    assert isinstance(optimal_hour, int) or optimal_hour is None
    if optimal_hour is not None:
        assert 0 <= optimal_hour < 24

@pytest.mark.asyncio
async def test_gas_limit_optimization(gas_optimizer, sample_transaction):
    """Test gas limit optimization"""
    optimized_limit = await gas_optimizer.optimize_gas_limit(sample_transaction)
    
    assert isinstance(optimized_limit, int)
    assert optimized_limit > 0
    
    # Should be higher than base estimate due to safety margin
    base_estimate = await gas_optimizer.web3.eth.estimate_gas(sample_transaction)
    assert optimized_limit >= base_estimate

@pytest.mark.asyncio
async def test_transaction_replacement(gas_optimizer):
    """Test transaction replacement analysis"""
    old_gas_price = 50000000000  # 50 Gwei
    current_gas_price = await gas_optimizer.get_current_gas_price()
    
    # Test with non-existent transaction (should return False)
    should_replace = await gas_optimizer.should_replace_transaction(
        old_gas_price,
        "0x0000000000000000000000000000000000000000000000000000000000000000"
    )
    assert isinstance(should_replace, bool)

@pytest.mark.asyncio
async def test_metrics_collection(gas_optimizer, sample_transaction):
    """Test metrics collection"""
    # Perform some gas optimizations
    await gas_optimizer.optimize_gas_limit(sample_transaction)
    
    # Check gas price gauge
    for metric in gas_optimizer._gas_price_gauge.collect():
        for sample in metric.samples:
            assert sample.value > 0
    
    # Check optimization time histogram
    for metric in gas_optimizer._optimization_time.collect():
        for sample in metric.samples:
            assert sample.value > 0

@pytest.mark.asyncio
async def test_concurrent_optimization(gas_optimizer, sample_transaction):
    """Test concurrent gas optimization"""
    async def optimize():
        return await gas_optimizer.optimize_gas_limit(sample_transaction)
    
    # Run multiple optimizations concurrently
    results = await asyncio.gather(*[optimize() for _ in range(5)])
    
    assert len(results) == 5
    assert all(isinstance(limit, int) and limit > 0 for limit in results)

@pytest.mark.asyncio
async def test_error_handling(gas_optimizer):
    """Test error handling in gas optimization"""
    # Test with invalid transaction
    invalid_tx = {'to': '0x123'}  # Invalid transaction
    
    with pytest.raises(Exception):
        await gas_optimizer.estimate_gas_cost(invalid_tx)
    
    # Test with invalid network
    with pytest.raises(Exception):
        await gas_optimizer.get_current_gas_price("invalid_network") 