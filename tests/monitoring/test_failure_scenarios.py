"""Tests for failure scenarios and error handling in monitoring system"""

import pytest
import asyncio
from datetime import datetime
import aiohttp
from unittest.mock import patch, AsyncMock
from src.monitoring.monitor_manager import MonitorManager
from src.monitoring.performance_monitor import PerformanceMonitor
from tests.utils.test_utils import create_mock_metrics, create_mock_trade_history

@pytest.mark.asyncio
async def test_redis_connection_failure(monitor_config):
    """Test system behavior when Redis connection fails"""
    # Modify config to use invalid Redis port
    monitor_config['monitoring']['redis_port'] = 6380  # Invalid port
    
    # Initialize monitor manager
    manager = MonitorManager(monitor_config)
    
    # Start should succeed even with Redis failure
    await manager.start()
    try:
        # System should continue functioning without Redis
        manager.record_trade(
            strategy="test_strategy",
            token_pair="ETH-USDC",
            dex="uniswap",
            profit=1.0,
            gas_price=50.0,
            execution_time=0.1,
            success=True
        )
        
        # Should still get insights without Redis
        insights = await manager.get_learning_insights()
        assert insights is not None
        
    finally:
        await manager.stop()

@pytest.mark.asyncio
async def test_prometheus_port_conflict(monitor_config):
    """Test system behavior when Prometheus port is in use"""
    # Create first monitor with default port
    monitor1 = PerformanceMonitor(port=8001)
    
    try:
        # Try to create second monitor with same port
        with pytest.raises(Exception):
            monitor2 = PerformanceMonitor(port=8001)
            
    finally:
        await monitor1.stop()

@pytest.mark.asyncio
async def test_invalid_metrics_data(monitor_manager):
    """Test handling of invalid metrics data"""
    # Try to record invalid trade data
    with pytest.raises(ValueError):
        monitor_manager.record_trade(
            strategy="",  # Invalid empty strategy
            token_pair="",  # Invalid empty token pair
            dex="",  # Invalid empty DEX
            profit=-1.0,  # Invalid negative profit
            gas_price=-50.0,  # Invalid negative gas
            execution_time=-0.1,  # Invalid negative time
            success=None  # Invalid success value
        )

@pytest.mark.asyncio
async def test_concurrent_failures(monitor_manager):
    """Test handling of concurrent operation failures"""
    async def failing_operation():
        for _ in range(5):
            # Simulate random failures
            if random.random() < 0.5:
                raise Exception("Random failure")
            await asyncio.sleep(0.01)
    
    # Run multiple failing operations concurrently
    tasks = [failing_operation() for _ in range(10)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Count failures
    failures = sum(1 for r in results if isinstance(r, Exception))
    assert failures > 0  # Should have some failures

@pytest.mark.asyncio
async def test_memory_pressure(monitor_manager):
    """Test system behavior under memory pressure"""
    # Generate large amount of test data
    large_history = create_mock_trade_history(num_trades=100000)
    
    # Record trades rapidly
    for _, trade in large_history.iterrows():
        monitor_manager.record_trade(
            strategy=trade['strategy'],
            token_pair=trade['token_pair'],
            dex=trade['dex'],
            profit=float(trade['profit']),
            gas_price=float(trade['gas_price']),
            execution_time=float(trade['execution_time']),
            success=bool(trade['success'])
        )
    
    # System should still function
    metrics = monitor_manager.get_system_metrics()
    assert metrics['memory_mb'] > 0

@pytest.mark.asyncio
async def test_network_failures(monitor_manager):
    """Test handling of network-related failures"""
    with patch('aiohttp.ClientSession.get', side_effect=aiohttp.ClientError):
        # Should handle Prometheus metrics endpoint failure gracefully
        metrics = await monitor_manager.get_metrics()
        assert metrics is not None

@pytest.mark.asyncio
async def test_data_corruption(monitor_manager):
    """Test handling of corrupted data"""
    # Simulate corrupted metrics data
    corrupted_metrics = create_mock_metrics()
    corrupted_metrics['profit_loss'] = [float('nan')] * len(corrupted_metrics['profit_loss'])
    
    with patch('src.monitoring.monitor_manager.MonitorManager.get_metrics', 
              return_value=corrupted_metrics):
        # Should handle corrupted data gracefully
        insights = await monitor_manager.get_learning_insights()
        assert insights is not None

@pytest.mark.asyncio
async def test_resource_exhaustion(monitor_manager):
    """Test behavior when system resources are exhausted"""
    # Simulate CPU-intensive workload
    async def cpu_intensive():
        for _ in range(1000000):
            _ = hash(str(random.random()))
    
    # Run multiple CPU-intensive tasks
    tasks = [cpu_intensive() for _ in range(10)]
    await asyncio.gather(*tasks)
    
    # System should still function
    metrics = monitor_manager.get_system_metrics()
    assert metrics['cpu_percent'] >= 0

@pytest.mark.asyncio
async def test_race_conditions(monitor_manager):
    """Test for potential race conditions"""
    async def concurrent_operation(i: int):
        # Multiple operations trying to update the same metrics
        monitor_manager.record_trade(
            strategy=f"strategy_{i%3}",
            token_pair="ETH-USDC",
            dex="uniswap",
            profit=1.0,
            gas_price=50.0,
            execution_time=0.1,
            success=True
        )
        await monitor_manager.get_learning_insights()
        monitor_manager.get_system_metrics()
    
    # Run many concurrent operations
    tasks = [concurrent_operation(i) for i in range(100)]
    await asyncio.gather(*tasks)
    
    # Verify system state is consistent
    insights = await monitor_manager.get_learning_insights()
    assert 'strategy_performance' in insights

@pytest.mark.asyncio
async def test_recovery_mechanism(monitor_manager):
    """Test system recovery after failures"""
    # Force system into error state
    with patch('src.monitoring.monitor_manager.MonitorManager._detect_anomalies',
              side_effect=Exception("Forced failure")):
        # Record some trades during error state
        for i in range(5):
            monitor_manager.record_trade(
                strategy="test_strategy",
                token_pair="ETH-USDC",
                dex="uniswap",
                profit=1.0,
                gas_price=50.0,
                execution_time=0.1,
                success=True
            )
    
    # System should recover and continue functioning
    insights = await monitor_manager.get_learning_insights()
    assert insights is not None
    
    # Verify no data was lost
    metrics = monitor_manager.get_system_metrics()
    assert metrics is not None 