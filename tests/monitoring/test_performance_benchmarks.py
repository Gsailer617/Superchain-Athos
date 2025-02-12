"""Performance benchmarks for monitoring system"""

import pytest
import asyncio
import time
import numpy as np
from typing import List, Dict, Any
from src.monitoring.monitor_manager import MonitorManager
from src.monitoring.performance_monitor import PerformanceMonitor
from tests.utils.test_utils import (
    create_mock_metrics,
    create_mock_trade_history,
    create_mock_learning_insights
)

@pytest.mark.benchmark
async def test_metrics_recording_performance(benchmark, monitor_manager):
    """Benchmark metrics recording performance"""
    def record_metrics():
        monitor_manager.record_trade(
            strategy="test_strategy",
            token_pair="ETH-USDC",
            dex="uniswap",
            profit=1.0,
            gas_price=50.0,
            execution_time=0.1,
            success=True
        )
    
    # Run benchmark
    result = benchmark(record_metrics)
    assert result is None

@pytest.mark.benchmark
async def test_insights_retrieval_performance(benchmark, monitor_manager):
    """Benchmark learning insights retrieval performance"""
    # Pre-populate with test data
    history = create_mock_trade_history(num_trades=1000)
    for _, trade in history.iterrows():
        monitor_manager.record_trade(
            strategy=trade['strategy'],
            token_pair=trade['token_pair'],
            dex=trade['dex'],
            profit=float(trade['profit']),
            gas_price=float(trade['gas_price']),
            execution_time=float(trade['execution_time']),
            success=bool(trade['success'])
        )
    
    async def get_insights():
        return await monitor_manager.get_learning_insights()
    
    # Run benchmark
    result = benchmark(asyncio.run, get_insights())
    assert result is not None

@pytest.mark.benchmark
async def test_concurrent_operations_performance(benchmark, monitor_manager):
    """Benchmark performance under concurrent operations"""
    async def concurrent_workload():
        tasks = []
        # Record trade
        tasks.append(asyncio.create_task(
            monitor_manager.record_trade(
                strategy="test_strategy",
                token_pair="ETH-USDC",
                dex="uniswap",
                profit=1.0,
                gas_price=50.0,
                execution_time=0.1,
                success=True
            )
        ))
        # Get insights
        tasks.append(asyncio.create_task(
            monitor_manager.get_learning_insights()
        ))
        # Get metrics
        tasks.append(asyncio.create_task(
            monitor_manager.get_system_metrics()
        ))
        await asyncio.gather(*tasks)
    
    # Run benchmark
    result = benchmark(asyncio.run, concurrent_workload())
    assert result is None

@pytest.mark.benchmark
async def test_anomaly_detection_performance(benchmark, monitor_manager):
    """Benchmark anomaly detection performance"""
    # Pre-populate with test data including anomalies
    history = create_mock_trade_history(num_trades=1000)
    # Inject anomalies
    anomaly_indices = np.random.choice(len(history), size=50, replace=False)
    for idx in anomaly_indices:
        history.iloc[idx, history.columns.get_loc('profit')] = -1000.0  # Anomalous profit
    
    for _, trade in history.iterrows():
        monitor_manager.record_trade(
            strategy=trade['strategy'],
            token_pair=trade['token_pair'],
            dex=trade['dex'],
            profit=float(trade['profit']),
            gas_price=float(trade['gas_price']),
            execution_time=float(trade['execution_time']),
            success=bool(trade['success'])
        )
    
    async def detect_anomalies():
        insights = await monitor_manager.get_learning_insights()
        return insights.get('anomaly_scores', [])
    
    # Run benchmark
    result = benchmark(asyncio.run, detect_anomalies())
    assert len(result) > 0

@pytest.mark.benchmark
async def test_optimization_suggestions_performance(benchmark, monitor_manager):
    """Benchmark optimization suggestions generation performance"""
    # Pre-populate with poor performance data
    history = create_mock_trade_history(num_trades=1000)
    history['profit'] = -100.0  # Force poor performance
    
    for _, trade in history.iterrows():
        monitor_manager.record_trade(
            strategy=trade['strategy'],
            token_pair=trade['token_pair'],
            dex=trade['dex'],
            profit=float(trade['profit']),
            gas_price=float(trade['gas_price']),
            execution_time=float(trade['execution_time']),
            success=bool(trade['success'])
        )
    
    async def generate_suggestions():
        insights = await monitor_manager.get_learning_insights()
        return insights.get('optimization_suggestions', [])
    
    # Run benchmark
    result = benchmark(asyncio.run, generate_suggestions())
    assert len(result) > 0

@pytest.mark.benchmark
async def test_memory_usage_performance(benchmark, monitor_manager):
    """Benchmark memory usage under load"""
    def memory_test():
        # Generate large history
        history = create_mock_trade_history(num_trades=10000)
        
        # Record all trades
        for _, trade in history.iterrows():
            monitor_manager.record_trade(
                strategy=trade['strategy'],
                token_pair=trade['token_pair'],
                dex=trade['dex'],
                profit=float(trade['profit']),
                gas_price=float(trade['gas_price']),
                execution_time=float(trade['execution_time']),
                success=bool(trade['success'])
            )
        
        # Get metrics to check memory
        return monitor_manager.get_system_metrics()['memory_mb']
    
    # Run benchmark
    result = benchmark(memory_test)
    assert result > 0

@pytest.mark.benchmark
async def test_prometheus_metrics_performance(benchmark, monitor_manager):
    """Benchmark Prometheus metrics collection performance"""
    def collect_metrics():
        return monitor_manager.performance_monitor.get_prometheus_metrics()
    
    # Run benchmark
    result = benchmark(collect_metrics)
    assert result is not None

@pytest.mark.benchmark
async def test_redis_cache_performance(benchmark, monitor_manager):
    """Benchmark Redis cache operations performance"""
    if not monitor_manager.cache:
        pytest.skip("Redis cache not enabled")
    
    async def cache_operation():
        # Write to cache
        await monitor_manager.cache.set('test_key', 'test_value')
        # Read from cache
        value = await monitor_manager.cache.get('test_key')
        return value
    
    # Run benchmark
    result = benchmark(asyncio.run, cache_operation())
    assert result == 'test_value'

@pytest.mark.benchmark
async def test_system_metrics_performance(benchmark, monitor_manager):
    """Benchmark system metrics collection performance"""
    def collect_system_metrics():
        return monitor_manager.get_system_metrics()
    
    # Run benchmark
    result = benchmark(collect_system_metrics)
    assert result is not None

@pytest.mark.benchmark
async def test_trade_history_query_performance(benchmark, monitor_manager):
    """Benchmark trade history query performance"""
    # Pre-populate with test data
    history = create_mock_trade_history(num_trades=10000)
    for _, trade in history.iterrows():
        monitor_manager.record_trade(
            strategy=trade['strategy'],
            token_pair=trade['token_pair'],
            dex=trade['dex'],
            profit=float(trade['profit']),
            gas_price=float(trade['gas_price']),
            execution_time=float(trade['execution_time']),
            success=bool(trade['success'])
        )
    
    def query_history():
        return monitor_manager.trade_history.get_history(
            start_time=time.time() - 3600,  # Last hour
            strategy="test_strategy"
        )
    
    # Run benchmark
    result = benchmark(query_history)
    assert len(result) > 0 