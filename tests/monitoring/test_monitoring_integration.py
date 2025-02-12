import pytest
import asyncio
from src.monitoring.performance_monitor import PerformanceMonitor
from prometheus_client.parser import text_string_to_metric_families
import aiohttp
import json
from unittest.mock import patch, MagicMock

@pytest.fixture
async def performance_monitor():
    """Fixture providing a configured performance monitor instance"""
    monitor = PerformanceMonitor(port=8001)  # Use different port for testing
    yield monitor
    await monitor.stop()

@pytest.mark.asyncio
async def test_prometheus_metrics_exposure(performance_monitor):
    """Test that metrics are properly exposed via Prometheus endpoint"""
    async with aiohttp.ClientSession() as session:
        async with session.get('http://localhost:8001/metrics') as response:
            assert response.status == 200
            metrics_text = await response.text()
            
            # Parse metrics and verify core metrics exist
            metrics = list(text_string_to_metric_families(metrics_text))
            metric_names = {m.name for m in metrics}
            
            expected_metrics = {
                'arbitrage_transactions_total',
                'system_cpu_usage_percent',
                'system_memory_usage_bytes',
                'strategy_success_rate',
                'network_latency_seconds',
                'error_count_total'
            }
            
            assert expected_metrics.issubset(metric_names)

@pytest.mark.asyncio
async def test_error_recording(performance_monitor):
    """Test error metric recording"""
    error_type = "validation_error"
    error_message = "Invalid transaction parameters"
    
    await performance_monitor.record_error(error_type, error_message)
    
    async with aiohttp.ClientSession() as session:
        async with session.get('http://localhost:8001/metrics') as response:
            metrics_text = await response.text()
            assert 'error_count_total{error_type="validation_error"} 1.0' in metrics_text
            assert 'error_details{' in metrics_text
            assert 'last_error_timestamp' in metrics_text

@pytest.mark.asyncio
async def test_network_health_monitoring(performance_monitor):
    """Test network health monitoring"""
    node = "mainnet-1"
    await performance_monitor.update_network_health(
        node=node,
        is_healthy=True,
        latency=0.15
    )
    
    async with aiohttp.ClientSession() as session:
        async with session.get('http://localhost:8001/metrics') as response:
            metrics_text = await response.text()
            assert 'rpc_node_health{node="mainnet-1"} 1.0' in metrics_text
            assert 'network_latency_seconds_bucket' in metrics_text

@pytest.mark.asyncio
async def test_gas_optimization_tracking(performance_monitor):
    """Test gas optimization metrics"""
    await performance_monitor.record_gas_savings(
        optimization_type="multicall",
        gas_saved=50000
    )
    
    async with aiohttp.ClientSession() as session:
        async with session.get('http://localhost:8001/metrics') as response:
            metrics_text = await response.text()
            assert 'gas_savings_total{optimization_type="multicall"} 50000' in metrics_text

@pytest.mark.asyncio
async def test_system_metrics_collection(performance_monitor):
    """Test system metrics collection"""
    # Wait for monitoring loop to collect some data
    await asyncio.sleep(6)
    
    async with aiohttp.ClientSession() as session:
        async with session.get('http://localhost:8001/metrics') as response:
            metrics_text = await response.text()
            assert 'system_cpu_usage_percent' in metrics_text
            assert 'system_memory_usage_bytes' in metrics_text

@pytest.mark.asyncio
async def test_strategy_performance_tracking(performance_monitor):
    """Test strategy performance metrics"""
    strategy = "flash_loan_v2"
    
    # Record a successful transaction
    await performance_monitor.record_transaction(
        tx_hash="0x123",
        status="success",
        value=1.5,
        gas_used=100000,
        execution_time=0.5,
        strategy=strategy
    )
    
    # Update strategy success rate
    performance_monitor.strategy_success_rate.labels(strategy=strategy).set(0.95)
    
    async with aiohttp.ClientSession() as session:
        async with session.get('http://localhost:8001/metrics') as response:
            metrics_text = await response.text()
            assert f'strategy_success_rate{{strategy="{strategy}"}} 0.95' in metrics_text
            assert f'strategy_profit_usd{{strategy="{strategy}"}}' in metrics_text

@pytest.mark.asyncio
async def test_monitoring_shutdown(performance_monitor):
    """Test clean shutdown of monitoring"""
    assert performance_monitor._running
    assert performance_monitor._monitor_thread.is_alive()
    
    await performance_monitor.stop()
    
    assert not performance_monitor._running
    assert not performance_monitor._monitor_thread.is_alive() 