import pytest
import asyncio
from datetime import datetime, timedelta
from src.monitoring.performance_monitor import PerformanceMonitor
from prometheus_client.parser import text_string_to_metric_families

@pytest.fixture
async def monitor():
    """Create a test instance of PerformanceMonitor"""
    monitor = PerformanceMonitor(port=8001)  # Use different port for tests
    yield monitor
    await monitor.stop()

@pytest.mark.asyncio
async def test_record_transaction(monitor):
    """Test recording a transaction with metrics"""
    # Record a test transaction
    await monitor.record_transaction(
        tx_hash="0x123",
        status="success",
        value=1.5,
        gas_used=100000,
        execution_time=0.5,
        strategy="test_strategy"
    )
    
    # Get metrics
    metrics = monitor.get_metrics()
    
    # Verify transaction metrics
    assert metrics['transactions_total']['success'] == 1
    assert metrics['transaction_value'][-1] == 1.5
    assert metrics['gas_used'][-1] == 100000
    assert metrics['execution_time'][-1] == 0.5

@pytest.mark.asyncio
async def test_update_system_metrics(monitor):
    """Test updating system metrics"""
    # Update system metrics
    await monitor.update_system_metrics()
    
    # Get metrics
    metrics = monitor.get_metrics()
    
    # Verify system metrics exist and are within reasonable ranges
    assert 0 <= metrics['cpu_percent'] <= 100
    assert metrics['memory_usage'] > 0

@pytest.mark.asyncio
async def test_record_network_latency(monitor):
    """Test recording network latency"""
    test_latency = 0.1
    await monitor.record_network_latency("test_operation", test_latency)
    
    metrics = monitor.get_metrics()
    assert metrics['network_latency'][-1] == test_latency

@pytest.mark.asyncio
async def test_update_strategy_metrics(monitor):
    """Test updating strategy metrics"""
    await monitor.update_strategy_metrics(
        strategy="test_strategy",
        success_rate=0.85,
        total_profit=100.0
    )
    
    metrics = monitor.get_metrics()
    assert metrics['strategy_success_rate']['test_strategy'] == 0.85

@pytest.mark.asyncio
async def test_prometheus_metrics(monitor):
    """Test Prometheus metrics are properly registered"""
    # Get metrics as text
    metrics_text = ""
    for metric in monitor.get_prometheus_metrics():
        metrics_text += metric + "\n"
    
    # Parse metrics
    metrics_families = list(text_string_to_metric_families(metrics_text))
    
    # Verify essential metrics exist
    metric_names = [family.name for family in metrics_families]
    assert 'arbitrage_transactions_total' in metric_names
    assert 'arbitrage_gas_used' in metric_names
    assert 'system_cpu_usage_percent' in metric_names
    assert 'system_memory_usage_bytes' in metric_names

@pytest.mark.asyncio
async def test_metrics_persistence(monitor):
    """Test metrics are properly stored and retrieved"""
    # Record multiple transactions
    for i in range(3):
        await monitor.record_transaction(
            tx_hash=f"0x{i}",
            status="success",
            value=1.0 * i,
            gas_used=100000 * i,
            execution_time=0.1 * i,
            strategy="test_strategy"
        )
    
    metrics = monitor.get_metrics()
    
    # Verify metrics history
    assert len(metrics['transaction_value']) == 3
    assert len(metrics['gas_used']) == 3
    assert len(metrics['execution_time']) == 3

@pytest.mark.asyncio
async def test_error_handling(monitor):
    """Test error handling in metrics recording"""
    # Test with invalid values
    with pytest.raises(ValueError):
        await monitor.record_transaction(
            tx_hash="0x123",
            status="invalid_status",  # Invalid status
            value=-1.0,  # Invalid negative value
            gas_used=-100000,  # Invalid negative gas
            execution_time=-0.5,  # Invalid negative time
            strategy=""  # Invalid empty strategy
        )

@pytest.mark.asyncio
async def test_concurrent_updates(monitor):
    """Test concurrent metric updates"""
    async def update_metrics():
        for i in range(10):
            await monitor.record_transaction(
                tx_hash=f"0x{i}",
                status="success",
                value=1.0,
                gas_used=100000,
                execution_time=0.1,
                strategy="test_strategy"
            )
            await asyncio.sleep(0.01)
    
    # Run multiple concurrent updates
    tasks = [update_metrics() for _ in range(3)]
    await asyncio.gather(*tasks)
    
    metrics = monitor.get_metrics()
    assert metrics['transactions_total']['success'] == 30  # 3 tasks * 10 updates

@pytest.mark.asyncio
async def test_long_running_monitoring(monitor):
    """Test monitoring over a longer period"""
    # Start monitoring
    monitor_task = asyncio.create_task(monitor._monitoring_loop())
    
    # Let it run for a short while
    await asyncio.sleep(2)
    
    # Stop monitoring
    monitor._running = False
    await monitor_task
    
    # Verify metrics were collected
    metrics = monitor.get_metrics()
    assert len(metrics['cpu_percent']) > 0
    assert len(metrics['memory_usage']) > 0

@pytest.mark.benchmark
def test_metrics_performance(benchmark, monitor):
    """Benchmark metrics recording performance"""
    def record_metrics():
        asyncio.run(monitor.record_transaction(
            tx_hash="0x123",
            status="success",
            value=1.0,
            gas_used=100000,
            execution_time=0.1,
            strategy="test_strategy"
        ))
    
    # Run benchmark
    result = benchmark(record_metrics)
    assert result is None  # Verify function completed 