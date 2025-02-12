import pytest
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from src.monitoring.monitor_manager import MonitorManager
from src.monitoring.performance_monitor import PerformanceMonitor
from src.history.trade_history import TradeHistoryManager

@pytest.fixture
async def config():
    """Test configuration"""
    return {
        'monitoring': {
            'storage_path': 'tests/data/monitoring',
            'prometheus_port': 8002,
            'cache_enabled': True,
            'redis_host': 'localhost',
            'redis_port': 6379,
            'redis_db': 1,
            'max_memory_entries': 1000,
            'flush_interval': 10
        },
        'jaeger': {
            'host': 'localhost',
            'port': 6831
        }
    }

@pytest.fixture
async def monitor_manager(config):
    """Create test instance of MonitorManager"""
    manager = MonitorManager(config)
    await manager.start()
    yield manager
    await manager.stop()

@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_monitoring(monitor_manager):
    """Test end-to-end monitoring flow"""
    # Record a trade
    monitor_manager.record_trade(
        strategy="test_strategy",
        token_pair="ETH-USDC",
        dex="uniswap",
        profit=1.5,
        gas_price=50.0,
        execution_time=0.5,
        success=True
    )
    
    # Wait for metrics to be processed
    await asyncio.sleep(1)
    
    # Get learning insights
    insights = await monitor_manager.get_learning_insights()
    
    # Verify insights contain expected data
    assert 'strategy_performance' in insights
    assert 'test_strategy' in insights['strategy_performance']
    performance = insights['strategy_performance']['test_strategy'][-1]
    assert performance['profit'] == 1.5
    assert performance['execution_time'] == 0.5

@pytest.mark.integration
@pytest.mark.asyncio
async def test_prometheus_integration(monitor_manager):
    """Test Prometheus metrics integration"""
    # Record some test data
    for i in range(5):
        monitor_manager.record_trade(
            strategy="test_strategy",
            token_pair="ETH-USDC",
            dex="uniswap",
            profit=1.0 * i,
            gas_price=50.0,
            execution_time=0.1 * i,
            success=True
        )
    
    # Wait for metrics to be exported
    await asyncio.sleep(1)
    
    # Query Prometheus metrics endpoint
    async with aiohttp.ClientSession() as session:
        async with session.get(f'http://localhost:{monitor_manager.config["monitoring"]["prometheus_port"]}/metrics') as response:
            assert response.status == 200
            metrics_text = await response.text()
            
            # Verify essential metrics are present
            assert 'arbitrage_transactions_total' in metrics_text
            assert 'strategy_profit_eth' in metrics_text
            assert 'system_cpu_usage_percent' in metrics_text

@pytest.mark.integration
@pytest.mark.asyncio
async def test_anomaly_detection(monitor_manager):
    """Test anomaly detection system"""
    # Record normal trades
    for i in range(10):
        monitor_manager.record_trade(
            strategy="test_strategy",
            token_pair="ETH-USDC",
            dex="uniswap",
            profit=1.0,
            gas_price=50.0,
            execution_time=0.1,
            success=True
        )
    
    # Record anomalous trade
    monitor_manager.record_trade(
        strategy="test_strategy",
        token_pair="ETH-USDC",
        dex="uniswap",
        profit=-10.0,  # Anomalous negative profit
        gas_price=500.0,  # Anomalous high gas
        execution_time=5.0,  # Anomalous slow execution
        success=False
    )
    
    # Wait for anomaly detection
    await asyncio.sleep(1)
    
    # Get insights
    insights = await monitor_manager.get_learning_insights()
    
    # Verify anomaly was detected
    assert len(insights['anomaly_scores']) > 0
    assert -1 in insights['anomaly_scores']  # -1 indicates anomaly

@pytest.mark.integration
@pytest.mark.asyncio
async def test_optimization_suggestions(monitor_manager):
    """Test optimization suggestions generation"""
    # Record trades with poor performance
    for i in range(5):
        monitor_manager.record_trade(
            strategy="test_strategy",
            token_pair="ETH-USDC",
            dex="uniswap",
            profit=-0.1,
            gas_price=100.0,
            execution_time=2.0,
            success=False
        )
    
    # Wait for analysis
    await asyncio.sleep(1)
    
    # Get insights
    insights = await monitor_manager.get_learning_insights()
    
    # Verify optimization suggestions were generated
    assert len(insights['optimization_suggestions']) > 0
    suggestions = insights['optimization_suggestions']
    
    # Check for specific suggestion types
    suggestion_texts = ' '.join(suggestions)
    assert any('gas' in text.lower() for text in suggestions)  # Gas optimization
    assert any('execution' in text.lower() for text in suggestions)  # Execution optimization
    assert any('strategy' in text.lower() for text in suggestions)  # Strategy optimization

@pytest.mark.integration
@pytest.mark.asyncio
async def test_trade_history_integration(monitor_manager):
    """Test trade history integration"""
    # Record trades
    trades = []
    for i in range(5):
        trade = {
            'strategy': "test_strategy",
            'token_pair': "ETH-USDC",
            'dex': "uniswap",
            'profit': 1.0 * i,
            'gas_price': 50.0,
            'execution_time': 0.1 * i,
            'success': True
        }
        monitor_manager.record_trade(**trade)
        trades.append(trade)
    
    # Wait for processing
    await asyncio.sleep(1)
    
    # Verify trades were recorded in history
    history = monitor_manager.trade_history.get_history()
    assert len(history) >= len(trades)
    
    # Verify trade details were preserved
    for trade in trades:
        matching_trades = history[
            (history['strategy'] == trade['strategy']) &
            (history['token_pair'] == trade['token_pair']) &
            (history['profit'] == trade['profit'])
        ]
        assert len(matching_trades) > 0

@pytest.mark.integration
@pytest.mark.asyncio
async def test_system_metrics_collection(monitor_manager):
    """Test system metrics collection"""
    # Start monitoring
    await asyncio.sleep(2)  # Wait for metrics collection
    
    # Get system metrics
    metrics = monitor_manager.get_system_metrics()
    
    # Verify system metrics
    assert 'cpu_percent' in metrics
    assert 'memory_percent' in metrics
    assert 'memory_mb' in metrics
    assert 'threads' in metrics
    assert metrics['cpu_percent'] >= 0
    assert metrics['memory_mb'] > 0

@pytest.mark.integration
@pytest.mark.benchmark
async def test_monitoring_performance(benchmark, monitor_manager):
    """Benchmark monitoring system performance"""
    def monitor_operation():
        asyncio.run(async_monitor_operation(monitor_manager))
    
    async def async_monitor_operation(manager):
        # Record trade
        manager.record_trade(
            strategy="test_strategy",
            token_pair="ETH-USDC",
            dex="uniswap",
            profit=1.0,
            gas_price=50.0,
            execution_time=0.1,
            success=True
        )
        
        # Get insights
        await manager.get_learning_insights()
        
        # Get system metrics
        manager.get_system_metrics()
    
    # Run benchmark
    result = benchmark(monitor_operation)
    assert result is None  # Verify completion 