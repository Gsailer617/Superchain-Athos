import pytest
from src.monitoring.performance_monitor import PerformanceMonitor
from src.core.web3_config import Web3Config
from src.market.strategies import ArbitrageStrategy
import asyncio
import time

@pytest.fixture
def performance_monitor():
    monitor = PerformanceMonitor(port=8001)  # Use different port for tests
    yield monitor
    monitor.stop()

@pytest.fixture
async def web3_config():
    config = Web3Config()
    await config.initialize()
    return config

@pytest.mark.benchmark
async def test_price_calculation_performance(benchmark, web3_config):
    """Benchmark price calculation performance"""
    strategy = ArbitrageStrategy(web3_config)
    
    def run_price_calc():
        return asyncio.run(strategy.calculate_price_difference())
    
    result = benchmark(run_price_calc)
    assert result is not None

@pytest.mark.benchmark
def test_transaction_monitoring(benchmark, performance_monitor):
    """Benchmark transaction monitoring overhead"""
    def record_transaction():
        performance_monitor.record_transaction(
            success=True,
            gas_price=50.0,
            execution_time=0.5,
            profit=100.0
        )
    
    benchmark(record_transaction)

@pytest.mark.benchmark
async def test_gas_optimization_performance(benchmark, web3_config):
    """Benchmark gas optimization calculations"""
    from src.gas.optimizer import GasOptimizer
    
    optimizer = GasOptimizer(web3_config)
    
    def optimize_gas():
        return asyncio.run(optimizer.calculate_optimal_gas_price())
    
    result = benchmark(optimize_gas)
    assert result > 0

@pytest.mark.asyncio
async def test_end_to_end_arbitrage_performance(performance_monitor, web3_config):
    """Test end-to-end arbitrage performance"""
    from src.agent.token_discovery import TokenDiscovery
    from src.execution.transaction_builder import TransactionBuilder
    
    start_time = time.time()
    
    # Initialize components
    token_discovery = TokenDiscovery()
    strategy = ArbitrageStrategy(web3_config)
    tx_builder = TransactionBuilder(web3_config)
    
    # Measure token discovery time
    tokens = await token_discovery.discover_arbitrage_opportunities()
    assert len(tokens) > 0
    
    # Measure price calculation time
    price_diff = await strategy.calculate_price_difference()
    assert price_diff is not None
    
    # Measure transaction building time
    tx = await tx_builder.build_arbitrage_transaction(tokens[0])
    assert tx is not None
    
    execution_time = time.time() - start_time
    
    # Record metrics
    performance_monitor.record_transaction(
        success=True,
        gas_price=50.0,
        execution_time=execution_time
    )
    
    # Assert reasonable execution time
    assert execution_time < 5.0  # Should complete within 5 seconds 