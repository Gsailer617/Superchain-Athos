import pytest
import asyncio
import time
from typing import List, Dict, Any
from src.monitoring.performance_monitor import PerformanceMonitor
from src.core.web3_config import Web3Config
from src.market.strategies import ArbitrageStrategy
from src.gas.optimizer import GasOptimizer
from src.agent.token_discovery import TokenDiscovery
from src.execution.transaction_builder import TransactionBuilder

@pytest.fixture
def performance_monitor():
    monitor = PerformanceMonitor(port=8002)  # Different port for strategy tests
    yield monitor
    monitor.stop()

@pytest.fixture
async def web3_config():
    config = Web3Config()
    await config.initialize()
    return config

@pytest.fixture
def token_pairs() -> List[Dict[str, str]]:
    return [
        {"token0": "WETH", "token1": "USDC"},
        {"token0": "WETH", "token1": "USDT"},
        {"token0": "WBTC", "token1": "USDC"},
        {"token0": "WETH", "token1": "DAI"}
    ]

class TestStrategyBenchmarks:
    @pytest.mark.benchmark
    @pytest.mark.parametrize("dex_pair", [
        ("Uniswap V3", "SushiSwap"),
        ("Uniswap V3", "Curve"),
        ("Balancer", "SushiSwap"),
        ("Curve", "Balancer")
    ])
    async def test_cross_dex_arbitrage_performance(self, benchmark, web3_config, 
                                                 performance_monitor, dex_pair, token_pairs):
        """Benchmark cross-DEX arbitrage strategy performance"""
        strategy = ArbitrageStrategy(web3_config)
        
        async def run_cross_dex_strategy():
            results = []
            for pair in token_pairs:
                start_time = time.time()
                
                # Price discovery
                price_a = await strategy.get_token_price(pair["token0"], pair["token1"], dex_pair[0])
                price_b = await strategy.get_token_price(pair["token0"], pair["token1"], dex_pair[1])
                
                # Calculate arbitrage opportunity
                opportunity = await strategy.calculate_arbitrage_opportunity(
                    pair["token0"], 
                    pair["token1"],
                    price_a,
                    price_b,
                    dex_pair[0],
                    dex_pair[1]
                )
                
                execution_time = time.time() - start_time
                
                if opportunity and opportunity["profit"] > 0:
                    performance_monitor.record_transaction(
                        success=True,
                        gas_price=opportunity["estimated_gas"],
                        execution_time=execution_time,
                        profit=opportunity["profit"],
                        strategy=f"cross_dex_{dex_pair[0]}_{dex_pair[1]}",
                        dex=f"{dex_pair[0]}-{dex_pair[1]}",
                        token_pair=f"{pair['token0']}-{pair['token1']}"
                    )
                
                results.append(opportunity)
            return results
        
        result = benchmark(asyncio.run, run_cross_dex_strategy())
        assert result is not None
        assert len(result) == len(token_pairs)

    @pytest.mark.benchmark
    @pytest.mark.parametrize("pool_type", ["Concentrated", "Wide Range", "Stable"])
    async def test_uniswap_v3_strategy_performance(self, benchmark, web3_config, 
                                                 performance_monitor, pool_type, token_pairs):
        """Benchmark Uniswap V3-specific strategies with different liquidity ranges"""
        strategy = ArbitrageStrategy(web3_config)
        
        async def run_uniswap_v3_strategy():
            results = []
            for pair in token_pairs:
                start_time = time.time()
                
                # Get pool data
                pool_data = await strategy.get_uniswap_v3_pool_data(
                    pair["token0"],
                    pair["token1"],
                    pool_type
                )
                
                # Calculate optimal position
                position = await strategy.calculate_optimal_position(
                    pool_data,
                    pool_type
                )
                
                # Simulate trade
                trade_result = await strategy.simulate_uniswap_v3_trade(
                    position,
                    pool_data
                )
                
                execution_time = time.time() - start_time
                
                if trade_result and trade_result["profit"] > 0:
                    performance_monitor.record_transaction(
                        success=True,
                        gas_price=trade_result["estimated_gas"],
                        execution_time=execution_time,
                        profit=trade_result["profit"],
                        strategy=f"uniswap_v3_{pool_type.lower()}",
                        dex="Uniswap V3",
                        token_pair=f"{pair['token0']}-{pair['token1']}"
                    )
                
                results.append(trade_result)
            return results
        
        result = benchmark(asyncio.run, run_uniswap_v3_strategy())
        assert result is not None
        assert len(result) == len(token_pairs)

    @pytest.mark.benchmark
    async def test_flash_loan_arbitrage_performance(self, benchmark, web3_config, 
                                                  performance_monitor, token_pairs):
        """Benchmark flash loan arbitrage strategy performance"""
        strategy = ArbitrageStrategy(web3_config)
        tx_builder = TransactionBuilder(web3_config)
        
        async def run_flash_loan_strategy():
            results = []
            for pair in token_pairs:
                start_time = time.time()
                
                # Find best flash loan provider
                loan_data = await strategy.find_optimal_flash_loan(
                    pair["token0"],
                    pair["token1"]
                )
                
                # Calculate optimal route
                route = await strategy.calculate_arbitrage_route(
                    loan_data,
                    pair["token0"],
                    pair["token1"]
                )
                
                # Build flash loan transaction
                if route and route["profit"] > 0:
                    tx = await tx_builder.build_flash_loan_transaction(
                        loan_data,
                        route
                    )
                    
                    execution_time = time.time() - start_time
                    
                    performance_monitor.record_transaction(
                        success=bool(tx),
                        gas_price=route["estimated_gas"],
                        execution_time=execution_time,
                        profit=route["profit"],
                        strategy="flash_loan_arbitrage",
                        dex=route["dex_route"],
                        token_pair=f"{pair['token0']}-{pair['token1']}"
                    )
                
                results.append(route)
            return results
        
        result = benchmark(asyncio.run, run_flash_loan_strategy())
        assert result is not None
        assert len(result) == len(token_pairs)

    @pytest.mark.benchmark
    async def test_multi_hop_arbitrage_performance(self, benchmark, web3_config, 
                                                 performance_monitor, token_pairs):
        """Benchmark multi-hop arbitrage strategy performance"""
        strategy = ArbitrageStrategy(web3_config)
        optimizer = GasOptimizer(web3_config)
        
        async def run_multi_hop_strategy():
            results = []
            for pair in token_pairs:
                start_time = time.time()
                
                # Find optimal path
                path = await strategy.find_optimal_multi_hop_path(
                    pair["token0"],
                    pair["token1"],
                    max_hops=3
                )
                
                if path:
                    # Optimize gas for multi-hop
                    gas_strategy = await optimizer.optimize_multi_hop_gas(path)
                    
                    # Calculate total profit
                    profit = await strategy.calculate_multi_hop_profit(
                        path,
                        gas_strategy
                    )
                    
                    execution_time = time.time() - start_time
                    
                    if profit > 0:
                        performance_monitor.record_transaction(
                            success=True,
                            gas_price=gas_strategy["total_gas"],
                            execution_time=execution_time,
                            profit=profit,
                            strategy="multi_hop_arbitrage",
                            dex=path["dex_sequence"],
                            token_pair=f"{pair['token0']}-{pair['token1']}"
                        )
                
                results.append({"path": path, "profit": profit if path else 0})
            return results
        
        result = benchmark(asyncio.run, run_multi_hop_strategy())
        assert result is not None
        assert len(result) == len(token_pairs) 