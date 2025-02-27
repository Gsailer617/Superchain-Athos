"""
Example script demonstrating the enhanced trade history module with gas and execution integration.
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import random
from pathlib import Path

# Import our modules
from src.history.trade_history import EnhancedTradeHistoryManager
from src.history.trade_analytics import TradeAnalytics, TradeGasExecutionIntegrator
from src.history.enhanced_trade_metrics import EnhancedTradeMetrics, GasMetrics, ExecutionMetrics, TokenMetrics

# Create output directory
output_dir = Path("output/trade_history_example")
output_dir.mkdir(parents=True, exist_ok=True)

# Sample data generation
def generate_sample_trade(
    strategy: str,
    token_pair: str,
    dex: str,
    optimization_mode: str,
    success: bool = True
) -> EnhancedTradeMetrics:
    """Generate a sample trade for testing"""
    # Random profit between -0.5 and 2.0 USD
    profit = random.uniform(-0.5, 2.0) if success else random.uniform(-2.0, -0.1)
    
    # Gas metrics
    gas_price = random.randint(20, 100) * 10**9  # 20-100 gwei
    gas_used = random.randint(100000, 500000)
    effective_gas_price = gas_price
    
    # EIP-1559 parameters
    if random.random() > 0.5:  # 50% chance of EIP-1559 tx
        max_fee_per_gas = gas_price + random.randint(5, 20) * 10**9
        max_priority_fee_per_gas = random.randint(1, 10) * 10**9
        effective_gas_price = min(max_fee_per_gas, max_priority_fee_per_gas + gas_price)
    else:
        max_fee_per_gas = None
        max_priority_fee_per_gas = None
    
    # Calculate gas costs
    gas_cost_wei = gas_used * effective_gas_price
    gas_cost_eth = gas_cost_wei / 10**18
    eth_price = random.uniform(1800, 2200)  # ETH price in USD
    gas_cost_usd = gas_cost_eth * eth_price
    
    # Network congestion (0-100%)
    network_congestion = random.uniform(10, 90)
    
    # Optimization savings (0-20%)
    if optimization_mode == "economy":
        optimization_savings = random.uniform(10, 20)
    elif optimization_mode == "normal":
        optimization_savings = random.uniform(5, 10)
    elif optimization_mode == "performance":
        optimization_savings = random.uniform(0, 5)
    else:  # urgent
        optimization_savings = 0
    
    # Create gas metrics
    gas = GasMetrics(
        gas_used=gas_used,
        gas_price=gas_price,
        max_fee_per_gas=max_fee_per_gas,
        max_priority_fee_per_gas=max_priority_fee_per_gas,
        effective_gas_price=effective_gas_price,
        gas_cost_wei=gas_cost_wei,
        gas_cost_eth=gas_cost_eth,
        gas_cost_usd=gas_cost_usd,
        optimization_mode=optimization_mode,
        optimization_savings=optimization_savings,
        network_congestion=network_congestion
    )
    
    # Execution metrics
    tx_hash = f"0x{''.join(random.choices('0123456789abcdef', k=64))}"
    block_number = random.randint(15000000, 16000000)
    execution_time = random.uniform(0.1, 2.0)
    confirmation_time = random.uniform(5, 60) if success else 0
    confirmation_blocks = random.randint(1, 12) if success else 0
    retry_count = random.randint(0, 3)
    
    execution = ExecutionMetrics(
        tx_hash=tx_hash,
        block_number=block_number,
        status=success,
        chain_id=1,  # Ethereum mainnet
        nonce=random.randint(1, 1000),
        execution_time=execution_time,
        confirmation_time=confirmation_time,
        confirmation_blocks=confirmation_blocks,
        retry_count=retry_count,
        simulated=True,
        simulation_success=success,
        error=None if success else "Transaction reverted"
    )
    
    # Token metrics
    token_in_symbol, token_out_symbol = token_pair.split("/")
    
    tokens = TokenMetrics(
        token_in=f"0x{''.join(random.choices('0123456789abcdef', k=40))}",
        token_out=f"0x{''.join(random.choices('0123456789abcdef', k=40))}",
        amount_in=random.uniform(0.01, 1.0),
        amount_out=random.uniform(0.01, 1.0),
        token_in_symbol=token_in_symbol,
        token_out_symbol=token_out_symbol,
        token_in_decimals=18,
        token_out_decimals=18,
        token_in_price_usd=random.uniform(1, 2000),
        token_out_price_usd=random.uniform(1, 2000),
        slippage=random.uniform(0, 1),
        price_impact=random.uniform(0, 2)
    )
    
    # Create enhanced trade metrics
    return EnhancedTradeMetrics(
        timestamp=datetime.now() - timedelta(minutes=random.randint(0, 60*24*7)),  # Random time in last week
        strategy=strategy,
        token_pair=token_pair,
        dex=dex,
        profit=profit,
        success=success,
        gas=gas,
        execution=execution,
        tokens=tokens,
        route=[],
        additional_data={"sample": True}
    )

async def main():
    print("Enhanced Trade History Example")
    print("==============================")
    
    # Initialize trade history manager
    history_manager = EnhancedTradeHistoryManager(
        storage_path="output/trade_history_example/history",
        enable_async=True,
        backup_enabled=True
    )
    
    # Initialize analytics
    analytics = TradeAnalytics(
        trade_history_manager=history_manager,
        reports_path="output/trade_history_example/reports"
    )
    
    # Initialize integrator
    integrator = TradeGasExecutionIntegrator(
        trade_history_manager=history_manager
    )
    
    # Generate sample trades
    print("Generating sample trades...")
    
    strategies = ["arbitrage", "liquidation", "flash_loan"]
    token_pairs = ["ETH/USDC", "WBTC/ETH", "LINK/ETH", "UNI/USDC"]
    dexes = ["uniswap", "sushiswap", "curve", "balancer"]
    optimization_modes = ["economy", "normal", "performance", "urgent"]
    
    # Generate 100 sample trades
    for i in range(100):
        strategy = random.choice(strategies)
        token_pair = random.choice(token_pairs)
        dex = random.choice(dexes)
        optimization_mode = random.choice(optimization_modes)
        success = random.random() > 0.2  # 80% success rate
        
        trade = generate_sample_trade(
            strategy=strategy,
            token_pair=token_pair,
            dex=dex,
            optimization_mode=optimization_mode,
            success=success
        )
        
        # Record trade
        await history_manager.record_trade_async(trade)
        
        if i % 10 == 0:
            print(f"Generated {i+1} trades...")
    
    print("Sample trades generated!")
    
    # Flush trades to disk
    await history_manager.flush_to_disk_async()
    
    # Generate performance report
    print("\nGenerating performance report...")
    report = await analytics.generate_async_report(
        timeframe='7d',
        include_gas_metrics=True,
        include_charts=True,
        save_report=True
    )
    
    # Print summary
    print("\nPerformance Summary:")
    print("-------------------")
    for key, value in report['summary'].items():
        print(f"{key}: {value}")
    
    # Analyze gas optimization impact
    print("\nAnalyzing gas optimization impact...")
    gas_impact = analytics.analyze_gas_optimization_impact(timeframe='7d')
    
    if gas_impact:
        print("\nGas Optimization Impact:")
        print("-----------------------")
        print(f"Estimated savings: ${gas_impact.get('estimated_savings_usd', 0):.2f} USD")
        print(f"Average savings percentage: {gas_impact.get('average_savings_percentage', 0):.2f}%")
    
    # Compare strategies
    print("\nComparing strategies...")
    strategy_comparison = analytics.compare_strategies(timeframe='7d')
    
    if strategy_comparison and 'strategy_comparison' in strategy_comparison:
        print("\nStrategy Comparison:")
        print("-------------------")
        strategies_data = strategy_comparison['strategy_comparison']
        if 'strategy' in strategies_data:
            for i, strategy in enumerate(strategies_data['strategy']):
                print(f"\nStrategy: {strategy}")
                if ('profit', 'sum') in strategies_data:
                    print(f"Total profit: ${strategies_data[('profit', 'sum')][i]:.2f} USD")
                if ('success', 'mean') in strategies_data:
                    print(f"Success rate: {strategies_data[('success', 'mean')][i]*100:.2f}%")
    
    # Recommend gas strategy for a specific strategy
    print("\nRecommending gas strategy for 'arbitrage'...")
    recommendation = integrator.recommend_gas_strategy(
        strategy="arbitrage",
        timeframe='7d'
    )
    
    print("\nGas Strategy Recommendation:")
    print("--------------------------")
    print(f"Recommended mode: {recommendation.get('recommended_mode', 'normal')}")
    print(f"Confidence: {recommendation.get('confidence', 0)*100:.2f}%")
    if 'average_profit' in recommendation:
        print(f"Average profit with this mode: ${recommendation['average_profit']:.2f} USD")
    
    # Visualize performance
    print("\nVisualizing performance...")
    viz_data = analytics.visualize_performance(
        timeframe='7d',
        metrics=['profit', 'gas_cost_usd', 'success_rate'],
        save_path="output/trade_history_example/performance_chart.png",
        show_plot=False
    )
    
    print(f"Performance chart saved to output/trade_history_example/performance_chart.png")
    
    # Export trade history to CSV
    csv_path = history_manager.export_to_csv(
        filepath="output/trade_history_example/trade_history.csv",
        timeframe='7d'
    )
    
    print(f"\nTrade history exported to {csv_path}")
    
    # Create backup
    backup_path = history_manager.create_backup()
    print(f"Backup created at {backup_path}")
    
    # Clean up
    print("\nCleaning up resources...")
    analytics.close()
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    asyncio.run(main()) 