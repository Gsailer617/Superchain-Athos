# Enhanced Trade History Module

The Enhanced Trade History Module provides comprehensive trade tracking, analytics, and integration with gas optimization and execution modules. It enables detailed tracking of trade metrics, performance analysis, and data-driven optimization of trading strategies.

## Features

### Enhanced Trade Metrics

- **Comprehensive Metrics**: Track detailed metrics for each trade, including gas usage, execution details, and token information.
- **EIP-1559 Support**: Full support for EIP-1559 transactions with tracking of max fee, priority fee, and effective gas price.
- **Cost Tracking**: Track gas costs in wei, ETH, and USD for accurate profitability analysis.
- **Execution Metrics**: Record transaction hashes, block numbers, confirmation times, and retry counts.
- **Token Metrics**: Track token amounts, prices, slippage, and price impact.

### Trade History Management

- **Asynchronous Support**: Fully async-compatible API for non-blocking operations.
- **Efficient Storage**: Parquet-based storage for efficient data compression and fast querying.
- **Automatic Backups**: Scheduled backups to prevent data loss.
- **Data Retention**: Automatic cleanup of old data to manage storage usage.
- **Import/Export**: CSV import/export functionality for data sharing and backup.

### Advanced Analytics

- **Performance Analysis**: Comprehensive analysis of trading performance across strategies, token pairs, and DEXes.
- **Gas Optimization Analysis**: Analyze the impact of different gas optimization strategies on profitability.
- **Strategy Comparison**: Compare the performance of different trading strategies.
- **Visualization**: Generate visualizations of performance metrics over time.
- **Custom Reports**: Generate and export detailed performance reports.

### Gas and Execution Integration

- **Execution Result Recording**: Directly record execution results from the execution module.
- **Gas Strategy Recommendations**: Get data-driven recommendations for optimal gas strategies based on historical performance.
- **Cross-Module Analytics**: Analyze the relationship between gas optimization and trading performance.

## Usage Examples

### Basic Usage

```python
from src.history import EnhancedTradeHistoryManager, EnhancedTradeMetrics

# Initialize the trade history manager
history_manager = EnhancedTradeHistoryManager(
    storage_path="data/trade_history",
    enable_async=True
)

# Record a trade
history_manager.record_trade(metrics)

# Get historical data
df = history_manager.get_history(
    start_time=datetime.now() - timedelta(days=7),
    strategy="arbitrage"
)

# Analyze performance
performance = history_manager.analyze_performance(timeframe='24h')
```

### Asynchronous Usage

```python
import asyncio
from src.history import EnhancedTradeHistoryManager

async def record_trades():
    history_manager = EnhancedTradeHistoryManager(enable_async=True)
    
    # Record a trade asynchronously
    await history_manager.record_trade_async(metrics)
    
    # Flush to disk asynchronously
    await history_manager.flush_to_disk_async()

asyncio.run(record_trades())
```

### Analytics

```python
from src.history import TradeAnalytics

# Initialize analytics
analytics = TradeAnalytics(trade_history_manager=history_manager)

# Generate performance report
report = analytics.generate_performance_report(
    timeframe='7d',
    include_gas_metrics=True,
    include_charts=True
)

# Visualize performance
analytics.visualize_performance(
    timeframe='30d',
    metrics=['profit', 'gas_cost_usd', 'success_rate'],
    save_path="reports/performance_chart.png"
)

# Analyze gas optimization impact
impact = analytics.analyze_gas_optimization_impact(timeframe='30d')

# Compare strategies
comparison = analytics.compare_strategies(timeframe='30d')
```

### Gas and Execution Integration

```python
from src.history import TradeGasExecutionIntegrator

# Initialize integrator
integrator = TradeGasExecutionIntegrator(trade_history_manager=history_manager)

# Record execution result
await integrator.record_execution_result(
    execution_result=result,
    strategy="arbitrage"
)

# Get gas strategy recommendation
recommendation = integrator.recommend_gas_strategy(
    strategy="arbitrage",
    token_pair="ETH/USDC",
    timeframe='7d'
)
```

## Dependencies

- pandas
- numpy
- pyarrow
- matplotlib
- seaborn
- aiofiles
- structlog

## Integration with Other Modules

The Enhanced Trade History Module integrates seamlessly with:

- **Gas Module**: Record and analyze gas optimization strategies and their impact on profitability.
- **Execution Module**: Record execution results and analyze execution performance.
- **Strategy Module**: Provide historical data and analytics for strategy optimization.

## Performance Considerations

- Use asynchronous methods for high-frequency trading to avoid blocking the main execution thread.
- Configure appropriate `flush_interval` and `max_memory_entries` based on your trading frequency.
- Enable automatic backups to prevent data loss.
- Use the cleanup functionality to manage disk usage for long-running systems.

## Advanced Configuration

```python
# Advanced configuration
history_manager = EnhancedTradeHistoryManager(
    storage_path="data/trade_history",
    max_memory_entries=20000,  # Keep more trades in memory
    flush_interval=50,         # Flush to disk more frequently
    enable_async=True,
    backup_enabled=True,
    backup_interval=1000,      # Backup every 1000 trades
    backup_path="data/backups",
    thread_pool_size=8         # More threads for parallel operations
)
``` 