# Trade History Migration Guide

This guide explains how to migrate from the legacy `TradeHistoryManager` to the enhanced `TradeMonitor` system.

## Overview

The `TradeMonitor` now includes all functionality from the previous `TradeHistoryManager`, with additional features:
- Real-time monitoring and metrics
- Advanced ML-based analysis
- Anomaly detection
- Prometheus metrics integration
- Performance alerting
- Historical data management
- Learning feature generation

## Migration Steps

1. **Backup Your Data**
   The migration script automatically creates a backup of your old data, but it's recommended to manually backup your data first:
   ```bash
   cp -r /path/to/old/trade/history /path/to/backup/location
   ```

2. **Install Required Dependencies**
   ```bash
   pip install pandas numpy scikit-learn plotly structlog
   ```

3. **Run the Migration**
   ```python
   from monitoring.migrations.trade_history_migration import migrate_trade_history
   
   await migrate_trade_history(
       old_storage_path="/path/to/old/trade/history",
       new_storage_path="/path/to/new/monitoring/data",
       backup=True  # Optional, defaults to True
   )
   ```

4. **Verify Migration**
   ```python
   from monitoring.migrations.trade_history_migration import verify_migration
   
   result = await verify_migration(
       old_storage_path="/path/to/old/trade/history",
       new_storage_path="/path/to/new/monitoring/data"
   )
   print(result)
   ```

5. **Check Migration Status**
   ```python
   from monitoring.migrations.trade_history_migration import get_migration_status
   
   status = get_migration_status("/path/to/new/monitoring/data")
   print(status)
   ```

## Code Changes Required

1. **Update Imports**
   ```python
   # Old import
   from history.trade_history import TradeHistoryManager
   
   # New import
   from monitoring.specialized.trade_monitor import TradeMonitor
   ```

2. **Initialize Monitor**
   ```python
   # Old initialization
   history_manager = TradeHistoryManager(
       storage_path="data/trade_history",
       max_memory_entries=10000,
       flush_interval=100
   )
   
   # New initialization
   trade_monitor = TradeMonitor(
       storage_path="data/monitoring/trades",
       historical_storage_path="data/monitoring/historical_trades",  # Optional
       max_memory_entries=10000,
       flush_interval=100,
       enable_prometheus=True,  # Optional
       performance_thresholds={  # Optional
           'min_profit': -0.1,
           'max_gas_price': 100.0,
           'max_execution_time': 5.0,
           'min_success_rate': 95.0
       },
       ml_config={  # Optional
           'profit_prediction': {
               'window_size': 24,
               'train_size': 0.8
           }
       }
   )
   ```

3. **Update Method Calls**
   ```python
   # Recording trades
   # Old way
   history_manager.record_trade(metrics)
   
   # New way
   await trade_monitor.record_trade(metrics)
   
   # Getting history
   # Old way
   df = history_manager.get_history(start_time, end_time)
   
   # New way
   df = trade_monitor.get_trade_history(
       start_time=start_time,
       end_time=end_time,
       include_memory=True  # Optional
   )
   ```

## New Features Available

1. **Real-time Monitoring**
   ```python
   # Start monitoring
   await trade_monitor.start_monitoring()
   
   # Stop monitoring
   await trade_monitor.stop_monitoring()
   ```

2. **Performance Analysis**
   ```python
   # Get performance metrics
   metrics = trade_monitor.analyze_performance(timeframe='24h')
   ```

3. **ML Predictions**
   ```python
   # Predict profit for a trade
   prediction = trade_monitor.predict_profit(trade_features)
   
   # Detect anomalies
   anomalies = trade_monitor.detect_anomalies()
   ```

4. **Metrics Export**
   ```python
   # Export metrics to Prometheus
   trade_monitor.export_metrics()
   ```

## Troubleshooting

1. **Missing Data**
   If some data is missing after migration, check:
   - Migration verification results
   - Old data backup
   - Migration logs

2. **Performance Issues**
   If experiencing performance issues:
   - Adjust `max_memory_entries`
   - Tune `flush_interval`
   - Consider using `include_memory=False` for large historical queries

3. **Migration Errors**
   Common issues and solutions:
   - File permissions: Ensure write access to new location
   - Disk space: Verify sufficient space for migration
   - Data format: Check if old data follows expected schema

## Support

For issues or questions:
1. Check migration logs
2. Verify data integrity
3. Contact support team

## Rollback

To rollback to the old system:
1. Stop the new monitor
2. Restore from backup
3. Revert code changes

## Best Practices

1. Always backup data before migration
2. Run migration during low-traffic periods
3. Verify data integrity after migration
4. Monitor system performance after migration
5. Keep old system running in parallel initially 