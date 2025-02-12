import pytest
import asyncio
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from src.history.trade_history import TradeHistoryManager, TradeMetrics
from decimal import Decimal

@pytest.fixture
async def trade_history():
    """Create trade history manager instance for testing"""
    manager = TradeHistoryManager()
    yield manager
    await manager.cleanup()

@pytest.mark.asyncio
async def test_trade_recording():
    """Test basic trade recording functionality"""
    manager = TradeHistoryManager()
    
    # Record a trade
    trade = {
        'tx_hash': '0x123',
        'timestamp': datetime.now(),
        'token_pair': 'ETH/USDT',
        'buy_price': 2000.0,
        'sell_price': 2020.0,
        'amount': 1.0,
        'profit_usd': 20.0,
        'gas_used': 150000,
        'gas_price': 50 * 10**9
    }
    
    await manager.record_trade(trade)
    
    # Verify trade was recorded
    trades = await manager.get_trades(limit=1)
    assert len(trades) == 1
    assert trades[0]['tx_hash'] == trade['tx_hash']
    assert trades[0]['profit_usd'] == trade['profit_usd']

@pytest.mark.asyncio
async def test_trade_metrics():
    """Test trade metrics calculation"""
    manager = TradeHistoryManager()
    
    # Record multiple trades
    trades = [
        {
            'tx_hash': f'0x{i}',
            'timestamp': datetime.now() - timedelta(hours=i),
            'token_pair': 'ETH/USDT',
            'buy_price': 2000.0,
            'sell_price': 2020.0,
            'amount': 1.0,
            'profit_usd': 20.0,
            'gas_used': 150000,
            'gas_price': 50 * 10**9
        }
        for i in range(5)
    ]
    
    for trade in trades:
        await manager.record_trade(trade)
    
    # Get metrics
    metrics = await manager.get_metrics(timeframe_hours=24)
    assert isinstance(metrics, TradeMetrics)
    assert metrics.total_trades == 5
    assert metrics.total_profit_usd == 100.0  # 5 trades * $20
    assert metrics.avg_profit_usd == 20.0

@pytest.mark.asyncio
async def test_trade_filtering():
    """Test trade history filtering"""
    manager = TradeHistoryManager()
    
    # Record trades with different pairs
    pairs = ['ETH/USDT', 'BTC/USDT', 'ETH/BTC']
    for pair in pairs:
        await manager.record_trade({
            'tx_hash': f'0x{pair}',
            'timestamp': datetime.now(),
            'token_pair': pair,
            'buy_price': 1000.0,
            'sell_price': 1010.0,
            'amount': 1.0,
            'profit_usd': 10.0,
            'gas_used': 150000,
            'gas_price': 50 * 10**9
        })
    
    # Filter by pair
    eth_usdt_trades = await manager.get_trades(token_pair='ETH/USDT')
    assert len(eth_usdt_trades) == 1
    assert eth_usdt_trades[0]['token_pair'] == 'ETH/USDT'

@pytest.mark.asyncio
async def test_profit_analysis():
    """Test profit analysis functionality"""
    manager = TradeHistoryManager()
    
    # Record trades with varying profits
    profits = [-10.0, 5.0, 15.0, 25.0, -5.0]
    for i, profit in enumerate(profits):
        await manager.record_trade({
            'tx_hash': f'0x{i}',
            'timestamp': datetime.now() - timedelta(hours=i),
            'token_pair': 'ETH/USDT',
            'buy_price': 2000.0,
            'sell_price': 2000.0 + profit,
            'amount': 1.0,
            'profit_usd': profit,
            'gas_used': 150000,
            'gas_price': 50 * 10**9
        })
    
    # Analyze profits
    analysis = await manager.analyze_profits(timeframe_hours=24)
    assert analysis['total_profit_usd'] == sum(profits)
    assert analysis['profitable_trades'] == 3
    assert analysis['unprofitable_trades'] == 2
    assert 0 <= analysis['success_rate'] <= 1

@pytest.mark.asyncio
async def test_gas_usage_analysis():
    """Test gas usage analysis"""
    manager = TradeHistoryManager()
    
    # Record trades with different gas usage
    for i in range(5):
        await manager.record_trade({
            'tx_hash': f'0x{i}',
            'timestamp': datetime.now() - timedelta(hours=i),
            'token_pair': 'ETH/USDT',
            'buy_price': 2000.0,
            'sell_price': 2020.0,
            'amount': 1.0,
            'profit_usd': 20.0,
            'gas_used': 150000 + i * 10000,
            'gas_price': (50 + i * 5) * 10**9
        })
    
    # Analyze gas usage
    gas_stats = await manager.analyze_gas_usage(timeframe_hours=24)
    assert 'avg_gas_used' in gas_stats
    assert 'avg_gas_price' in gas_stats
    assert 'total_gas_cost_eth' in gas_stats
    assert gas_stats['avg_gas_used'] > 150000

@pytest.mark.asyncio
async def test_performance_metrics():
    """Test trading performance metrics"""
    manager = TradeHistoryManager()
    
    # Record trades over time
    timestamps = [
        datetime.now() - timedelta(hours=i)
        for i in range(24)
    ]
    
    for i, timestamp in enumerate(timestamps):
        await manager.record_trade({
            'tx_hash': f'0x{i}',
            'timestamp': timestamp,
            'token_pair': 'ETH/USDT',
            'buy_price': 2000.0,
            'sell_price': 2020.0,
            'amount': 1.0,
            'profit_usd': 20.0,
            'gas_used': 150000,
            'gas_price': 50 * 10**9
        })
    
    # Get hourly performance
    hourly = await manager.get_hourly_performance(last_n_hours=24)
    assert len(hourly) == 24
    assert all('timestamp' in h and 'profit_usd' in h for h in hourly)

@pytest.mark.asyncio
async def test_trade_export():
    """Test trade history export functionality"""
    manager = TradeHistoryManager()
    
    # Record some trades
    for i in range(5):
        await manager.record_trade({
            'tx_hash': f'0x{i}',
            'timestamp': datetime.now() - timedelta(hours=i),
            'token_pair': 'ETH/USDT',
            'buy_price': 2000.0,
            'sell_price': 2020.0,
            'amount': 1.0,
            'profit_usd': 20.0,
            'gas_used': 150000,
            'gas_price': 50 * 10**9
        })
    
    # Export trades
    exported = await manager.export_trades(
        start_time=datetime.now() - timedelta(days=1),
        format='csv'
    )
    assert len(exported.split('\n')) > 5  # Header + 5 trades

@pytest.mark.asyncio
async def test_trade_aggregation():
    """Test trade data aggregation"""
    manager = TradeHistoryManager()
    
    # Record trades for different pairs
    pairs = ['ETH/USDT', 'BTC/USDT', 'ETH/BTC']
    for pair in pairs:
        for i in range(3):  # 3 trades per pair
            await manager.record_trade({
                'tx_hash': f'0x{pair}{i}',
                'timestamp': datetime.now() - timedelta(hours=i),
                'token_pair': pair,
                'buy_price': 1000.0,
                'sell_price': 1010.0,
                'amount': 1.0,
                'profit_usd': 10.0,
                'gas_used': 150000,
                'gas_price': 50 * 10**9
            })
    
    # Get aggregated stats
    aggregated = await manager.get_aggregated_stats(
        group_by='token_pair',
        timeframe_hours=24
    )
    
    assert len(aggregated) == len(pairs)
    for pair_stats in aggregated:
        assert pair_stats['trade_count'] == 3
        assert pair_stats['total_profit_usd'] == 30.0

@pytest.mark.asyncio
async def test_trade_cleanup():
    """Test old trade cleanup functionality"""
    manager = TradeHistoryManager()
    
    # Record old and new trades
    old_time = datetime.now() - timedelta(days=31)
    new_time = datetime.now()
    
    trades = [
        {
            'tx_hash': '0x_old',
            'timestamp': old_time,
            'token_pair': 'ETH/USDT',
            'profit_usd': 10.0,
            'gas_used': 150000,
            'gas_price': 50 * 10**9
        },
        {
            'tx_hash': '0x_new',
            'timestamp': new_time,
            'token_pair': 'ETH/USDT',
            'profit_usd': 10.0,
            'gas_used': 150000,
            'gas_price': 50 * 10**9
        }
    ]
    
    for trade in trades:
        await manager.record_trade(trade)
    
    # Cleanup old trades
    cleaned = await manager.cleanup_old_trades(days=30)
    assert cleaned == 1  # One old trade removed
    
    # Verify only new trade remains
    remaining = await manager.get_trades()
    assert len(remaining) == 1
    assert remaining[0]['tx_hash'] == '0x_new' 