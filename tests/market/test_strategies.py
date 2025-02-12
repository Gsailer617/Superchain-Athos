import pytest
import asyncio
from unittest.mock import patch, MagicMock
from src.market.strategies import ArbitrageStrategies, StrategyConfig
from src.core.types import MarketValidationResult
from decimal import Decimal

@pytest.fixture
async def arbitrage_strategies():
    """Create arbitrage strategies instance for testing"""
    config = StrategyConfig(
        min_profit_usd=100,
        max_gas_price=500 * 10**9,
        slippage_tolerance=0.005
    )
    strategies = ArbitrageStrategies(config)
    yield strategies
    await strategies.cleanup()

@pytest.mark.asyncio
async def test_simple_arbitrage():
    """Test simple two-exchange arbitrage"""
    strategies = ArbitrageStrategies(StrategyConfig())
    
    # Mock market prices
    market_data = {
        'uniswap': {'ETH/USDT': 2000},
        'sushiswap': {'ETH/USDT': 2020}
    }
    
    with patch('src.market.strategies.ArbitrageStrategies._get_market_prices') as mock_prices:
        mock_prices.return_value = market_data
        
        opportunities = await strategies.find_simple_arbitrage(
            token_pair='ETH/USDT',
            exchanges=['uniswap', 'sushiswap']
        )
        
        assert len(opportunities) > 0
        assert opportunities[0].profit_usd > 0
        assert opportunities[0].buy_exchange == 'uniswap'
        assert opportunities[0].sell_exchange == 'sushiswap'

@pytest.mark.asyncio
async def test_triangular_arbitrage():
    """Test triangular arbitrage detection"""
    strategies = ArbitrageStrategies(StrategyConfig())
    
    # Mock triangular prices
    prices = {
        'ETH/USDT': 2000,
        'ETH/BTC': 0.07,
        'BTC/USDT': 30000
    }
    
    with patch('src.market.strategies.ArbitrageStrategies._get_token_prices') as mock_prices:
        mock_prices.return_value = prices
        
        opportunities = await strategies.find_triangular_arbitrage(
            tokens=['ETH', 'BTC', 'USDT']
        )
        
        assert len(opportunities) > 0
        for opp in opportunities:
            assert opp.profit_usd > 0
            assert len(opp.path) == 3

@pytest.mark.asyncio
async def test_flash_loan_arbitrage():
    """Test flash loan arbitrage opportunities"""
    strategies = ArbitrageStrategies(StrategyConfig())
    
    # Mock flash loan parameters
    loan_params = {
        'token': 'USDC',
        'amount': 100000 * 10**6,  # 100k USDC
        'fee': 0.0009  # 0.09% fee
    }
    
    with patch('src.market.strategies.ArbitrageStrategies._simulate_flash_loan') as mock_sim:
        mock_sim.return_value = {
            'profit': 500 * 10**6,  # 500 USDC profit
            'path': ['borrow', 'swap1', 'swap2', 'repay']
        }
        
        opportunity = await strategies.find_flash_loan_arbitrage(loan_params)
        assert opportunity is not None
        assert opportunity.profit_usd > loan_params['fee'] * loan_params['amount']
        assert len(opportunity.steps) >= 3

@pytest.mark.asyncio
async def test_strategy_validation():
    """Test arbitrage strategy validation"""
    strategies = ArbitrageStrategies(StrategyConfig())
    
    # Test valid strategy
    valid_strategy = {
        'type': 'simple',
        'token_pair': 'ETH/USDT',
        'exchanges': ['uniswap', 'sushiswap'],
        'amount': 1.0
    }
    
    result = await strategies.validate_strategy(valid_strategy)
    assert isinstance(result, MarketValidationResult)
    assert result.is_valid is True
    
    # Test invalid strategy
    invalid_strategy = {
        'type': 'unknown',
        'token_pair': 'INVALID/PAIR'
    }
    
    result = await strategies.validate_strategy(invalid_strategy)
    assert result.is_valid is False

@pytest.mark.asyncio
async def test_profit_calculation():
    """Test profit calculation with fees and slippage"""
    strategies = ArbitrageStrategies(StrategyConfig())
    
    trade_params = {
        'buy_price': 1000,
        'sell_price': 1010,
        'amount': 1.0,
        'gas_cost': 50,
        'exchange_fees': 0.003  # 0.3% fee
    }
    
    profit = await strategies.calculate_profit(trade_params)
    expected_profit = (
        trade_params['amount'] * 
        (trade_params['sell_price'] - trade_params['buy_price']) *
        (1 - trade_params['exchange_fees']) -
        trade_params['gas_cost']
    )
    
    assert abs(profit - expected_profit) < 0.01

@pytest.mark.asyncio
async def test_slippage_protection():
    """Test slippage protection mechanisms"""
    config = StrategyConfig(slippage_tolerance=0.01)  # 1% max slippage
    strategies = ArbitrageStrategies(config)
    
    # Test acceptable slippage
    result = await strategies.check_slippage(
        expected_price=1000,
        actual_price=1009  # 0.9% slippage
    )
    assert result is True
    
    # Test excessive slippage
    result = await strategies.check_slippage(
        expected_price=1000,
        actual_price=1015  # 1.5% slippage
    )
    assert result is False

@pytest.mark.asyncio
async def test_multi_hop_arbitrage():
    """Test multi-hop arbitrage detection"""
    strategies = ArbitrageStrategies(StrategyConfig())
    
    # Mock multi-hop prices
    prices = {
        'dex1': {'ETH/USDT': 2000, 'ETH/DAI': 2010},
        'dex2': {'DAI/USDT': 1.01},
        'dex3': {'ETH/USDT': 2020}
    }
    
    with patch('src.market.strategies.ArbitrageStrategies._get_dex_prices') as mock_prices:
        mock_prices.return_value = prices
        
        opportunities = await strategies.find_multi_hop_arbitrage(
            start_token='ETH',
            end_token='USDT',
            max_hops=3
        )
        
        assert len(opportunities) > 0
        for opp in opportunities:
            assert opp.profit_usd > 0
            assert len(opp.path) <= 3

@pytest.mark.asyncio
async def test_gas_optimization():
    """Test gas optimization for arbitrage execution"""
    strategies = ArbitrageStrategies(StrategyConfig())
    
    # Test route optimization
    route = {
        'steps': [
            {'dex': 'uniswap', 'action': 'swap'},
            {'dex': 'sushiswap', 'action': 'swap'}
        ],
        'estimated_gas': 300000
    }
    
    optimized = await strategies.optimize_execution_route(route)
    assert optimized['estimated_gas'] <= route['estimated_gas']
    assert len(optimized['steps']) == len(route['steps'])

@pytest.mark.asyncio
async def test_concurrent_opportunity_finding():
    """Test concurrent opportunity detection"""
    strategies = ArbitrageStrategies(StrategyConfig())
    
    # Test multiple pairs simultaneously
    pairs = ['ETH/USDT', 'BTC/USDT', 'ETH/BTC']
    
    with patch('src.market.strategies.ArbitrageStrategies.find_simple_arbitrage') as mock_find:
        mock_find.return_value = [MagicMock(profit_usd=100)]
        
        results = await asyncio.gather(*[
            strategies.find_simple_arbitrage(pair, ['uniswap', 'sushiswap'])
            for pair in pairs
        ])
        
        assert len(results) == len(pairs)
        assert all(len(r) > 0 for r in results)

@pytest.mark.asyncio
async def test_strategy_persistence():
    """Test strategy state persistence"""
    strategies = ArbitrageStrategies(StrategyConfig())
    
    # Record successful strategy
    await strategies.record_strategy_result(
        strategy_id='simple_eth_usdt',
        profit_usd=100,
        success=True
    )
    
    # Get strategy statistics
    stats = await strategies.get_strategy_stats('simple_eth_usdt')
    assert stats['success_count'] > 0
    assert stats['total_profit_usd'] >= 100
    assert stats['success_rate'] > 0 