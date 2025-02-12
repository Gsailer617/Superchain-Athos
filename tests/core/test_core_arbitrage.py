from __future__ import annotations

import pytest
import pytest_asyncio
import asyncio
import logging
from typing import Dict, Any, Optional, TypedDict, Union, AsyncGenerator, cast
from decimal import Decimal
from unittest.mock import patch, AsyncMock
from datetime import datetime
import torch

# Import from project root
from SuperchainArbitrageAgent import (
    SuperchainArbitrageAgent,
    MarketValidationResult,
    ExecutionResult,
    ExecutionStatus
)

# Import from tests directory
from tests.utils.test_utils import (
    create_mock_web3,
    get_test_config,
    create_mock_market_data,
    MockMarketAnalyzer,
    MockArbitrageModel
)

# Type definitions
class MarketData(TypedDict):
    liquidity: float
    volatility: float
    price: Decimal
    volume: float
    
class OpportunityData(TypedDict):
    token_pair: tuple[str, str]
    amount: float
    predicted_profit: float
    confidence: float
    risk_score: float
    execution_strategy: list
    market_analysis: Dict[str, Any]

class TradeData(TypedDict):
    profit: float
    gas_used: float
    tokens: list[str]
    tokens_involved: list[str]
    timestamp: datetime
    tx_hash: str
    execution_time: float

@pytest.fixture(scope="function")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()

@pytest_asyncio.fixture
async def agent() -> AsyncGenerator[SuperchainArbitrageAgent, None]:
    """Create a test instance of SuperchainArbitrageAgent."""
    agent = SuperchainArbitrageAgent(config_path='tests/fixtures/test_config.json')
    
    # Replace market analyzer with mock
    agent.market_analyzer = MockMarketAnalyzer(agent.config)
    
    yield agent

@pytest.fixture
def mock_market_data() -> Dict[str, Any]:
    """Create mock market data for testing."""
    return {
        'market_analysis': {
            'liquidity': 1000000.0,
            'volatility': 0.1,
            'price': float(Decimal('1800.50')),
            'volume': 500000.0
        },
        'mev': {
            'sandwich_risk': 0.1,
            'frontrunning_risk': 0.2,
            'backrunning_risk': 0.1
        },
        'gas': {
            'base_fee': 1.5,
            'priority_fee': 0.5,
            'block_utilization': 0.8
        },
        'liquidity': {
            'depth': 1000000.0,
            'concentration': 0.7,
            'imbalance': 0.1
        },
        'token_economics': {
            'supply_ratio': 0.8,
            'holder_distribution': 0.6,
            'volume_profile': 0.7
        }
    }

@pytest.mark.asyncio
async def test_agent_initialization(agent: SuperchainArbitrageAgent) -> None:
    """Test agent initialization and basic functionality"""
    assert agent is not None
    assert hasattr(agent, 'market_analyzer')
    assert hasattr(agent, 'model')
    assert hasattr(agent, 'training_manager')

@pytest.fixture
async def test_agent():
    """Fixture for test agent with mocked dependencies"""
    from SuperchainArbitrageAgent import SuperchainArbitrageAgent
    
    # Create agent with test config
    agent = SuperchainArbitrageAgent(config_path='tests/fixtures/test_config.json')
    
    # Replace components with mocks
    agent.market_analyzer = MockMarketAnalyzer(agent.config)
    agent.model = MockArbitrageModel(agent.device)
    
    return agent

@pytest.mark.asyncio
async def test_analyze_opportunity(test_agent):
    """Test opportunity analysis"""
    # Create proper market data with all required fields
    market_data = {
        'sentiment': torch.zeros(8, dtype=torch.long),
        'historical': torch.zeros((1, 8, 8), dtype=torch.float),
        'cross_chain': torch.zeros(12, dtype=torch.long),
        'mev': torch.zeros(8, dtype=torch.long),
        'gas': torch.zeros(6, dtype=torch.float),
        'liquidity': torch.zeros(10, dtype=torch.float),
        'token_economics': torch.zeros(12, dtype=torch.long),
        'token_pair': ('0x1234...', '0x5678...'),
        'price': 100.0,
        'liquidity_value': 10000.0,
        'gas_price': 50000000000
    }
    
    token_pair = market_data['token_pair']
    amount = 1.0
    
    opportunity = await test_agent.analyze_opportunity(token_pair, amount, market_data)
    assert opportunity is not None
    assert isinstance(opportunity, dict)
    assert 'predicted_profit' in opportunity
    assert 'confidence' in opportunity
    assert 'risk_score' in opportunity

@pytest.mark.asyncio
async def test_validate_market_conditions(test_agent):
    """Test market condition validation"""
    market_data = create_mock_market_data()
    opportunity = {
        'token_pair': market_data['token_pair'],
        'amount': 1.0,
        'entry_price': market_data['price'],
        'min_required_liquidity': market_data['liquidity'] * 0.8,
        'gas_price': market_data['gas_price']
    }
    
    result = await test_agent.validate_market_conditions(opportunity)
    assert result.is_valid
    assert result.current_price is not None
    assert result.current_liquidity is not None

@pytest.mark.asyncio
async def test_execute_arbitrage(test_agent):
    """Test arbitrage execution"""
    market_data = create_mock_market_data()
    opportunity = {
        'token_pair': market_data['token_pair'],
        'amount': 1.0,
        'predicted_profit': 0.1,
        'confidence': 0.8,
        'risk_score': 0.2,
        'market_data': market_data
    }
    
    result = await test_agent.execute_arbitrage(opportunity)
    assert result is not None
    assert hasattr(result, 'success')

@pytest.mark.asyncio
async def test_concurrent_operations(test_agent):
    """Test concurrent operations"""
    # Create proper market data with all required fields
    market_data = {
        'sentiment': torch.zeros(8, dtype=torch.long),
        'historical': torch.zeros((1, 8, 8), dtype=torch.float),
        'cross_chain': torch.zeros(12, dtype=torch.long),
        'mev': torch.zeros(8, dtype=torch.long),
        'gas': torch.zeros(6, dtype=torch.float),
        'liquidity': torch.zeros(10, dtype=torch.float),
        'token_economics': torch.zeros(12, dtype=torch.long),
        'token_pair': ('0x1234...', '0x5678...'),
        'price': 100.0,
        'liquidity_value': 10000.0,
        'gas_price': 50000000000
    }
    
    token_pair = market_data['token_pair']
    amounts = [0.1, 0.5, 1.0]
    
    # Run concurrent analyses
    tasks = [
        test_agent.analyze_opportunity(token_pair, amount, market_data)
        for amount in amounts
    ]
    
    import asyncio
    results = await asyncio.gather(*tasks)
    assert all(r is not None for r in results)
    assert len(results) == len(amounts)

@pytest.mark.asyncio
async def test_error_handling(test_agent):
    """Test error handling with invalid inputs"""
    # Test with invalid token pair format
    with pytest.raises(KeyError, match='token_pair must be a tuple of two token addresses'):
        await test_agent.analyze_opportunity(
            'invalid_pair',  # Not a tuple
            1.0,
            {'sentiment': torch.zeros(8, dtype=torch.long)}
        )
    
    # Test with negative amount
    market_data = {
        'sentiment': torch.zeros(8, dtype=torch.long),
        'historical': torch.zeros((1, 8, 8), dtype=torch.float),
        'cross_chain': torch.zeros(12, dtype=torch.long),
        'mev': torch.zeros(8, dtype=torch.long),
        'gas': torch.zeros(6, dtype=torch.float),
        'liquidity': torch.zeros(10, dtype=torch.float),
        'token_economics': torch.zeros(12, dtype=torch.long)
    }
    with pytest.raises(ValueError, match='amount must be positive'):
        await test_agent.analyze_opportunity(
            ('0x1234...', '0x5678...'),
            -1.0,
            market_data
        )
    
    # Test with empty market data
    with pytest.raises(KeyError, match='market_data is required and must be a dictionary'):
        await test_agent.analyze_opportunity(
            ('0x1234...', '0x5678...'),
            1.0,
            {}
        )
    
    # Test with missing market data
    with pytest.raises(KeyError, match='market_data is required and must be a dictionary'):
        await test_agent.analyze_opportunity(
            ('0x1234...', '0x5678...'),
            1.0,
            None
        )

@pytest.mark.asyncio
async def test_monitoring_capabilities(test_agent):
    """Test monitoring capabilities"""
    # Create proper market data
    mock_market_data = {
        'sentiment': torch.zeros(8, dtype=torch.long),
        'historical': torch.zeros((1, 8, 8), dtype=torch.float),
        'cross_chain': torch.zeros(12, dtype=torch.long),
        'mev': torch.zeros(8, dtype=torch.long),
        'gas': torch.zeros(6, dtype=torch.float),
        'liquidity': torch.zeros(10, dtype=torch.float),
        'token_economics': torch.zeros(12, dtype=torch.long),
        'token_pair': ('0x1234...', '0x5678...'),
        'price': 100.0,
        'liquidity_value': 10000.0,
        'gas_price': 50000000000
    }
    
    # Mock the monitoring loop to run only once
    with patch.object(test_agent.market_analyzer, 'fetch_market_data', return_value=mock_market_data):
        with patch.object(test_agent, '_process_opportunities', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = []
            
            # Start monitoring in background
            monitor_task = asyncio.create_task(test_agent.monitor_superchain())
            
            # Let it run for a short time
            await asyncio.sleep(0.5)  # Increased sleep time to ensure task runs
            
            # Cancel the monitoring task
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
            
            # Verify monitoring behavior
            mock_process.assert_called()

if __name__ == "__main__":
    pytest.main(["-v", __file__]) 
    pytest.main(["-v", __file__]) 