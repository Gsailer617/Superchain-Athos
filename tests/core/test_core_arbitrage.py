from __future__ import annotations

import pytest
import pytest_asyncio
import asyncio
import logging
from typing import Dict, Any, Optional, TypedDict, Union, AsyncGenerator, cast
from decimal import Decimal
from unittest.mock import patch, AsyncMock
from datetime import datetime

# Import from project root
from SuperchainArbitrageAgent import (
    SuperchainArbitrageAgent,
    MarketValidationResult,
    ExecutionResult,
    ExecutionStatus
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
async def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest_asyncio.fixture
async def agent() -> AsyncGenerator[SuperchainArbitrageAgent, None]:
    """Create a test instance of SuperchainArbitrageAgent."""
    agent = SuperchainArbitrageAgent(config_path='tests/fixtures/test_config.json')
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

@pytest.mark.asyncio
async def test_analyze_opportunity(
    agent: SuperchainArbitrageAgent,
    mock_market_data: Dict[str, Any]
) -> None:
    """Test opportunity analysis capabilities"""
    token_pair = ('WETH', 'USDC')
    amount = 1.0
    
    opportunity = await agent.analyze_opportunity(token_pair, amount, mock_market_data)
    
    assert opportunity is not None
    assert 'predicted_profit' in opportunity
    assert 'confidence' in opportunity
    assert 'risk_score' in opportunity
    assert cast(float, opportunity['predicted_profit']) >= 0
    assert 0 <= cast(float, opportunity['confidence']) <= 1
    assert 0 <= cast(float, opportunity['risk_score']) <= 1

@pytest.mark.asyncio
async def test_validate_market_conditions(
    agent: SuperchainArbitrageAgent,
    mock_market_data: Dict[str, Any]
) -> None:
    """Test market validation capabilities"""
    opportunity = {
        'token_pair': ('WETH', 'USDC'),
        'amount': 1.0,
        'predicted_profit': 0.02,
        'confidence': 0.8,
        'risk_score': 0.3,
        'execution_strategy': [],
        'market_analysis': mock_market_data['market_analysis']
    }
    
    validation_result = await agent.validate_market_conditions(opportunity)
    
    assert isinstance(validation_result, MarketValidationResult)
    assert hasattr(validation_result, 'is_valid')
    assert hasattr(validation_result, 'reason')
    if not validation_result.is_valid:
        assert validation_result.reason is not None

@pytest.mark.asyncio
async def test_execute_arbitrage(
    agent: SuperchainArbitrageAgent,
    mock_market_data: Dict[str, Any]
) -> None:
    """Test arbitrage execution capabilities"""
    opportunity = {
        'token_pair': ('WETH', 'USDC'),
        'amount': 1.0,
        'predicted_profit': 0.02,
        'confidence': 0.8,
        'risk_score': 0.3,
        'execution_strategy': [],
        'market_analysis': mock_market_data['market_analysis']
    }
    
    execution_result = await agent.execute_arbitrage(opportunity)
    
    assert isinstance(execution_result, ExecutionResult)
    assert hasattr(execution_result, 'status')
    assert hasattr(execution_result, 'success')
    assert isinstance(execution_result.status, ExecutionStatus)

@pytest.mark.asyncio
async def test_concurrent_operations(
    agent: SuperchainArbitrageAgent,
    mock_market_data: Dict[str, Any]
) -> None:
    """Test handling of concurrent operations"""
    async def analyze_market():
        token_pair = ('WETH', 'USDC')
        amount = 1.0
        return await agent.analyze_opportunity(token_pair, amount, mock_market_data)
    
    # Run multiple analyses concurrently
    results = await asyncio.gather(
        *[analyze_market() for _ in range(5)]
    )
    
    assert len(results) == 5
    assert all(r is not None for r in results)
    assert all(isinstance(r, dict) and 'predicted_profit' in r for r in results)

@pytest.mark.asyncio
async def test_error_handling(agent: SuperchainArbitrageAgent) -> None:
    """Test error handling and recovery"""
    # Test with invalid input
    with pytest.raises(ValueError):
        await agent.analyze_opportunity(('INVALID', 'TOKEN'), 1.0, {})
    
    # Test with network error
    with patch.object(agent.market_analyzer, 'fetch_market_data', side_effect=Exception('Network error')):
        with pytest.raises(Exception):
            await agent.monitor_superchain()

@pytest.mark.asyncio
async def test_monitoring_capabilities(
    agent: SuperchainArbitrageAgent,
    mock_market_data: Dict[str, Any]
) -> None:
    """Test monitoring capabilities"""
    # Mock the monitoring loop to run only once
    with patch.object(agent.market_analyzer, 'fetch_market_data', return_value=mock_market_data):
        with patch.object(agent, '_process_opportunities') as mock_process:
            mock_process.return_value = []
            
            # Start monitoring in background
            monitor_task = asyncio.create_task(agent.monitor_superchain())
            
            # Let it run for a short time
            await asyncio.sleep(0.1)
            
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