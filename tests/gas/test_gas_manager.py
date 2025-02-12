import pytest
import asyncio
from unittest.mock import patch, MagicMock
from src.gas.gas_manager import GasManager
from src.core.types import ExecutionStatus
from decimal import Decimal

@pytest.fixture
async def gas_manager():
    """Create a gas manager instance for testing"""
    manager = GasManager()
    yield manager
    await manager.close()  # Cleanup

@pytest.mark.asyncio
async def test_gas_price_estimation():
    """Test gas price estimation with different network conditions"""
    manager = GasManager()
    
    # Mock web3 gas price calls
    with patch('web3.eth.Eth.gas_price') as mock_gas_price:
        mock_gas_price.return_value = 50 * 10**9  # 50 Gwei
        
        price = await manager.estimate_gas_price()
        assert 45 * 10**9 <= price <= 55 * 10**9  # Allow for buffer

@pytest.mark.asyncio
async def test_gas_limit_calculation():
    """Test gas limit calculation for different transaction types"""
    manager = GasManager()
    
    # Test simple transfer
    simple_tx = {
        'to': '0x742d35Cc6634C0532925a3b844Bc454e4438f44e',
        'value': 1000000000000000000,  # 1 ETH
        'data': '0x'
    }
    
    limit = await manager.calculate_gas_limit(simple_tx)
    assert 21000 <= limit <= 25000  # Standard transfer range
    
    # Test contract interaction
    contract_tx = {
        'to': '0x742d35Cc6634C0532925a3b844Bc454e4438f44e',
        'data': '0xa9059cbb0000000000000000000000742d35cc6634c0532925a3b844bc454e4438f44e0000000000000000000000000000000000000000000000000de0b6b3a7640000'  # ERC20 transfer
    }
    
    limit = await manager.calculate_gas_limit(contract_tx)
    assert 50000 <= limit <= 100000  # Contract interaction range

@pytest.mark.asyncio
async def test_gas_optimization_strategies():
    """Test different gas optimization strategies"""
    manager = GasManager()
    
    # Test aggressive optimization
    with patch('web3.eth.Eth.gas_price') as mock_gas_price:
        mock_gas_price.return_value = 100 * 10**9  # 100 Gwei
        
        optimized = await manager.optimize_gas_usage(
            strategy='aggressive',
            base_gas_limit=100000
        )
        assert optimized < 100000  # Should reduce gas limit
    
    # Test balanced optimization
    optimized = await manager.optimize_gas_usage(
        strategy='balanced',
        base_gas_limit=100000
    )
    assert 80000 <= optimized <= 100000  # Should maintain reasonable limits

@pytest.mark.asyncio
async def test_gas_price_monitoring():
    """Test gas price monitoring and alerts"""
    manager = GasManager()
    
    # Test high gas price alert
    with patch('web3.eth.Eth.gas_price') as mock_gas_price:
        mock_gas_price.return_value = 500 * 10**9  # 500 Gwei
        
        is_high = await manager.check_gas_price_threshold()
        assert is_high is True
    
    # Test normal gas price
    with patch('web3.eth.Eth.gas_price') as mock_gas_price:
        mock_gas_price.return_value = 30 * 10**9  # 30 Gwei
        
        is_high = await manager.check_gas_price_threshold()
        assert is_high is False

@pytest.mark.asyncio
async def test_gas_usage_tracking():
    """Test gas usage tracking and statistics"""
    manager = GasManager()
    
    # Record some gas usage
    await manager.record_gas_usage(
        tx_hash='0x123',
        gas_used=100000,
        gas_price=50 * 10**9
    )
    
    # Get statistics
    stats = await manager.get_gas_usage_stats()
    assert stats['total_gas_used'] >= 100000
    assert stats['avg_gas_price'] >= 50 * 10**9
    assert 'total_transactions' in stats

@pytest.mark.asyncio
async def test_gas_saving_strategies():
    """Test gas saving strategies implementation"""
    manager = GasManager()
    
    # Test multicall optimization
    txs = [
        {'to': '0x123', 'value': 1000000, 'data': '0x'},
        {'to': '0x456', 'value': 2000000, 'data': '0x'}
    ]
    
    optimized_tx = await manager.optimize_batch_transactions(txs)
    assert optimized_tx['to'] != txs[0]['to']  # Should be multicall contract
    assert len(optimized_tx['data']) > 0  # Should contain encoded batch data

@pytest.mark.asyncio
async def test_gas_price_prediction():
    """Test gas price prediction model"""
    manager = GasManager()
    
    # Test short-term prediction
    prediction = await manager.predict_gas_price(
        timeframe='short',  # 5-10 minutes
        confidence=0.8
    )
    assert isinstance(prediction, dict)
    assert 'predicted_price' in prediction
    assert 'confidence_interval' in prediction
    
    # Test long-term prediction
    prediction = await manager.predict_gas_price(
        timeframe='long',  # 1-2 hours
        confidence=0.9
    )
    assert isinstance(prediction, dict)
    assert prediction['confidence_interval'][1] > prediction['confidence_interval'][0]

@pytest.mark.asyncio
async def test_emergency_conditions():
    """Test gas manager behavior under emergency conditions"""
    manager = GasManager()
    
    # Test network congestion handling
    with patch('web3.eth.Eth.gas_price') as mock_gas_price:
        mock_gas_price.return_value = 1000 * 10**9  # 1000 Gwei
        
        status = await manager.check_network_conditions()
        assert status == ExecutionStatus.HALTED
        
        # Should reject transactions
        with pytest.raises(Exception):
            await manager.validate_gas_price(800 * 10**9)

@pytest.mark.asyncio
async def test_gas_refund_calculation():
    """Test gas refund calculations for various operations"""
    manager = GasManager()
    
    # Test storage clearing refund
    refund = await manager.calculate_gas_refund(
        original_gas=100000,
        storage_cleared=True
    )
    assert refund > 0
    assert refund <= 50000  # Max refund is 50% of gas used
    
    # Test no refund case
    refund = await manager.calculate_gas_refund(
        original_gas=100000,
        storage_cleared=False
    )
    assert refund == 0 