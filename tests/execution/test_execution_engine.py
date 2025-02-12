import pytest
import asyncio
from unittest.mock import patch, MagicMock
from src.execution.execution_engine import ExecutionEngine
from src.core.types import ExecutionStatus, ExecutionResult
from decimal import Decimal

@pytest.fixture
async def execution_engine():
    """Create execution engine instance for testing"""
    engine = ExecutionEngine()
    yield engine
    await engine.cleanup()

@pytest.mark.asyncio
async def test_transaction_execution():
    """Test basic transaction execution"""
    engine = ExecutionEngine()
    
    # Test successful transaction
    tx_data = {
        'to': '0x742d35Cc6634C0532925a3b844Bc454e4438f44e',
        'value': 1000000000000000000,  # 1 ETH
        'data': '0x',
        'gas_price': 50 * 10**9,
        'gas_limit': 21000
    }
    
    with patch('web3.eth.Eth.send_raw_transaction') as mock_send:
        mock_send.return_value = '0x123'  # Transaction hash
        
        result = await engine.execute_transaction(tx_data)
        assert isinstance(result, ExecutionResult)
        assert result.status == ExecutionStatus.SUCCESS
        assert result.tx_hash == '0x123'

@pytest.mark.asyncio
async def test_batch_execution():
    """Test batch transaction execution"""
    engine = ExecutionEngine()
    
    # Prepare batch of transactions
    transactions = [
        {
            'to': '0x123',
            'value': 1000000,
            'data': '0x'
        },
        {
            'to': '0x456',
            'value': 2000000,
            'data': '0x'
        }
    ]
    
    with patch('web3.eth.Eth.send_raw_transaction') as mock_send:
        mock_send.return_value = '0x789'
        
        results = await engine.execute_batch(transactions)
        assert len(results) == 2
        assert all(r.status == ExecutionStatus.SUCCESS for r in results)

@pytest.mark.asyncio
async def test_execution_validation():
    """Test transaction validation before execution"""
    engine = ExecutionEngine()
    
    # Test invalid transaction
    invalid_tx = {
        'to': 'invalid_address',
        'value': -1000  # Invalid value
    }
    
    result = await engine.validate_transaction(invalid_tx)
    assert result.status == ExecutionStatus.VALIDATION_FAILED
    
    # Test valid transaction
    valid_tx = {
        'to': '0x742d35Cc6634C0532925a3b844Bc454e4438f44e',
        'value': 1000000000000000000,
        'data': '0x'
    }
    
    result = await engine.validate_transaction(valid_tx)
    assert result.status == ExecutionStatus.VALIDATED

@pytest.mark.asyncio
async def test_nonce_management():
    """Test nonce management for transactions"""
    engine = ExecutionEngine()
    
    with patch('web3.eth.Eth.get_transaction_count') as mock_nonce:
        mock_nonce.return_value = 10
        
        # Get next nonce
        nonce = await engine.get_next_nonce()
        assert nonce == 10
        
        # Test nonce increment
        next_nonce = await engine.get_next_nonce()
        assert next_nonce == 11

@pytest.mark.asyncio
async def test_transaction_monitoring():
    """Test transaction monitoring and confirmation"""
    engine = ExecutionEngine()
    
    tx_hash = '0x123'
    
    with patch('web3.eth.Eth.get_transaction_receipt') as mock_receipt:
        mock_receipt.return_value = {
            'status': 1,
            'blockNumber': 1000,
            'gasUsed': 21000
        }
        
        result = await engine.wait_for_transaction(tx_hash)
        assert result.status == ExecutionStatus.SUCCESS
        assert result.block_number == 1000
        assert result.gas_used == 21000

@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling during execution"""
    engine = ExecutionEngine()
    
    # Test network error
    with patch('web3.eth.Eth.send_raw_transaction', side_effect=Exception('Network error')):
        result = await engine.execute_transaction({
            'to': '0x742d35Cc6634C0532925a3b844Bc454e4438f44e',
            'value': 1000000000000000000
        })
        assert result.status == ExecutionStatus.FAILED
        assert 'network error' in result.error.lower()
    
    # Test out of gas error
    with patch('web3.eth.Eth.send_raw_transaction', side_effect=Exception('out of gas')):
        result = await engine.execute_transaction({
            'to': '0x742d35Cc6634C0532925a3b844Bc454e4438f44e',
            'value': 1000000000000000000
        })
        assert result.status == ExecutionStatus.GAS_LIMIT_EXCEEDED

@pytest.mark.asyncio
async def test_concurrent_execution():
    """Test concurrent transaction execution"""
    engine = ExecutionEngine()
    
    # Prepare multiple transactions
    transactions = [
        {'to': f'0x{i}', 'value': 1000000} for i in range(5)
    ]
    
    with patch('web3.eth.Eth.send_raw_transaction') as mock_send:
        mock_send.side_effect = lambda tx: f'0x{len(tx)}'  # Unique hash per tx
        
        # Execute concurrently
        results = await asyncio.gather(*[
            engine.execute_transaction(tx) for tx in transactions
        ])
        
        assert len(results) == 5
        assert len(set(r.tx_hash for r in results)) == 5  # Unique hashes

@pytest.mark.asyncio
async def test_execution_retry():
    """Test transaction retry mechanism"""
    engine = ExecutionEngine()
    
    tx_data = {
        'to': '0x742d35Cc6634C0532925a3b844Bc454e4438f44e',
        'value': 1000000000000000000
    }
    
    # Mock failing then succeeding
    with patch('web3.eth.Eth.send_raw_transaction') as mock_send:
        mock_send.side_effect = [
            Exception('Temporary error'),
            '0x123'  # Success on retry
        ]
        
        result = await engine.execute_with_retry(tx_data, max_retries=2)
        assert result.status == ExecutionStatus.SUCCESS
        assert result.retry_count == 1

@pytest.mark.asyncio
async def test_gas_price_bumping():
    """Test gas price bumping for stuck transactions"""
    engine = ExecutionEngine()
    
    tx_data = {
        'to': '0x742d35Cc6634C0532925a3b844Bc454e4438f44e',
        'value': 1000000000000000000,
        'gas_price': 50 * 10**9
    }
    
    # Test gas price bump
    bumped_tx = await engine.bump_gas_price(tx_data, multiplier=1.2)
    assert bumped_tx['gas_price'] == int(tx_data['gas_price'] * 1.2)
    
    # Test max gas price limit
    with pytest.raises(Exception):
        await engine.bump_gas_price(tx_data, multiplier=100)

@pytest.mark.asyncio
async def test_transaction_simulation():
    """Test transaction simulation before execution"""
    engine = ExecutionEngine()
    
    tx_data = {
        'to': '0x742d35Cc6634C0532925a3b844Bc454e4438f44e',
        'value': 1000000000000000000,
        'data': '0x'
    }
    
    # Test successful simulation
    with patch('web3.eth.Eth.call') as mock_call:
        mock_call.return_value = b''  # Success
        
        result = await engine.simulate_transaction(tx_data)
        assert result.status == ExecutionStatus.SIMULATION_SUCCESS
        
    # Test failed simulation
    with patch('web3.eth.Eth.call', side_effect=Exception('Revert')):
        result = await engine.simulate_transaction(tx_data)
        assert result.status == ExecutionStatus.SIMULATION_FAILED 