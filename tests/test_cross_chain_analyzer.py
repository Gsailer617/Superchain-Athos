import pytest
import asyncio
from unittest.mock import Mock, patch
import time
from src.analysis.cross_chain_analyzer import CrossChainAnalyzer, ChainMetrics, PerformanceMetrics

@pytest.fixture
def analyzer():
    return CrossChainAnalyzer()

@pytest.fixture
def mock_web3():
    mock = Mock()
    mock.eth.wait_for_transaction_receipt.return_value = {
        'status': 1,
        'blockNumber': 1000,
        'gasUsed': 100000,
        'effectiveGasPrice': 50000000000
    }
    mock.eth.get_transaction.return_value = {
        'value': 1000000000000000000,
        'from': '0x1234...',
        'to': '0x5678...'
    }
    mock.eth.block_number = 1010
    return mock

@pytest.mark.asyncio
async def test_performance_metrics_recording():
    """Test recording and retrieving performance metrics"""
    metrics = PerformanceMetrics()
    
    # Record some operation times
    metrics.record_operation_time('price_check', 0.5)
    metrics.record_operation_time('price_check', 1.0)
    metrics.record_operation_time('bridge_check', 2.0)
    
    # Test average calculations
    assert metrics.get_avg_operation_time('price_check') == 0.75
    assert metrics.get_avg_operation_time('bridge_check') == 2.0
    assert metrics.get_avg_operation_time('nonexistent') == 0.0

@pytest.mark.asyncio
async def test_chain_metrics_tracking():
    """Test tracking chain-specific metrics"""
    metrics = PerformanceMetrics()
    chain_metrics = ChainMetrics(
        avg_block_time=2.0,
        avg_gas_price=50.0,
        success_rate=0.95,
        failed_txs=5,
        total_txs=100
    )
    
    metrics.update_chain_metrics('ethereum', chain_metrics)
    assert 'ethereum' in metrics.chain_metrics
    assert metrics.chain_metrics['ethereum'].success_rate == 0.95
    assert metrics.chain_metrics['ethereum'].total_txs == 100

@pytest.mark.asyncio
async def test_error_handling(analyzer):
    """Test error handling and metrics updates"""
    # Test connection error handling
    await analyzer._handle_connection_error('ethereum', Exception('Connection failed'))
    metrics = analyzer.metrics.chain_metrics.get('ethereum')
    assert metrics is not None
    assert 'Connection failed' in metrics.last_error
    
    # Test timeout error handling
    await analyzer._handle_timeout_error('price_check', Exception('Timeout'))
    avg_time = analyzer.metrics.get_avg_operation_time('price_check')
    assert avg_time == 30.0  # Timeout is recorded as 30s

@pytest.mark.asyncio
async def test_transaction_monitoring(analyzer, mock_web3):
    """Test transaction monitoring and status updates"""
    with patch('src.analysis.cross_chain_analyzer.get_chain_connector') as mock_connector:
        mock_connector.return_value.get_connection.return_value = mock_web3
        
        # Monitor a transaction
        result = await analyzer._get_transaction_status('ethereum', '0x1234...')
        
        assert result['status'] == 'success'
        assert result['confirmations'] == 10
        
        # Check metrics were updated
        metrics = analyzer.metrics.chain_metrics.get('ethereum')
        assert metrics is not None
        assert metrics.total_txs == 1
        assert metrics.success_rate == 1.0

@pytest.mark.asyncio
async def test_performance_report(analyzer):
    """Test generating performance report"""
    # Add some test data
    chain_metrics = ChainMetrics(
        avg_block_time=2.0,
        avg_gas_price=50.0,
        success_rate=0.98,
        failed_txs=2,
        total_txs=100,
        last_updated=time.time()
    )
    analyzer.metrics.update_chain_metrics('ethereum', chain_metrics)
    analyzer.metrics.record_operation_time('price_check', 1.0)
    
    report = analyzer.get_performance_report()
    
    assert report['overall_health'] == 'healthy'
    assert 'ethereum' in report['chain_metrics']
    assert report['chain_metrics']['ethereum']['success_rate'] == 0.98
    assert 'price_check' in report['operation_times']

@pytest.mark.asyncio
async def test_bridge_error_handling(analyzer):
    """Test bridge error handling and cache clearing"""
    # Add test data to cache
    analyzer.bridge_liquidity_cache['ethereum:polygon'] = 1000000
    
    # Simulate bridge error
    await analyzer._handle_bridge_error('ethereum', 'polygon', Exception('Bridge error'))
    
    # Check cache was cleared
    assert 'ethereum:polygon' not in analyzer.bridge_liquidity_cache

@pytest.mark.asyncio
async def test_validation_error_handling(analyzer):
    """Test validation error handling and metrics updates"""
    tx_data = {'chain': 'ethereum', 'hash': '0x1234...'}
    await analyzer._handle_validation_error(tx_data, Exception('Validation failed'))
    
    metrics = analyzer.metrics.chain_metrics.get('ethereum')
    assert metrics is not None
    assert metrics.failed_txs == 1
    assert 'Validation failed' in metrics.last_error 