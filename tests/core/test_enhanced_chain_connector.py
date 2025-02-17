import pytest
from unittest.mock import Mock, patch, AsyncMock
from web3 import Web3, AsyncWeb3
from web3.providers import HTTPProvider, WebsocketProvider
import asyncio
import time

from src.core.enhanced_chain_connector import EnhancedChainConnector, ChainMetrics
from src.config.chain_specs import ChainSpec, GasModel

@pytest.fixture
def chain_spec():
    """Test chain specification"""
    return ChainSpec(
        name="test_chain",
        chain_id=1,
        rpc_url="http://localhost:8545",
        ws_url="ws://localhost:8546",
        block_time=12,
        gas_fee_model=GasModel.EIP1559,
        is_l2=False
    )

@pytest.fixture
def mock_web3():
    """Mock Web3 instance"""
    mock = Mock(spec=Web3)
    mock.eth.chain_id = 1
    mock.eth.gas_price = 50000000000  # 50 gwei
    mock.eth.max_priority_fee = 2000000000  # 2 gwei
    mock.eth.block_number = 1000
    mock.is_connected.return_value = True
    
    # Mock latest block
    mock.eth.get_block.return_value = {
        'baseFeePerGas': 40000000000  # 40 gwei
    }
    
    return mock

@pytest.fixture
def mock_async_web3():
    """Mock AsyncWeb3 instance"""
    mock = AsyncMock(spec=AsyncWeb3)
    mock.eth.chain_id = 1
    mock.eth.gas_price = 50000000000  # 50 gwei
    mock.eth.max_priority_fee = 2000000000  # 2 gwei
    mock.eth.block_number = 1000
    mock.is_connected.return_value = True
    
    # Mock latest block
    mock.eth.get_block.return_value = {
        'baseFeePerGas': 40000000000  # 40 gwei
    }
    
    return mock

class TestEnhancedChainConnector:
    """Test suite for enhanced chain connector"""
    
    @pytest.mark.asyncio
    async def test_get_async_web3_success(self, chain_spec, mock_async_web3):
        """Test successful async Web3 connection"""
        with patch('web3.AsyncWeb3', return_value=mock_async_web3):
            connector = EnhancedChainConnector()
            web3 = await connector.get_async_web3("test_chain")
            
            assert web3 is not None
            assert await web3.eth.chain_id == chain_spec.chain_id
    
    def test_get_web3_success(self, chain_spec, mock_web3):
        """Test successful Web3 connection"""
        with patch('web3.Web3', return_value=mock_web3):
            connector = EnhancedChainConnector()
            web3 = connector.get_web3("test_chain")
            
            assert web3 is not None
            assert web3.eth.chain_id == chain_spec.chain_id
    
    @pytest.mark.asyncio
    async def test_connection_caching(self, chain_spec, mock_async_web3):
        """Test that connections are cached"""
        with patch('web3.AsyncWeb3', return_value=mock_async_web3):
            connector = EnhancedChainConnector()
            
            # Get connection twice
            web3_1 = await connector.get_async_web3("test_chain")
            web3_2 = await connector.get_async_web3("test_chain")
            
            # Should return same instance
            assert web3_1 is web3_2
    
    @pytest.mark.asyncio
    async def test_fallback_rpcs(self, chain_spec):
        """Test fallback RPC functionality"""
        primary_error = ConnectionError("Primary RPC failed")
        fallback_mock = AsyncMock(spec=AsyncWeb3)
        fallback_mock.eth.chain_id = chain_spec.chain_id
        fallback_mock.is_connected.return_value = True
        
        with patch('web3.AsyncWeb3') as mock_web3:
            # Make primary RPC fail
            mock_web3.side_effect = [primary_error, fallback_mock]
            
            connector = EnhancedChainConnector()
            web3 = await connector.get_async_web3("test_chain")
            
            assert web3 is not None
            assert web3 is fallback_mock
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, chain_spec, mock_async_web3):
        """Test metrics collection"""
        with patch('web3.AsyncWeb3', return_value=mock_async_web3):
            connector = EnhancedChainConnector()
            
            # Make some requests
            web3 = await connector.get_async_web3("test_chain")
            await web3.eth.get_block_number()
            
            # Get metrics
            metrics = connector.get_metrics("test_chain")
            
            assert isinstance(metrics, ChainMetrics)
            assert metrics.total_requests > 0
            assert metrics.success_rate == 1.0
            assert metrics.avg_response_time > 0
    
    @pytest.mark.asyncio
    async def test_chain_health_monitoring(self, chain_spec, mock_async_web3):
        """Test chain health monitoring"""
        with patch('web3.AsyncWeb3', return_value=mock_async_web3):
            connector = EnhancedChainConnector()
            
            # Monitor chain health
            health = await connector.monitor_chain_health("test_chain")
            
            assert health['status'] == 'healthy'
            assert 'latest_block' in health
            assert 'gas_prices' in health
            assert 'block_time' in health
            assert 'congestion_level' in health
    
    @pytest.mark.asyncio
    async def test_gas_price_estimation(self, chain_spec, mock_async_web3):
        """Test gas price estimation for different models"""
        with patch('web3.AsyncWeb3', return_value=mock_async_web3):
            connector = EnhancedChainConnector()
            
            # Test EIP-1559 gas prices
            chain_spec.gas_fee_model = GasModel.EIP1559
            gas_prices = await connector._get_chain_gas_prices(mock_async_web3, chain_spec)
            
            assert 'base_fee' in gas_prices
            assert 'max_priority_fee' in gas_prices
            assert 'average' in gas_prices
            assert 'fast' in gas_prices
            assert 'fastest' in gas_prices
            
            # Test legacy gas prices
            chain_spec.gas_fee_model = GasModel.LEGACY
            gas_prices = await connector._get_chain_gas_prices(mock_async_web3, chain_spec)
            
            assert 'average' in gas_prices
            assert 'fast' in gas_prices
            assert 'fastest' in gas_prices
    
    @pytest.mark.asyncio
    async def test_congestion_calculation(self, chain_spec):
        """Test congestion level calculation"""
        connector = EnhancedChainConnector()
        
        # Test normal conditions
        block_times = [12.0] * 10  # Target block time
        gas_prices = [50.0] * 10   # Stable gas price
        
        congestion = connector._calculate_congestion_level(
            block_times,
            gas_prices,
            chain_spec
        )
        assert congestion == 0.0  # No congestion
        
        # Test congested conditions
        block_times = [24.0] * 10  # 2x target block time
        gas_prices = [50.0, 60.0, 70.0, 80.0, 90.0] * 2  # Rising gas prices
        
        congestion = connector._calculate_congestion_level(
            block_times,
            gas_prices,
            chain_spec
        )
        assert congestion > 0.5  # High congestion
    
    @pytest.mark.asyncio
    async def test_connection_error_handling(self, chain_spec):
        """Test connection error handling"""
        with patch('web3.AsyncWeb3', side_effect=ConnectionError):
            connector = EnhancedChainConnector()
            
            with pytest.raises(ConnectionError):
                await connector.get_async_web3("test_chain")
            
            # Check metrics were updated
            metrics = connector.get_metrics("test_chain")
            assert metrics is not None
            assert metrics.failed_requests > 0
            assert metrics.success_rate < 1.0
    
    @pytest.mark.asyncio
    async def test_chain_specific_middleware(self, chain_spec, mock_async_web3):
        """Test chain-specific middleware handling"""
        with patch('web3.AsyncWeb3', return_value=mock_async_web3):
            connector = EnhancedChainConnector()
            
            # Test L1 chain (should add PoA middleware)
            chain_spec.is_l2 = False
            web3 = await connector.get_async_web3("test_chain")
            assert len(web3.middleware_onion._middleware) > 0
            
            # Test L2 chain (should not add PoA middleware)
            chain_spec.is_l2 = True
            web3 = await connector.get_async_web3("test_chain")
            assert len(web3.middleware_onion._middleware) == 1  # Only metrics middleware
    
    def test_cleanup(self, mock_web3, mock_async_web3):
        """Test cleanup of connections"""
        with patch('web3.Web3', return_value=mock_web3), \
             patch('web3.AsyncWeb3', return_value=mock_async_web3):
            connector = EnhancedChainConnector()
            
            # Create some connections
            web3_sync = connector.get_web3("test_chain")
            web3_async = asyncio.run(connector.get_async_web3("test_chain"))
            
            # Close connections
            connector.close()
            
            # Verify cleanup
            assert not connector._connections
            assert not connector._async_connections
            assert not connector._metrics
    
    @pytest.mark.asyncio
    async def test_websocket_support(self, chain_spec, mock_async_web3):
        """Test WebSocket connection support"""
        with patch('web3.AsyncWeb3', return_value=mock_async_web3), \
             patch('web3.providers.WebsocketProvider') as mock_ws:
            connector = EnhancedChainConnector()
            
            # Test WebSocket connection
            chain_spec.rpc_url = "ws://localhost:8546"
            web3 = await connector.get_async_web3("test_chain")
            
            assert mock_ws.called
            assert isinstance(web3.provider, WebsocketProvider)
    
    @pytest.mark.asyncio
    async def test_connection_pooling(self, chain_spec, mock_async_web3):
        """Test connection pooling under load"""
        with patch('web3.AsyncWeb3', return_value=mock_async_web3):
            connector = EnhancedChainConnector(max_workers=5)
            
            # Make multiple concurrent requests
            tasks = [
                connector.get_async_web3("test_chain")
                for _ in range(10)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # All requests should succeed and use cached connection
            assert all(result is not None for result in results)
            assert len(set(id(result) for result in results)) == 1  # Same instance
    
    @pytest.mark.asyncio
    async def test_metrics_persistence(self, chain_spec, mock_async_web3):
        """Test metrics persistence across requests"""
        with patch('web3.AsyncWeb3', return_value=mock_async_web3):
            connector = EnhancedChainConnector()
            
            # Make multiple requests
            for _ in range(5):
                web3 = await connector.get_async_web3("test_chain")
                await web3.eth.get_block_number()
                time.sleep(0.1)  # Simulate request time
            
            # Check metrics
            metrics = connector.get_metrics("test_chain")
            assert metrics.total_requests == 5
            assert len(metrics.block_times) > 0
            assert len(metrics.gas_prices) > 0
            
            # Metrics should persist after new requests
            web3 = await connector.get_async_web3("test_chain")
            await web3.eth.get_block_number()
            
            updated_metrics = connector.get_metrics("test_chain")
            assert updated_metrics.total_requests == 6
            assert len(updated_metrics.block_times) > len(metrics.block_times) 