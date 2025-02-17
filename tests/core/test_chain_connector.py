import pytest
from unittest.mock import Mock, patch
from web3 import Web3, AsyncWeb3
from web3.providers.rpc import HTTPProvider
from web3.providers.websocket.websocket import WebsocketProvider

from src.core.chain_connector import ChainConnector, ConnectionMetrics
from src.config.chain_specs import ChainSpec, RPCConfig, NetworkConfig, BlockConfig

@pytest.fixture
def chain_spec():
    """Test chain specification"""
    return ChainSpec(
        name="test_chain",
        chain_id=1,
        native_currency="ETH",
        native_currency_decimals=18,
        rpc=RPCConfig(
            http=[
                "http://localhost:8545",
                "http://localhost:8546"
            ],
            ws=[
                "ws://localhost:8547"
            ],
            timeout=10,
            retry_count=3
        ),
        network=NetworkConfig(
            is_testnet=False,
            supports_eip1559=True
        ),
        block=BlockConfig(
            target_block_time=12.0,
            safe_confirmations=12
        ),
        is_l2=False
    )

@pytest.fixture
def mock_web3():
    """Mock Web3 instance"""
    mock = Mock(spec=Web3)
    mock.eth.chain_id = 1
    mock.is_connected.return_value = True
    return mock

@pytest.fixture
def mock_async_web3():
    """Mock AsyncWeb3 instance"""
    mock = Mock(spec=AsyncWeb3)
    mock.eth.chain_id = 1
    mock.is_connected.return_value = True
    return mock

class TestChainConnector:
    """Test suite for chain connector"""
    
    def test_get_web3_success(self, chain_spec, mock_web3):
        """Test successful Web3 connection"""
        with patch('web3.Web3', return_value=mock_web3):
            connector = ChainConnector()
            web3 = connector.get_web3("test_chain")
            
            assert web3 is not None
            assert web3.eth.chain_id == chain_spec.chain_id
    
    def test_get_web3_invalid_chain(self):
        """Test getting Web3 connection for invalid chain"""
        connector = ChainConnector()
        
        with pytest.raises(ValueError):
            connector.get_web3("invalid_chain")
    
    @pytest.mark.asyncio
    async def test_get_async_web3_success(self, chain_spec, mock_async_web3):
        """Test successful AsyncWeb3 connection"""
        with patch('web3.AsyncWeb3', return_value=mock_async_web3):
            connector = ChainConnector()
            web3 = await connector.get_async_web3("test_chain")
            
            assert web3 is not None
            assert web3.eth.chain_id == chain_spec.chain_id
    
    @pytest.mark.asyncio
    async def test_get_async_web3_invalid_chain(self):
        """Test getting AsyncWeb3 connection for invalid chain"""
        connector = ChainConnector()
        
        with pytest.raises(ValueError):
            await connector.get_async_web3("invalid_chain")
    
    def test_connection_caching(self, chain_spec, mock_web3):
        """Test that connections are cached"""
        with patch('web3.Web3', return_value=mock_web3):
            connector = ChainConnector()
            
            # Get connection twice
            web3_1 = connector.get_web3("test_chain")
            web3_2 = connector.get_web3("test_chain")
            
            # Should return same instance
            assert web3_1 is web3_2
    
    def test_connection_metrics(self, chain_spec, mock_web3):
        """Test connection metrics tracking"""
        with patch('web3.Web3', return_value=mock_web3):
            connector = ChainConnector()
            
            # Get connection and update metrics
            web3 = connector.get_web3("test_chain")
            connector.update_metrics("test_chain", latency=0.1)
            
            # Get metrics
            metrics = connector.get_metrics("test_chain")
            
            assert isinstance(metrics, ConnectionMetrics)
            assert metrics.latency == 0.1
            assert metrics.success_rate == 1.0
            assert metrics.total_requests == 1
    
    def test_failed_connection_metrics(self, chain_spec):
        """Test metrics for failed connections"""
        with patch('web3.Web3', side_effect=ConnectionError):
            connector = ChainConnector()
            
            # Try to get connection
            with pytest.raises(ConnectionError):
                connector.get_web3("test_chain")
            
            # Update metrics with error
            connector.update_metrics(
                "test_chain",
                success=False,
                error="Connection failed"
            )
            
            # Get metrics
            metrics = connector.get_metrics("test_chain")
            
            assert isinstance(metrics, ConnectionMetrics)
            assert metrics.success_rate == 0.0
            assert metrics.failed_requests == 1
            assert metrics.last_error == "Connection failed"
    
    @pytest.mark.asyncio
    async def test_check_all_connections(self, chain_spec, mock_async_web3):
        """Test checking all connections"""
        with patch('web3.AsyncWeb3', return_value=mock_async_web3):
            connector = ChainConnector()
            
            # Check connections
            results = await connector.check_all_connections()
            
            assert isinstance(results, dict)
            assert all(isinstance(status, bool) for status in results.values())
    
    def test_close_connections(self, chain_spec, mock_web3, mock_async_web3):
        """Test closing all connections"""
        with patch('web3.Web3', return_value=mock_web3), \
             patch('web3.AsyncWeb3', return_value=mock_async_web3):
            connector = ChainConnector()
            
            # Get some connections
            web3 = connector.get_web3("test_chain")
            
            # Close connections
            connector.close()
            
            # Check that connections are cleared
            assert not connector._connections
            assert not connector._async_connections
            assert not connector._metrics
    
    def test_provider_type_handling(self, chain_spec):
        """Test handling of different provider types"""
        connector = ChainConnector()
        
        # Test HTTP provider
        with patch('web3.providers.rpc.HTTPProvider') as mock_http:
            mock_http.return_value.close = Mock()
            with patch('web3.Web3') as mock_web3:
                mock_web3.return_value.provider = mock_http.return_value
                mock_web3.return_value.is_connected.return_value = True
                mock_web3.return_value.eth.chain_id = chain_spec.chain_id
                
                web3 = connector.get_web3("test_chain")
                assert isinstance(web3.provider, HTTPProvider)
        
        # Test WebSocket provider
        with patch('web3.providers.websocket.websocket.WebsocketProvider') as mock_ws:
            mock_ws.return_value.close = Mock()
            with patch('web3.Web3') as mock_web3:
                mock_web3.return_value.provider = mock_ws.return_value
                mock_web3.return_value.is_connected.return_value = True
                mock_web3.return_value.eth.chain_id = chain_spec.chain_id
                
                web3 = connector.get_web3("test_chain")
                assert isinstance(web3.provider, WebsocketProvider) 