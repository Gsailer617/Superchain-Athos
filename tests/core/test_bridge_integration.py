import pytest
from unittest.mock import Mock, patch
from web3 import Web3
from decimal import Decimal

from src.core.bridge_integration import BridgeIntegration
from src.core.bridge_adapter import BridgeState
from src.core.register_adapters import register_bridge_adapters

@pytest.fixture
def web3_mock():
    """Mock Web3 instance"""
    mock = Mock(spec=Web3)
    mock.eth.chain_id = 1
    mock.eth.gas_price = 50000000000  # 50 gwei
    mock.eth.get_balance.return_value = 1000000000000000000  # 1 ETH
    return mock

@pytest.fixture
def bridge_integration(web3_mock):
    """Initialize bridge integration with mock Web3"""
    register_bridge_adapters()  # Ensure adapters are registered
    return BridgeIntegration(web3_mock)

class TestBridgeIntegration:
    """Test suite for bridge integration"""
    
    def test_get_supported_bridges(self, bridge_integration):
        """Test getting supported bridges"""
        bridges = bridge_integration.get_supported_bridges(
            "ethereum",
            "base",
            "USDC"
        )
        
        assert isinstance(bridges, list)
        assert len(bridges) > 0
        for bridge_name, adapter in bridges:
            assert isinstance(bridge_name, str)
            assert hasattr(adapter, 'validate_transfer')
    
    def test_get_supported_bridges_invalid_chain(self, bridge_integration):
        """Test getting supported bridges with invalid chain"""
        bridges = bridge_integration.get_supported_bridges(
            "invalid_chain",
            "base",
            "USDC"
        )
        
        assert isinstance(bridges, list)
        assert len(bridges) == 0
    
    def test_get_optimal_bridge(self, bridge_integration):
        """Test getting optimal bridge"""
        bridge_name, adapter, info = bridge_integration.get_optimal_bridge(
            "ethereum",
            "base",
            "USDC",
            1000.0
        )
        
        assert isinstance(bridge_name, str)
        assert hasattr(adapter, 'prepare_transfer')
        assert isinstance(info, dict)
        assert 'estimated_time' in info
        assert 'fees' in info
        assert 'liquidity' in info
        assert 'score' in info
        assert 'score_breakdown' in info
    
    def test_get_optimal_bridge_with_constraints(self, bridge_integration):
        """Test getting optimal bridge with time and cost constraints"""
        bridge_name, adapter, info = bridge_integration.get_optimal_bridge(
            "ethereum",
            "base",
            "USDC",
            1000.0,
            max_time=3600,  # 1 hour
            max_cost=0.1  # 0.1 ETH
        )
        
        if bridge_name:
            assert info['estimated_time'] <= 3600
            assert info['fees']['total'] <= 0.1
    
    def test_prepare_bridge_transfer(self, bridge_integration):
        """Test preparing bridge transfer"""
        tx_params, info = bridge_integration.prepare_bridge_transfer(
            "ethereum",
            "base",
            "USDC",
            1000.0,
            "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
        )
        
        assert isinstance(tx_params, dict)
        assert 'to' in tx_params
        assert 'data' in tx_params
        assert 'value' in tx_params
        assert isinstance(info, dict)
        assert 'bridge' in info
        assert 'estimated_time' in info
        assert 'fees' in info
        assert 'state' in info
    
    def test_prepare_bridge_transfer_specific_bridge(self, bridge_integration):
        """Test preparing bridge transfer with specific bridge"""
        tx_params, info = bridge_integration.prepare_bridge_transfer(
            "ethereum",
            "base",
            "USDC",
            1000.0,
            "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
            bridge_name="debridge"
        )
        
        assert isinstance(tx_params, dict)
        assert info['bridge'] == "debridge"
    
    def test_prepare_bridge_transfer_invalid_bridge(self, bridge_integration):
        """Test preparing bridge transfer with invalid bridge"""
        tx_params, info = bridge_integration.prepare_bridge_transfer(
            "ethereum",
            "base",
            "USDC",
            1000.0,
            "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
            bridge_name="invalid_bridge"
        )
        
        assert tx_params is None
        assert 'error' in info
    
    def test_monitor_bridge_transfer(self, bridge_integration):
        """Test monitoring bridge transfer"""
        status = bridge_integration.monitor_bridge_transfer(
            "ethereum",
            "base",
            "0x1234567890abcdef",
            "debridge"
        )
        
        assert isinstance(status, dict)
        assert 'source_status' in status
        assert 'bridge_state' in status
        assert 'message_verified' in status
        assert 'confirmations' in status
    
    def test_monitor_bridge_transfer_invalid_bridge(self, bridge_integration):
        """Test monitoring bridge transfer with invalid bridge"""
        status = bridge_integration.monitor_bridge_transfer(
            "ethereum",
            "base",
            "0x1234567890abcdef",
            "invalid_bridge"
        )
        
        assert isinstance(status, dict)
        assert 'error' in status
    
    @patch('src.core.bridge_adapter.BridgeAdapter.get_bridge_state')
    def test_bridge_state_handling(self, mock_state, bridge_integration):
        """Test handling of different bridge states"""
        # Test each bridge state
        for state in BridgeState:
            mock_state.return_value = state
            
            bridge_name, adapter, info = bridge_integration.get_optimal_bridge(
                "ethereum",
                "base",
                "USDC",
                1000.0
            )
            
            if state == BridgeState.ACTIVE:
                assert bridge_name is not None
                assert adapter is not None
            else:
                assert bridge_name is None or info.get('score', 0) == float('-inf')
    
    def test_bridge_scoring(self, bridge_integration):
        """Test bridge scoring logic"""
        _, _, info = bridge_integration.get_optimal_bridge(
            "ethereum",
            "base",
            "USDC",
            1000.0
        )
        
        if 'score_breakdown' in info:
            scores = info['score_breakdown']
            assert 'cost_score' in scores
            assert 'time_score' in scores
            assert 'liquidity_score' in scores
            assert all(0 <= score <= 1 for score in scores.values())
    
    @pytest.mark.parametrize("chain_pair", [
        ("ethereum", "base"),
        ("ethereum", "polygon"),
        ("base", "polygon"),
        ("arbitrum", "optimism"),
        ("polygon", "bnb")
    ])
    def test_cross_chain_compatibility(self, bridge_integration, chain_pair):
        """Test bridge compatibility across different chain pairs"""
        source_chain, target_chain = chain_pair
        bridges = bridge_integration.get_supported_bridges(
            source_chain,
            target_chain,
            "USDC"
        )
        
        assert isinstance(bridges, list)
        if bridges:
            bridge_name, adapter = bridges[0]
            assert bridge_name in ["debridge", "superbridge", "across"] 