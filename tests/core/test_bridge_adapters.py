import pytest
from unittest.mock import Mock, patch
from web3 import Web3
from eth_typing import ChecksumAddress
from decimal import Decimal

from src.core.bridge_adapter import BridgeConfig, BridgeState, BridgeMetrics
from src.core.debridge_adapter import DeBridgeAdapter
from src.core.superbridge_adapter import SuperbridgeAdapter
from src.core.across_adapter import AcrossAdapter
from src.core.mode_bridge_adapter import ModeBridgeAdapter
from src.core.sonic_bridge_adapter import SonicBridgeAdapter

@pytest.fixture
def web3_mock():
    """Mock Web3 instance"""
    mock = Mock(spec=Web3)
    mock.eth.chain_id = 1
    mock.eth.gas_price = 50000000000  # 50 gwei
    mock.eth.get_balance.return_value = 1000000000000000000  # 1 ETH
    return mock

@pytest.fixture
def bridge_config():
    """Basic bridge configuration"""
    return BridgeConfig(
        name="test_bridge",
        supported_chains=["ethereum", "base", "polygon"],
        min_amount=100.0,
        max_amount=1000000.0,
        fee_multiplier=1.0,
        gas_limit_multiplier=1.2,
        confirmation_blocks=1
    )

class TestDeBridgeAdapter:
    """Test suite for DeBridge adapter"""
    
    def test_validate_transfer_success(self, web3_mock, bridge_config):
        """Test successful transfer validation"""
        adapter = DeBridgeAdapter(bridge_config, web3_mock)
        
        result = adapter.validate_transfer(
            "ethereum",
            "base",
            "USDC",
            1000.0
        )
        
        assert result is True
    
    def test_validate_transfer_unsupported_chain(self, web3_mock, bridge_config):
        """Test validation with unsupported chain"""
        adapter = DeBridgeAdapter(bridge_config, web3_mock)
        
        result = adapter.validate_transfer(
            "ethereum",
            "unsupported",
            "USDC",
            1000.0
        )
        
        assert result is False
    
    def test_validate_transfer_amount_limits(self, web3_mock, bridge_config):
        """Test amount validation limits"""
        adapter = DeBridgeAdapter(bridge_config, web3_mock)
        
        # Test below minimum
        assert not adapter.validate_transfer(
            "ethereum",
            "base",
            "USDC",
            50.0
        )
        
        # Test above maximum
        assert not adapter.validate_transfer(
            "ethereum",
            "base",
            "USDC",
            2000000.0
        )
    
    def test_estimate_fees(self, web3_mock, bridge_config):
        """Test fee estimation"""
        adapter = DeBridgeAdapter(bridge_config, web3_mock)
        
        fees = adapter.estimate_fees(
            "ethereum",
            "base",
            "USDC",
            1000.0
        )
        
        assert isinstance(fees, dict)
        assert 'base_fee' in fees
        assert 'gas_fee' in fees
        assert 'total' in fees
        assert fees['total'] == fees['base_fee'] + fees['gas_fee']
        assert fees['total'] > 0
    
    def test_estimate_time(self, web3_mock, bridge_config):
        """Test transfer time estimation"""
        adapter = DeBridgeAdapter(bridge_config, web3_mock)
        
        time = adapter.estimate_time("ethereum", "base")
        
        assert isinstance(time, int)
        assert time > 0
    
    @patch('web3.eth.Eth.contract')
    def test_prepare_transfer(self, mock_contract, web3_mock, bridge_config):
        """Test transfer preparation"""
        adapter = DeBridgeAdapter(bridge_config, web3_mock)
        
        tx_params = adapter.prepare_transfer(
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
    
    def test_get_bridge_state(self, web3_mock, bridge_config):
        """Test bridge state monitoring"""
        adapter = DeBridgeAdapter(bridge_config, web3_mock)
        
        state = adapter.get_bridge_state("ethereum", "base")
        
        assert isinstance(state, BridgeState)
    
    def test_monitor_liquidity(self, web3_mock, bridge_config):
        """Test liquidity monitoring"""
        adapter = DeBridgeAdapter(bridge_config, web3_mock)
        
        liquidity = adapter.monitor_liquidity("ethereum", "USDC")
        
        assert isinstance(liquidity, float)
        assert liquidity >= 0
        assert adapter.metrics.liquidity == liquidity

class TestSuperbridgeAdapter:
    """Test suite for Superbridge adapter"""
    
    def test_validate_transfer_success(self, web3_mock, bridge_config):
        """Test successful transfer validation"""
        adapter = SuperbridgeAdapter(bridge_config, web3_mock)
        
        result = adapter.validate_transfer(
            "ethereum",
            "base",
            "USDC",
            1000.0
        )
        
        assert result is True
    
    def test_validate_transfer_unsupported_chain(self, web3_mock, bridge_config):
        """Test validation with unsupported chain"""
        adapter = SuperbridgeAdapter(bridge_config, web3_mock)
        
        result = adapter.validate_transfer(
            "ethereum",
            "unsupported",
            "USDC",
            1000.0
        )
        
        assert result is False
    
    def test_estimate_fees(self, web3_mock, bridge_config):
        """Test fee estimation"""
        adapter = SuperbridgeAdapter(bridge_config, web3_mock)
        
        fees = adapter.estimate_fees(
            "ethereum",
            "base",
            "USDC",
            1000.0
        )
        
        assert isinstance(fees, dict)
        assert 'lz_fee' in fees
        assert 'protocol_fee' in fees
        assert 'gas_fee' in fees
        assert 'total' in fees
        assert fees['total'] == fees['lz_fee'] + fees['protocol_fee'] + fees['gas_fee']
    
    def test_estimate_time(self, web3_mock, bridge_config):
        """Test transfer time estimation"""
        adapter = SuperbridgeAdapter(bridge_config, web3_mock)
        
        time = adapter.estimate_time("ethereum", "base")
        
        assert isinstance(time, int)
        assert time > 0
    
    @patch('web3.eth.Eth.contract')
    def test_prepare_transfer(self, mock_contract, web3_mock, bridge_config):
        """Test transfer preparation"""
        adapter = SuperbridgeAdapter(bridge_config, web3_mock)
        
        tx_params = adapter.prepare_transfer(
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
    
    def test_verify_message(self, web3_mock, bridge_config):
        """Test message verification"""
        adapter = SuperbridgeAdapter(bridge_config, web3_mock)
        
        result = adapter.verify_message(
            "ethereum",
            "base",
            "0x1234567890abcdef",
            b"test_proof"
        )
        
        assert isinstance(result, bool)
    
    def test_get_bridge_state_active(self, web3_mock, bridge_config):
        """Test bridge state monitoring - active state"""
        adapter = SuperbridgeAdapter(bridge_config, web3_mock)
        
        state = adapter.get_bridge_state("ethereum", "base")
        
        assert isinstance(state, BridgeState)
        assert state == BridgeState.ACTIVE
    
    def test_monitor_liquidity(self, web3_mock, bridge_config):
        """Test liquidity monitoring"""
        adapter = SuperbridgeAdapter(bridge_config, web3_mock)
        
        liquidity = adapter.monitor_liquidity("ethereum", "USDC")
        
        assert isinstance(liquidity, float)
        assert liquidity >= 0
        assert adapter.metrics.liquidity == liquidity

class TestAcrossAdapter:
    """Test suite for Across adapter"""
    
    def test_validate_transfer_success(self, web3_mock, bridge_config):
        """Test successful transfer validation"""
        adapter = AcrossAdapter(bridge_config, web3_mock)
        
        result = adapter.validate_transfer(
            "ethereum",
            "base",
            "USDC",
            1000.0
        )
        
        assert result is True
    
    def test_estimate_fees(self, web3_mock, bridge_config):
        """Test fee estimation"""
        adapter = AcrossAdapter(bridge_config, web3_mock)
        
        fees = adapter.estimate_fees(
            "ethereum",
            "base",
            "USDC",
            1000.0
        )
        
        assert isinstance(fees, dict)
        assert 'relayer_fee' in fees
        assert 'lp_fee' in fees
        assert 'gas_fee' in fees
        assert 'total' in fees
        assert fees['total'] == fees['relayer_fee'] + fees['lp_fee'] + fees['gas_fee']
    
    def test_estimate_time(self, web3_mock, bridge_config):
        """Test transfer time estimation"""
        adapter = AcrossAdapter(bridge_config, web3_mock)
        
        time = adapter.estimate_time("ethereum", "base")
        
        assert isinstance(time, int)
        assert time > 0
    
    @patch('web3.eth.Eth.contract')
    def test_prepare_transfer(self, mock_contract, web3_mock, bridge_config):
        """Test transfer preparation"""
        adapter = AcrossAdapter(bridge_config, web3_mock)
        
        tx_params = adapter.prepare_transfer(
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
    
    def test_verify_message(self, web3_mock, bridge_config):
        """Test message verification"""
        adapter = AcrossAdapter(bridge_config, web3_mock)
        
        result = adapter.verify_message(
            "ethereum",
            "base",
            "0x1234567890abcdef",
            b"test_proof"
        )
        
        assert isinstance(result, bool)
    
    def test_get_bridge_state(self, web3_mock, bridge_config):
        """Test bridge state monitoring"""
        adapter = AcrossAdapter(bridge_config, web3_mock)
        
        state = adapter.get_bridge_state("ethereum", "base")
        
        assert isinstance(state, BridgeState)
    
    def test_monitor_liquidity(self, web3_mock, bridge_config):
        """Test liquidity monitoring"""
        adapter = AcrossAdapter(bridge_config, web3_mock)
        
        liquidity = adapter.monitor_liquidity("ethereum", "USDC")
        
        assert isinstance(liquidity, float)
        assert liquidity >= 0
        assert adapter.metrics.liquidity == liquidity
    
    def test_calculate_relayer_fee(self, web3_mock, bridge_config):
        """Test relayer fee calculation"""
        adapter = AcrossAdapter(bridge_config, web3_mock)
        
        fee = adapter._calculate_relayer_fee(1000.0, "USDC", "base")
        
        assert isinstance(fee, float)
        assert fee > 0
        assert fee < 1000.0  # Fee should be less than amount
    
    def test_calculate_lp_fee(self, web3_mock, bridge_config):
        """Test LP fee calculation"""
        adapter = AcrossAdapter(bridge_config, web3_mock)
        
        fee = adapter._calculate_lp_fee(1000.0, "USDC", "base")
        
        assert isinstance(fee, float)
        assert fee > 0
        assert fee < 1000.0  # Fee should be less than amount

class TestModeBridgeAdapter:
    """Test suite for Mode bridge adapter"""
    
    def test_validate_transfer_success(self, web3_mock, bridge_config):
        """Test successful transfer validation"""
        adapter = ModeBridgeAdapter(bridge_config, web3_mock)
        
        result = adapter.validate_transfer(
            "ethereum",
            "mode",
            "USDC",
            1000.0
        )
        
        assert result is True
    
    def test_validate_transfer_unsupported_chain(self, web3_mock, bridge_config):
        """Test validation with unsupported chain"""
        adapter = ModeBridgeAdapter(bridge_config, web3_mock)
        
        result = adapter.validate_transfer(
            "mode",
            "unsupported",
            "USDC",
            1000.0
        )
        
        assert result is False
    
    def test_estimate_fees(self, web3_mock, bridge_config):
        """Test fee estimation"""
        adapter = ModeBridgeAdapter(bridge_config, web3_mock)
        
        fees = adapter.estimate_fees(
            "ethereum",
            "mode",
            "USDC",
            1000.0
        )
        
        assert isinstance(fees, dict)
        assert 'l1_da_cost' in fees
        assert 'l2_execution_cost' in fees
        assert 'bridge_fee' in fees
        assert 'total' in fees
        assert fees['total'] == fees['l1_da_cost'] + fees['l2_execution_cost'] + fees['bridge_fee']
    
    def test_estimate_time(self, web3_mock, bridge_config):
        """Test transfer time estimation"""
        adapter = ModeBridgeAdapter(bridge_config, web3_mock)
        
        time = adapter.estimate_time("ethereum", "mode")
        
        assert isinstance(time, int)
        assert time > 0
    
    @patch('web3.eth.Eth.contract')
    def test_prepare_transfer(self, mock_contract, web3_mock, bridge_config):
        """Test transfer preparation"""
        adapter = ModeBridgeAdapter(bridge_config, web3_mock)
        
        tx_params = adapter.prepare_transfer(
            "ethereum",
            "mode",
            "USDC",
            1000.0,
            "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
        )
        
        assert isinstance(tx_params, dict)
        assert 'to' in tx_params
        assert 'data' in tx_params
        assert 'value' in tx_params
    
    def test_verify_message(self, web3_mock, bridge_config):
        """Test message verification"""
        adapter = ModeBridgeAdapter(bridge_config, web3_mock)
        
        result = adapter.verify_message(
            "ethereum",
            "mode",
            "0x1234567890abcdef",
            b"test_proof"
        )
        
        assert isinstance(result, bool)
    
    def test_get_bridge_state(self, web3_mock, bridge_config):
        """Test bridge state monitoring"""
        adapter = ModeBridgeAdapter(bridge_config, web3_mock)
        
        state = adapter.get_bridge_state("ethereum", "mode")
        
        assert isinstance(state, BridgeState)
    
    def test_monitor_liquidity(self, web3_mock, bridge_config):
        """Test liquidity monitoring"""
        adapter = ModeBridgeAdapter(bridge_config, web3_mock)
        
        liquidity = adapter.monitor_liquidity("mode", "USDC")
        
        assert isinstance(liquidity, float)
        assert liquidity >= 0
        assert adapter.metrics.liquidity == liquidity

class TestSonicBridgeAdapter:
    """Test suite for Sonic bridge adapter"""
    
    def test_validate_transfer_success(self, web3_mock, bridge_config):
        """Test successful transfer validation"""
        adapter = SonicBridgeAdapter(bridge_config, web3_mock)
        
        result = adapter.validate_transfer(
            "ethereum",
            "sonic",
            "USDC",
            1000.0
        )
        
        assert result is True
    
    def test_validate_transfer_unsupported_chain(self, web3_mock, bridge_config):
        """Test validation with unsupported chain"""
        adapter = SonicBridgeAdapter(bridge_config, web3_mock)
        
        result = adapter.validate_transfer(
            "sonic",
            "unsupported",
            "USDC",
            1000.0
        )
        
        assert result is False
    
    def test_estimate_fees(self, web3_mock, bridge_config):
        """Test fee estimation"""
        adapter = SonicBridgeAdapter(bridge_config, web3_mock)
        
        fees = adapter.estimate_fees(
            "ethereum",
            "sonic",
            "USDC",
            1000.0
        )
        
        assert isinstance(fees, dict)
        assert 'bridge_fee' in fees
        assert 'gas_cost' in fees
        assert 'lp_fee' in fees
        assert 'total' in fees
        assert fees['total'] == fees['bridge_fee'] + fees['gas_cost'] + fees['lp_fee']
    
    def test_estimate_time(self, web3_mock, bridge_config):
        """Test transfer time estimation"""
        adapter = SonicBridgeAdapter(bridge_config, web3_mock)
        
        time = adapter.estimate_time("ethereum", "sonic")
        
        assert isinstance(time, int)
        assert time > 0
    
    @patch('web3.eth.Eth.contract')
    def test_prepare_transfer(self, mock_contract, web3_mock, bridge_config):
        """Test transfer preparation"""
        adapter = SonicBridgeAdapter(bridge_config, web3_mock)
        
        tx_params = adapter.prepare_transfer(
            "ethereum",
            "sonic",
            "USDC",
            1000.0,
            "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
        )
        
        assert isinstance(tx_params, dict)
        assert 'to' in tx_params
        assert 'data' in tx_params
        assert 'value' in tx_params
    
    def test_verify_message(self, web3_mock, bridge_config):
        """Test message verification"""
        adapter = SonicBridgeAdapter(bridge_config, web3_mock)
        
        result = adapter.verify_message(
            "ethereum",
            "sonic",
            "0x1234567890abcdef",
            b"test_proof"
        )
        
        assert isinstance(result, bool)
    
    def test_get_bridge_state(self, web3_mock, bridge_config):
        """Test bridge state monitoring"""
        adapter = SonicBridgeAdapter(bridge_config, web3_mock)
        
        state = adapter.get_bridge_state("ethereum", "sonic")
        
        assert isinstance(state, BridgeState)
    
    def test_monitor_liquidity(self, web3_mock, bridge_config):
        """Test liquidity monitoring"""
        adapter = SonicBridgeAdapter(bridge_config, web3_mock)
        
        liquidity = adapter.monitor_liquidity("sonic", "USDC")
        
        assert isinstance(liquidity, float)
        assert liquidity >= 0
        assert adapter.metrics.liquidity == liquidity
    
    def test_liquidity_pool_status(self, web3_mock, bridge_config):
        """Test liquidity pool status monitoring"""
        adapter = SonicBridgeAdapter(bridge_config, web3_mock)
        
        state = adapter.get_bridge_state("ethereum", "sonic")
        
        # If liquidity pool is configured, check its status
        if adapter.sonic_config.liquidity_pool:
            assert state in [BridgeState.ACTIVE, BridgeState.LOW_LIQUIDITY] 