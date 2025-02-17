import pytest
from unittest.mock import Mock, patch, AsyncMock
from web3 import Web3, AsyncWeb3
from web3.types import Wei
from eth_typing import ChecksumAddress
from hexbytes import HexBytes
import time

from src.core.stargate_adapter import StargateAdapter
from src.core.bridge_adapter import BridgeConfig, BridgeState

@pytest.fixture
def web3_mock():
    """Mock Web3 instance"""
    mock = Mock(spec=Web3)
    mock.eth.chain_id = 1
    mock.eth.gas_price = 50000000000  # 50 gwei
    mock.eth.max_priority_fee = 2000000000  # 2 gwei
    mock.eth.block_number = 1000
    mock.eth.default_account = "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
    mock.is_connected.return_value = True
    
    # Mock latest block
    mock.eth.get_block.return_value = {
        'baseFeePerGas': 40000000000  # 40 gwei
    }
    
    return mock

@pytest.fixture
def bridge_config():
    """Test bridge configuration"""
    return BridgeConfig(
        name="stargate",
        supported_chains=["ethereum", "base", "arbitrum", "optimism", "polygon"],
        min_amount=100.0,  # $100 minimum
        max_amount=1000000.0,  # $1M maximum
        stargate_config={
            'pool_ids': {
                'USDC': 1,
                'USDT': 2,
                'ETH': 3
            },
            'router_version': 'v2'
        }
    )

class TestStargateAdapter:
    """Test suite for Stargate adapter"""
    
    def test_initialize_adapter(self, web3_mock, bridge_config):
        """Test adapter initialization"""
        adapter = StargateAdapter(bridge_config, web3_mock)
        
        assert adapter.stargate_config['pool_ids']['USDC'] == 1
        assert adapter.stargate_config['pool_ids']['USDT'] == 2
        assert adapter.stargate_config['pool_ids']['ETH'] == 3
        assert adapter.stargate_config['router_version'] == 'v2'
    
    def test_validate_transfer_success(self, web3_mock, bridge_config):
        """Test successful transfer validation"""
        adapter = StargateAdapter(bridge_config, web3_mock)
        
        result = adapter.validate_transfer(
            "ethereum",
            "base",
            "USDC",
            1000.0
        )
        
        assert result is True
    
    def test_validate_transfer_unsupported_chain(self, web3_mock, bridge_config):
        """Test transfer validation with unsupported chain"""
        adapter = StargateAdapter(bridge_config, web3_mock)
        
        result = adapter.validate_transfer(
            "ethereum",
            "unsupported",
            "USDC",
            1000.0
        )
        
        assert result is False
    
    def test_validate_transfer_unsupported_token(self, web3_mock, bridge_config):
        """Test transfer validation with unsupported token"""
        adapter = StargateAdapter(bridge_config, web3_mock)
        
        result = adapter.validate_transfer(
            "ethereum",
            "base",
            "INVALID",
            1000.0
        )
        
        assert result is False
    
    def test_validate_transfer_amount_limits(self, web3_mock, bridge_config):
        """Test transfer validation with amount limits"""
        adapter = StargateAdapter(bridge_config, web3_mock)
        
        # Test below minimum
        result = adapter.validate_transfer(
            "ethereum",
            "base",
            "USDC",
            10.0  # Below min_amount
        )
        assert result is False
        
        # Test above maximum
        result = adapter.validate_transfer(
            "ethereum",
            "base",
            "USDC",
            2000000.0  # Above max_amount
        )
        assert result is False
    
    def test_estimate_fees(self, web3_mock, bridge_config):
        """Test fee estimation"""
        adapter = StargateAdapter(bridge_config, web3_mock)
        
        fees = adapter.estimate_fees(
            "ethereum",
            "base",
            "USDC",
            1000.0
        )
        
        assert isinstance(fees, dict)
        assert 'lz_fee' in fees
        assert 'protocol_fee' in fees
        assert 'lp_fee' in fees
        assert 'total' in fees
        assert fees['total'] == fees['lz_fee'] + fees['protocol_fee'] + fees['lp_fee']
    
    def test_estimate_time(self, web3_mock, bridge_config):
        """Test time estimation"""
        adapter = StargateAdapter(bridge_config, web3_mock)
        
        time = adapter.estimate_time(
            "ethereum",
            "base"
        )
        
        assert isinstance(time, int)
        assert time > 0
    
    @patch('web3.eth.Eth.contract')
    def test_prepare_transfer(self, mock_contract, web3_mock, bridge_config):
        """Test transfer preparation"""
        # Mock contract functions
        mock_contract.return_value.functions.swap.return_value.build_transaction.return_value = {
            'to': '0x742d35Cc6634C0532925a3b844Bc454e4438f44e',
            'data': b'contract_data',
            'value': 1000000000000000000,  # 1 ETH
            'gas': 200000
        }
        
        adapter = StargateAdapter(bridge_config, web3_mock)
        
        tx = adapter.prepare_transfer(
            "ethereum",
            "base",
            "USDC",
            1000.0,
            "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
        )
        
        assert isinstance(tx, dict)
        assert 'to' in tx
        assert 'data' in tx
        assert 'value' in tx
        assert 'gas' in tx
    
    def test_verify_message(self, web3_mock, bridge_config):
        """Test message verification"""
        adapter = StargateAdapter(bridge_config, web3_mock)
        
        result = adapter.verify_message(
            "ethereum",
            "base",
            "0x1234567890abcdef",
            b'proof_data'
        )
        
        assert isinstance(result, bool)
    
    def test_get_bridge_state_active(self, web3_mock, bridge_config):
        """Test getting active bridge state"""
        adapter = StargateAdapter(bridge_config, web3_mock)
        
        state = adapter.get_bridge_state(
            "ethereum",
            "base"
        )
        
        assert state == BridgeState.ACTIVE
    
    @patch('web3.eth.Eth.contract')
    def test_monitor_liquidity(self, mock_contract, web3_mock, bridge_config):
        """Test liquidity monitoring"""
        # Mock contract functions
        mock_contract.return_value.functions.totalLiquidity.return_value.call.return_value = 1000000000000000000  # 1 ETH
        
        adapter = StargateAdapter(bridge_config, web3_mock)
        
        liquidity = adapter.monitor_liquidity(
            "ethereum",
            "USDC"
        )
        
        assert isinstance(liquidity, float)
        assert liquidity > 0
    
    @patch('web3.eth.Eth.get_transaction')
    def test_recover_failed_transfer(self, mock_get_tx, web3_mock, bridge_config):
        """Test transfer recovery"""
        # Mock transaction data
        mock_get_tx.return_value = {
            'input': b'transaction_data',
            'nonce': 1,
            'gasPrice': 50000000000,
            'hash': HexBytes('0x1234567890abcdef'),
            'value': 1000000000000000000
        }
        
        adapter = StargateAdapter(bridge_config, web3_mock)
        
        result = adapter.recover_failed_transfer(
            "ethereum",
            "base",
            "0x1234567890abcdef"
        )
        
        assert isinstance(result, str) or result is None 