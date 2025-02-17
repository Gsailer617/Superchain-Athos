import pytest
from unittest.mock import Mock, patch, AsyncMock
from web3 import Web3
from web3.types import Wei
from eth_typing import ChecksumAddress
from hexbytes import HexBytes
import time

from src.core.across_adapter import AcrossAdapter
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
        name="across",
        supported_chains=["ethereum", "base", "arbitrum", "optimism", "polygon"],
        min_amount=100.0,  # $100 minimum
        max_amount=1000000.0,  # $1M maximum
        across_config={
            'relayer_fee_pct': 0.04,
            'lp_fee_pct': 0.02,
            'verification_gas_limit': 2000000,
            'supported_tokens': {
                'USDC': {
                    'ethereum': '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
                    'base': '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913',
                    'arbitrum': '0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8',
                    'optimism': '0x7F5c764cBc14f9669B88837ca1490cCa17c31607',
                    'polygon': '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174'
                },
                'USDT': {
                    'ethereum': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
                    'base': '0x4200000000000000000000000000000000000000',
                    'arbitrum': '0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9',
                    'optimism': '0x94b008aA00579c1307B0EF2c499aD98a8ce58e58',
                    'polygon': '0xc2132D05D31c914a87C6611C10748AEb04B58e8F'
                }
            }
        }
    )

class TestAcrossAdapter:
    """Test suite for Across adapter"""
    
    def test_initialize_adapter(self, web3_mock, bridge_config):
        """Test adapter initialization"""
        adapter = AcrossAdapter(bridge_config, web3_mock)
        
        assert adapter.relayer_fee_pct == 0.04
        assert adapter.lp_fee_pct == 0.02
        assert adapter.verification_gas_limit == 2000000
    
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
    
    def test_validate_transfer_unsupported_chain(self, web3_mock, bridge_config):
        """Test transfer validation with unsupported chain"""
        adapter = AcrossAdapter(bridge_config, web3_mock)
        
        result = adapter.validate_transfer(
            "ethereum",
            "unsupported",
            "USDC",
            1000.0
        )
        
        assert result is False
    
    def test_validate_transfer_unsupported_token(self, web3_mock, bridge_config):
        """Test transfer validation with unsupported token"""
        adapter = AcrossAdapter(bridge_config, web3_mock)
        
        result = adapter.validate_transfer(
            "ethereum",
            "base",
            "INVALID",
            1000.0
        )
        
        assert result is False
    
    def test_validate_transfer_amount_limits(self, web3_mock, bridge_config):
        """Test transfer validation with amount limits"""
        adapter = AcrossAdapter(bridge_config, web3_mock)
        
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
        """Test time estimation"""
        adapter = AcrossAdapter(bridge_config, web3_mock)
        
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
        mock_contract.return_value.functions.deposit.return_value.build_transaction.return_value = {
            'to': '0x742d35Cc6634C0532925a3b844Bc454e4438f44e',
            'data': b'contract_data',
            'value': 1000000000000000000,  # 1 ETH
            'gas': 200000
        }
        
        adapter = AcrossAdapter(bridge_config, web3_mock)
        
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
        adapter = AcrossAdapter(bridge_config, web3_mock)
        
        result = adapter.verify_message(
            "ethereum",
            "base",
            "0x1234567890abcdef",
            b'proof_data'
        )
        
        assert isinstance(result, bool)
    
    def test_get_bridge_state_active(self, web3_mock, bridge_config):
        """Test getting active bridge state"""
        adapter = AcrossAdapter(bridge_config, web3_mock)
        
        state = adapter.get_bridge_state(
            "ethereum",
            "base"
        )
        
        assert state == BridgeState.ACTIVE
    
    @patch('web3.eth.Eth.contract')
    def test_monitor_liquidity(self, mock_contract, web3_mock, bridge_config):
        """Test liquidity monitoring"""
        # Mock contract functions
        mock_contract.return_value.functions.getPoolMetrics.return_value.call.return_value = {
            'totalLiquidity': 1000000000000000000,  # 1 ETH
            'utilization': 500000000000000000  # 0.5 ETH
        }
        
        adapter = AcrossAdapter(bridge_config, web3_mock)
        
        liquidity = adapter.monitor_liquidity(
            "ethereum",
            "USDC"
        )
        
        assert isinstance(liquidity, float)
        assert liquidity > 0
        assert adapter.metrics.utilization > 0
    
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
        
        adapter = AcrossAdapter(bridge_config, web3_mock)
        
        result = adapter.recover_failed_transfer(
            "ethereum",
            "base",
            "0x1234567890abcdef"
        )
        
        assert isinstance(result, str) or result is None 