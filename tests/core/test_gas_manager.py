import pytest
from unittest.mock import Mock, patch
from web3 import Web3, AsyncWeb3
from web3.types import Wei, TxParams
from decimal import Decimal
from typing import cast
from enum import Enum

from src.core.gas_manager import GasManager, GasMetrics
from src.config.chain_specs import (
    ChainSpec,
    GasModel,
    GasConfig,
    ChainConfig,
    GasFeeModel
)

# Define enums for testing
class GasModel(Enum):
    EIP1559 = "eip1559"
    LEGACY = "legacy"
    OPTIMISTIC = "optimistic"
    ARBITRUM = "arbitrum"

class GasFeeModel(Enum):
    EIP1559 = "eip1559"
    LEGACY = "legacy"

# Define config classes for testing
class GasConfig:
    def __init__(
        self,
        model: GasModel,
        base_fee_enabled: bool,
        priority_fee_enabled: bool,
        max_fee_per_gas: int,
        max_priority_fee_per_gas: int,
        gas_limit_multiplier: float,
        gas_price_multiplier: float
    ):
        self.model = model
        self.base_fee_enabled = base_fee_enabled
        self.priority_fee_enabled = priority_fee_enabled
        self.max_fee_per_gas = max_fee_per_gas
        self.max_priority_fee_per_gas = max_priority_fee_per_gas
        self.gas_limit_multiplier = gas_limit_multiplier
        self.gas_price_multiplier = gas_price_multiplier
        self.l1_fee_overhead = None
        self.l1_fee_scalar = None

class ChainConfig:
    def __init__(
        self,
        name: str,
        chain_id: int,
        native_currency: str,
        block_time: int,
        confirmation_blocks: int,
        gas_fee_model: GasFeeModel,
        rpc_urls: list[str],
        gas: GasConfig
    ):
        self.name = name
        self.chain_id = chain_id
        self.native_currency = native_currency
        self.block_time = block_time
        self.confirmation_blocks = confirmation_blocks
        self.gas_fee_model = gas_fee_model
        self.rpc_urls = rpc_urls
        self.gas = gas

@pytest.fixture
def chain_spec():
    """Test chain specification"""
    return ChainConfig(
        name="test_chain",
        chain_id=1,
        native_currency="ETH",
        block_time=12,
        confirmation_blocks=1,
        gas_fee_model=GasFeeModel.EIP1559,
        rpc_urls=["http://localhost:8545"],
        gas=GasConfig(
            model=GasModel.EIP1559,
            base_fee_enabled=True,
            priority_fee_enabled=True,
            max_fee_per_gas=500_000_000_000,  # 500 gwei
            max_priority_fee_per_gas=10_000_000_000,  # 10 gwei
            gas_limit_multiplier=1.2,
            gas_price_multiplier=1.1
        )
    )

@pytest.fixture
def mock_web3():
    """Mock Web3 instance"""
    mock = Mock(spec=Web3)
    mock.eth.chain_id = 1
    mock.eth.gas_price = 50_000_000_000  # 50 gwei
    mock.eth.max_priority_fee = 2_000_000_000  # 2 gwei
    mock.eth.block_number = 1000
    
    # Mock latest block
    mock.eth.get_block.return_value = {
        'baseFeePerGas': 40_000_000_000  # 40 gwei
    }
    
    return mock

@pytest.fixture
def mock_async_web3():
    """Mock AsyncWeb3 instance"""
    mock = Mock(spec=AsyncWeb3)
    mock.eth.chain_id = 1
    mock.eth.gas_price = 50_000_000_000  # 50 gwei
    mock.eth.max_priority_fee = 2_000_000_000  # 2 gwei
    mock.eth.block_number = 1000
    
    # Mock latest block
    mock.eth.get_block.return_value = {
        'baseFeePerGas': 40_000_000_000  # 40 gwei
    }
    
    return mock

class TestGasManager:
    """Test suite for gas manager"""
    
    def test_estimate_gas_price_eip1559(self, chain_spec, mock_web3):
        """Test gas price estimation for EIP-1559 chain"""
        manager = GasManager()
        
        max_fee, priority_fee = manager.estimate_gas_price(
            "test_chain",
            mock_web3
        )
        
        assert max_fee > 0
        assert priority_fee > 0
        assert max_fee >= priority_fee
    
    def test_estimate_gas_price_legacy(self, chain_spec, mock_web3):
        """Test gas price estimation for legacy chain"""
        # Modify chain spec to use legacy gas model
        chain_spec.gas.model = GasModel.LEGACY
        
        manager = GasManager()
        
        max_fee, priority_fee = manager.estimate_gas_price(
            "test_chain",
            mock_web3
        )
        
        assert max_fee > 0
        assert priority_fee == 0  # Legacy chains don't use priority fees
    
    def test_estimate_gas_price_optimistic(self, chain_spec, mock_web3):
        """Test gas price estimation for optimistic rollup"""
        # Modify chain spec to use optimistic gas model
        chain_spec.gas.model = GasModel.OPTIMISTIC
        chain_spec.gas.l1_fee_overhead = 2100
        chain_spec.gas.l1_fee_scalar = 1000000
        
        manager = GasManager()
        
        max_fee, priority_fee = manager.estimate_gas_price(
            "test_chain",
            mock_web3
        )
        
        assert max_fee > 0
        assert priority_fee == 0  # Optimistic rollups don't use priority fees
    
    def test_estimate_gas_price_arbitrum(self, chain_spec, mock_web3):
        """Test gas price estimation for Arbitrum"""
        # Modify chain spec to use Arbitrum gas model
        chain_spec.gas.model = GasModel.ARBITRUM
        
        manager = GasManager()
        
        max_fee, priority_fee = manager.estimate_gas_price(
            "test_chain",
            mock_web3
        )
        
        assert max_fee > 0
        assert priority_fee == 0  # Arbitrum doesn't use priority fees
    
    def test_estimate_gas_price_speed(self, chain_spec, mock_web3):
        """Test gas price estimation with different speeds"""
        manager = GasManager()
        
        # Test slow speed
        max_fee_slow, _ = manager.estimate_gas_price(
            "test_chain",
            mock_web3,
            speed="slow"
        )
        
        # Test standard speed
        max_fee_standard, _ = manager.estimate_gas_price(
            "test_chain",
            mock_web3,
            speed="standard"
        )
        
        # Test fast speed
        max_fee_fast, _ = manager.estimate_gas_price(
            "test_chain",
            mock_web3,
            speed="fast"
        )
        
        assert max_fee_slow < max_fee_standard < max_fee_fast
    
    def test_estimate_gas_price_invalid_speed(self, chain_spec, mock_web3):
        """Test gas price estimation with invalid speed"""
        manager = GasManager()
        
        with pytest.raises(ValueError):
            manager.estimate_gas_price(
                "test_chain",
                mock_web3,
                speed="invalid"
            )
    
    def test_estimate_gas_limit(self, chain_spec, mock_web3):
        """Test gas limit estimation"""
        manager = GasManager()
        
        # Mock transaction
        tx = cast(TxParams, {
            'to': '0x742d35Cc6634C0532925a3b844Bc454e4438f44e',
            'value': Wei(1000000000000000000),  # 1 ETH
            'data': b''
        })
        
        # Mock estimate_gas
        mock_web3.eth.estimate_gas.return_value = 21000
        
        gas_limit = manager.estimate_gas_limit(
            "test_chain",
            mock_web3,
            tx
        )
        
        assert isinstance(gas_limit, int)
        assert gas_limit > 21000  # Should include safety margin
    
    def test_estimate_gas_limit_contract(self, chain_spec, mock_web3):
        """Test gas limit estimation for contract interaction"""
        manager = GasManager()
        
        # Mock transaction with contract data
        tx = cast(TxParams, {
            'to': '0x742d35Cc6634C0532925a3b844Bc454e4438f44e',
            'value': Wei(0),
            'data': b'contract_data'
        })
        
        # Mock estimate_gas failure
        mock_web3.eth.estimate_gas.side_effect = Exception("Estimation failed")
        
        gas_limit = manager.estimate_gas_limit(
            "test_chain",
            mock_web3,
            tx
        )
        
        assert isinstance(gas_limit, int)
        assert gas_limit == 250000  # Default for contract interactions
    
    @pytest.mark.asyncio
    async def test_estimate_gas_price_async(self, chain_spec, mock_async_web3):
        """Test async gas price estimation"""
        manager = GasManager()
        
        max_fee, priority_fee = await manager.estimate_gas_price_async(
            "test_chain",
            mock_async_web3
        )
        
        assert max_fee > 0
        assert priority_fee >= 0
    
    @pytest.mark.asyncio
    async def test_estimate_gas_limit_async(self, chain_spec, mock_async_web3):
        """Test async gas limit estimation"""
        manager = GasManager()
        
        # Mock transaction
        tx = cast(TxParams, {
            'to': '0x742d35Cc6634C0532925a3b844Bc454e4438f44e',
            'value': Wei(1000000000000000000),  # 1 ETH
            'data': b''
        })
        
        # Mock estimate_gas
        mock_async_web3.eth.estimate_gas.return_value = 21000
        
        gas_limit = await manager.estimate_gas_limit_async(
            "test_chain",
            mock_async_web3,
            tx
        )
        
        assert isinstance(gas_limit, int)
        assert gas_limit > 21000  # Should include safety margin
    
    def test_gas_price_caching(self, chain_spec, mock_web3):
        """Test gas price caching"""
        manager = GasManager()
        
        # First call
        max_fee_1, _ = manager.estimate_gas_price(
            "test_chain",
            mock_web3
        )
        
        # Second call within cache duration
        max_fee_2, _ = manager.estimate_gas_price(
            "test_chain",
            mock_web3
        )
        
        # Should return same value from cache
        assert max_fee_1 == max_fee_2
        assert mock_web3.eth.get_block.call_count == 1  # Called only once 