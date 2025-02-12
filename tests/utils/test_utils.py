"""Test utilities for mocking without affecting live system"""

from unittest.mock import MagicMock, AsyncMock
from typing import Dict, Any, Optional, List, Tuple
from web3 import Web3
import torch
from datetime import datetime, timedelta
from src.market.analyzer import MarketAnalyzer
from src.ml.model import ArbitrageModel
import pandas as pd
import numpy as np
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
import random

def create_mock_web3(
    base_fee: int = 20000000000,  # 20 Gwei
    gas_used: int = 12000000,
    gas_limit: int = 30000000,
    gas_price: int = 50000000000,  # 50 Gwei
    priority_fee: int = 2000000000  # 2 Gwei
) -> MagicMock:
    """Create a mock Web3 instance for testing without affecting live Web3"""
    mock_web3 = MagicMock(spec=Web3)
    
    # Mock eth object
    mock_web3.eth = MagicMock()
    mock_web3.eth.get_block = AsyncMock(return_value={
        'baseFeePerGas': base_fee,
        'gasUsed': gas_used,
        'gasLimit': gas_limit,
        'number': 1000000,
        'timestamp': 1677777777
    })
    mock_web3.eth.gas_price = gas_price
    mock_web3.eth.max_priority_fee = priority_fee
    mock_web3.eth.get_transaction_count = AsyncMock(return_value=100)
    
    return mock_web3

def create_mock_market_data() -> Dict[str, Any]:
    """Create mock market data for testing that matches MarketDataType"""
    token_pair = ('0x1234...', '0x5678...')
    
    return {
        'token_pair': token_pair,
        'price': 100.0,
        'volume_24h': 1000000.0,
        'liquidity': 500000.0,
        'volatility': 0.1,
        'market_cap': 1000000000.0,
        'tvl': 500000000.0,
        'fees_24h': 5000.0,
        'gas_price': 50000000000,
        'block_time': 12.0,
        'network_load': 0.8,
        'pending_tx_count': 100,
        'primary_protocol': 'uniswap',
        'protocols': ['uniswap', 'sushiswap', 'balancer'],
        'tokens': [
            {'address': token_pair[0], 'symbol': 'TOKEN1', 'decimals': 18},
            {'address': token_pair[1], 'symbol': 'TOKEN2', 'decimals': 18}
        ],
        'pools': [
            {
                'address': '0xpool1...',
                'protocol': 'uniswap',
                'tokens': token_pair,
                'liquidity': 500000.0,
                'volume_24h': 100000.0
            }
        ],
        # Optional validation tensors
        'actual_market': torch.randn(16),
        'actual_path': torch.randn(32),
        'actual_risk': torch.randn(8),
        'actual_execution': torch.randn(4)
    }

def create_mock_metrics() -> Dict[str, Any]:
    """Create mock metrics data for testing"""
    timestamps = pd.date_range(start='2024-01-01', periods=100, freq='H')
    return {
        'timestamps': timestamps,
        'profit_loss': np.random.normal(100, 20, 100),
        'gas_used': np.random.uniform(50000, 150000, 100),
        'execution_time': np.random.uniform(0.1, 2.0, 100),
        'success_rate': np.random.uniform(0.8, 1.0, 100),
        'cpu_percent': np.random.uniform(10, 90, 100),
        'memory_mb': np.random.uniform(100, 1000, 100),
        'network_latency': np.random.uniform(0.01, 0.5, 100)
    }

def create_mock_prometheus_registry() -> CollectorRegistry:
    """Create mock Prometheus registry with test metrics"""
    registry = CollectorRegistry()
    
    # Transaction metrics
    Counter('arbitrage_transactions_total', 'Total transactions', registry=registry).inc(10)
    Counter('successful_arbitrage_transactions_total', 'Successful transactions', registry=registry).inc(8)
    
    # Performance metrics
    Gauge('current_gas_price_gwei', 'Current gas price', registry=registry).set(50)
    Histogram('transaction_execution_time_seconds', 'Execution time', registry=registry).observe(0.5)
    
    # System metrics
    Gauge('system_cpu_usage_percent', 'CPU usage', registry=registry).set(45)
    Gauge('system_memory_usage_bytes', 'Memory usage', registry=registry).set(500 * 1024 * 1024)
    
    return registry

def create_mock_trade_history(num_trades: int = 100) -> pd.DataFrame:
    """Create mock trade history data"""
    return pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=num_trades, freq='H'),
        'strategy': [f'strategy_{i%3}' for i in range(num_trades)],
        'token_pair': ['ETH-USDC', 'ETH-USDT', 'WBTC-USDC'] * (num_trades // 3 + 1),
        'dex': ['uniswap', 'sushiswap', 'baseswap'] * (num_trades // 3 + 1),
        'profit': np.random.normal(100, 20, num_trades),
        'gas_price': np.random.uniform(50, 150, num_trades),
        'execution_time': np.random.uniform(0.1, 2.0, num_trades),
        'success': [random.random() > 0.1 for _ in range(num_trades)]
    })

def create_mock_learning_insights() -> Dict[str, Any]:
    """Create mock learning insights data"""
    timestamps = pd.date_range(start='2024-01-01', periods=100, freq='H')
    return {
        'timestamps': timestamps,
        'learning_progress': np.cumsum(np.random.uniform(0, 0.1, 100)),
        'strategy_performance': np.random.normal(100, 10, 100),
        'feature_importance': {
            'price_volatility': 0.3,
            'volume': 0.2,
            'liquidity': 0.25,
            'gas_price': 0.15,
            'network_congestion': 0.1
        },
        'predicted_performance': np.random.normal(110, 5, 100),
        'actual_performance': np.random.normal(100, 15, 100),
        'anomaly_scores': [-1 if random.random() < 0.1 else 1 for _ in range(100)],
        'optimization_suggestions': [
            'Consider increasing gas price for faster execution',
            'Strategy_1 showing suboptimal performance',
            'High network congestion detected'
        ]
    }

def get_test_config(override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get test configuration without affecting live config"""
    base_config = {
        'gas': {
            'max_priority_fee': 2000000000,  # 2 Gwei
            'max_fee_per_gas': 100000000000,  # 100 Gwei
            'gas_limit_buffer': 1.2
        },
        'network': {
            'rpc_url': 'mock://localhost:8545',
            'chain_id': 8453,
            'name': 'base'
        },
        'validation': {
            'max_price_movement': 0.02,
            'min_liquidity_ratio': 0.8,
            'max_gas_increase': 1.5,
            'max_slippage': 0.01
        },
        'model': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'hidden_size': 128
        },
        'trading': {
            'token_pairs': [
                {'token0': '0x1234...', 'token1': '0x5678...'}
            ],
            'test_amounts': [0.1, 0.5, 1.0],
            'min_profit_threshold': 0.01
        },
        'monitoring': {
            'storage_path': 'tests/data/monitoring',
            'prometheus_port': 8002,
            'cache_enabled': True,
            'redis_host': 'localhost',
            'redis_port': 6379,
            'redis_db': 1,
            'max_memory_entries': 1000,
            'flush_interval': 10
        },
        'jaeger': {
            'host': 'localhost',
            'port': 6831
        },
        'sentry': {
            'dsn': None,  # Disable Sentry for tests
            'traces_sample_rate': 0.0
        }
    }
    
    if override:
        for key, value in override.items():
            if isinstance(value, dict) and key in base_config:
                base_config[key].update(value)
            else:
                base_config[key] = value
                
    return base_config 

class MockArbitrageModel(ArbitrageModel):
    """Mock model for testing that simulates real blockchain behavior"""
    def __init__(self, device: Optional[torch.device] = None):
        device = device or torch.device('cpu')
        super().__init__(device)
        
    def forward(self, x: torch.Tensor, market_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Mock forward pass that simulates real blockchain analysis"""
        # Always return a valid prediction for testing
        # In real blockchain this would analyze actual market conditions
        return {
            'predicted_profit': 0.1,  # Simulated 10% profit
            'confidence': 0.9,  # High confidence for testing
            'risk_score': 0.2,  # Low risk for testing
            'execution_strategy': [1, 0, 1, 0]  # Mock execution path
        }
        
    async def predict(self, features: torch.Tensor) -> Dict[str, Any]:
        """Mock prediction that simulates blockchain analysis"""
        # In real blockchain this would analyze real market data
        return {
            'predicted_profit': 0.1,
            'confidence': 0.9,
            'risk_score': 0.2,
            'market_analysis': {
                'liquidity': 1000000.0,
                'volume': 500000.0,
                'price_impact': 0.001
            }
        }

class MockMarketAnalyzer(MarketAnalyzer):
    """Mock MarketAnalyzer that simulates blockchain market analysis"""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with simulated blockchain data"""
        mock_web3 = create_mock_web3()
        super().__init__(web3=mock_web3, config=config)
        self.volatility_history: List[Tuple[float, float]] = [
            (datetime.now().timestamp(), 0.1),
            (datetime.now().timestamp() + 60, 0.12)
        ]
        self.model = MockArbitrageModel()
        
    async def analyze_market(self, token_pair: Tuple[str, str], amount: float) -> Dict[str, Any]:
        """Mock market analysis simulating blockchain data"""
        # Simulate what we'd get from a real blockchain
        return {
            'sentiment': torch.zeros(8, dtype=torch.long),  # Market sentiment
            'historical': torch.zeros((1, 8, 8), dtype=torch.float),  # Price history
            'cross_chain': torch.zeros(12, dtype=torch.long),  # Cross-chain data
            'mev': torch.zeros(8, dtype=torch.long),  # MEV protection
            'gas': torch.zeros(6, dtype=torch.float),  # Gas optimization
            'liquidity': torch.zeros(10, dtype=torch.float),  # Pool liquidity
            'token_economics': torch.zeros(12, dtype=torch.long),  # Token metrics
            'price': 100.0,  # Current token price
            'volume_24h': 1000000.0,  # 24h volume
            'volatility': 0.1,  # Price volatility
            'gas_price': 50000000000,  # Current gas price
            'market_impact': 0.001,  # Simulated market impact
            'pool_depth': 1000000.0  # Simulated pool depth
        }
        
    async def fetch_market_data(self) -> Dict[str, Any]:
        """Mock market data fetch simulating blockchain data"""
        # This simulates what we'd get from real blockchain queries
        return {
            'sentiment': torch.zeros(8, dtype=torch.long),
            'historical': torch.zeros((1, 8, 8), dtype=torch.float),
            'cross_chain': torch.zeros(12, dtype=torch.long),
            'mev': torch.zeros(8, dtype=torch.long),
            'gas': torch.zeros(6, dtype=torch.float),
            'liquidity': torch.zeros(10, dtype=torch.float),
            'token_economics': torch.zeros(12, dtype=torch.long),
            'token_pair': ('0x1234...', '0x5678...'),  # Mock token pair
            'price': 100.0,  # Mock current price
            'liquidity_value': 10000.0,  # Mock liquidity
            'gas_price': 50000000000,  # Mock gas price
            'market_impact': 0.001,  # Mock market impact
            'pool_depth': 1000000.0  # Mock pool depth
        } 