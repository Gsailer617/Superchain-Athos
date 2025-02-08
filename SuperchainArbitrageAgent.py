import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from web3 import Web3
from typing import List, Dict, Tuple, Optional, Union, Callable, Any, AsyncGenerator, Coroutine, TypeVar, Literal
import asyncio
import logging
from datetime import datetime, timedelta
from eth_typing import Address, HexStr, ChecksumAddress
from web3.contract import Contract
from web3.exceptions import ContractLogicError, TransactionNotFound
import aiohttp
import json
import os
from dotenv import load_dotenv
from eth_abi.abi import encode, decode
from eth_utils.address import to_checksum_address
from eth_utils.currency import to_wei, from_wei
import traceback
from concurrent.futures import ThreadPoolExecutor
import time
from decimal import Decimal
import gc
from tenacity import retry, stop_after_attempt, wait_exponential
import threading
import networkx as nx
from functools import wraps
from dataclasses import dataclass
from cachetools import cached, TTLCache
import signal
import sys
import statistics
from web3.types import TxReceipt, Wei
from aiohttp import ClientSession
import math
from typing import Literal
from enum import Enum, auto
from src.core.web3_config import get_web3, get_async_web3

# Type Aliases
TokenAddress = str
DexName = str
FlashLoanProviderType = Literal['aave', 'balancer', 'radiant']
TokenPair = Tuple[TokenAddress, TokenAddress]

MarketDataType = Dict[str, Union[float, str, Dict[str, Any]]]
OpportunityType = Dict[str, Union[str, float, TokenPair, Dict[str, Any]]]
FlashLoanOpportunityType = Dict[str, Union[
    str,  # id, type, timestamp
    TokenPair,  # token_pair
    float,  # amount, provider_score, fees, profits
    FlashLoanProviderType,  # flash_loan_provider
    Dict[str, Any]  # provider_metrics
]]

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Custom Exception Classes
class ArbitrageError(Exception):
    """Base exception class for arbitrage operations"""
    pass

class ValidationError(ArbitrageError):
    """Validation related errors"""
    pass

class ExecutionError(ArbitrageError):
    """Execution related errors"""
    pass

class NetworkError(ArbitrageError):
    """Network and connection related errors"""
    pass

class ConfigurationError(ArbitrageError):
    """Configuration and setup related errors"""
    pass

def retry_async(retries: int = 3, delay: int = 1):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == retries - 1:
                        raise
                    await asyncio.sleep(delay * (2 ** attempt))
            return None
        return wrapper
    return decorator

def validate_input(validator: Callable) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            if not await validator(*args, **kwargs):
                raise ValidationError("Input validation failed")
            return await func(*args, **kwargs)
        return wrapper
    return decorator

class TimeSeriesFeatures:
    def __init__(self, window_size: int = 100, cleanup_threshold: int = 1000):
        self.window_size = window_size
        self.cleanup_threshold = cleanup_threshold
        self.price_history = []
        self.volume_history = []
        self.last_cleanup = datetime.now()
        
    def update(self, price: float, volume: float):
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        # Enforce window size limit
        if len(self.price_history) > self.window_size:
            self.price_history = self.price_history[-self.window_size:]
            self.volume_history = self.volume_history[-self.window_size:]
        
        # Periodic cleanup
        if len(self.price_history) > self.cleanup_threshold:
            self._cleanup()
            
    def _cleanup(self):
        if (datetime.now() - self.last_cleanup).total_seconds() < 3600:  # Once per hour
            return
            
        self.price_history = self.price_history[-self.window_size:]
        self.volume_history = self.volume_history[-self.window_size:]
        self.last_cleanup = datetime.now()
        
    def get_features(self) -> torch.Tensor:
        if len(self.price_history) < 2:
            return torch.zeros(5)
            
        returns = np.diff(self.price_history) / self.price_history[:-1]
        volatility = np.std(returns) * np.sqrt(365 * 24 * 60)
        volume_ma = np.mean(self.volume_history[-24:]) if len(self.volume_history) >= 24 else 0
        volume_trend = volume_ma / np.mean(self.volume_history) if len(self.volume_history) > 0 else 0
        
        return torch.tensor([
            volatility,
            volume_trend,
            self._calculate_momentum(),
            self._calculate_relative_strength(),
            self._calculate_liquidity_depth()
        ])
        
    def _calculate_momentum(self) -> float:
        if len(self.price_history) < 2:
            return 0
        return (self.price_history[-1] / self.price_history[0]) - 1
        
    def _calculate_relative_strength(self) -> float:
        if len(self.price_history) < 14:
            return 0.0
        gains = []
        losses = []
        for i in range(1, len(self.price_history)):
            change = self.price_history[i] - self.price_history[i-1]
            if change >= 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(-change)
        avg_gain = float(np.mean(gains[-14:]))
        avg_loss = float(np.mean(losses[-14:]))
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))
        
    def _calculate_liquidity_depth(self) -> float:
        if len(self.volume_history) < 24:
            return 0.0
        mean_volume = float(np.mean(self.volume_history[-24:]))
        std_volume = float(np.std(self.volume_history[-24:]))
        return mean_volume / std_volume if std_volume > 0 else 0.0

class CrossChainAnalyzer:
    def __init__(self):
        self.eth_correlation = []
        self.l2_metrics = {}
        
    def analyze_opportunities(self, token_pair: Tuple[str, str]) -> Dict:
        return {
            'eth_correlation': self._calculate_eth_correlation(),
            'l2_gas_efficiency': self._calculate_l2_efficiency(),
            'bridge_liquidity': self._get_bridge_liquidity(),
            'cross_chain_volume': self._get_cross_chain_volume()
        }
        
    def _calculate_eth_correlation(self) -> float:
        if len(self.eth_correlation) < 2:
            return 0
        return np.corrcoef(self.eth_correlation)[0, 1]
        
    def _calculate_l2_efficiency(self) -> float:
        gas_savings = 0.85  # Base typically saves 85% on gas compared to L1
        return gas_savings * self._get_l2_multiplier()
        
    def _get_bridge_liquidity(self) -> float:
        return 0.0  # Implement bridge liquidity analysis
        
    def _get_cross_chain_volume(self) -> float:
        return 0.0  # Implement cross-chain volume analysis
        
    def _get_l2_multiplier(self) -> float:
        return 1.2  # Base has good efficiency due to OP Stack

class MEVProtection:
    def __init__(self):
        self.sandwich_threshold = 0.02  # 2% price impact threshold
        self.frontrun_threshold = 0.01  # 1% slippage threshold
        
    def calculate_mev_risk(self, trade_params: Dict) -> Dict:
        """Calculate MEV risk metrics for a trade"""
        return {
            'sandwich_risk': self._estimate_sandwich_risk(trade_params),
            'frontrunning_risk': self._estimate_frontrunning_risk(trade_params),
            'backrunning_risk': self._estimate_backrunning_risk(trade_params),
            'optimal_block_position': self._calculate_optimal_block_position(trade_params)
        }
        
    def _estimate_sandwich_risk(self, trade_params: Dict) -> float:
        return min(trade_params.get('price_impact', 0) / self.sandwich_threshold, 1.0)
        
    def _estimate_frontrunning_risk(self, trade_params: Dict) -> float:
        return min(trade_params.get('slippage', 0) / self.frontrun_threshold, 1.0)
        
    def _estimate_backrunning_risk(self, trade_params: Dict) -> float:
        volume = trade_params.get('volume_24h', 0)
        return 1.0 / (1.0 + volume/1e6) if volume > 0 else 1.0
        
    def _calculate_optimal_block_position(self, trade_params: Dict) -> int:
        gas_price = trade_params.get('gas_price', 0)
        return 1 if gas_price > 100 else 0  # First position if gas price is high

class GasOptimizer:
    def __init__(self):
        self.base_fee_history = []
        self.priority_fee_history = []
        
    def optimize_execution(self, trade_params: Dict) -> Dict:
        return {
            'optimal_gas_price': self._calculate_optimal_gas_price(),
            'base_fee_prediction': self._predict_base_fee_next_blocks(),
            'priority_fee_strategy': self._calculate_priority_fee(),
            'block_space_analysis': self._analyze_block_space()
        }
        
    def _calculate_optimal_gas_price(self) -> int:
        return max(self.base_fee_history[-1] * 1.2 if self.base_fee_history else 0, 1)
        
    def _predict_base_fee_next_blocks(self) -> List[int]:
        return self.base_fee_history[-5:] if self.base_fee_history else []
        
    def _calculate_priority_fee(self) -> int:
        return int(np.mean(self.priority_fee_history[-10:]) if self.priority_fee_history else 1)
        
    def _analyze_block_space(self) -> Dict:
        return {'utilization': 0.8, 'congestion': 'medium'}

class TokenEconomicsAnalyzer:
    def __init__(self):
        self.supply_history = {}
        self.holder_data = {}
        
    def analyze_token_metrics(self, token: str) -> Dict:
        return {
            'supply_dynamics': self._analyze_supply_changes(token),
            'holder_concentration': self._analyze_holder_distribution(token),
            'vesting_schedules': self._track_vesting_events(token),
            'protocol_revenue': self._analyze_protocol_revenue(token)
        }
        
    def _analyze_supply_changes(self, token: str) -> Dict:
        return {'inflation_rate': 0.0, 'burn_rate': 0.0, 'circulating_ratio': 0.8}
        
    def _analyze_holder_distribution(self, token: str) -> Dict:
        return {'top_holders_share': 0.0, 'gini_coefficient': 0.5}
        
    def _track_vesting_events(self, token: str) -> List:
        return []
        
    def _analyze_protocol_revenue(self, token: str) -> Dict:
        return {'revenue_30d': 0.0, 'revenue_growth': 0.0}

class ArbitrageNetwork(nn.Module):
    """Enhanced neural network for predicting arbitrage opportunities"""
    def __init__(self, input_size: int = 24, hidden_size: int = 128):
        super().__init__()
        # Main prediction network with enhanced features
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 3)  # [profit_prediction, confidence, risk_score]
        )
        
        # Market Sentiment Analysis
        self.sentiment_network = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16)
        )
        
        # Historical Performance Analysis
        self.historical_lstm = nn.LSTM(
            input_size=8,
            hidden_size=32,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )
        
        # Cross-Chain Analysis
        self.cross_chain_network = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16)
        )
        
        # MEV Protection
        self.mev_protection = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 4)  # [sandwich_risk, frontrun_risk, backrun_risk, block_position]
        )
        
        # Gas Optimization
        self.gas_optimizer = nn.Sequential(
            nn.Linear(6, 24),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(24, 4)  # [optimal_gas, base_fee_pred, priority_fee, block_space]
        )
        
        # Liquidity Analysis
        self.liquidity_analyzer = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16)
        )
        
        # Token Economics
        self.token_economics = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 8)
        )
        
    def forward(self, x, market_data=None):
        """Forward pass with market data validation"""
        if market_data is None:
            raise ValueError("market_data is required for prediction")
            
        # Validate required features
        required_features = {
            'sentiment': (8,),
            'historical': (1, 8, 8),
            'cross_chain': (12,),
            'mev': (8,),
            'gas': (6,),
            'liquidity': (10,),
            'token_economics': (12,)
        }
        
        for feature, expected_shape in required_features.items():
            if feature not in market_data:
                raise ValueError(f"Missing required feature: {feature}")
            if market_data[feature].shape[-len(expected_shape):] != expected_shape:
                raise ValueError(f"Invalid shape for {feature}: expected {expected_shape}, got {market_data[feature].shape}")
        
        # Extract features from different components
        sentiment_features = self.sentiment_network(market_data['sentiment'])
        
        # Process historical data through LSTM
        historical_output, _ = self.historical_lstm(market_data['historical'])
        historical_features = historical_output[:, -1, :]  # Take last output
        
        # Cross-chain analysis
        cross_chain_features = self.cross_chain_network(market_data['cross_chain'])
        
        # MEV protection analysis
        mev_features = self.mev_protection(market_data['mev'])
        
        # Gas optimization
        gas_features = self.gas_optimizer(market_data['gas'])
        
        # Liquidity analysis
        liquidity_features = self.liquidity_analyzer(market_data['liquidity'])
        
        # Token economics
        token_features = self.token_economics(market_data['token_economics'])
        
        # Combine all features
        combined_features = torch.cat([
            x,
            sentiment_features,
            historical_features,
            cross_chain_features,
            mev_features,
            gas_features,
            liquidity_features,
            token_features
        ], dim=1)
        
        # Get predictions
        predictions = self.network(combined_features)
        
        return predictions

class DefiLlamaIntegration:
    def __init__(self):
        self.client = DefiLlama()
        self.cache_duration = 300  # 5 minutes cache
        self.tvl_cache = {}
        self.volume_cache = {}
        
    async def get_protocol_data(self, protocol_slug: str) -> Dict:
        """Get comprehensive protocol data from DeFiLlama"""
        try:
            current_time = datetime.now().timestamp()
            if (protocol_slug in self.tvl_cache and 
                current_time - self.tvl_cache[protocol_slug]['timestamp'] < self.cache_duration):
                return self.tvl_cache[protocol_slug]['data']
            
            protocol_data = await self.client.get_protocol(protocol_slug)
            tvl_data = await self.client.get_protocol_tvl(protocol_slug)
            
            data = {
                'tvl': protocol_data.get('tvl', 0),
                'volume_24h': protocol_data.get('volume24h', 0),
                'fees_24h': protocol_data.get('fees24h', 0),
                'mcap_tvl': protocol_data.get('mcapTvl', 0),
                'historical_tvl': tvl_data,
                'timestamp': current_time
            }
            
            self.tvl_cache[protocol_slug] = {
                'data': data,
                'timestamp': current_time
            }
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching DeFiLlama data for {protocol_slug}: {str(e)}")
            return None
            
    async def get_pool_data(self, pool_address: str, chain: str = 'base') -> Dict:
        """Get pool-specific data from DeFiLlama"""
        try:
            pool_data = await self.client.get_pool(pool_address, chain)
            return {
                'liquidity': pool_data.get('liquidity', 0),
                'apy': pool_data.get('apy', 0),
                'volume_24h': pool_data.get('volume24h', 0),
                'fee_apy': pool_data.get('feeApy', 0)
            }
        except Exception as e:
            logger.error(f"Error fetching pool data for {pool_address}: {str(e)}")
            return None
            
    async def get_token_data(self, token_address: str, chain: str = 'base') -> Dict:
        """Get token-specific data from DeFiLlama"""
        try:
            token_data = await self.client.get_token(token_address, chain)
            return {
                'price': token_data.get('price', 0),
                'market_cap': token_data.get('marketCap', 0),
                'volume_24h': token_data.get('volume24h', 0),
                'liquidity': token_data.get('liquidity', 0)
            }
        except Exception as e:
            logger.error(f"Error fetching token data: {str(e)}")
            return None

class RateLimiter:
    def __init__(self, calls_per_second: float = 1.0):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = 0.0
        self.lock = asyncio.Lock()
        
    async def wait(self):
        async with self.lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time
            if time_since_last_call < self.min_interval:
                await asyncio.sleep(self.min_interval - time_since_last_call)
            self.last_call_time = time.time()

class GasManager:
    def __init__(self, web3: Web3):
        self.web3 = web3
        self.gas_price_history = []
        self.max_history = 1000
        self.update_interval = 10  # seconds
        self.last_update = 0
        
    async def get_optimal_gas_price(self) -> int:
        current_time = time.time()
        if current_time - self.last_update > self.update_interval:
            try:
                base_fee = self.web3.eth.get_block('latest')['baseFeePerGas']
                max_priority_fee = self.web3.eth.max_priority_fee
                gas_price = base_fee + max_priority_fee
                
                self.gas_price_history.append({
                    'timestamp': current_time,
                    'price': gas_price
                })
                
                if len(self.gas_price_history) > self.max_history:
                    self.gas_price_history.pop(0)
                    
                self.last_update = current_time
                
            except Exception as e:
                logger.error(f"Error updating gas price: {str(e)}")
                if self.gas_price_history:
                    gas_price = self.gas_price_history[-1]['price']
                else:
                    gas_price = self.web3.eth.gas_price
                    
        return gas_price

class HuggingFaceInterface:
    """Interface for HuggingFace-based analysis and interaction"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('HF_API_KEY')
        self.api_url = "https://api-inference.huggingface.co/models/"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.summary_model = "facebook/bart-large-cnn"
        self.analysis_model = "bigscience/bloom"
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_summary(self, performance_data: Dict) -> str:
        """Generate a summary of agent's performance using BART"""
        try:
            prompt = self._create_summary_prompt(performance_data)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}{self.summary_model}",
                    headers=self.headers,
                    json={"inputs": prompt, "parameters": {"max_length": 150, "min_length": 50}}
                ) as response:
                    result = await response.json()
                    
            if isinstance(result, list) and len(result) > 0:
                return result[0]['summary_text']
            return "Error generating summary"
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Error generating summary"
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def process_training_suggestion(self, suggestion: str, current_params: Dict) -> Dict:
        """Process user suggestion using BLOOM"""
        try:
            prompt = self._create_training_prompt(suggestion, current_params)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}{self.analysis_model}",
                    headers=self.headers,
                    json={
                        "inputs": prompt,
                        "parameters": {
                            "max_length": 200,
                            "temperature": 0.3,
                            "return_full_text": False
                        }
                    }
                ) as response:
                    result = await response.json()
                    
            return self._parse_training_response(result[0]['generated_text'])
            
        except Exception as e:
            logger.error(f"Error processing training suggestion: {e}")
            return {}
            
    def _create_summary_prompt(self, data: Dict) -> str:
        """Create prompt for performance summary"""
        return f"""Summarize the following trading performance:
        Total Trades: {data.get('total_trades', 0)}
        Success Rate: {data.get('success_rate', 0):.2f}%
        Total Profit: {data.get('total_profit', 0):.4f} ETH
        Average Gas Cost: {data.get('avg_gas_cost', 0):.4f} ETH
        Best Trade: {data.get('best_trade', 'N/A')}
        Market Conditions: {data.get('market_conditions', 'N/A')}
        
        Focus on key metrics, trends, and actionable insights."""
        
    def _create_training_prompt(self, suggestion: str, current_params: Dict) -> str:
        """Create prompt for processing training suggestions"""
        return f"""Given the following trading bot parameters and user suggestion, provide parameter adjustments in JSON format.
        
        Current Parameters:
        {json.dumps(current_params, indent=2)}
        
        User Suggestion: {suggestion}
        
        Respond with a JSON object containing only the parameters that should be adjusted. Example:
        {{"learning_rate": 0.001, "batch_size": 64}}"""
        
    def _parse_training_response(self, response: str) -> Dict:
        """Parse model response into parameter adjustments"""
        try:
            # Find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
            return {}
        except Exception as e:
            logger.error(f"Error parsing training response: {e}")
            return {}

class FlashLoanArbitrageInterface:
    def __init__(self, web3: Web3, contract_address: str, private_key: str):
        self.web3 = web3
        self.contract_address = contract_address
        self.private_key = private_key
        self.contract = self.web3.eth.contract(
            address=contract_address,
            abi=self._load_contract_abi()
        )
        
    def _load_contract_abi(self) -> List:
        try:
            with open('abis/flash_loan_arbitrage.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading flash loan ABI: {str(e)}")
            return []

class MarketAnalyzer:
    """Consolidated market analysis functionality"""
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.price_history = []
        self.volume_history = []
        
    @retry_async()
    async def analyze_market(self, token_pair: Tuple[str, str], amount: float) -> Dict:
        """Comprehensive market analysis"""
        return {
            'time_series': self._analyze_time_series(),
            'cross_chain': await self._analyze_cross_chain(token_pair),
            'mev_risk': await self._analyze_mev_risk(amount),
            'gas': await self._optimize_gas(),
            'economics': await self._analyze_token_economics(token_pair)
        }
        
    def _analyze_time_series(self) -> Dict:
        return {
            'momentum': self._calculate_momentum(),
            'rsi': self._calculate_relative_strength(),
            'liquidity': self._calculate_liquidity_depth()
        }
        
    async def _analyze_cross_chain(self, token_pair: Tuple[str, str]) -> Dict:
        return {
            'eth_correlation': self._calculate_eth_correlation(),
            'l2_efficiency': self._calculate_l2_efficiency(),
            'bridge_liquidity': await self._get_bridge_liquidity(),
            'volume': await self._get_cross_chain_volume()
        }
        
    async def _analyze_mev_risk(self, amount: float) -> Dict:
        return {
            'sandwich_risk': self._estimate_sandwich_risk({'amount': amount}),
            'frontrunning_risk': self._estimate_frontrunning_risk({'amount': amount}),
            'backrunning_risk': self._estimate_backrunning_risk({'amount': amount}),
            'optimal_position': self._calculate_optimal_block_position({'amount': amount})
        }
        
    async def _optimize_gas(self) -> Dict:
        return {
            'optimal_price': self._calculate_optimal_gas_price(),
            'predicted_base_fee': self._predict_base_fee_next_blocks(),
            'priority_fee': self._calculate_priority_fee(),
            'block_space': self._analyze_block_space()
        }
        
    async def _analyze_token_economics(self, token_pair: Tuple[str, str]) -> Dict:
        return {
            'supply_changes': self._analyze_supply_changes(token_pair[0]),
            'holder_distribution': self._analyze_holder_distribution(token_pair[0]),
            'vesting_events': self._track_vesting_events(token_pair[0]),
            'protocol_revenue': self._analyze_protocol_revenue(token_pair[0])
        }

class SuperchainArbitrageAgent:
    """Main arbitrage agent using composition of specialized components"""
    
    def __init__(self, config_path: str = 'config.json'):
        """Initialize arbitrage agent with configuration"""
        # Load configuration
        self.config_path = config_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get Web3 from centralized provider
        self.web3 = get_web3()
        logger.info("Using centralized Web3 provider")
        
        # Initialize components using composition
        self.market_analyzer = MarketAnalyzer()
        self.model = ArbitrageModel(self.device)
        self.training_manager = TrainingManager(self.model, self.device)
        self.transaction_validator = TransactionValidator()
        self.transaction_builder = TransactionBuilder(self.web3)
        self.execution_engine = ExecutionEngine(
            web3=self.web3,
            gas_manager=self.gas_manager,
            market_validator=self.market_validator
        )
        
        # Initialize monitoring
        self.monitoring = {
            'active_executions': 0,
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_profit': 0.0,
            'total_gas_used': 0
        }
        
        # Thread safety
        self._execution_lock = asyncio.Lock()
        self.execution_semaphore = asyncio.Semaphore(3)
        
    async def analyze_opportunity(
        self,
        token_pair: TokenPair,
        amount: float,
        market_data: MarketDataType
    ) -> Optional[OpportunityType]:
        """Analyze arbitrage opportunity using market analyzer and model"""
        try:
            # Monitor market conditions
            self.market_analyzer.monitor_volatility(market_data)
            
            # Get market analysis with caching
            market_analysis = await self._cached_market_analysis(token_pair, amount)
            
            # Get model predictions with error handling
            predictions = await self._get_model_predictions(market_data)
            if predictions is None:
                logger.warning("Model predictions returned None")
                return None
                
            # Adjust confidence based on volatility
            confidence = float(predictions['confidence'])
            volatility_factor = self.market_analyzer.get_volatility_adjustment()
            adjusted_confidence = confidence * volatility_factor
            
            return {
                'token_pair': token_pair,
                'amount': amount,
                'predicted_profit': float(predictions['market_analysis'].mean()),
                'confidence': adjusted_confidence,
                'risk_score': float(predictions['risk_assessment'].mean()),
                'execution_strategy': predictions['execution_strategy'].tolist(),
                'market_analysis': market_analysis
            }
            
        except Exception as e:
            logger.error(f"Error analyzing opportunity: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    @cached(cache=TTLCache(maxsize=100, ttl=60))  # Cache for 60 seconds
    async def _cached_market_analysis(
        self,
        token_pair: TokenPair,
        amount: float
    ) -> Dict:
        """Cached market analysis to reduce RPC calls"""
        return await self.market_analyzer.analyze_market(token_pair, amount)

    async def _get_model_predictions(
        self,
        market_data: MarketDataType
    ) -> Optional[Dict]:
        """Get model predictions with error handling"""
        try:
            with torch.no_grad():
                predictions = self.model(market_data)
            return predictions
        except Exception as e:
            logger.error(f"Error getting model predictions: {str(e)}")
            return None

    async def execute_arbitrage(
        self,
        opportunity: Union[OpportunityType, FlashLoanOpportunityType]
    ) -> ExecutionResult:
        """Execute arbitrage using execution engine with enhanced error handling"""
        try:
            async with self._execution_lock:
                async with self.execution_semaphore:
                    try:
                        start_time = time.time()
                        self.active_executions += 1
                        
                        # Revalidate market conditions
                        validation_result = await self.validate_market_conditions(opportunity)
                        if not validation_result.is_valid:
                            logger.warning(f"Market conditions changed: {validation_result.reason}")
                            return ExecutionResult(
                                status=ExecutionStatus.MARKET_CONDITIONS_CHANGED,
                                success=False,
                                error_message=validation_result.reason
                            )
                        
                        # Execute with retry logic
                        execution_result = await self._execute_with_retry(opportunity)
                        execution_time = time.time() - start_time
                        
                        if execution_result.success:
                            await self._update_execution_metrics(execution_result)
                            return ExecutionResult(
                                status=ExecutionStatus.SUCCESS,
                                success=True,
                                gas_used=execution_result.gas_used,
                                execution_time=execution_time,
                                tx_hash=execution_result.tx_hash
                            )
                        else:
                            return ExecutionResult(
                                status=ExecutionStatus.EXECUTION_ERROR,
                                success=False,
                                error_message=execution_result.error_message,
                                execution_time=execution_time
                            )
                            
                    finally:
                        self.active_executions -= 1
                        
        except Exception as e:
            error_msg = f"Error executing arbitrage: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            await self.record_failure(opportunity, error_msg)
            
            return ExecutionResult(
                status=ExecutionStatus.NETWORK_ERROR,
                success=False,
                error_message=error_msg
            )

    async def validate_market_conditions(
        self,
        opportunity: Union[OpportunityType, FlashLoanOpportunityType],
        max_price_movement: float = 0.02,  # 2% max price movement
        min_liquidity_ratio: float = 0.8,  # 80% of original liquidity
        max_gas_increase: float = 1.5  # 50% max gas increase
    ) -> MarketValidationResult:
        """Validate current market conditions before execution with enhanced checks"""
        try:
            # Get current price with caching
            current_price = await self._get_cached_price(opportunity['token_pair'])
            entry_price = opportunity.get('entry_price', current_price)
            price_change = abs(current_price - entry_price) / entry_price
            
            if price_change > max_price_movement:
                return MarketValidationResult(
                    is_valid=False,
                    reason=f"Price moved {price_change*100:.2f}% since opportunity detection",
                    current_price=current_price,
                    price_change=price_change
                )
            
            # Check liquidity with caching
            current_liquidity = await self._get_cached_liquidity(opportunity['token_pair'])
            min_required_liquidity = opportunity.get('min_required_liquidity', current_liquidity * min_liquidity_ratio)
            
            if current_liquidity < min_required_liquidity:
                return MarketValidationResult(
                    is_valid=False,
                    reason=f"Insufficient liquidity: {current_liquidity} < {min_required_liquidity}",
                    current_liquidity=current_liquidity
                )
            
            # Check gas price with caching
            current_gas = await self._get_cached_gas_price()
            original_gas = opportunity.get('gas_price', current_gas)
            
            if current_gas > original_gas * max_gas_increase:
                return MarketValidationResult(
                    is_valid=False,
                    reason=f"Gas price too high: {current_gas} > {original_gas * max_gas_increase}",
                    current_gas=current_gas
                )
            
            # Check network conditions
            network_status = await self._check_network_status()
            if not network_status['is_healthy']:
                return MarketValidationResult(
                    is_valid=False,
                    reason=f"Network issues: {network_status['reason']}",
                    network_status=network_status
                )
            
            # Volatility check
            if hasattr(self, 'volatility_history') and self.volatility_history:
                recent_volatility = statistics.mean([
                    v for _, v in self.volatility_history[-5:]  # Last 5 readings
                ])
                if recent_volatility > self.VOLATILITY_THRESHOLD:
                    return MarketValidationResult(
                        is_valid=False,
                        reason=f"High market volatility: {recent_volatility:.4f}",
                        network_status={'volatility': recent_volatility}
                    )
            
            # All conditions valid
            return MarketValidationResult(
                is_valid=True,
                current_price=current_price,
                price_change=price_change,
                current_liquidity=current_liquidity,
                current_gas=current_gas,
                network_status=network_status
            )
            
        except Exception as e:
            error_msg = f"Error validating market conditions: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return MarketValidationResult(
                is_valid=False,
                reason=error_msg
            )

    @cached(cache=TTLCache(maxsize=100, ttl=30))  # Cache for 30 seconds
    async def _get_cached_price(self, token_pair: TokenPair) -> float:
        """Get cached token price"""
        return await self.get_current_price(token_pair)

    @cached(cache=TTLCache(maxsize=100, ttl=30))
    async def _get_cached_liquidity(self, token_pair: TokenPair) -> float:
        """Get cached liquidity"""
        return await self.get_current_liquidity(token_pair)

    @cached(cache=TTLCache(maxsize=1, ttl=5))  # Short cache for gas price
    async def _get_cached_gas_price(self) -> int:
        """Get cached gas price"""
        return await self.gas_manager.get_optimal_gas_price()

    async def _check_network_status(self) -> Dict[str, Any]:
        """Check network health status"""
        try:
            # Get latest block with timeout
            latest_block = await asyncio.wait_for(
                self.web3.eth.get_block('latest'),
                timeout=5.0
            )
            
            # Check block timestamp
            block_delay = time.time() - latest_block['timestamp']
            if block_delay > 60:  # More than 1 minute delay
                return {
                    'is_healthy': False,
                    'reason': f"Block delay: {block_delay:.1f}s"
                }
            
            # Check pending transactions
            pending_tx_count = await self.web3.eth.get_block_transaction_count('pending')
            if pending_tx_count > 10000:  # High pending tx count
                return {
                    'is_healthy': False,
                    'reason': f"High pending transactions: {pending_tx_count}"
                }
            
            return {
                'is_healthy': True,
                'block_number': latest_block['number'],
                'block_delay': block_delay,
                'pending_tx_count': pending_tx_count
            }
            
        except Exception as e:
            logger.error(f"Error checking network status: {str(e)}")
            return {
                'is_healthy': False,
                'reason': f"Network error: {str(e)}"
            }

    async def _update_execution_metrics(self, result: ExecutionResult):
        """Update execution metrics for monitoring"""
        self.monitoring['total_executions'] += 1
        if result.success:
            self.monitoring['successful_executions'] += 1
        else:
            self.monitoring['failed_executions'] += 1
        
        if result.gas_used:
            self.monitoring['total_gas_used'] += result.gas_used
            
        # Emit metrics for monitoring
        logger.info(f"Execution metrics updated: {json.dumps(self.monitoring)}")

    async def train_model(self):
        """Train model using training manager"""
        return await self.training_manager.train_step(
            self.training_manager.get_training_batch()
        )
        
    async def monitor_superchain(self) -> None:
        """Monitor blockchain for opportunities"""
        try:
            while True:
                # Fetch market data
                market_data = await self.market_analyzer.fetch_market_data()
                
                # Process opportunities
                opportunities = await self._process_opportunities(market_data)
                
                # Execute profitable opportunities
                for opportunity in opportunities:
                    if await self.validate_and_execute_opportunity(opportunity):
                        break
                        
                await asyncio.sleep(1)  # Rate limiting
                
        except asyncio.CancelledError:
            logger.info("Monitoring task cancelled")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {str(e)}")
            raise

    async def validate_and_execute_opportunity(
        self,
        opportunity: Union[OpportunityType, FlashLoanOpportunityType]
    ) -> bool:
        """Validate and execute opportunity"""
        # Validate opportunity
        if not self.transaction_validator.validate_opportunity(opportunity):
                return False
                
        # Execute if valid
        result = await self.execute_arbitrage(opportunity)
        return result.success
        
    async def _process_opportunities(
        self,
        market_data: MarketDataType
    ) -> List[OpportunityType]:
        """Process market data for opportunities"""
        opportunities = []
        
        for token_pair in self.token_pairs:
            for amount in self.test_amounts:
                opportunity = await self.analyze_opportunity(
                    token_pair,
                    amount,
                    market_data
                )
                if opportunity and opportunity['predicted_profit'] > 0:
                    opportunities.append(opportunity)
                    
        return sorted(
            opportunities,
            key=lambda x: x['predicted_profit'],
            reverse=True
        )
        
class ExecutionStatus(Enum):
    """Execution status codes for arbitrage operations"""
    SUCCESS = auto()
    INVALID_OPPORTUNITY = auto()
    MARKET_CONDITIONS_CHANGED = auto()
    EXECUTION_ERROR = auto()
    NETWORK_ERROR = auto()

@dataclass
class ExecutionResult:
    """Result of an arbitrage execution attempt"""
    status: ExecutionStatus
    success: bool
    error_message: Optional[str] = None
    gas_used: Optional[int] = None
    execution_time: Optional[float] = None
    tx_hash: Optional[str] = None

@dataclass
class MarketValidationResult:
    """Result of market condition validation"""
    is_valid: bool
    reason: Optional[str] = None
    current_price: Optional[float] = None
    price_change: Optional[float] = None
    current_liquidity: Optional[float] = None
    current_gas: Optional[int] = None
    network_status: Optional[Dict[str, Any]] = None