import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from web3 import Web3
from typing import List, Dict, Tuple, Optional, Union, Callable, Any, AsyncGenerator, Coroutine, TypeVar, Literal, cast
import asyncio
import logging
from datetime import datetime, timedelta
from eth_typing import Address, HexStr, ChecksumAddress
from web3.contract.contract import Contract
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
from src.core.types import ExecutionResult, ExecutionStatus, MarketValidationResult
from src.ml.model import ArbitrageModel
from src.utils.training_manager import TrainingManager
from src.validation.transaction_validator import TransactionValidator
from src.execution.transaction_builder import TransactionBuilder
from src.execution.execution_engine import ExecutionEngine
from src.gas.optimizer import AsyncGasOptimizer
from src.gas.gas_manager import GasManager
from src.validation.market_validator import MarketValidator
from src.market.analyzer import MarketAnalyzer, DefiLlamaIntegration
from threading import Lock
from src.agent.token_discovery import TokenDiscovery
from src.monitoring.monitor_manager import MonitorManager

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

# Neural network feature types
SentimentFeature = torch.Tensor  # Shape: (batch_size, 8)
HistoricalFeature = torch.Tensor  # Shape: (batch_size, 1, 8, 8)
CrossChainFeature = torch.Tensor  # Shape: (batch_size, 12)
MEVFeature = torch.Tensor  # Shape: (batch_size, 8)
GasFeature = torch.Tensor  # Shape: (batch_size, 6)
LiquidityFeature = torch.Tensor  # Shape: (batch_size, 10)
TokenEconomicsFeature = torch.Tensor  # Shape: (batch_size, 12)

# Neural network market data type
MarketDataBatch = Dict[str, Union[
    SentimentFeature,
    HistoricalFeature,
    CrossChainFeature,
    MEVFeature,
    GasFeature,
    LiquidityFeature,
    TokenEconomicsFeature
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
        
    def forward(self, x: torch.Tensor, market_data: Optional[MarketDataBatch] = None) -> torch.Tensor:
        """Forward pass with market data validation"""
        if market_data is None:
            raise ValueError("market_data is required for prediction")
            
        # Validate required features
        required_shapes = {
            'sentiment': (8,),
            'historical': (1, 8, 8),
            'cross_chain': (12,),
            'mev': (8,),
            'gas': (6,),
            'liquidity': (10,),
            'token_economics': (12,)
        }
        
        for feature_name, expected_shape in required_shapes.items():
            if feature_name not in market_data:
                raise ValueError(f"Missing required feature: {feature_name}")
            feature_tensor = market_data[feature_name]
            if not isinstance(feature_tensor, torch.Tensor):
                raise ValueError(f"Feature {feature_name} must be a torch.Tensor")
            if feature_tensor.shape[-len(expected_shape):] != expected_shape:
                raise ValueError(f"Invalid shape for {feature_name}: expected {expected_shape}, got {feature_tensor.shape}")
        
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

class HuggingFaceInterface:
    """Interface for HuggingFace-based analysis and interaction"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize HuggingFace interface"""
        self.api_key = api_key if api_key is not None else os.getenv('HF_API_KEY', '')
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
                summary = result[0].get('summary_text', '')
                if summary:
                    return summary
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
    """Interface for flash loan arbitrage operations"""
    
    def __init__(self, web3: Web3, contract_address: str, private_key: str):
        """Initialize flash loan interface"""
        self.web3 = web3
        self.contract_address = Web3.to_checksum_address(contract_address)
        self.private_key = private_key
        self.contract = self.web3.eth.contract(
            address=self.contract_address,
            abi=self._load_contract_abi()
        )
        
    def _load_contract_abi(self) -> List:
        """Load flash loan contract ABI"""
        try:
            with open('abis/flash_loan_arbitrage.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading flash loan ABI: {str(e)}")
            return []

def thread_safe_cache(func):
    """Thread-safe caching decorator"""
    cache = TTLCache(maxsize=100, ttl=30)
    lock = Lock()
    
    @wraps(func)
    async def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        with lock:
            if key in cache:
                return cache[key]
            result = await func(*args, **kwargs)
            cache[key] = result
            return result
    return wrapper

class SuperchainArbitrageAgent:
    """Main arbitrage agent using composition of specialized components"""
    
    def __init__(self, config_path: str = 'config.json'):
        """Initialize the arbitrage agent"""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Constants
        self.MIN_CONFIDENCE = 0.8  # 80% minimum confidence threshold
        self.VOLATILITY_THRESHOLD = 0.1  # 10% volatility threshold
        
        # Initialize Web3
        self.web3 = get_web3()
        self.async_web3 = get_async_web3()
        
        # Initialize token discovery
        self.token_discovery = TokenDiscovery(self.config)
        
        # Initialize gas manager and market validator first
        self.gas_manager = GasManager(self.web3, self.config)
        self.market_validator = MarketValidator(self.web3, self.config)
        
        # Initialize components with config and dependencies
        self.market_analyzer = MarketAnalyzer(config=self.config)
        self.validator = TransactionValidator(web3=self.web3, config=self.config)
        self.builder = TransactionBuilder(config=self.config)
        self.execution_engine = ExecutionEngine(
            web3=self.web3,
            gas_manager=self.gas_manager,
            market_validator=self.market_validator
        )
        
        # Initialize monitoring and learning components
        self.monitor_manager = MonitorManager(
            config=self.config,
            storage_path=self.config.get('monitoring', {}).get('storage_path', 'data/monitoring'),
            prometheus_port=self.config.get('monitoring', {}).get('prometheus_port', 8000),
            cache_enabled=self.config.get('monitoring', {}).get('cache_enabled', True)
        )
        
        # Initialize neural network components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ArbitrageModel(self.device)
        self.training_manager = TrainingManager(self.model, self.device)
        
        # Initialize time series features tracking
        self.time_series = TimeSeriesFeatures(
            window_size=self.config.get('learning', {}).get('window_size', 100),
            cleanup_threshold=self.config.get('learning', {}).get('cleanup_threshold', 1000)
        )
        
        # Initialize specialized analyzers
        self.cross_chain = CrossChainAnalyzer()
        self.mev_protection = MEVProtection()
        self.gas_optimizer = GasOptimizer()
        self.token_economics = TokenEconomicsAnalyzer()
        
        # Initialize HuggingFace interface for advanced analysis
        self.hf_interface = HuggingFaceInterface(
            api_key=self.config.get('huggingface', {}).get('api_key')
        )
        
        # Thread safety
        self._execution_lock = asyncio.Lock()
        self.execution_semaphore = asyncio.Semaphore(3)
        
        # Initialize execution tracking
        self.active_executions = 0
        self.volatility_history: List[Tuple[float, float]] = []
        
        # Initialize token pairs and test amounts
        self.token_pairs = self._load_initial_token_pairs()
        self.test_amounts = self._load_test_amounts()
        
        # Initialize learning state
        self.learning_state = {
            'model_version': 0,
            'total_training_steps': 0,
            'performance_history': [],
            'strategy_metrics': {},
            'feature_importance': {},
            'anomaly_scores': []
        }

    async def start(self):
        """Start the arbitrage agent with all components"""
        try:
            # Start monitoring components
            await self.monitor_manager.start()
            
            # Start market monitoring
            monitoring_task = asyncio.create_task(self.monitor_superchain())
            
            # Start learning loop
            learning_task = asyncio.create_task(self._learning_loop())
            
            # Start optimization loop
            optimization_task = asyncio.create_task(self._optimization_loop())
            
            # Wait for all tasks
            await asyncio.gather(
                monitoring_task,
                learning_task,
                optimization_task
            )
            
        except Exception as e:
            logger.error(f"Error starting arbitrage agent: {str(e)}")
            raise

    async def _learning_loop(self):
        """Main learning loop for continuous improvement"""
        try:
            while True:
                # Get latest performance metrics
                metrics = await self.monitor_manager.get_learning_insights()
                
                # Update learning state
                self._update_learning_state(metrics)
                
                # Train model if enough data
                if len(self.learning_state['performance_history']) >= 100:
                    await self.train_model()
                    
                    # Generate and apply optimizations
                    await self._apply_learning_optimizations()
                    
                    # Update feature importance
                    feature_importance = self.model.get_feature_importance()
                    self.monitor_manager.update_feature_importance(feature_importance)
                    
                    # Log learning progress
                    logger.info(
                        "Learning progress",
                        model_version=self.learning_state['model_version'],
                        training_steps=self.learning_state['total_training_steps']
                    )
                
                await asyncio.sleep(300)  # Sleep for 5 minutes
                
        except asyncio.CancelledError:
            logger.info("Learning loop cancelled")
        except Exception as e:
            logger.error(f"Error in learning loop: {str(e)}")
            raise

    async def _optimization_loop(self):
        """Continuous optimization loop"""
        try:
            while True:
                # Get current insights
                insights = await self.monitor_manager.get_learning_insights()
                
                # Apply optimizations based on insights
                if insights.get('optimization_suggestions'):
                    for suggestion in insights['optimization_suggestions']:
                        await self._apply_optimization(suggestion)
                
                # Update strategies based on performance
                await self._update_strategies()
                
                # Generate performance summary
                summary = await self.hf_interface.generate_summary(self.learning_state)
                logger.info(f"Performance summary: {summary}")
                
                await asyncio.sleep(600)  # Sleep for 10 minutes
                
        except asyncio.CancelledError:
            logger.info("Optimization loop cancelled")
        except Exception as e:
            logger.error(f"Error in optimization loop: {str(e)}")
            raise

    def _update_learning_state(self, metrics: Dict[str, Any]):
        """Update learning state with new metrics"""
        try:
            # Update performance history
            self.learning_state['performance_history'].append({
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics
            })
            
            # Trim history if too long
            max_history = self.config.get('learning', {}).get('max_history', 1000)
            if len(self.learning_state['performance_history']) > max_history:
                self.learning_state['performance_history'] = \
                    self.learning_state['performance_history'][-max_history:]
            
            # Update strategy metrics
            for strategy, perf in metrics.get('strategy_performance', {}).items():
                if strategy not in self.learning_state['strategy_metrics']:
                    self.learning_state['strategy_metrics'][strategy] = []
                self.learning_state['strategy_metrics'][strategy].append(perf)
            
            # Update anomaly scores
            self.learning_state['anomaly_scores'] = \
                metrics.get('anomaly_scores', self.learning_state['anomaly_scores'])
            
            # Update feature importance
            if 'feature_importance' in metrics:
                self.learning_state['feature_importance'] = metrics['feature_importance']
                
        except Exception as e:
            logger.error(f"Error updating learning state: {str(e)}")

    async def _apply_learning_optimizations(self):
        """Apply optimizations based on learning insights"""
        try:
            # Get current insights
            insights = await self.monitor_manager.get_learning_insights()
            
            # Update model parameters based on performance
            if insights.get('strategy_performance'):
                await self._update_model_parameters(insights['strategy_performance'])
            
            # Adjust trading parameters
            if insights.get('optimization_suggestions'):
                await self._adjust_trading_parameters(insights['optimization_suggestions'])
            
            # Update feature selection
            if insights.get('feature_importance'):
                self._update_feature_selection(insights['feature_importance'])
                
        except Exception as e:
            logger.error(f"Error applying learning optimizations: {str(e)}")

    async def _update_model_parameters(self, performance: Dict[str, Any]):
        """Update model parameters based on performance metrics"""
        try:
            # Calculate performance metrics
            avg_profit = np.mean([p.get('profit', 0) for p in performance.values()])
            success_rate = np.mean([p.get('success', 0) for p in performance.values()])
            
            # Adjust model parameters
            if avg_profit < 0 or success_rate < 0.5:
                # Increase learning rate for faster adaptation
                self.model.adjust_learning_rate(factor=1.5)
            else:
                # Decrease learning rate for stability
                self.model.adjust_learning_rate(factor=0.9)
                
            self.learning_state['model_version'] += 1
            
        except Exception as e:
            logger.error(f"Error updating model parameters: {str(e)}")

    async def _adjust_trading_parameters(self, suggestions: List[str]):
        """Adjust trading parameters based on optimization suggestions"""
        try:
            for suggestion in suggestions:
                if "gas" in suggestion.lower():
                    # Adjust gas optimization parameters
                    self.gas_manager.adjust_optimization_params()
                elif "execution" in suggestion.lower():
                    # Adjust execution parameters
                    self.execution_engine.adjust_execution_params()
                elif "strategy" in suggestion.lower():
                    # Adjust strategy parameters
                    await self._update_strategies()
                    
        except Exception as e:
            logger.error(f"Error adjusting trading parameters: {str(e)}")

    def _update_feature_selection(self, importance: Dict[str, float]):
        """Update feature selection based on importance scores"""
        try:
            # Sort features by importance
            sorted_features = sorted(
                importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Keep top features
            top_features = sorted_features[:self.config.get('learning', {}).get('max_features', 20)]
            
            # Update model feature selection
            self.model.update_feature_selection([f[0] for f in top_features])
            
        except Exception as e:
            logger.error(f"Error updating feature selection: {str(e)}")

    async def _update_strategies(self):
        """Update trading strategies based on performance"""
        try:
            # Get strategy performance
            strategy_metrics = self.learning_state['strategy_metrics']
            
            # Calculate strategy scores
            strategy_scores = {}
            for strategy, metrics in strategy_metrics.items():
                if not metrics:
                    continue
                    
                recent_metrics = metrics[-10:]  # Last 10 executions
                avg_profit = np.mean([m.get('profit', 0) for m in recent_metrics])
                success_rate = np.mean([m.get('success', 0) for m in recent_metrics])
                
                strategy_scores[strategy] = avg_profit * success_rate
            
            # Update strategy weights
            total_score = sum(strategy_scores.values())
            if total_score > 0:
                for strategy, score in strategy_scores.items():
                    weight = score / total_score
                    await self._update_strategy_weight(strategy, weight)
                    
        except Exception as e:
            logger.error(f"Error updating strategies: {str(e)}")

    async def _update_strategy_weight(self, strategy: str, weight: float):
        """Update individual strategy weight"""
        try:
            # Update strategy configuration
            if 'strategies' not in self.config:
                self.config['strategies'] = {}
            if strategy not in self.config['strategies']:
                self.config['strategies'][strategy] = {}
                
            self.config['strategies'][strategy]['weight'] = weight
            
            # Log update
            logger.info(
                f"Updated strategy weight",
                strategy=strategy,
                weight=weight
            )
            
        except Exception as e:
            logger.error(f"Error updating strategy weight: {str(e)}")

    def _load_initial_token_pairs(self) -> List[TokenPair]:
        """Load initial token pairs from config"""
        pairs = self.config.get('trading', {}).get('token_pairs', [])
        return [(pair['token0'], pair['token1']) for pair in pairs]
        
    async def _update_token_pairs(self) -> None:
        """Update token pairs with discovered tokens"""
        try:
            # Get configured pairs
            configured_pairs = self._load_initial_token_pairs()
            
            # Discover new tokens
            discovered_tokens = await self.token_discovery.discover_new_tokens()
            logger.info(f"Discovered {len(discovered_tokens)} new tokens")
            
            # Validate discovered tokens
            valid_tokens = []
            for token in discovered_tokens:
                if await self.token_discovery.validate_token(token['address']):
                    valid_tokens.append(token['address'])
            
            # Create all viable token pairs
            discovered_pairs = []
            weth_address = self.config['weth_address']
            
            # 1. Create pairs with WETH (important for liquidity)
            discovered_pairs.extend([(weth_address, token) for token in valid_tokens])
            
            # 2. Create pairs between all valid tokens
            for i, token1 in enumerate(valid_tokens):
                for token2 in valid_tokens[i+1:]:  # Avoid duplicate pairs
                    # Validate if this pair has enough liquidity
                    if await self._validate_pair_liquidity((token1, token2)):
                        discovered_pairs.append((token1, token2))
            
            # Combine configured and discovered pairs
            all_pairs = configured_pairs + discovered_pairs
            
            # Log detailed statistics
            logger.info(
                f"Token pairs updated:\n"
                f"- Configured pairs: {len(configured_pairs)}\n"
                f"- WETH pairs: {len([p for p in discovered_pairs if weth_address in p])}\n"
                f"- Non-WETH pairs: {len([p for p in discovered_pairs if weth_address not in p])}\n"
                f"- Total pairs: {len(all_pairs)}"
            )
            
            # Update token pairs
            self.token_pairs = all_pairs
            
        except Exception as e:
            logger.error(f"Error updating token pairs: {str(e)}")
            logger.error(traceback.format_exc())
            
    async def _validate_pair_liquidity(self, token_pair: TokenPair) -> bool:
        """Validate if a token pair has sufficient liquidity"""
        try:
            # Get liquidity from market analyzer
            liquidity = await self.market_analyzer.get_current_liquidity(token_pair)
            min_liquidity = self.config.get('min_pair_liquidity', 10000)  # Default $10k
            
            if liquidity >= min_liquidity:
                logger.debug(f"Pair {token_pair} has sufficient liquidity: ${liquidity:,.2f}")
                return True
            else:
                logger.debug(f"Pair {token_pair} has insufficient liquidity: ${liquidity:,.2f} < ${min_liquidity:,.2f}")
                return False
                
        except Exception as e:
            logger.error(f"Error validating pair liquidity for {token_pair}: {str(e)}")
            return False
        
    def _load_test_amounts(self) -> List[float]:
        """Load test amounts from config"""
        return self.config.get('trading', {}).get('test_amounts', [0.1, 0.5, 1.0])
        
    async def _execute_with_retry(
        self,
        opportunity: Union[OpportunityType, FlashLoanOpportunityType],
        retry_count: int = 0
    ) -> ExecutionResult:
        """Execute transaction with retry logic"""
        try:
            tx_params = await self.builder.build_transaction(opportunity)
            if not tx_params:
                return ExecutionResult(
                    status=ExecutionStatus.EXECUTION_ERROR,
                    success=False,
                    error_message="Failed to build transaction"
                )
            
            return await self.execution_engine.execute_transaction(tx_params, retry_count)
            
        except Exception as e:
            error_msg = f"Error executing transaction: {str(e)}"
            logger.error(error_msg)
            return ExecutionResult(
                status=ExecutionStatus.EXECUTION_ERROR,
                success=False,
                error_message=error_msg
            )
            
    async def record_failure(
        self,
        opportunity: Union[OpportunityType, FlashLoanOpportunityType],
        error_msg: str
    ) -> None:
        """Record execution failure"""
        self.monitoring['failed_executions'] += 1
        logger.error(f"Execution failed: {error_msg}")
        
    async def get_current_price(self, token_pair: TokenPair) -> float:
        """Get current price for token pair"""
        return await self.market_analyzer.get_current_price(token_pair)
        
    async def get_current_liquidity(self, token_pair: TokenPair) -> float:
        """Get current liquidity for token pair"""
        return await self.market_analyzer.get_current_liquidity(token_pair)
        
    async def analyze_opportunity(
        self,
        token_pair: TokenPair,
        amount: float,
        market_data: MarketDataType
    ) -> Optional[OpportunityType]:
        """Analyze arbitrage opportunity with enhanced validation"""
        try:
            # Validate token pair format first
            if not isinstance(token_pair, tuple) or len(token_pair) != 2:
                raise KeyError('token_pair must be a tuple of two token addresses')
            
            # Then validate amount
            if amount <= 0:
                raise ValueError('amount must be positive')
            
            # Finally validate market data
            if not market_data or not isinstance(market_data, dict):
                raise KeyError('market_data is required and must be a dictionary')
            
            # Get market analysis with caching
            market_analysis = await self._cached_market_analysis(token_pair, amount)
            if not market_analysis:
                logger.warning("Market analysis returned None")
                return None
            
            # Get model predictions with error handling
            predictions = await self._get_model_predictions(market_data)
            if not predictions:
                logger.warning("Model predictions returned None")
                return None
            
            # Calculate opportunity metrics
            predicted_profit = predictions.get('profit', 0.0)
            confidence = predictions.get('confidence', 0.0)
            risk_score = predictions.get('risk_score', 1.0)
            
            if predicted_profit <= 0 or confidence < self.MIN_CONFIDENCE:
                logger.debug(f"Opportunity rejected: profit={predicted_profit}, confidence={confidence}")
                return None
            
            # Create opportunity
            opportunity: OpportunityType = {
                'token_pair': token_pair,
                'amount': amount,
                'predicted_profit': predicted_profit,
                'confidence': confidence,
                'risk_score': risk_score,
                'execution_strategy': predictions.get('execution_strategy', []),
                'market_analysis': market_analysis
            }
            
            return opportunity
            
        except Exception as e:
            logger.error(f"Error analyzing opportunity: {str(e)}")
            logger.error(traceback.format_exc())
            raise  # Re-raise to maintain error propagation

    @thread_safe_cache
    async def _cached_market_analysis(
        self,
        token_pair: TokenPair,
        amount: float
    ) -> Dict:
        """Cached market analysis to reduce RPC calls"""
        try:
            return await self.market_analyzer.analyze_market(token_pair, amount)
        except Exception as e:
            logger.error(f"Error in market analysis: {str(e)}")
            return {}

    async def _get_model_predictions(
        self,
        market_data: MarketDataType
    ) -> Optional[Dict]:
        """Get model predictions with error handling"""
        try:
            async with self._execution_lock:  # Ensure thread-safe model access
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

    def _safe_float_conversion(self, value: Any, default: float = 0.0) -> float:
        """Safely convert a value to float"""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
            
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
            token_pair = cast(TokenPair, opportunity['token_pair'])
            current_price = await self._get_cached_price(token_pair)
            entry_price = self._safe_float_conversion(opportunity.get('entry_price'), current_price)
            price_change = abs(current_price - entry_price) / entry_price
            
            if price_change > max_price_movement:
                return MarketValidationResult(
                    is_valid=False,
                    reason=f"Price moved {price_change*100:.2f}% since opportunity detection",
                    current_price=current_price,
                    price_change=price_change
                )
            
            # Check liquidity with caching
            current_liquidity = await self._get_cached_liquidity(token_pair)
            min_required_liquidity = self._safe_float_conversion(
                opportunity.get('min_required_liquidity'),
                current_liquidity * min_liquidity_ratio
            )
            
            if current_liquidity < min_required_liquidity:
                return MarketValidationResult(
                    is_valid=False,
                    reason=f"Insufficient liquidity: {current_liquidity} < {min_required_liquidity}",
                    current_liquidity=current_liquidity
                )
            
            # Check gas price with caching
            current_gas = await self._get_cached_gas_price()
            original_gas = self._safe_float_conversion(opportunity.get('gas_price'), current_gas)
            
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
        try:
            gas_settings = await self.gas_manager.optimize_gas_settings({})
            gas_price = gas_settings.get('maxFeePerGas')
            if gas_price is None:
                gas_price = self.web3.eth.gas_price
            return int(gas_price)
        except Exception as e:
            logger.error(f"Error getting gas price: {str(e)}")
            return int(self.web3.eth.gas_price)

    async def _check_network_status(self) -> Dict[str, Any]:
        """Check network health status"""
        try:
            # Get latest block
            try:
                latest_block = self.web3.eth.get_block('latest')
            except Exception as e:
                logger.error(f"Failed to get latest block: {str(e)}")
                return {
                    'is_healthy': False,
                    'reason': "Failed to get latest block"
                }
            
            if not latest_block:
                return {
                    'is_healthy': False,
                    'reason': "No latest block returned"
                }
            
            # Check block timestamp
            try:
                block_timestamp = int(latest_block.get('timestamp', 0))
            except (ValueError, TypeError, AttributeError):
                block_timestamp = 0
                
            block_delay = time.time() - block_timestamp
            if block_delay > 60:  # More than 1 minute delay
                return {
                    'is_healthy': False,
                    'reason': f"Block delay: {block_delay:.1f}s"
                }
            
            # Check pending transactions
            try:
                pending_tx_count = self.web3.eth.get_block_transaction_count('pending')
            except Exception:
                pending_tx_count = 0
                
            if pending_tx_count > 10000:  # High pending tx count
                return {
                    'is_healthy': False,
                    'reason': f"High pending transaction count: {pending_tx_count}"
                }
            
            # Get block number
            try:
                block_number = int(latest_block.get('number', 0))
            except (ValueError, TypeError, AttributeError):
                block_number = 0
            
            return {
                'is_healthy': True,
                'block_number': block_number,
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

    async def train_model(self) -> None:
        """Train model using training manager"""
        batch = self.training_manager.get_training_batch()
        if batch is not None:
            await self.training_manager.train_step(batch)
            
    async def _apply_optimization(self, suggestion: str) -> None:
        """Apply optimization based on suggestion"""
        try:
            if 'gas' in suggestion.lower():
                await self._optimize_gas_settings()
            elif 'execution' in suggestion.lower():
                await self._optimize_execution_settings()
            elif 'strategy' in suggestion.lower():
                await self._optimize_strategy_settings()
            elif 'liquidity' in suggestion.lower():
                await self._optimize_liquidity_settings()
                
        except Exception as e:
            logger.error(f"Error applying optimization: {str(e)}")

    async def _optimize_gas_settings(self) -> None:
        """Optimize gas-related settings"""
        try:
            # Get current gas metrics
            gas_metrics = await self.monitor_manager.get_gas_metrics()
            
            # Calculate optimal gas settings
            optimal_settings = self.gas_manager.calculate_optimal_settings(gas_metrics)
            
            # Update gas manager settings
            await self.gas_manager.update_settings(optimal_settings)
            
            logger.info("Updated gas optimization settings", settings=optimal_settings)
            
        except Exception as e:
            logger.error(f"Error optimizing gas settings: {str(e)}")

    async def _optimize_execution_settings(self) -> None:
        """Optimize execution-related settings"""
        try:
            # Get execution metrics
            execution_metrics = await self.monitor_manager.get_execution_metrics()
            
            # Calculate optimal settings
            optimal_settings = self.execution_engine.calculate_optimal_settings(execution_metrics)
            
            # Update execution engine settings
            await self.execution_engine.update_settings(optimal_settings)
            
            logger.info("Updated execution optimization settings", settings=optimal_settings)
            
        except Exception as e:
            logger.error(f"Error optimizing execution settings: {str(e)}")

    async def _optimize_strategy_settings(self) -> None:
        """Optimize strategy-related settings"""
        try:
            # Get strategy metrics
            strategy_metrics = await self.monitor_manager.get_strategy_metrics()
            
            # Calculate optimal settings
            optimal_settings = self.model.calculate_optimal_settings(strategy_metrics)
            
            # Update model settings
            self.model.update_settings(optimal_settings)
            
            logger.info("Updated strategy optimization settings", settings=optimal_settings)
            
        except Exception as e:
            logger.error(f"Error optimizing strategy settings: {str(e)}")

    async def _optimize_liquidity_settings(self) -> None:
        """Optimize liquidity-related settings"""
        try:
            # Get liquidity metrics
            liquidity_metrics = await self.monitor_manager.get_liquidity_metrics()
            
            # Calculate optimal settings
            optimal_settings = self.market_analyzer.calculate_optimal_settings(liquidity_metrics)
            
            # Update market analyzer settings
            await self.market_analyzer.update_settings(optimal_settings)
            
            logger.info("Updated liquidity optimization settings", settings=optimal_settings)
            
        except Exception as e:
            logger.error(f"Error optimizing liquidity settings: {str(e)}")

    async def _check_circuit_breakers(self) -> bool:
        """Check if any circuit breakers should be triggered"""
        try:
            # Get current metrics
            metrics = await self.monitor_manager.get_learning_insights()
            
            # Check for anomalies
            if len(metrics.get('anomaly_scores', [])) > 0:
                recent_anomalies = metrics['anomaly_scores'][-10:]  # Last 10 scores
                if sum(1 for score in recent_anomalies if score == -1) >= 3:
                    logger.warning("Circuit breaker triggered: Too many recent anomalies")
                    return True
            
            # Check performance degradation
            if metrics.get('strategy_performance'):
                recent_performance = []
                for strategy in metrics['strategy_performance'].values():
                    if isinstance(strategy, list) and len(strategy) >= 10:
                        recent_metrics = strategy[-10:]  # Last 10 trades
                        avg_profit = sum(m.get('profit', 0) for m in recent_metrics) / len(recent_metrics)
                        recent_performance.append(avg_profit)
                
                if recent_performance and sum(1 for p in recent_performance if p < 0) > len(recent_performance) * 0.5:
                    logger.warning("Circuit breaker triggered: Performance degradation")
                    return True
            
            # Check gas costs
            gas_metrics = await self.monitor_manager.get_gas_metrics()
            if gas_metrics.get('average_cost', 0) > self.config.get('max_gas_cost', 1000000):
                logger.warning("Circuit breaker triggered: Excessive gas costs")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking circuit breakers: {str(e)}")
            return True  # Fail-safe: trigger circuit breaker on error

    async def _handle_circuit_breaker(self) -> None:
        """Handle circuit breaker activation"""
        try:
            logger.warning("Circuit breaker activated - pausing operations")
            
            # Stop active executions
            self.active_executions = 0
            
            # Reset volatile state
            self.volatility_history.clear()
            
            # Notify monitoring
            await self.monitor_manager.record_circuit_breaker()
            
            # Wait for recovery period
            await asyncio.sleep(self.config.get('circuit_breaker_cooldown', 300))  # 5 minutes default
            
            # Check if safe to resume
            if await self._check_recovery_conditions():
                logger.info("Circuit breaker reset - resuming operations")
            else:
                logger.warning("Recovery conditions not met - maintaining circuit breaker")
                
        except Exception as e:
            logger.error(f"Error handling circuit breaker: {str(e)}")

    async def _check_recovery_conditions(self) -> bool:
        """Check if conditions are safe for recovery"""
        try:
            # Get current metrics
            metrics = await self.monitor_manager.get_learning_insights()
            
            # Check market conditions
            market_validation = await self.market_validator.validate_market_conditions({})
            if not market_validation.is_valid:
                return False
            
            # Check gas prices
            gas_metrics = await self.monitor_manager.get_gas_metrics()
            if gas_metrics.get('average_cost', 0) > self.config.get('max_gas_cost', 1000000):
                return False
            
            # Check recent performance
            if metrics.get('strategy_performance'):
                recent_performance = []
                for strategy in metrics['strategy_performance'].values():
                    if isinstance(strategy, list) and len(strategy) >= 5:
                        recent_metrics = strategy[-5:]  # Last 5 trades
                        avg_profit = sum(m.get('profit', 0) for m in recent_metrics) / len(recent_metrics)
                        recent_performance.append(avg_profit)
                
                if recent_performance and sum(1 for p in recent_performance if p < 0) > 0:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking recovery conditions: {str(e)}")
            return False

    async def monitor_superchain(self) -> None:
        """Monitor blockchain for opportunities with enhanced safety"""
        try:
            last_discovery_time = 0
            discovery_interval = 300  # 5 minutes
            
            while True:
                try:
                    # Check circuit breakers
                    if await self._check_circuit_breakers():
                        await self._handle_circuit_breaker()
                        continue
                    
                    current_time = time.time()
                    
                    # Update token pairs periodically
                    if current_time - last_discovery_time > discovery_interval:
                        await self._update_token_pairs()
                        last_discovery_time = current_time
                        logger.info(f"Token pairs updated. Now tracking {len(self.token_pairs)} pairs")
                    
                    # Fetch market data
                    market_data = await self.market_analyzer.fetch_market_data()
                    
                    # Process opportunities with enhanced monitoring
                    opportunities = await self._process_opportunities(market_data)
                    
                    # Record monitoring metrics
                    await self.monitor_manager.record_monitoring_cycle({
                        'opportunities_found': len(opportunities),
                        'token_pairs_tracked': len(self.token_pairs),
                        'market_conditions': market_data.get('market_conditions', {}),
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Execute profitable opportunities
                    for opportunity in opportunities:
                        if await self.validate_and_execute_opportunity(opportunity):
                            break
                            
                    await asyncio.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {str(e)}")
                    await asyncio.sleep(5)  # Back off on error
                    
        except asyncio.CancelledError:
            logger.info("Monitoring task cancelled")
        except Exception as e:
            logger.error(f"Fatal error in monitoring loop: {str(e)}")
            raise

    async def validate_and_execute_opportunity(
        self,
        opportunity: Union[OpportunityType, FlashLoanOpportunityType]
    ) -> bool:
        """Validate and execute opportunity"""
        # Validate opportunity
        if not await self.validator.validate_transaction(opportunity):
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
                if opportunity is not None:
                    predicted_profit = self._safe_float_conversion(opportunity.get('predicted_profit'))
                    if predicted_profit > 0:
                        opportunities.append(opportunity)
                    
        return sorted(
            opportunities,
            key=lambda x: self._safe_float_conversion(x.get('predicted_profit')),
            reverse=True
        )

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise ConfigurationError(f"Failed to load config: {str(e)}")