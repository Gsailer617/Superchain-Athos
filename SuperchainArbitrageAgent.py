import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from web3 import Web3
from typing import List, Dict, Tuple, Optional
import asyncio
import logging
from datetime import datetime, timedelta
from eth_typing import Address
from web3.contract import Contract
import aiohttp
import json
from defillama import DefiLlama
from visualization import ArbitrageVisualizer
import os
from telegram_bot import telegram_bot
import ssl

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ArbitrageNetwork(nn.Module):
    """Enhanced neural network for predicting arbitrage opportunities"""
    def __init__(self, input_size: int = 12, hidden_size: int = 128):
        super().__init__()
        # Main prediction network
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 3)  # [profit_prediction, confidence, risk_score]
        )
        
        # DEX attention mechanism
        self.dex_attention = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.Tanh(),
            nn.Linear(32, 10)  # Attention weights for supported DEXes
        )
        
    def forward(self, x):
        # Main prediction
        predictions = self.network(x)
        
        # DEX attention weights
        dex_weights = torch.softmax(self.dex_attention(x), dim=1)
        
        return predictions, dex_weights

class SuperchainArbitrageAgent(nn.Module):
    """Advanced AI agent for discovering and executing arbitrage opportunities on Superchain"""
    
    # Enhanced DEX support with configurations
    supported_dexes = {
        'uniswap_v3': {
            'address': '0x8aD414D56502Ec3Ea68B3968F5396C8dEB2f3CC8',  # Base Uniswap V3 Factory
            'type': 'UniswapV3',
            'fee_tiers': [100, 500, 3000, 10000],
            'router': '0x2626664c2603336E57B271c5C0b26F421741e481'  # Base Uniswap V3 Router
        },
        'sushiswap': {
            'address': '0xc35DADB65012eC5796536bD9864eD8773aBc74C4',  # Base SushiSwap Factory
            'type': 'UniswapV2',
            'fee': 300,  # 0.3%
            'router': '0x6BDED42c6DA8FD5E8B11852d05271A5241e32594'  # Base SushiSwap Router
        },
        'baseswap': {
            'address': '0xFDa619b6d20975be80A10332cD39b9a4b0FAa8BB',  # BaseSwap Factory
            'type': 'UniswapV2',
            'fee': 300,
            'router': '0x327Df1E6de05895d2ab08513aaDD9313Fe505d86'  # BaseSwap Router
        },
        'aerodrome': {
            'address': '0x420DD381b31aEf6683db6B902084cB0FFEe076d6',  # Aerodrome Factory
            'type': 'UniswapV2',
            'fee': 300,
            'router': '0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43'  # Aerodrome Router
        },
        'pancakeswap': {
            'address': '0x0BFbCF9fa4f9C56B0F40a671Ad40E0805A091865',  # Base PancakeSwap Factory
            'type': 'UniswapV2',
            'fee': 250,  # 0.25%
            'router': '0x678Aa4bF4E210cf2166753e054d5b7c31cc7fa86'  # Base PancakeSwap Router
        },
        'swapbased': {
            'address': '0x36905172C5D3C7A2e91c6819A27A1Ab346A0aeEE',  # SwapBased Factory
            'type': 'UniswapV2',
            'fee': 300,
            'router': '0x327Df1E6de05895d2ab08513aaDD9313Fe505d86'  # SwapBased Router
        },
        'alienbase': {
            'address': '0x3E84D913803b02A4a7f027165E8cA42C14C0FdE7',  # AlienBase Factory
            'type': 'UniswapV2',
            'fee': 300,
            'router': '0xb0505d78A2A6d49fA6E4905BA9D6efC472B642C2'  # AlienBase Router
        },
        'maverick': {
            'address': '0x32D02Fc7722E81F6Ac6bD1b70Ef5277eD3566fb9',  # Maverick Factory
            'type': 'UniswapV3',
            'fee_tiers': [100, 500, 3000, 10000],
            'router': '0x32D02Fc7722E81F6Ac6bD1b70Ef5277eD3566fb9'  # Maverick Router
        },
        'synthswap': {
            'address': '0x9B2Cc8e6a2Bbb56d6bE4682891a91B0e48633F96',  # SynthSwap Factory
            'type': 'UniswapV2',
            'fee': 300,
            'router': '0x5589D08c49cDddF48bB9f6E648c587d5D4533B28'  # SynthSwap Router
        },
        'horizon_dex': {
            'address': '0x0F633F78147b933B743d2d660A56BA3Ff5683C24',  # Horizon DEX Factory
            'type': 'UniswapV2',
            'fee': 300,
            'router': '0xE8C1365E4EF99F8Bf69c5Bb34Ea54b622A43f367'  # Horizon DEX Router
        }
    }
    
    def __init__(self, web3_provider: str, model_params=None):
        super().__init__()
        # Initialize Web3 with Base RPC endpoint
        self.web3_provider = "https://mainnet.base.org"
        self.web3 = Web3(Web3.HTTPProvider(self.web3_provider))
        self.defillama_client = DefiLlama()
        
        # Initialize dex_analytics
        self.dex_analytics = {
            dex_name: {
                'tokens': set(),
                'liquidity': {},
                'volume': {},
                'pairs': set()
            }
            for dex_name in self.supported_dexes.keys()
        }
        
        # Initialize training components with provided parameters
        if model_params is not None:
            self.optimizer = torch.optim.Adam(model_params, lr=0.001)
        else:
            # Initialize default model if no parameters provided
            self.model = nn.Sequential(
                nn.Linear(10, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        self.criterion = nn.MSELoss()
        
        # Enhanced data collection
        self.training_buffer = {
            'features': [],
            'labels': [],
            'timestamps': [],
            'market_conditions': [],
            'execution_results': []
        }
        self.buffer_size = 10000  # Keep last 10000 data points
        self.min_samples_for_training = 100
        
        # Adaptive learning settings
        self.adaptive_settings = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'training_frequency': 100,  # Train every 100 new data points
            'validation_split': 0.2,
            'early_stopping_patience': 5,
            'min_delta': 0.001
        }
        
        # Performance tracking
        self.performance_metrics = {
            'training_losses': [],
            'validation_losses': [],
            'prediction_accuracy': [],
            'profit_history': [],
            'risk_adjusted_returns': []
        }
        
        # Flash loan configurations
        self.flash_loan_providers = {
            'aave': {
                'address': '0x...',  # Aave lending pool address
                'router': '0x...',   # Aave router
                'min_amount': 0.1,   # Minimum flash loan amount in ETH
                'fee': 0.0009       # 0.09% fee
            },
            'balancer': {
                'address': '0x...',  # Balancer vault address
                'router': '0x...',   # Balancer router
                'min_amount': 0.05,  # Minimum amount
                'fee': 0.0001       # 0.01% fee
            },
            'dodo': {
                'address': '0x...',  # DODO vault address
                'router': '0x...',   # DODO router
                'min_amount': 0.01,  # Minimum amount
                'fee': 0.0002       # 0.02% fee
            }
        }
        
        # Flash loan specific settings
        self.flash_loan_settings = {
            'min_profit_threshold': 0.005,  # 0.5% minimum profit after fees
            'max_loan_amount': 1000,        # Maximum flash loan in ETH
            'min_loan_amount': 0.1,         # Minimum flash loan in ETH
            'max_routes': 3,                # Maximum number of DEX routes
            'preferred_providers': ['aave', 'balancer'],  # Preferred flash loan providers
            'max_gas_impact': 0.2,          # Maximum gas impact on profit
            'min_net_profit_usd': 50        # Minimum profit in USD after all fees
        }
        
        # Neural network layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.prediction_head = nn.Linear(32, 3)  # profit, confidence, risk
        self.dex_attention = nn.Linear(32, 10)  # 10 supported DEXes
        
        self.model = ArbitrageNetwork()
        self.training_data = []
        self.performance_history = []
        self.telegram_bot = telegram_bot
        
        # Token discovery and analysis settings
        self.token_discovery = {
            'min_liquidity': 50000,  # Minimum liquidity in USD
            'min_holders': 100,      # Minimum number of holders
            'min_age_days': 3,       # Minimum token age in days
            'blacklisted_tokens': set(),  # Known scam tokens
            'discovered_tokens': {},  # Tracked tokens and their metrics
            'pair_history': {}       # Historical performance of token pairs
        }
        
        # Risk management settings
        self.risk_settings = {
            'min_confidence': 0.85,
            'max_risk_score': 0.3,
            'min_profit_threshold': 0.002,  # 0.2%
            'min_sharpe_ratio': 1.5,
            'max_drawdown': 0.1,
            'volatility_threshold': 0.02,
            'min_liquidity_usd': 50000,
            'max_price_impact': 0.01,  # 1%
            'max_slippage': 0.005     # 0.5%
        }
        
        # Initialize visualizer
        self.visualizer = ArbitrageVisualizer()
        self.visualizer.run_in_thread()
        
    def forward(self, x):
        """Forward pass through the network"""
        x = self.feature_extractor(x)
        predictions = self.prediction_head(x)
        dex_weights = torch.softmax(self.dex_attention(x), dim=0)  # Change dim=1 to dim=0 for 1D input
        return predictions, dex_weights
        
    async def close(self):
        """Close any open connections"""
        try:
            if hasattr(self.web3.provider, 'close'):
                await self.web3.provider.close()
        except Exception as e:
            logger.error(f"Error closing web3 provider: {str(e)}")
        
    async def fetch_market_data(self) -> Dict:
        """Enhanced market data fetching with TVL and token metrics"""
        try:
            async with aiohttp.ClientSession() as session:
                tasks = []
                
                # Fetch DEX data with retries
                for dex_name, dex_info in self.supported_dexes.items():
                    task = self.fetch_dex_data_with_retry(session, dex_name, dex_info)
                    tasks.append(task)
                
                # Fetch TVL data with timeout
                tvl_task = asyncio.wait_for(self.fetch_tvl_data(), timeout=10)
                tasks.append(tvl_task)
                
                # Fetch token discovery data with timeout
                discovery_task = asyncio.wait_for(self.discover_new_tokens(session), timeout=15)
                tasks.append(discovery_task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Filter out failed requests
                dex_data = [r for r in results[:-2] if not isinstance(r, Exception)]
                tvl_data = results[-2] if not isinstance(results[-2], Exception) else {}
                new_tokens = results[-1] if not isinstance(results[-1], Exception) else []
                
                return {
                    'dex_data': dex_data,
                    'tvl_data': tvl_data,
                    'new_tokens': new_tokens,
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error in fetch_market_data: {str(e)}")
            return {
                'dex_data': [],
                'tvl_data': {},
                'new_tokens': [],
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
            
    async def fetch_tvl_data(self) -> Dict:
        """Fetch TVL data from DeFiLlama"""
        try:
            tvl_data = {}
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                for dex_name in self.supported_dexes:
                    try:
                        async with session.get(f"https://api.llama.fi/protocol/{dex_name}") as response:
                            if response.status == 200:
                                data = await response.json()
                                tvl_data[dex_name] = data.get('tvl', 0)
                            else:
                                logger.warning(f"Failed to fetch TVL for {dex_name}: {response.status}")
                                tvl_data[dex_name] = 0
                    except Exception as e:
                        logger.error(f"Error fetching TVL for {dex_name}: {str(e)}")
                        tvl_data[dex_name] = 0
            return tvl_data
        except Exception as e:
            logger.error(f"Error fetching TVL data: {str(e)}")
            return {}
            
    async def discover_new_tokens(self, session: aiohttp.ClientSession) -> List[Dict]:
        """Enhanced token discovery with security checks"""
        try:
            new_tokens = []
            
            # Fetch new token listings from various sources
            # Implement token discovery logic here
            
            # Analyze each new token
            for token in new_tokens:
                if await self.validate_token(token):
                    self.token_discovery['discovered_tokens'][token['address']] = {
                        'symbol': token['symbol'],
                        'discovery_time': datetime.now(),
                        'liquidity': token['liquidity'],
                        'holders': token['holders'],
                        'score': await self.calculate_token_score(token)
                    }
            
            return new_tokens
        except Exception as e:
            logger.error(f"Error in token discovery: {str(e)}")
            return []
            
    async def validate_token(self, token: Dict) -> bool:
        """Enhanced token validation with comprehensive checks"""
        try:
            # Basic requirements check
            if (token['liquidity'] < self.token_discovery['min_liquidity'] or
                token['holders'] < self.token_discovery['min_holders'] or
                token['address'] in self.token_discovery['blacklisted_tokens']):
                return False
            
            # Contract security checks
            contract_security = await self.check_contract_security(token['address'])
            if not contract_security['is_safe']:
                self.token_discovery['blacklisted_tokens'].add(token['address'])
                return False
            
            # Additional validation checks
            if not await self.validate_token_extended(token):
                return False
            
            # Token age check
            creation_time = await self.get_token_creation_time(token['address'])
            if (datetime.now() - creation_time) < timedelta(days=self.token_discovery['min_age_days']):
                return False
            
            # Trading volume check
            if not await self.check_trading_volume(token):
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating token {token.get('symbol', 'unknown')}: {str(e)}")
            return False
            
    async def analyze_opportunity(self, 
                                token_pair: Tuple[str, str],
                                amount: float,
                                market_data: Dict) -> Dict:
        """Enhanced opportunity analysis with advanced metrics"""
        # Prepare input features
        features = torch.tensor([
            amount,
            market_data['price_impact'],
            market_data['liquidity'],
            market_data['volatility'],
            market_data['gas_price'],
            market_data['block_time'],
            market_data['price_difference'],
            market_data['volume_24h'],
            market_data['tvl'],
            market_data['token_score'],
            market_data['market_sentiment'],
            market_data['network_congestion']
        ], dtype=torch.float32)
        
        # Get model predictions and DEX weights
        with torch.no_grad():
            predictions, dex_weights = self.model(features)
            profit_pred, confidence, risk_score = predictions.numpy()
            
        # Calculate additional metrics
        metrics = self.calculate_advanced_metrics(
            profit_pred, confidence, risk_score, market_data
        )
        
        opportunity = {
            'id': f"opp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(token_pair))}",
            'profit_prediction': float(profit_pred),
            'confidence': float(confidence),
            'risk_score': float(risk_score),
            'token_pair': token_pair,
            'amount': amount,
            'timestamp': datetime.now().isoformat(),
            'gas_price': market_data['gas_price'],
            'volume_24h': market_data['volume_24h'],
            'dex_weights': dex_weights.numpy().tolist(),
            'metrics': metrics,
            'type': 'DEX Arbitrage',
            'tokens_involved': list(token_pair),
            'expected_profit': float(profit_pred),
            'gas_cost': market_data['gas_price'] * 21000,  # Estimate gas cost
            'price_impact': market_data['price_impact'],
            'slippage': market_data.get('slippage', 0),
            'liquidity': market_data['liquidity'],
            'volatility': market_data.get('volatility', 0)
        }
        
        # Notify via Telegram bot
        await self.telegram_bot.notify_opportunity(opportunity)
        
        return opportunity
        
    def calculate_advanced_metrics(self, profit_pred: float, confidence: float, 
                                 risk_score: float, market_data: Dict) -> Dict:
        """Calculate advanced trading metrics"""
        return {
            'sharpe_ratio': self.calculate_sharpe_ratio(profit_pred, market_data),
            'sortino_ratio': self.calculate_sortino_ratio(profit_pred, market_data),
            'profit_factor': self.calculate_profit_factor(),
            'win_rate': self.calculate_win_rate(),
            'expected_value': profit_pred * confidence * (1 - risk_score),
            'risk_adjusted_return': profit_pred / (risk_score + 1e-6),
            'market_impact': self.estimate_market_impact(market_data),
            'execution_probability': self.estimate_execution_probability(market_data)
        }
        
    async def execute_arbitrage(self, opportunity: Dict) -> bool:
        """Enhanced arbitrage execution with safety checks"""
        if not self.telegram_bot.is_trade_profitable(opportunity):
            return False
            
        try:
            # Prepare transaction with optimal path
            flash_loan_params = await self.prepare_flash_loan(
                token_pair=opportunity['token_pair'],
                amount=opportunity['amount'],
                dex_weights=opportunity['dex_weights']
            )
            
            # Execute with advanced safety checks
            trade_result = await self.safe_execute_transaction(flash_loan_params)
            
            # Prepare trade execution notification
            trade_execution = {
                'id': opportunity['id'],
                'type': opportunity['type'],
                'status': 'completed' if trade_result['success'] else 'failed',
                'profit': trade_result.get('profit', 0),
                'gas_used': trade_result.get('gas_used', 0),
                'execution_time': trade_result.get('execution_time', 0),
                'error': trade_result.get('error'),
                'tokens_involved': opportunity['tokens_involved']
            }
            
            # Notify execution status and update performance
            await self.telegram_bot.notify_trade_execution(trade_execution)
            self.telegram_bot.update_performance(trade_execution)
            
            if trade_result['success']:
                await self.record_success(opportunity)
                return True
                
        except Exception as e:
            logger.error(f"Arbitrage execution failed: {str(e)}")
            await self.record_failure(opportunity, str(e))
            
        return False
        
    def validate_opportunity(self, opportunity: Dict) -> bool:
        """Validate opportunity against risk settings"""
        return (
            opportunity['confidence'] > self.risk_settings['min_confidence'] and
            opportunity['risk_score'] < self.risk_settings['max_risk_score'] and
            opportunity['profit_prediction'] > self.risk_settings['min_profit_threshold'] and
            opportunity['metrics']['sharpe_ratio'] > self.risk_settings['min_sharpe_ratio'] and
            opportunity['metrics']['execution_probability'] > 0.9
        )
        
    async def analyze_flash_loan_opportunity(self, 
                                          token_pair: Tuple[str, str],
                                          amount: float,
                                          market_data: Dict) -> Dict:
        """Analyze potential flash loan arbitrage opportunity"""
        try:
            # Calculate flash loan fees for each provider
            provider_fees = {
                name: amount * config['fee']
                for name, config in self.flash_loan_providers.items()
            }
            
            # Get the best provider based on fees
            best_provider = min(provider_fees.items(), key=lambda x: x[1])
            
            # Calculate potential profit with flash loan
            features = torch.tensor([
                amount,
                market_data['price_impact'],
                market_data['liquidity'],
                market_data['volatility'],
                market_data['gas_price'],
                market_data['block_time'],
                market_data['price_difference'],
                market_data['volume_24h'],
                market_data['tvl'],
                market_data['token_score'],
                market_data['market_sentiment'],
                market_data['network_congestion']
            ], dtype=torch.float32)
            
            # Get model predictions
            with torch.no_grad():
                predictions, dex_weights = self.model(features)
                profit_pred, confidence, risk_score = predictions.numpy()
            
            # Calculate net profit after flash loan fees
            flash_loan_fee = best_provider[1]
            gas_cost = market_data['gas_price'] * 300000  # Higher gas estimate for flash loans
            net_profit = profit_pred - flash_loan_fee - gas_cost
            
            opportunity = {
                'id': f"flash_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(token_pair))}",
                'type': 'Flash Loan Arbitrage',
                'token_pair': token_pair,
                'amount': amount,
                'flash_loan_provider': best_provider[0],
                'flash_loan_fee': flash_loan_fee,
                'gross_profit': profit_pred,
                'net_profit': net_profit,
                'gas_cost': gas_cost,
                'confidence': float(confidence),
                'risk_score': float(risk_score),
                'dex_weights': dex_weights.numpy().tolist(),
                'timestamp': datetime.now().isoformat(),
                'metrics': self.calculate_advanced_metrics(profit_pred, confidence, risk_score, market_data)
            }
            
            return opportunity
            
        except Exception as e:
            logger.error(f"Error analyzing flash loan opportunity: {str(e)}")
            return None
            
    async def execute_flash_loan(self, opportunity: Dict) -> bool:
        """Execute flash loan arbitrage"""
        try:
            provider = self.flash_loan_providers[opportunity['flash_loan_provider']]
            
            # Prepare flash loan parameters
            flash_params = {
                'token': opportunity['token_pair'][0],
                'amount': opportunity['amount'],
                'provider_address': provider['address'],
                'router': provider['router'],
                'dex_routes': self.get_optimal_dex_route(opportunity['dex_weights'])
            }
            
            # Build flash loan transaction
            flash_tx = await self.build_flash_loan_transaction(flash_params)
            
            # Execute with safety checks
            if await self.validate_flash_loan_safety(flash_tx):
                result = await self.safe_execute_transaction(flash_tx)
                
                # Prepare execution notification
                execution = {
                    'id': opportunity['id'],
                    'type': 'Flash Loan Arbitrage',
                    'status': 'completed' if result['success'] else 'failed',
                    'profit': result.get('profit', 0),
                    'gas_used': result.get('gas_used', 0),
                    'execution_time': result.get('execution_time', 0),
                    'error': result.get('error'),
                    'flash_loan_provider': opportunity['flash_loan_provider']
                }
                
                # Notify execution status
                await self.telegram_bot.notify_trade_execution(execution)
                self.telegram_bot.update_performance(execution)
                
                return result['success']
                
            return False
            
        except Exception as e:
            logger.error(f"Flash loan execution failed: {str(e)}")
            return False
            
    def validate_flash_loan_safety(self, flash_tx: Dict) -> bool:
        """Validate flash loan transaction safety"""
        try:
            # Verify flash loan parameters
            if flash_tx['amount'] < self.flash_loan_settings['min_loan_amount']:
                return False
                
            if flash_tx['amount'] > self.flash_loan_settings['max_loan_amount']:
                return False
                
            # Verify expected profit meets minimum threshold
            if flash_tx['expected_profit'] < self.flash_loan_settings['min_profit_threshold']:
                return False
                
            # Verify gas impact
            gas_impact = flash_tx['gas_cost'] / flash_tx['expected_profit']
            if gas_impact > self.flash_loan_settings['max_gas_impact']:
                return False
                
            # Verify net profit meets minimum USD threshold
            if flash_tx['net_profit_usd'] < self.flash_loan_settings['min_net_profit_usd']:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating flash loan safety: {str(e)}")
            return False
            
    async def monitor_superchain(self):
        """Enhanced continuous monitoring focusing on flash loan opportunities"""
        while True:
            try:
                if self.telegram_bot.system_paused:
                    await asyncio.sleep(5)
                    continue
                    
                # Fetch market data
                market_data = await self.fetch_market_data()
                
                # First, look for flash loan opportunities
                flash_opportunities = []
                for token_pair in self.get_viable_pairs():
                    for amount in self.get_optimal_flash_loan_amounts(token_pair, market_data):
                        opportunity = await self.analyze_flash_loan_opportunity(
                            token_pair=token_pair,
                            amount=amount,
                            market_data=market_data
                        )
                        if opportunity:
                            flash_opportunities.append(opportunity)
                
                # Execute profitable flash loan opportunities
                for opportunity in self.prioritize_opportunities(flash_opportunities):
                    if self.validate_flash_loan_safety(opportunity):
                        if self.telegram_bot.auto_trade:
                            await self.execute_flash_loan(opportunity)
                
                # Only look for regular arbitrage if no flash loan opportunities
                if not flash_opportunities:
                    regular_opportunities = []
                    for token_pair in self.get_viable_pairs():
                        opportunity = await self.analyze_opportunity(
                            token_pair=token_pair,
                            amount=self.get_optimal_regular_amount(token_pair, market_data),
                            market_data=market_data
                        )
                        # Only consider opportunities with guaranteed profit
                        if opportunity and opportunity['confidence'] > 0.95 and opportunity['risk_score'] < 0.1:
                            regular_opportunities.append(opportunity)
                    
                    # Execute regular opportunities with guaranteed profit
                    for opportunity in self.prioritize_opportunities(regular_opportunities):
                        if self.telegram_bot.is_trade_profitable(opportunity):
                            if self.telegram_bot.auto_trade:
                                await self.execute_arbitrage(opportunity)
                
                await asyncio.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Monitoring error: {str(e)}")
                await asyncio.sleep(5)
                
    def prioritize_opportunities(self, opportunities: List[Dict]) -> List[Dict]:
        """Prioritize opportunities based on multiple factors"""
        return sorted(
            opportunities,
            key=lambda x: (
                x['profit_prediction'] * 
                x['confidence'] * 
                (1 - x['risk_score']) * 
                x['metrics']['execution_probability']
            ),
            reverse=True
        )
        
    async def update_models_and_metrics(self):
        """Update ML models and performance metrics"""
        if len(self.training_data) % 100 == 0:  # Update every 100 new data points
            await self.train_model()
            self.save_model()
            self.update_performance_metrics()
            self.visualizer.update_metrics(self.performance_history)
            
    def cleanup_historical_data(self):
        """Clean up old historical data"""
        # Implement cleanup logic here
        pass

    async def record_success(self, opportunity: Dict):
        """Enhanced success recording with training data collection"""
        # Original success recording
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'profit': opportunity['profit_prediction'],
            'token_pair': opportunity['token_pair'],
            'amount': opportunity['amount']
        })
        
        # Collect training data
        await self.collect_training_data(
            features=self.extract_features(opportunity),
            label=opportunity['profit_prediction'],
            market_condition=opportunity.get('market_data', {}),
            execution_result={'success': True, 'actual_profit': opportunity['profit_prediction']}
        )
        
        # Update visualization
        success_rate = sum(1 for p in self.performance_history[-100:] if p['success']) / min(len(self.performance_history), 100)
        opportunity['success_rate'] = success_rate
        self.visualizer.update_data(opportunity)
        
        # Trigger training if needed
        await self.check_and_train()

    async def record_failure(self, opportunity: Dict, error: str):
        """Enhanced failure recording with training data collection"""
        # Original failure recording
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'error': error,
            'token_pair': opportunity['token_pair'],
            'amount': opportunity['amount']
        })
        
        # Collect training data
        await self.collect_training_data(
            features=self.extract_features(opportunity),
            label=0.0,  # Failed trades get 0 profit
            market_condition=opportunity.get('market_data', {}),
            execution_result={'success': False, 'error': error}
        )
        
        # Update visualization
        success_rate = sum(1 for p in self.performance_history[-100:] if p['success']) / min(len(self.performance_history), 100)
        opportunity['success_rate'] = success_rate
        self.visualizer.update_data(opportunity)
        
        # Trigger training if needed
        await self.check_and_train()

    async def collect_training_data(self, features: torch.Tensor, label: float,
                                  market_condition: Dict, execution_result: Dict):
        """Collect and maintain training data buffer"""
        self.training_buffer['features'].append(features)
        self.training_buffer['labels'].append(label)
        self.training_buffer['timestamps'].append(datetime.now().isoformat())
        self.training_buffer['market_conditions'].append(market_condition)
        self.training_buffer['execution_results'].append(execution_result)
        
        # Maintain buffer size
        if len(self.training_buffer['features']) > self.buffer_size:
            for key in self.training_buffer:
                self.training_buffer[key] = self.training_buffer[key][-self.buffer_size:]

    def extract_features(self, opportunity: Dict) -> torch.Tensor:
        """Extract features from opportunity for training"""
        return torch.tensor([
            opportunity['amount'],
            opportunity.get('price_impact', 0),
            opportunity.get('liquidity', 0),
            opportunity.get('volatility', 0),
            opportunity.get('gas_price', 0),
            opportunity.get('volume_24h', 0),
            opportunity.get('tvl', 0),
            opportunity.get('market_sentiment', 0),
            opportunity.get('network_congestion', 0),
            opportunity.get('confidence', 0),
            opportunity.get('risk_score', 0),
            opportunity.get('slippage', 0)
        ], dtype=torch.float32)

    async def check_and_train(self):
        """Check if we should trigger training and execute if needed"""
        if len(self.training_buffer['features']) >= self.min_samples_for_training and \
           len(self.training_buffer['features']) % 100 == 0:  # Train every 100 new samples
            await self.train_model()
            self.save_model()

    async def train_model(self):
        """Enhanced training with validation and early stopping"""
        try:
            # Prepare data
            features = torch.stack(self.training_buffer['features'])
            labels = torch.tensor(self.training_buffer['labels'])
            
            # Split into train/validation
            split_idx = int(len(features) * 0.8)  # 80% train, 20% validation
            train_features, val_features = features[:split_idx], features[split_idx:]
            train_labels, val_labels = labels[:split_idx], labels[split_idx:]
            
            # Training loop
            best_val_loss = float('inf')
            patience = 5
            patience_counter = 0
            
            for epoch in range(100):  # Max 100 epochs
                # Train
                self.train()
                self.optimizer.zero_grad()
                train_pred = self.model(train_features)[0]  # Only take predictions, not dex_weights
                train_loss = self.criterion(train_pred, train_labels)
                train_loss.backward()
                self.optimizer.step()
                
                # Validate
                self.eval()
                with torch.no_grad():
                    val_pred = self.model(val_features)[0]
                    val_loss = self.criterion(val_pred, val_labels)
                
                # Early stopping check
                if val_loss < best_val_loss - 0.001:  # Minimum improvement threshold
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
                
                # Update learning rate
                self.scheduler.step(val_loss)
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
            
            logger.info(f"Training completed. Final validation loss: {val_loss.item():.4f}")
            
        except Exception as e:
            logger.error(f"Error in training: {str(e)}")

    def save_model(self, path: str = 'models/arbitrage_model.pt'):
        """Save the neural network model to disk"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'training_data': self.training_data,
                'performance_history': self.performance_history,
                'token_discovery': self.token_discovery,
                'timestamp': datetime.now().isoformat()
            }, path)
            logger.info(f"Model saved successfully to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    def load_model(self, path: str = 'models/arbitrage_model.pt'):
        """Load the neural network model from disk"""
        try:
            if os.path.exists(path):
                checkpoint = torch.load(path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.training_data = checkpoint.get('training_data', [])
                self.performance_history = checkpoint.get('performance_history', [])
                self.token_discovery = checkpoint.get('token_discovery', self.token_discovery)
                logger.info(f"Model loaded successfully from {path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def calculate_advanced_metrics(self, profit_pred: float, confidence: float, 
                                 risk_score: float, market_data: Dict) -> Dict:
        """Calculate advanced trading metrics"""
        return {
            'sharpe_ratio': self.calculate_sharpe_ratio(profit_pred, market_data),
            'sortino_ratio': self.calculate_sortino_ratio(profit_pred, market_data),
            'profit_factor': self.calculate_profit_factor(),
            'win_rate': self.calculate_win_rate(),
            'expected_value': profit_pred * confidence * (1 - risk_score),
            'risk_adjusted_return': profit_pred / (risk_score + 1e-6),
            'market_impact': self.estimate_market_impact(market_data),
            'execution_probability': self.estimate_execution_probability(market_data)
        }

    async def fetch_dex_data_with_retry(self, session: aiohttp.ClientSession, dex_name: str, dex_info: Dict) -> Dict:
        """Fetch DEX data with retry mechanism"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                # Get contract ABI based on DEX type
                abi = self.get_dex_abi(dex_info['type'])
                contract = self.web3.eth.contract(
                    address=Web3.to_checksum_address(dex_info['address']),
                    abi=abi
                )
                
                # Verify contract deployment
                try:
                    code = self.web3.eth.get_code(contract.address)
                    if code == b'':
                        logger.error(f"Contract not deployed at {contract.address} for {dex_name}")
                        return {
                            'name': dex_name,
                            'error': 'Contract not deployed',
                            'timestamp': datetime.now().isoformat()
                        }
                except Exception as e:
                    logger.error(f"Error verifying contract for {dex_name}: {str(e)}")
                    return {
                        'name': dex_name,
                        'error': f'Contract verification failed: {str(e)}',
                        'timestamp': datetime.now().isoformat()
                    }
                
                # Fetch pairs/pools based on DEX type
                if dex_info['type'] == 'UniswapV3':
                    pools = []
                    try:
                        pool_count = contract.functions.allPoolsLength().call()
                        for pool_index in range(min(pool_count, 100)):  # Limit to 100 pools
                            try:
                                pool = contract.functions.allPools(pool_index).call()
                                for fee_tier in dex_info['fee_tiers']:
                                    pools.append({
                                        'address': pool,
                                        'fee_tier': fee_tier
                                    })
                            except Exception as e:
                                logger.error(f"Error fetching pool {pool_index} for {dex_name}: {str(e)}")
                                continue
                    except Exception as e:
                        logger.error(f"Error fetching pools for {dex_name}: {str(e)}")
                    data = {'pools': pools}
                else:  # UniswapV2
                    pairs = []
                    try:
                        pair_count = contract.functions.allPairsLength().call()
                        for pair_index in range(min(pair_count, 100)):  # Limit to 100 pairs
                            try:
                                pair = contract.functions.allPairs(pair_index).call()
                                pairs.append({
                                    'address': pair,
                                    'fee': dex_info['fee']
                                })
                            except Exception as e:
                                logger.error(f"Error fetching pair {pair_index} for {dex_name}: {str(e)}")
                                continue
                    except Exception as e:
                        logger.error(f"Error fetching pairs for {dex_name}: {str(e)}")
                    data = {'pairs': pairs}
                
                return {
                    'name': dex_name,
                    'type': dex_info['type'],
                    'data': data,
                    'timestamp': datetime.now().isoformat()
                }
                    
            except Exception as e:
                logger.error(f"Error fetching {dex_name} data (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    return {
                        'name': dex_name,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
        
        return None

    def get_dex_abi(self, dex_type: str) -> List:
        """Get ABI based on DEX type"""
        if dex_type == 'UniswapV3':
            return [
                {
                    "inputs": [],
                    "name": "allPoolsLength",
                    "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
                    "name": "allPools",
                    "outputs": [{"internalType": "address", "name": "", "type": "address"}],
                    "stateMutability": "view",
                    "type": "function"
                }
            ]
        else:  # UniswapV2
            return [
                {
                    "constant": True,
                    "inputs": [],
                    "name": "allPairsLength",
                    "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
                    "payable": False,
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "constant": True,
                    "inputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
                    "name": "allPairs",
                    "outputs": [{"internalType": "address", "name": "", "type": "address"}],
                    "payable": False,
                    "stateMutability": "view",
                    "type": "function"
                }
            ]

    def get_viable_pairs(self) -> List[Tuple[str, str]]:
        """Get list of viable trading pairs based on liquidity and volume"""
        viable_pairs = []
        
        try:
            # Get common tokens across supported DEXes
            common_tokens = set()
            for dex_name, dex_info in self.supported_dexes.items():
                if dex_name in self.dex_analytics:
                    dex_tokens = set(self.dex_analytics[dex_name].get('tokens', []))
                    if not common_tokens:
                        common_tokens = dex_tokens
                    else:
                        common_tokens &= dex_tokens
            
            # Filter pairs based on criteria
            for token1 in common_tokens:
                for token2 in common_tokens:
                    if token1 != token2:
                        pair = (token1, token2)
                        if self._is_viable_pair(pair):
                            viable_pairs.append(pair)
            
            return viable_pairs
            
        except Exception as e:
            logger.error(f"Error getting viable pairs: {str(e)}")
            return []

    def _is_viable_pair(self, pair: Tuple[str, str]) -> bool:
        """Check if a trading pair meets viability criteria"""
        try:
            token1, token2 = pair
            
            # Check minimum liquidity
            if not self._check_liquidity(token1, token2):
                return False
                
            # Check trading volume
            if not self._check_volume(token1, token2):
                return False
                
            # Check if pair is blacklisted
            if pair in self.token_discovery['blacklisted_tokens']:
                return False
                
            # Check token age
            if not self._check_token_age(token1) or not self._check_token_age(token2):
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking pair viability: {str(e)}")
            return False

    def _check_liquidity(self, token1: str, token2: str) -> bool:
        """Check if pair has sufficient liquidity across DEXes"""
        try:
            total_liquidity = 0
            for dex_name, dex_info in self.supported_dexes.items():
                if dex_name in self.dex_analytics:
                    pair_key = f"{token1}/{token2}"
                    liquidity = self.dex_analytics[dex_name].get('liquidity', {}).get(pair_key, 0)
                    total_liquidity += liquidity
            
            return total_liquidity >= self.token_discovery['min_liquidity']
            
        except Exception as e:
            logger.error(f"Error checking liquidity: {str(e)}")
            return False

    def _check_volume(self, token1: str, token2: str) -> bool:
        """Check if pair has sufficient trading volume"""
        try:
            total_volume = 0
            for dex_name in self.supported_dexes:
                if dex_name in self.dex_analytics:
                    pair_key = f"{token1}/{token2}"
                    volume = self.dex_analytics[dex_name].get('volume', {}).get(pair_key, 0)
                    total_volume += volume
            
            return total_volume > 0  # You can set a minimum volume threshold here
            
        except Exception as e:
            logger.error(f"Error checking volume: {str(e)}")
            return False

    def _check_token_age(self, token: str) -> bool:
        """Check if token meets minimum age requirement"""
        try:
            if token in self.token_discovery['discovered_tokens']:
                discovery_time = datetime.fromisoformat(
                    self.token_discovery['discovered_tokens'][token]['discovery_time']
                )
                age_days = (datetime.now() - discovery_time).days
                return age_days >= self.token_discovery['min_age_days']
            return False
            
        except Exception as e:
            logger.error(f"Error checking token age: {str(e)}")
            return False

async def main():
    # Initialize agent
    agent = SuperchainArbitrageAgent(web3_provider="YOUR_RPC_ENDPOINT")
    
    try:
        # Load existing model if available
        try:
            agent.load_model()
            logger.info("Loaded existing model")
        except:
            logger.info("Starting with fresh model")
        
        # Start visualization dashboard
        agent.visualizer.run_in_thread()
        
        # Initialize Telegram bot
        if not await telegram_bot.initialize():
            logger.error("Failed to initialize Telegram bot")
            agent.visualizer.stop()
            return
            
        # Start monitoring in the background
        monitoring_task = asyncio.create_task(agent.monitor_superchain())
        
        try:
            # Keep the main loop running
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            # Cleanup
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass
            
            # Cleanup components
            await agent.close()
            if telegram_bot.application:
                await telegram_bot.application.stop()
                await telegram_bot.application.shutdown()
            await telegram_bot.stop()
            agent.visualizer.stop()
            
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True) 