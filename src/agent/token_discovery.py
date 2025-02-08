"""
Token Discovery Module

This module provides comprehensive token discovery and validation functionality,
integrating multiple data sources, caching, and sentiment analysis.

Key Features:
- Asynchronous & parallel processing for token discovery
- Redis-based caching with expiration policies
- Rate limiting for external API calls
- Prometheus metrics for monitoring
- Social sentiment analysis using NLP
"""

import aiohttp
import asyncio
from web3 import Web3
from web3.middleware import async_geth_poa_middleware, geth_poa_middleware  # Required for Base operations
from web3.contract import Contract
from eth_typing import HexStr, BlockNumber
from hexbytes import HexBytes
from typing import Dict, List, Set, Optional, Tuple, Union, Any, cast, TypedDict, TypeVar, Literal
from eth_typing import HexStr, ChecksumAddress
from web3.types import FilterParams, _Hash32
from eth_utils.address import to_checksum_address
import logging
from datetime import datetime, timedelta
import json
import time
from eth_utils.address import to_checksum_address
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
import redis
from transformers import pipeline
from dataclasses import dataclass, asdict, field
import aioredis
import statistics
from concurrent.futures import ThreadPoolExecutor
import os
from dotenv import load_dotenv
from src.core.web3_config import get_web3, get_async_web3
from prometheus_client import Counter, Histogram, Gauge
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class SentimentScore:
    score: float
    confidence: float
    sources: Dict[str, float]

@dataclass
class ValidationResult:
    is_valid: bool
    security_score: float
    social_sentiment: SentimentScore
    timestamp: float
    metadata: Dict[str, Any]

@dataclass
class CacheConfig:
    """Configuration for token validation caching"""
    duration: int = 1800  # 30 minutes default
    refresh_threshold: int = 1500  # 25 minutes
    max_size: int = 10000  # Maximum cache entries

@dataclass
class ValidationConfig:
    """Configuration for token validation thresholds"""
    min_liquidity: float = 50000  # $50k minimum
    min_holders: int = 100
    max_top10_concentration: float = 0.8
    max_gini: float = 0.9
    min_market_cap: float = 1000000  # $1M minimum
    min_daily_volume: float = 50000  # $50k minimum daily volume

class HolderData(TypedDict, total=False):
    locked_holders: List[Dict[str, float]]
    total_holders: int
    distribution: Dict[str, float]
    gini_coefficient: float
    top_10_concentration: float

class TokenDiscovery:
    """
    TokenDiscovery class handles token discovery, validation, and analysis.
    
    This class integrates multiple data sources and validation methods to discover
    and validate new tokens. It includes caching, rate limiting, and metrics
    collection for monitoring purposes.
    
    Key Components:
    - Token discovery from multiple sources (DEX, events, social media)
    - Security validation with multiple checks
    - Liquidity analysis across DEXes
    - Social sentiment analysis using NLP
    - Holder distribution analysis
    """
    
    def __init__(self, config: Dict):
        """
        Initialize TokenDiscovery with configuration.
        
        Args:
            config: Configuration dictionary containing API keys and settings
        """
        # Get Web3 instances from centralized provider
        self.web3 = get_web3()
        self.async_web3 = get_async_web3()
        logger.info("Using centralized Web3 provider in TokenDiscovery")
        
        self.config = config
        self.discovered_tokens: Set[str] = set()
        self.token_metadata: Dict[str, Dict] = {}
        self.blacklisted_tokens: Set[str] = set()
        self.last_scan_time = datetime.now()
        self.scan_interval = timedelta(minutes=5)
        
        # Cache configuration
        self.cache_config = CacheConfig(
            duration=config.get('cache_duration', 1800),
            refresh_threshold=config.get('cache_refresh_threshold', 1500),
            max_size=config.get('cache_max_size', 10000)
        )
        
        # Validation configuration
        self.validation_config = ValidationConfig(
            min_liquidity=config.get('min_liquidity', 50000),
            min_holders=config.get('min_holders', 100),
            max_top10_concentration=config.get('max_top10_concentration', 0.8),
            max_gini=config.get('max_gini_coefficient', 0.9),
            min_market_cap=config.get('min_market_cap', 1000000),
            min_daily_volume=config.get('min_daily_volume', 50000)
        )
        
        # Initialize Redis with improved configuration
        self.redis = aioredis.from_url(
            config.get('redis_url', 'redis://localhost'),
            encoding='utf-8',
            decode_responses=True,
            max_connections=20,
            socket_timeout=5.0,
            retry_on_timeout=True
        )
        
        # Initialize components with improved configuration
        self._init_nlp()
        self._init_rate_limiters()
        self._init_metrics()
        self._init_worker_pool()
        
        # Trading configuration
        self.min_token_amount = config.get('min_token_amount', 0.1)  # Min ETH amount
        self.wallet_address = config['wallet_address']
        self.router_address = config['router_address'] 
        self.router_abi = config['router_abi']
        self.weth_address = config['weth_address']
        
        # Initialize metrics
        self.tokens_discovered = Counter('tokens_discovered_total', 'Total number of tokens discovered')
        self.scan_duration = Histogram('token_scan_duration_seconds', 'Time taken to scan for new tokens')
        self.active_tokens = Gauge('active_tokens', 'Number of currently tracked tokens')
        
    def _init_nlp(self) -> None:
        """Initialize NLP components for sentiment analysis"""
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="finiteautomata/bertweet-base-sentiment-analysis",
            device=-1  # Use CPU by default
        )
        
    def _init_rate_limiters(self) -> None:
        """Initialize rate limiters for external APIs"""
        self.rate_limiters = {
            'etherscan': asyncio.Semaphore(5),
            'dexscreener': asyncio.Semaphore(3),
            'defillama': asyncio.Semaphore(5),  # 5 concurrent requests
            'twitter': asyncio.Semaphore(10),
            'telegram': asyncio.Semaphore(5)
        }
        
    def _init_metrics(self) -> None:
        """Initialize Prometheus metrics collectors"""
        self.metrics = {
            'tokens_scanned': Counter(
                'tokens_scanned_total',
                'Total number of tokens scanned'
            ),
            'validation_time': Histogram(
                'token_validation_seconds',
                'Time spent validating tokens',
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
            ),
            'api_errors': Counter(
                'api_errors_total',
                'Total number of API errors',
                ['source']
            ),
            'cache_hits': Counter(
                'cache_hits_total',
                'Total number of cache hits'
            ),
            'security_score': Gauge(
                'token_security_score',
                'Token security score',
                ['token_address']
            ),
            'api_calls': Counter(
                'api_calls_total',
                'Total number of API calls',
                ['source']
            )
        }
        
    def _init_worker_pool(self) -> None:
        """Initialize thread pool for CPU-bound tasks"""
        self.worker_pool = ThreadPoolExecutor(
            max_workers=10,
            thread_name_prefix="TokenDiscovery"
        )
        
    async def discover_new_tokens(self) -> List[Dict]:
        """Discover new tokens using distributed processing"""
        try:
            logger.info("Starting token discovery process")
            
            # Parallel token discovery from multiple sources
            discovery_tasks = [
                self._scan_dex_listings(),
                self._scan_token_events(),
                self._scan_social_media(),
                self._scan_trending_pairs()
            ]
            
            results = await asyncio.gather(*discovery_tasks, return_exceptions=True)
            
            # Process results
            discovered_tokens = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error in discovery task: {result}")
                    continue
                discovered_tokens.extend(cast(List[Dict[str, Any]], result))
            
            return discovered_tokens
            
        except Exception as e:
            logger.error(f"Error in token discovery: {e}")
            return []
            
    async def validate_token(self, token_address: str) -> bool:
        """Comprehensive token validation with caching"""
        try:
            # Use system time for cache management
            current_time = time.time()
            
            # Check cache first
            cached_result = await self.get_cached_validation(token_address)
            if cached_result:
                self.metrics['cache_hits'].inc()
                return cached_result['is_valid']
            
            with self.metrics['validation_time'].time():
                # Basic checks
                if token_address in self.blacklisted_tokens:
                    return False
                
                # Security validation
                security_score = await self._check_security(token_address)
                self.metrics['security_score'].labels(token_address=token_address).set(security_score)
                
                # Liquidity validation
                liquidity_data = await self._check_liquidity(token_address)
                if not self._validate_liquidity(liquidity_data):
                    return False
                
                # Social sentiment analysis
                sentiment = await self.get_social_sentiment(token_address)
                
                # Store validation result with system timestamp for cache management
                result = ValidationResult(
                    is_valid=True,
                    security_score=security_score,
                    social_sentiment=sentiment,
                    timestamp=current_time,  # Use system time for cache expiration
                    metadata={
                        'liquidity': liquidity_data,
                        'holder_data': await self._analyze_holders(token_address)
                    }
                )
                
                await self._cache_validation_result(token_address, result)
            return True
            
        except Exception as e:
            logger.error(f"Token validation failed for {token_address}: {str(e)}")
            return False
            
    async def get_cached_validation(self, token_address: str) -> Optional[Dict]:
        """
        Get cached validation result with improved error handling and refresh logic.
        
        Args:
            token_address: Token address to check
            
        Returns:
            Optional[Dict]: Cached validation result or None if not found/expired
        """
        try:
            cached = await self.redis.get(f"token_validation:{token_address}")
            if not cached:
                return None
                
            result = json.loads(cached)
            current_time = time.time()
            
            # Check if cache is still valid
            if current_time - result['timestamp'] < self.cache_config.duration:
                self.metrics['cache_hits'].inc()
                
                # Check if cache needs refresh (approaching expiration)
                if current_time - result['timestamp'] > self.cache_config.refresh_threshold:
                    # Trigger async refresh without blocking
                    asyncio.create_task(self._refresh_validation(token_address, result))
                    
                return result
                
            # Cache expired
            await self.redis.delete(f"token_validation:{token_address}")
            return None
            
        except Exception as e:
            logger.error(f"Cache retrieval failed for {token_address}: {str(e)}")
            return None
            
    async def _refresh_validation(self, token_address: str, old_result: Dict) -> None:
        """
        Asynchronously refresh validation data approaching expiration.
        
        Args:
            token_address: Token address to refresh
            old_result: Previous validation result
        """
        try:
            # Perform validation in background
            new_result = await self._perform_validation(token_address)
            
            if new_result:
                # Update cache with new result
                await self._cache_validation_result(token_address, new_result)
                
        except Exception as e:
            logger.error(f"Validation refresh failed for {token_address}: {str(e)}")
            
    async def _cache_validation_result(self, token_address: str, result: ValidationResult) -> None:
        """Cache validation result"""
        try:
            await self.redis.setex(
                f"token_validation:{token_address}",
                self.cache_config.duration,
                json.dumps(asdict(result))
            )
        except Exception as e:
            logger.error(f"Cache storage failed: {str(e)}")
            
    async def _analyze_twitter_sentiment(self, token_address: str) -> float:
        """Analyze Twitter sentiment for token"""
        try:
            async with self.rate_limiters['twitter']:
                tweets = await self._fetch_token_tweets(token_address)
                if not tweets:
                    return 0.5
                sentiments = self.sentiment_analyzer(tweets)
                if not sentiments:
                    return 0.5
                scores: List[float] = [float(s.get('score', 0.5)) if isinstance(s, dict) else 0.5 for s in sentiments]
                return statistics.mean(scores)
        except Exception as e:
            logger.error(f"Twitter sentiment analysis failed: {str(e)}")
            return 0.5
            
    async def _fetch_telegram_messages(self, token_address: str) -> List[str]:
        """Fetch recent messages about a token from Telegram"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config['telegram_api_endpoint']}/messages",
                    params={'token': token_address, 'limit': 100}
                ) as response:
                    data = await response.json()
                    return [msg['text'] for msg in data.get('messages', [])]
        except Exception as e:
            logger.error(f"Error fetching Telegram messages: {str(e)}")
            return []

    async def _analyze_telegram_sentiment(self, token_address: str) -> float:
        """Analyze Telegram sentiment for token"""
        try:
            async with self.rate_limiters['telegram']:
                messages = await self._fetch_telegram_messages(token_address)
                if not messages:
                    return 0.5
                sentiments = self.sentiment_analyzer(messages)
                if not sentiments:
                    return 0.5
                scores: List[float] = [float(s.get('score', 0.5)) if isinstance(s, dict) else 0.5 for s in sentiments]
                return statistics.mean(scores)
        except Exception as e:
            logger.error(f"Telegram sentiment analysis failed: {str(e)}")
            return 0.5
            
    async def get_social_sentiment(self, token_address: str) -> SentimentScore:
        """Get combined social sentiment from X and Telegram"""
        try:
            # Only get sentiment from X and Telegram
            x_sentiment = await self._analyze_twitter_sentiment(token_address)
            telegram_sentiment = await self._analyze_telegram_sentiment(token_address)
            
            # Calculate combined score
            sources = {
                'x': x_sentiment,
                'telegram': telegram_sentiment
            }
            
            avg_score = statistics.mean(sources.values())
            confidence = len([s for s in sources.values() if s > 0.5]) / len(sources)
            
            return SentimentScore(
                score=avg_score,
                confidence=confidence,
                sources=sources
            )
        except Exception as e:
            logger.error(f"Error getting social sentiment: {str(e)}")
            return SentimentScore(score=0.5, confidence=0.0, sources={})
        
    def get_discovery_metrics(self) -> Dict[str, float]:
        """Get token discovery metrics"""
        return {
            'total_tokens_scanned': self.metrics['tokens_scanned']._value.get(),
            'validation_success_rate': self._calculate_success_rate(),
            'average_validation_time': self.metrics['validation_time'].observe(),
            'api_error_rate': self._calculate_error_rate()
        }
        
    def get_api_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get API usage metrics"""
        return {
            'requests_per_second': {
                source: self._calculate_request_rate(source)
                for source in self.rate_limiters.keys()
            }
        }
        
    def _calculate_success_rate(self) -> float:
        """Calculate token validation success rate"""
        total = self.metrics['tokens_scanned']._value.get()
        if total == 0:
            return 0.0
        return (total - sum(self.metrics['api_errors']._value.values())) / total
        
    def _calculate_error_rate(self) -> float:
        """Calculate API error rate"""
        total = self.metrics['tokens_scanned']._value.get()
        if total == 0:
            return 0.0
        return sum(self.metrics['api_errors']._value.values()) / total
        
    def _calculate_request_rate(self, source: str) -> float:
        """Calculate requests per second for a source"""
        try:
            return self.metrics['api_calls'].labels(source=source)._value.get() / 60
        except Exception:
            return 0.0
            
    async def _scan_dex_listings(self) -> List[Dict]:
        """Scan major DEXes for new token listings"""
        async with aiohttp.ClientSession() as session:
                # Scan multiple DEXes in parallel
                dexes = ['uniswap', 'sushiswap', 'pancakeswap']
                tasks = [self._scan_single_dex(session, dex) for dex in dexes]
                results = await asyncio.gather(*tasks)
                
                tokens = []
                for dex_tokens in results:
                    tokens.extend(dex_tokens)
                    
                return tokens
                
    async def _scan_single_dex(self, session: aiohttp.ClientSession, dex: str) -> List[Dict]:
        """Scan a single DEX for new token listings"""
        try:
            async with self.rate_limiters['dexscreener']:
                async with session.get(
                    f"{self.config['dexscreener_api']}/pairs/{dex}",
                    params={'limit': 100, 'sort': 'created'}
                ) as response:
                    data = await response.json()
                    return [
                        {'address': pair['token_address'], 'dex': dex}
                        for pair in data.get('pairs', [])
                    ]
        except Exception as e:
            logger.error(f"Error scanning {dex}: {str(e)}")
            return []
            
    async def _scan_token_events(self) -> List[Dict]:
        """Scan blockchain events for new token deployments"""
        try:
            # Get latest block
            latest_block = await self.async_web3.eth.get_block_number()
            from_block = latest_block - 1000  # Last 1000 blocks
            
            # Get token creation events
            event_filter = {
                'fromBlock': from_block,
                'toBlock': 'latest',
                'address': None  # Match any address
            }
            creation_filter = self.web3.eth.contract().events.Transfer().create_filter(**event_filter)
            
            events = await self.async_web3.eth.get_logs(event_filter)
            
            tokens = []
            for event in events:
                token_data = await self._process_token_event(event)
                if token_data:
                    tokens.append(token_data)
                    
            return tokens
            
        except Exception as e:
            logger.error(f"Error scanning token events: {str(e)}")
            return []
            
    async def _scan_social_media(self) -> List[Dict]:
        """Scan social media for trending tokens"""
        try:
            # Scan Twitter
            twitter_tokens = await self._scan_twitter()
            
            # Scan Telegram channels
            telegram_tokens = await self._scan_telegram()
            
            return twitter_tokens + telegram_tokens
            
        except Exception as e:
            logger.error(f"Error scanning social media: {str(e)}")
            return []
            
    async def _verify_contract(self, token_address: str) -> bool:
        """Verify contract code and check for security issues"""
        try:
            # Get contract code using async web3
            code = await self.async_web3.eth.get_code(Web3.to_checksum_address(token_address))
            if len(code) == 0:
                return False
                
            # Check for proxy implementation
            if await self._is_proxy_contract(token_address):
                impl_address = await self._get_implementation_address(token_address)
                if impl_address:
                    token_address = impl_address
                    
            # Verify source code
            async with self.rate_limiters['etherscan']:
                async with aiohttp.ClientSession() as session:
                    is_verified = await self._check_contract_verification(session, token_address)
                    if not is_verified:
                        return False
                        
            return True
            
        except Exception as e:
            logger.error(f"Error verifying contract {token_address}: {str(e)}")
            return False
            
    async def _check_security(self, token_address: str) -> float:
        """Check token contract security and calculate security score"""
        try:
            security_checks = {
                'honeypot': await self._check_honeypot(token_address),
                'ownership_renounced': await self._check_ownership(token_address),
                'mint_function': await self._check_mint_function(token_address),
                'blacklist_function': await self._check_blacklist_function(token_address),
                'proxy_safety': await self._check_proxy_safety(token_address),
                'hidden_owner': await self._check_hidden_owner(token_address)
            }
            
            # Calculate weighted security score
            weights = {
                'honeypot': 0.3,
                'ownership_renounced': 0.2,
                'mint_function': 0.15,
                'blacklist_function': 0.15,
                'proxy_safety': 0.1,
                'hidden_owner': 0.1
            }
            
            score = sum(
                weights[check] * (0 if result else 100)
                for check, result in security_checks.items()
            )
            
            return score
            
        except Exception as e:
            logger.error(f"Error checking security for {token_address}: {str(e)}")
            return 0
            
    async def _check_liquidity(self, token_address: str) -> Dict:
        """Check token liquidity across multiple DEXes"""
        try:
            async with aiohttp.ClientSession() as session:
                dex_tasks = [
                    self._get_dex_liquidity(session, token_address, dex)
                    for dex in ['uniswap', 'sushiswap', 'pancakeswap']
                ]
                
                results = await asyncio.gather(*dex_tasks)
                
                total_liquidity = sum(result.get('liquidity', 0) for result in results)
                locked_liquidity = sum(result.get('locked_liquidity', 0) for result in results)
                
                return {
                    'total_liquidity': total_liquidity,
                    'locked_liquidity': locked_liquidity,
                    'dex_distribution': results
                }
                
        except Exception as e:
            logger.error(f"Error checking liquidity for {token_address}: {str(e)}")
            return {}
            
    def _validate_liquidity(self, liquidity_data: Dict) -> bool:
        """Validate liquidity metrics"""
        try:
            if liquidity_data.get('total_liquidity', 0) < self.validation_config.min_liquidity:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating liquidity: {str(e)}")
            return False
            
    async def _analyze_holders(self, token_address: str) -> Dict[str, Any]:
        """Analyze token holder distribution"""
        try:
            async with self.rate_limiters['etherscan']:
                async with aiohttp.ClientSession() as session:
                    holders = await self._get_token_holders(session, token_address)
                    
            total_supply = sum(float(holder.get('balance', 0)) for holder in holders)
            holder_data = await self._analyze_holders(token_address)
            locked = sum(h['balance'] for h in holder_data.get('locked_holders', []))
            circulating = total_supply - locked
            
            return {
                'total_holders': len(holders),
                'distribution': self._calculate_holder_distribution(holders, total_supply),
                'gini_coefficient': self._calculate_gini(holders),
                'top_10_concentration': self._calculate_top_concentration(holders, 10)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing holders for {token_address}: {str(e)}")
            return {}
            
    def _calculate_gini(self, holders: List[Dict]) -> float:
        """Calculate Gini coefficient for holder distribution"""
        try:
            balances = sorted(holder['balance'] for holder in holders)
            n = len(balances)
            if n == 0:
                return 0
                
            cumsum = np.cumsum(balances)
            return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
            
        except Exception as e:
            logger.error(f"Error calculating Gini coefficient: {str(e)}")
            return 1.0
            
    def _calculate_top_concentration(self, holders: List[Dict], n: int) -> float:
        """Calculate concentration of top N holders"""
        try:
            if not holders:
                return 0
                
            sorted_holders = sorted(holders, key=lambda x: x['balance'], reverse=True)
            top_n = sorted_holders[:n]
            total_supply = sum(holder['balance'] for holder in holders)
            
            return sum(holder['balance'] for holder in top_n) / total_supply
            
        except Exception as e:
            logger.error(f"Error calculating top concentration: {str(e)}")
            return 1.0
            
    async def _scan_twitter(self) -> List[Dict]:
        """Scan Twitter for trending tokens"""
        try:
            headers = {'Authorization': f"Bearer {self.config['twitter_api_key']}"}
            async with aiohttp.ClientSession() as session:
                response = await session.get(
                'https://api.twitter.com/2/tweets/search/recent',
                headers=headers,
                params={
                    'query': 'crypto OR token OR defi lang:en',
                    'max_results': 100
                }
                )
                data = await response.json()
                return await self._extract_token_mentions(data)
                
        except Exception as e:
            logger.error(f"Error scanning Twitter: {str(e)}")
            return []
            
    async def _scan_telegram(self) -> List[Dict]:
        """Scan Telegram channels for token mentions"""
        try:
            channels = self.config.get('telegram_channels', [])
            tasks = [self._scan_telegram_channel(channel) for channel in channels]
            results = await asyncio.gather(*tasks)
            
            tokens = []
            for channel_tokens in results:
                tokens.extend(channel_tokens)
            return tokens
            
        except Exception as e:
            logger.error(f"Error scanning Telegram: {str(e)}")
            return []
            
    async def _check_honeypot(self, token_address: str) -> bool:
        """Check if token is a honeypot"""
        try:
            # Simulate buy transaction
            buy_success = await self._simulate_trade(token_address, True)
            if not buy_success:
                return True
                
            # Simulate sell transaction
            sell_success = await self._simulate_trade(token_address, False)
            if not sell_success:
                return True
                
            # Check transfer restrictions
            if await self._has_transfer_restrictions(token_address):
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking honeypot for {token_address}: {str(e)}")
            return True
            
    async def _simulate_trade(self, token_address: str, is_buy: bool) -> bool:
        """Simulate a trade to check for restrictions"""
        try:
            # Prepare simulation parameters
            amount = Web3.to_wei(0.1, 'ether') if is_buy else self.min_token_amount
            
            # Create transaction parameters
            params = {
                'wallet_address': self.wallet_address,
                'dex_address': self.router_address,
                'value': amount if is_buy else 0,
                'gas_estimate': 500000,  # Conservative estimate
                'data': self._encode_swap_data(token_address, amount, is_buy)
            }
            
            # Simulate the transaction
            result = await asyncio.to_thread(self.web3.eth.call, {
                'to': params['dex_address'],
                'from': params['wallet_address'],
                'value': params['value'],
                'data': params['data'],
                'gas': params['gas_estimate']
            })
            
            if not result or result == HexBytes("0x"):
                logger.warning(f"{'Buy' if is_buy else 'Sell'} simulation failed for {token_address}: empty result")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error simulating {'buy' if is_buy else 'sell'} trade for {token_address}: {str(e)}")
            return False

    def _encode_swap_data(self, token_address: str, amount: int, is_buy: bool) -> str:
        """Encode swap function data for simulation"""
        try:
            router = self.web3.eth.contract(
                address=self.router_address,
                abi=self.router_abi
            )
            
            path = [self.weth_address, token_address] if is_buy else [token_address, self.weth_address]
            deadline = int(time.time()) + 300  # 5 minutes
            
            if is_buy:
                return router.encodeABI(
                    fn_name='swapExactETHForTokens',
                    args=[
                        0,  # Min amount out (0 for simulation)
                        path,
                        self.wallet_address,
                        deadline
                    ]
                )
            else:
                return router.encodeABI(
                    fn_name='swapExactTokensForETH',
                    args=[
                        amount,
                        0,  # Min amount out (0 for simulation)
                        path,
                        self.wallet_address,
                        deadline
                    ]
                )
                
        except Exception as e:
            logger.error(f"Error encoding swap data: {str(e)}")
            return '0x'
            
    async def _has_transfer_restrictions(self, token_address: str) -> bool:
        """Check for transfer restrictions in token contract"""
        try:
            code = await self.web3.eth.get_code.call(token_address)
            code_hex = code.hex()
            
            # Check for common restriction patterns
            restriction_patterns = [
                '0x42966c68',  # burn function
                '0x75f0a674',  # blacklist function
                '0x8456cb59',  # pause function
            ]
            
            for pattern in restriction_patterns:
                if pattern in code_hex:
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error checking transfer restrictions for {token_address}: {str(e)}")
            return True
            
    async def _check_ownership(self, token_address: str) -> bool:
        """Check if contract ownership is renounced"""
        try:
            contract = self.web3.eth.contract(
                address=Web3.to_checksum_address(token_address),
                abi=self.config['token_abi']
            )
            
            try:
                owner = await contract.functions.owner().call()
                return owner == '0x0000000000000000000000000000000000000000'
            except Exception:
                # No owner function, check for Ownable pattern
                return False
                
        except Exception as e:
            logger.error(f"Error checking ownership for {token_address}: {str(e)}")
            return False
            
    async def _get_dex_liquidity(self, session: aiohttp.ClientSession, token_address: str, dex: str) -> Dict:
        """Get token liquidity information from a specific DEX"""
        try:
            async with session.get(
                f"https://api.dexscreener.com/latest/dex/tokens/{token_address}",
                params={'dex': dex}
            ) as response:
                data = await response.json()
                
                if 'pairs' not in data:
                    return {'liquidity': 0, 'locked_liquidity': 0}
                    
                total_liquidity = sum(
                    float(pair.get('liquidity', {}).get('usd', 0))
                    for pair in data['pairs']
                )
                
                locked_liquidity = await self._get_locked_liquidity(
                    session, token_address, dex
                )
                
                return {
                    'liquidity': total_liquidity,
                    'locked_liquidity': locked_liquidity,
                    'pairs': len(data['pairs'])
                }
                
        except Exception as e:
            logger.error(f"Error getting DEX liquidity for {token_address} on {dex}: {str(e)}")
            return {'liquidity': 0, 'locked_liquidity': 0}
            
    async def _get_locked_liquidity(self, session: aiohttp.ClientSession, token_address: str, dex: str) -> float:
        """Get amount of locked liquidity for a token"""
        try:
            # Check common liquidity lockers
            lockers = [
                'unicrypt',
                'team.finance',
                'pinksale',
                'dxsale'
            ]
            
            tasks = []
            for locker in lockers:
                tasks.append(self._check_locker(session, token_address, locker))
                
            results = await asyncio.gather(*tasks)
            return sum(results)
            
        except Exception as e:
            logger.error(f"Error getting locked liquidity for {token_address}: {str(e)}")
            return 0.0
            
    async def _check_locker(self, session: aiohttp.ClientSession, token_address: str, locker: str) -> float:
        """Check specific liquidity locker for locked amount"""
        try:
            if locker == 'unicrypt':
                return await self._check_unicrypt(session, token_address)
            elif locker == 'team.finance':
                return await self._check_teamfinance(session, token_address)
            elif locker == 'pinksale':
                return await self._check_pinksale(token_address)
            elif locker == 'dxsale':
                return await self._check_dxsale(token_address)
            return 0.0
            
        except Exception as e:
            logger.error(f"Error checking {locker} for {token_address}: {str(e)}")
            return 0.0
            
    async def _check_unicrypt(self, session: aiohttp.ClientSession, token_address: str) -> float:
        """Check Unicrypt locker for locked liquidity"""
        try:
            async with session.get(
                f"https://api.unicrypt.network/api/v1/lock/{token_address}"
            ) as response:
                data = await response.json()
                if not data.get('success'):
                    return 0.0
                    
                locks = data.get('result', [])
                total_locked = sum(
                    float(lock.get('amount', 0)) * float(lock.get('token_price', 0))
                    for lock in locks
                    if not self._is_lock_expired(lock)
                )
                
                return total_locked
                
        except Exception as e:
            logger.error(f"Error checking Unicrypt for {token_address}: {str(e)}")
            return 0.0
            
    async def _check_teamfinance(self, session: aiohttp.ClientSession, token_address: str) -> float:
        """Check Team Finance locker for locked liquidity"""
        try:
            async with session.get(
                f"https://api.team.finance/api/v1/locks/{token_address}"
            ) as response:
                data = await response.json()
                if not data.get('success'):
                    return 0.0
                    
                locks = data.get('locks', [])
                return sum(
                    float(lock.get('amount', 0)) * float(lock.get('price', 0))
                    for lock in locks
                    if not self._is_lock_expired(lock)
                )
                
        except Exception as e:
            logger.error(f"Error checking Team Finance for {token_address}: {str(e)}")
            return 0.0
            
    def _is_lock_expired(self, lock: Dict) -> bool:
        """Check if a liquidity lock has expired"""
        try:
            unlock_time = int(lock.get('unlock_time', 0))
            return unlock_time < datetime.now().timestamp()
        except Exception:
            return True
            
    async def _analyze_volume(self, token_address: str) -> Dict:
        """Analyze trading volume patterns"""
        try:
            async with aiohttp.ClientSession() as session:
                # Get volume data from multiple sources
                dex_volume = await self._get_dex_volume(session, token_address)
                cex_volume = await self._get_cex_volume(session, token_address)
                
                # Calculate volume metrics
                total_volume = dex_volume + cex_volume
                volume_distribution = {
                    'dex_volume': dex_volume,
                    'cex_volume': cex_volume,
                    'total_volume': total_volume,
                    'dex_ratio': dex_volume / total_volume if total_volume > 0 else 0
                }
                
                return volume_distribution
                
        except Exception as e:
            logger.error(f"Error analyzing volume for {token_address}: {str(e)}")
            return {}
            
    async def _get_dex_volume(self, session: aiohttp.ClientSession, token_address: str) -> float:
        """Get total DEX trading volume"""
        try:
            total_volume = 0
            dexes = ['uniswap', 'sushiswap', 'pancakeswap']
            
            for dex in dexes:
                async with session.get(
                    f"https://api.dexscreener.com/latest/dex/tokens/{token_address}",
                    params={'dex': dex}
                ) as response:
                    data = await response.json()
                    pairs = data.get('pairs', [])
                    
                    dex_volume = sum(
                        float(pair.get('volume', {}).get('h24', 0))
                        for pair in pairs
                    )
                    
                    total_volume += dex_volume
                    
            return total_volume
            
        except Exception as e:
            logger.error(f"Error getting DEX volume for {token_address}: {str(e)}")
            return 0.0
            
    async def _get_cex_volume(self, session: aiohttp.ClientSession, token_address: str) -> float:
        """Get total CEX trading volume"""
        try:
            # Check major CEXes
            cex_apis = {
                'binance': f"https://api.binance.com/api/v3/ticker/24hr?symbol={token_address}USDT",
                'kucoin': f"https://api.kucoin.com/api/v1/market/stats?symbol={token_address}-USDT",
                'huobi': f"https://api.huobi.pro/market/detail?symbol={token_address.lower()}usdt"
            }
            
            total_volume = 0
            for cex, url in cex_apis.items():
                try:
                    async with session.get(url) as response:
                        data = await response.json()
                        volume = self._extract_cex_volume(cex, data)
                        total_volume += volume
                except Exception:
                    continue
                    
            return total_volume
            
        except Exception as e:
            logger.error(f"Error getting CEX volume for {token_address}: {str(e)}")
            return 0.0
            
    def _extract_cex_volume(self, cex: str, data: Dict) -> float:
        """Extract 24h volume from CEX API response"""
        try:
            if cex == 'binance':
                return float(data.get('volume', 0)) * float(data.get('lastPrice', 0))
            elif cex == 'kucoin':
                return float(data.get('data', {}).get('volValue', 0))
            elif cex == 'huobi':
                return float(data.get('tick', {}).get('vol', 0))
            return 0.0
        except Exception:
            return 0.0
            
    def _validate_volume(self, volume_data: Dict) -> bool:
        """Validate volume metrics"""
        try:
            min_volume = self.config.get('min_daily_volume', 10000)  # $10k minimum
            min_dex_ratio = self.config.get('min_dex_volume_ratio', 0.3)  # 30% minimum on DEXes
            
            if volume_data.get('total_volume', 0) < min_volume:
                return False
                
            if volume_data.get('dex_ratio', 0) < min_dex_ratio:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating volume: {str(e)}")
            return False
            
    def _validate_holder_distribution(self, holder_data: Dict) -> bool:
        """Validate holder distribution metrics"""
        try:
            if holder_data.get('total_holders', 0) < self.validation_config.min_holders:
                return False
                
            if holder_data.get('top_10_concentration', 1.0) > self.validation_config.max_top10_concentration:
                return False
                
            if holder_data.get('gini_coefficient', 1.0) > self.validation_config.max_gini:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating holder distribution: {str(e)}")
            return False
            
    async def simulate_transaction(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a transaction using eth_call"""
        try:
            result = await asyncio.to_thread(self.web3.eth.call, {
                'to': params['dex_address'],
                'from': params['wallet_address'],
                'value': params['value'],
                'data': params['data'],
                'gas': params['gas_estimate']
            })
            return {'success': True, 'result': result}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _scan_trending_pairs(self) -> List[Dict]:
        """Scan for trending trading pairs"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.config['trending_pairs_endpoint']) as response:
                    data = await response.json()
                    return [
                        {'address': pair['token_address'], 'symbol': pair['symbol']}
                        for pair in data.get('pairs', [])
                    ]
        except Exception as e:
            logger.error(f"Error scanning trending pairs: {str(e)}")
            return []

    async def _perform_validation(self, token_address: str) -> Optional[ValidationResult]:
        """Perform comprehensive token validation"""
        try:
            security_score = await self._check_security(token_address)
            sentiment = await self.get_social_sentiment(token_address)
            liquidity_data = await self._check_liquidity(token_address)
            holder_data = await self._analyze_holders(token_address)
            
            return ValidationResult(
                is_valid=True,
                security_score=security_score,
                social_sentiment=sentiment,
                timestamp=time.time(),
                metadata={
                    'liquidity': liquidity_data,
                    'holder_data': holder_data
                }
            )
        except Exception as e:
            logger.error(f"Validation failed for {token_address}: {str(e)}")
            return None

    async def _fetch_token_tweets(self, token_address: str) -> List[str]:
        """Fetch recent tweets about a token"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config['twitter_api_endpoint']}/search",
                    params={'q': token_address, 'count': 100}
                ) as response:
                    data = await response.json()
                    return [tweet['text'] for tweet in data.get('statuses', [])]
        except Exception as e:
            logger.error(f"Error fetching tweets: {str(e)}")
            return []

    async def _analyze_historical_volume(self, token_address: str) -> Dict[str, float]:
        """Analyze historical trading volume patterns for reliability"""
        try:
            async with aiohttp.ClientSession() as session:
                # Get 30 days of volume data
                async with session.get(
                    f"{self.config['dexscreener_api']}/tokens/{token_address}/history",
                    params={'interval': '1d', 'limit': 30}
                ) as response:
                    data = await response.json()
                    volumes = [float(d['volume']) for d in data.get('history', [])]
                    
                    if not volumes:
                        return {
                            'avg_daily_volume': 0,
                            'volume_stability': 0,
                            'volume_trend': 0,
                            'is_reliable': False
                        }

                    avg_volume = statistics.mean(volumes)
                    volume_std = statistics.stdev(volumes) if len(volumes) > 1 else float('inf')
                    
                    # Calculate volume trend (positive is good)
                    volume_trend = (sum(volumes[-7:]) / 7) - (sum(volumes[:7]) / 7) if len(volumes) >= 14 else 0
                    
                    # Volume stability score (lower variance is better)
                    volume_stability = 1 - min(1, volume_std / avg_volume if avg_volume > 0 else float('inf'))
                    
                    return {
                        'avg_daily_volume': avg_volume,
                        'volume_stability': volume_stability,
                        'volume_trend': volume_trend,
                        'is_reliable': (
                            avg_volume >= self.validation_config.min_daily_volume and
                            volume_stability >= 0.7 and
                            volume_trend >= 0
                        )
                    }
        except Exception as e:
            logger.error(f"Error analyzing historical volume: {str(e)}")
            return {
                'avg_daily_volume': 0,
                'volume_stability': 0,
                'volume_trend':0,
                'is_reliable': False
            }

    async def _detect_rugpull_risk(self, token_address: str) -> Dict[str, Any]:
        """Advanced rugpull risk detection"""
        try:
            risk_factors = {
                'contract_risk': await self._analyze_contract_risk(token_address),
                'liquidity_risk': await self._analyze_liquidity_risk(token_address),
                'ownership_risk': await self._analyze_ownership_risk(token_address),
                'trading_risk': await self._analyze_trading_patterns(token_address)
            }
            
            # Weight the risk factors
            weights = {
                'contract_risk': 0.35,
                'liquidity_risk': 0.30,
                'ownership_risk': 0.20,
                'trading_risk': 0.15
            }
            
            total_risk = sum(
                risk * weights[factor]
                for factor, risk in risk_factors.items()
            )
            
            return {
                'risk_score': total_risk,
                'risk_factors': risk_factors,
                'is_safe': total_risk < 0.3,  # 30% risk threshold
                'warnings': self._generate_risk_warnings(risk_factors)
            }
            
        except Exception as e:
            logger.error(f"Error in rugpull detection: {str(e)}")
            return {
                'risk_score': 1.0,
                'risk_factors': {},
                'is_safe': False,
                'warnings': ['Unable to complete risk analysis']
            }
            
    async def _analyze_contract_risk(self, token_address: str) -> float:
        """Analyze smart contract risk factors"""
        try:
            # Check for risky functions
            code = await self.web3.eth.get_code.call(token_address)
            code_hex = code.hex()
            risky_patterns = [
                'selfdestruct',
                'delegatecall',
                'transfer.*(owner|admin)',
                'mint(?!.*capped)',
                '_blacklist_'
            ]
            
            risk_score = 0.0
            for pattern in risky_patterns:
                if pattern.encode() in code:
                    risk_score += 0.2
                    
            return min(1.0, risk_score)
            
        except Exception as e:
            logger.error(f"Error analyzing contract risk: {str(e)}")
            return 1.0
            
    async def _analyze_liquidity_risk(self, token_address: str) -> float:
        """Analyze liquidity-based risk factors"""
        try:
            liquidity_data = await self._check_liquidity(token_address)
            
            risk_score = 0.0
            
            # Check total liquidity
            if liquidity_data['total_liquidity'] < self.validation_config.min_liquidity:
                risk_score += 0.3
                
            # Check DEX distribution
            if len(liquidity_data['dex_distribution']) < 2:
                risk_score += 0.3
                
            return min(1.0, risk_score)
            
        except Exception as e:
            logger.error(f"Error analyzing liquidity risk: {str(e)}")
            return 1.0
            
    async def _analyze_ownership_risk(self, token_address: str) -> float:
        """Analyze ownership and control risk factors"""
        try:
            holder_data = await self._analyze_holders(token_address)
            
            risk_score = 0.0
            
            # Check holder concentration
            if holder_data['top_10_concentration'] > self.validation_config.max_top10_concentration:
                risk_score += 0.4
                
            # Check Gini coefficient
            if holder_data['gini_coefficient'] > self.validation_config.max_gini:
                risk_score += 0.3
                
            # Check if ownership is renounced
            if not await self._check_ownership(token_address):
                risk_score += 0.3
                
            return min(1.0, risk_score)
            
        except Exception as e:
            logger.error(f"Error analyzing ownership risk: {str(e)}")
            return 1.0
            
    async def _analyze_trading_patterns(self, token_address: str) -> float:
        """Analyze trading patterns for suspicious activity"""
        try:
            volume_data = await self._analyze_historical_volume(token_address)
            
            risk_score = 0.0
            
            # Check volume stability
            if volume_data['volume_stability'] < 0.7:
                risk_score += 0.3
                
            # Check volume trend
            if volume_data['volume_trend'] < 0:
                risk_score += 0.3
                
            # Check average volume
            if volume_data['avg_daily_volume'] < self.validation_config.min_daily_volume:
                risk_score += 0.4
                
            return min(1.0, risk_score)
            
        except Exception as e:
            logger.error(f"Error analyzing trading patterns: {str(e)}")
            return 1.0
            
    def _generate_risk_warnings(self, risk_factors: Dict[str, float]) -> List[str]:
        """Generate human-readable risk warnings"""
        warnings = []
        
        if risk_factors.get('contract_risk', 0) > 0.5:
            warnings.append('High-risk functions detected in contract code')
            
        if risk_factors.get('liquidity_risk', 0) > 0.5:
            warnings.append('Insufficient or poorly distributed liquidity')
            
        if risk_factors.get('ownership_risk', 0) > 0.5:
            warnings.append('Concentrated ownership or control risks')
            
        if risk_factors.get('trading_risk', 0) > 0.5:
            warnings.append('Suspicious trading patterns detected')
            
        return warnings

    async def _get_token_holders(self, session: aiohttp.ClientSession, token_address: str) -> List[Dict]:
        """Get list of token holders using Web3"""
        try:
            # Get Transfer events for the token
            contract = self.web3.eth.contract(
                address=Web3.to_checksum_address(token_address),
                abi=self.config['token_abi']
            )
            
            # Get current block
            current_block = await self.async_web3.eth.get_block_number()
            from_block = max(0, current_block - 1000000)  # Last ~1M blocks
            
            # Get Transfer events with proper typing
            transfer_filter = {
                'fromBlock': BlockNumber(from_block),
                'toBlock': BlockNumber(0),  # 'latest' block
                'address': Web3.to_checksum_address(token_address)
            }
            
            events = await self.async_web3.eth.get_logs(cast(FilterParams, transfer_filter))
            
            # Process events to get current holders
            holders = {}
            for event in events:
                decoded_event = contract.events.Transfer().process_log(event)
                from_addr = decoded_event.args['from']
                to_addr = decoded_event.args['to']
                value = decoded_event.args['value']
                
                if from_addr in holders:
                    holders[from_addr] = max(0, holders[from_addr] - value)
                if to_addr not in holders:
                    holders[to_addr] = 0
                holders[to_addr] += value
            
            # Convert to list format
            return [
                {'address': addr, 'balance': bal}
                for addr, bal in holders.items()
                if bal > 0
            ]
            
        except Exception as e:
            logger.error(f"Error getting token holders: {str(e)}")
            return []

    def _calculate_holder_distribution(self, holders: List[Dict], total_supply: int) -> Dict[str, float]:
        """Calculate holder distribution metrics"""
        try:
            if not holders or total_supply == 0:
                return {
                    'top_10_percent': 1.0,
                    'top_50_percent': 1.0,
                    'gini': 1.0
                }

            # Sort holders by balance
            sorted_holders = sorted(holders, key=lambda x: float(x.get('balance', 0)), reverse=True)
            
            # Calculate percentages
            top_10_balance = sum(float(h.get('balance', 0)) for h in sorted_holders[:10])
            top_50_balance = sum(float(h.get('balance', 0)) for h in sorted_holders[:50])
            
            return {
                'top_10_percent': top_10_balance / total_supply,
                'top_50_percent': top_50_balance / total_supply,
                'gini': self._calculate_gini(sorted_holders)
            }
            
        except Exception as e:
            logger.error(f"Error calculating holder distribution: {str(e)}")
            return {
                'top_10_percent': 1.0,
                'top_50_percent': 1.0,
                'gini': 1.0
            }

    async def _analyze_market_cap(self, token_address: str) -> Dict[str, Any]:
        """Analyze market cap and related metrics"""
        try:
            async with aiohttp.ClientSession() as session:
                # Get token supply and price data
                supply = await self._get_token_supply(token_address)
                price_data = await self._get_token_price(session, token_address)
                
                if not supply or not price_data:
                    return {
                        'market_cap': 0,
                        'fully_diluted_cap': 0,
                        'price': 0,
                        'is_valid': False
                    }
                
                market_cap = supply['circulating'] * price_data['price']
                fully_diluted = supply['total'] * price_data['price']
                
                return {
                    'market_cap': market_cap,
                    'fully_diluted_cap': fully_diluted,
                    'price': price_data['price'],
                    'is_valid': (
                        market_cap >= self.validation_config.min_market_cap and
                        fully_diluted <= market_cap * 3  # Max 3x dilution
                    )
                }
                
        except Exception as e:
            logger.error(f"Error analyzing market cap: {str(e)}")
            return {
                'market_cap': 0,
                'fully_diluted_cap': 0,
                'price': 0,
                'is_valid': False
            }
            
    async def _get_token_supply(self, token_address: str) -> Optional[Dict[str, float]]:
        """Get token supply information"""
        try:
            contract = self.web3.eth.contract(
                address=Web3.to_checksum_address(token_address),
                abi=self.config['token_abi']
            )
            
            total_supply = await contract.functions.totalSupply().call()
            
            # Try to get circulating supply from contract
            try:
                circulating = await contract.functions.circulatingSupply().call()
            except Exception:
                # Estimate circulating supply from holder data
                holder_data = await self._analyze_holders(token_address)
                locked = sum(h['balance'] for h in holder_data.get('locked_holders', []))
                circulating = total_supply - locked
            
            return {
                'total': total_supply,
                'circulating': circulating
            }
            
        except Exception as e:
            logger.error(f"Error getting token supply: {str(e)}")
            return None
            
    async def _get_token_price(self, session: aiohttp.ClientSession, token_address: str) -> Optional[Dict[str, float]]:
        """Get token price information"""
        try:
            async with session.get(
                f"{self.config['dexscreener_api']}/tokens/{token_address}"
            ) as response:
                data = await response.json()
                pairs = data.get('pairs', [])
                
                if not pairs:
                    return None
                    
                # Use the most liquid pair for price
                best_pair = max(pairs, key=lambda p: float(p.get('liquidity', {}).get('usd', 0)))
                
                return {
                    'price': float(best_pair['priceUsd']),
                    'price_change_24h': float(best_pair['priceChange']['h24'])
                }
                
        except Exception as e:
            logger.error(f"Error getting token price: {str(e)}")
            return None

    async def _process_token_event(self, event) -> Optional[Dict]:
        """Process a token transfer event to extract token data"""
        try:
            token_address = event.address
            if token_address in self.discovered_tokens:
                return None
                
            if await self._verify_contract(token_address):
                self.discovered_tokens.add(token_address)
                return {
                    'address': token_address,
                    'source': 'event',
                    'block_number': event.blockNumber
                }
            return None
            
        except Exception as e:
            logger.error(f"Error processing token event: {str(e)}")
            return None

    async def _is_proxy_contract(self, token_address: str) -> bool:
        """Check if contract is a proxy by looking for proxy patterns"""
        try:
            code = await self.async_web3.eth.get_code(Web3.to_checksum_address(token_address))
            code_hex = code.hex()
            
            # Common proxy patterns
            proxy_patterns = [
                '0x5c60da1b',  # implementation()
                '0x360894a1',  # proxiableUUID()
                '0xf851a440',  # admin()
            ]
            
            return any(pattern in code_hex for pattern in proxy_patterns)
            
        except Exception as e:
            logger.error(f"Error checking proxy status: {str(e)}")
            return False

    async def _get_implementation_address(self, token_address: str) -> Optional[str]:
        """Get implementation address of a proxy contract"""
        try:
            code = await self.async_web3.eth.get_code(Web3.to_checksum_address(token_address))
            code_hex = code.hex()
            
            # Check for proxy patterns
            if '0x5c60da1b' in code_hex:
                # Extract implementation address
                implementation_address = code_hex[code_hex.find('5c60da1b') + 4:code_hex.find('5c60da1b') + 44]
                return Web3.to_checksum_address(implementation_address)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting implementation address: {str(e)}")
            return None
            
    async def _check_contract_verification(self, session: aiohttp.ClientSession, token_address: str) -> bool:
        """Check if contract is verified on Base Scan"""
        try:
            async with session.get(
                f"{self.config['basescan_api']}/api",
                params={
                    'module': 'contract',
                    'action': 'verify',
                    'contractaddress': token_address,
                    'apikey': self.config['basescan_key']
                }
            ) as response:
                data = await response.json()
                return data.get('status') == '1' and data.get('result', [{}])[0].get('SourceCode')
        except Exception as e:
            logger.error(f"Error checking contract verification: {str(e)}")
            return False

    async def _check_mint_function(self, token_address: str) -> bool:
        """Check for unrestricted mint function"""
        try:
            code = await self.async_web3.eth.get_code(Web3.to_checksum_address(token_address))
            code_hex = code.hex()
            return '0x40c10f19' in code_hex  # mint(address,uint256)
        except Exception as e:
            logger.error(f"Error checking mint function: {str(e)}")
            return True

    async def _check_blacklist_function(self, token_address: str) -> bool:
        """Check for blacklist functionality"""
        try:
            code = await self.async_web3.eth.get_code(Web3.to_checksum_address(token_address))
            code_hex = code.hex()
            blacklist_signatures = [
                '0x50f6d37e',  # addToBlacklist(address)
                '0xa1755b99'   # setBlacklist(address,bool)
            ]
            return any(sig in code_hex for sig in blacklist_signatures)
        except Exception as e:
            logger.error(f"Error checking blacklist function: {str(e)}")
            return True

    async def _check_proxy_safety(self, token_address: str) -> bool:
        """Check if proxy implementation is safe"""
        try:
            if not await self._is_proxy_contract(token_address):
                return True
            impl_address = await self._get_implementation_address(token_address)
            if not impl_address:
                return False
            return await self._verify_contract(impl_address)
        except Exception as e:
            logger.error(f"Error checking proxy safety: {str(e)}")
            return False

    async def _check_hidden_owner(self, token_address: str) -> bool:
        """Check for hidden owner patterns"""
        try:
            code = await self.async_web3.eth.get_code(Web3.to_checksum_address(token_address))
            code_hex = code.hex()
            hidden_patterns = [
                'delegatecall',
                'selfdestruct',
                'suicide'
            ]
            return any(pattern in code_hex.lower() for pattern in hidden_patterns)
        except Exception as e:
            logger.error(f"Error checking hidden owner: {str(e)}")
            return True

    async def _extract_token_mentions(self, data: Dict) -> List[Dict]:
        """Extract token addresses from social media data"""
        try:
            mentions = []
            for item in data.get('statuses', []):
                text = item.get('text', '')
                # Look for Ethereum addresses
                addresses = re.findall(r'0x[a-fA-F0-9]{40}', text)
                for addr in addresses:
                    if await self._verify_contract(addr):
                        mentions.append({
                            'address': addr,
                            'source': 'social',
                            'timestamp': item.get('created_at')
                        })
            return mentions
        except Exception as e:
            logger.error(f"Error extracting token mentions: {str(e)}")
            return []

    async def _scan_telegram_channel(self, channel: str) -> List[Dict]:
        """Scan a single Telegram channel for token mentions"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config['telegram_api']}/messages",
                    params={
                        'channel': channel,
                        'limit': 100
                    }
                ) as response:
                    data = await response.json()
                    return await self._extract_token_mentions(data)
        except Exception as e:
            logger.error(f"Error scanning Telegram channel {channel}: {str(e)}")
            return []

    async def get_contract(self, address: str) -> Contract:
        """Get contract instance with proper address conversion"""
        checksum_address = cast(ChecksumAddress, Web3.to_checksum_address(address))
        return self.web3.eth.contract(address=checksum_address, abi=self.config['token_abi'])

    async def get_code(self, address: str) -> bytes:
        """Get contract bytecode with proper address conversion"""
        checksum_address = cast(ChecksumAddress, Web3.to_checksum_address(address))
        code = await self.async_web3.eth.get_code(checksum_address)
        return bytes(code)

    async def _check_pinksale(self, token_address: str) -> bool:
        """Check if token has an active presale on PinkSale"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config['pinksale_api']}/presales",
                    params={'token': Web3.to_checksum_address(token_address)}
                ) as response:
                    data = await response.json()
                    return bool(data.get('active_presale'))
        except Exception as e:
            logger.error(f"Error checking PinkSale: {str(e)}")
            return False

    async def _check_dxsale(self, token_address: str) -> bool:
        """Check if token has an active presale on DxSale"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config['dxsale_api']}/presales",
                    params={'token': Web3.to_checksum_address(token_address)}
                ) as response:
                    data = await response.json()
                    return bool(data.get('active_presale'))
        except Exception as e:
            logger.error(f"Error checking DxSale: {str(e)}")
            return False 