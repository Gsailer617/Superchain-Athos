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

import asyncio
import aiohttp
from web3 import Web3
from web3.contract.contract import Contract
from eth_typing import HexStr, BlockNumber, ChecksumAddress
from hexbytes import HexBytes
from typing import Dict, List, Set, Optional, Tuple, Union, Any, cast, TypedDict, TypeVar, Literal, Iterable, Sequence
from web3.types import FilterParams, _Hash32, EventData, LogReceipt
from eth_utils.address import to_checksum_address
import logging
from datetime import datetime, timedelta
import json
import time
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
from dataclasses import dataclass, asdict, field
import statistics
import os
from dotenv import load_dotenv
from src.core.web3_config import get_web3, get_async_web3
import re
import aioredis
from concurrent.futures import ThreadPoolExecutor
from src.utils.rate_limiter import RateLimiterRegistry, AsyncRateLimiter
from src.utils.metrics import MetricsManager
from src.utils.cache import AsyncCache, CacheConfig
from transformers import pipeline
import torch
from asyncio import Semaphore
from functools import lru_cache, wraps
from src.ml.parallel_processor import DistributedProcessor, ParallelConfig
from bloom_filter import BloomFilter
import hashlib
import zlib
from src.modules.token_economics import TokenEconomics

logger = logging.getLogger(__name__)

@dataclass
class TokenData:
    """Token data structure"""
    address: str
    source: str
    block_number: Optional[int] = None
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    chain_id: int = 1

@dataclass
class ValidationResult:
    """Validation result structure"""
    is_valid: bool
    security_score: float
    social_sentiment: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    chain_id: int = 1

class TokenDiscovery:
    def __init__(self, config: Dict[str, Any], web3: Optional[Web3] = None):
        self.config = config
        self.web3 = web3 or get_web3()
        self.async_web3 = get_async_web3()
        self.discovered_tokens: Set[str] = set()
        self.metrics = MetricsManager()
        
        # Initialize components
        self._init_nlp()
        self._init_rate_limiters()
        self._init_worker_pool()
        
        # Enhanced tiered caching with different TTLs based on importance
        self.cache_config = {
            'high_priority': CacheConfig(
                duration=1800,  # 30 minutes
            max_size=1000,
            refresh_threshold=300  # 5 minutes
            ),
            'medium_priority': CacheConfig(
                duration=3600,  # 1 hour
                max_size=5000,
                refresh_threshold=600  # 10 minutes
            ),
            'low_priority': CacheConfig(
                duration=86400,  # 24 hours
                max_size=10000,
                refresh_threshold=3600  # 1 hour
            )
        }
        
        # Bloom filter for quick existence checks (reduces Redis load)
        self.token_filter = BloomFilter(
            max_elements=100000,
            error_rate=0.001
        )
        
        # Initialize Redis and cache
        try:
            self.redis = aioredis.from_url(
                self.config.get('redis_url', 'redis://localhost:6379'),
                encoding='utf-8',
                decode_responses=True
            )
            self.cache = AsyncCache(self.config.get('redis_url', 'redis://localhost:6379'))
            
            # Prime the bloom filter with existing tokens
            self._prime_bloom_filter()
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            self.redis = None
            self.cache = None
        
        # Enhanced validation tracking with circuit breakers
        self.validation_stats: Dict[str, Dict[str, Any]] = {}
        self.validation_threshold = 0.7  # 70% confidence required
        
        # Circuit breakers for external services
        self.circuit_breakers = {
            'etherscan': {'failures': 0, 'threshold': 5, 'open': False, 'last_attempt': 0},
            'defillama': {'failures': 0, 'threshold': 5, 'open': False, 'last_attempt': 0},
            'coingecko': {'failures': 0, 'threshold': 5, 'open': False, 'last_attempt': 0}
        }
        
        # Improved semaphores for rate limiting
        self._setup_rate_limiters()
        
        # Initialize token economics service for validation
        self.token_economics = TokenEconomics(self.web3)
        
        # Initialize parallel processor for distributed tasks
        parallel_config = ParallelConfig(
            num_cpus=self.config.get('num_cpus', 4),
            num_gpus=self.config.get('num_gpus', 0),
            use_dask=self.config.get('use_dask', True)
        )
        self.parallel_processor = DistributedProcessor(parallel_config)

    async def _prime_bloom_filter(self):
        """Prime bloom filter with existing tokens from Redis"""
        if self.redis:
            try:
                # Get all token addresses from Redis
                keys = await self.redis.keys("token:*")
                for key in keys:
                    token_address = key.split(":")[-1]
                    self.token_filter.add(token_address)
                logger.info(f"Primed bloom filter with {len(keys)} tokens")
            except Exception as e:
                logger.error(f"Failed to prime bloom filter: {e}")

    def _init_nlp(self) -> None:
        """Initialize NLP components"""
        try:
            # Use existing sentiment analyzer if loaded in a previous instance
            if not hasattr(TokenDiscovery, '_sentiment_analyzer'):
                TokenDiscovery._sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )
            self.sentiment_analyzer = TokenDiscovery._sentiment_analyzer
        except Exception as e:
            logger.error(f"Failed to initialize sentiment analyzer: {e}")
            self.sentiment_analyzer = None

    def _init_rate_limiters(self) -> None:
        """Initialize rate limiters"""
        # Use the global rate limiter registry instead of creating local rate limiters
        self.rate_limiter_registry = RateLimiterRegistry()

    def _init_worker_pool(self) -> None:
        """Initialize thread pool for CPU-bound tasks with dynamic sizing"""
        # Dynamic thread pool sizing based on system load
        import psutil
        cpu_count = psutil.cpu_count(logical=False) or 4
        load_avg = psutil.getloadavg()[0] / cpu_count
        
        # More workers when system is less loaded, fewer when loaded
        optimal_workers = max(4, int(cpu_count * (1 - min(load_avg, 0.8))))
        
        self.worker_pool = ThreadPoolExecutor(
            max_workers=optimal_workers,
            thread_name_prefix="TokenDiscovery"
        )
        logger.info(f"Initialized worker pool with {optimal_workers} workers")

    def _setup_rate_limiters(self) -> None:
        """Configure rate limiters for different APIs using the global registry"""
        # External APIs with various rate limits
        rate_limiters = {
            'telegram': {'max_requests': 5, 'requests_per_second': 0.5, 'burst_size': 5},
            'discord': {'max_requests': 10, 'requests_per_second': 1.0, 'burst_size': 15},
            'defillama': {'max_requests': 2, 'requests_per_second': 0.2, 'burst_size': 3},
            'etherscan': {'max_requests': 5, 'requests_per_second': 0.5, 'burst_size': 10},
            'coingecko': {'max_requests': 10, 'requests_per_second': 1.0, 'burst_size': 20},
            'twitter': {'max_requests': 2, 'requests_per_second': 0.2, 'burst_size': 5},
            'dexscreener': {'max_requests': 3, 'requests_per_second': 0.3, 'burst_size': 5}
        }
        
        # Register rate limiters in the global registry
        for name, config in rate_limiters.items():
            self.rate_limiter_registry.get_limiter(
                name,
                max_requests=config['max_requests'],
                requests_per_second=config['requests_per_second'],
                burst_size=config['burst_size']
            )

    async def discover_new_tokens(self, chain_id: int = 1) -> List[Dict]:
        """
        Discover new tokens from multiple sources with enhanced error handling,
        parallel processing, and progressive streaming of results
        
        Args:
            chain_id: Blockchain ID to discover tokens on
        
        Returns:
            List of discovered token data
        """
        try:
            # Record start time for metrics
            start_time = time.time()
            
            # Subscribe to real-time discovery streamer
            discovery_queue = asyncio.Queue()
            processor_task = asyncio.create_task(self._process_discovery_queue(discovery_queue))
            
            # Create tasks for different sources with backpressure management
            sources = [
                self._scan_dex_listings(chain_id, discovery_queue),
                self._scan_token_events(chain_id, discovery_queue),
                self._scan_social_media(chain_id, discovery_queue),
                self._scan_token_transfers(chain_id, discovery_queue)
            ]
            
            # Add optional sources if enabled in config
            if self.config.get('enable_mempool_monitoring', False):
                sources.append(self._scan_mempool(chain_id, discovery_queue))
                
            if self.config.get('enable_contract_creation_monitoring', False):
                sources.append(self._scan_contract_creation(chain_id, discovery_queue))
            
            # Execute tasks with timeout and backpressure
            try:
                producers = asyncio.gather(
                    *sources,
                    return_exceptions=True
                )
                
                # Start producer tasks with timeout
                timeout = self.config.get('discovery_timeout', 60)
                await asyncio.wait_for(producers, timeout=timeout)
                
            except asyncio.TimeoutError:
                logger.warning("Token discovery timed out, but will continue processing found tokens")
                self.metrics.record_api_error('discovery', 'timeout')
            finally:
                # Signal queue completion
                await discovery_queue.put(None)
                
            # Wait for all tokens to be processed
            discovered_tokens = await processor_task
                
            # Update metrics
            discovery_duration = time.time() - start_time
            self.metrics.validation_duration.observe(discovery_duration)
            self.metrics.record_discovery('total', len(discovered_tokens))
            logger.info(f"Discovered {len(discovered_tokens)} tokens in {discovery_duration:.2f}s")
            
            return discovered_tokens
            
        except Exception as e:
            logger.error(f"Error in token discovery: {e}", exc_info=True)
            self.metrics.record_api_error('discovery', str(e))
            return []

    async def _process_discovery_queue(self, queue: asyncio.Queue) -> List[Dict]:
        """Process discovered tokens as they stream in from multiple sources"""
        discovered_tokens = []
        processing_tasks = set()
        max_concurrent = self.config.get('max_concurrent_validations', 10)
        
        while True:
            # Get next token from queue
            token_data = await queue.get()
            
            # Check for end of stream
            if token_data is None:
                break
                
            # Skip if already in bloom filter (fast check before validation)
            if token_data['address'] in self.token_filter:
                queue.task_done()
                    continue
                    
            # Add to bloom filter
            self.token_filter.add(token_data['address'])
            
            # Manage max concurrent validations with backpressure
            while len(processing_tasks) >= max_concurrent:
                # Wait for at least one task to complete
                done, processing_tasks = await asyncio.wait(
                    processing_tasks,
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Add results from completed tasks
                for task in done:
                    try:
                        result = await task
                        if result:
                            discovered_tokens.append(result)
            except Exception as e:
                        logger.error(f"Error in validation task: {e}")
            
            # Create task for validating this token
            task = asyncio.create_task(self._validate_and_process_token(token_data))
            processing_tasks.add(task)
            task.add_done_callback(lambda t: processing_tasks.remove(t) if t in processing_tasks else None)
            
            queue.task_done()
        
        # Wait for any remaining tasks
        if processing_tasks:
            done, _ = await asyncio.wait(processing_tasks)
            for task in done:
                try:
                    result = await task
                    if result:
                        discovered_tokens.append(result)
        except Exception as e:
                    logger.error(f"Error in validation task: {e}")
        
        return discovered_tokens

    async def _validate_and_process_token(self, token_data: Dict) -> Optional[Dict]:
        """Validate and process a single token with progressive validation"""
        try:
            # Quick validation first (inexpensive checks)
            if not await self._quick_validation(token_data):
                return None
            
            # Medium validation (more thorough but still efficient)
            if not await self._medium_validation(token_data):
                return None
                
            # Full validation (expensive checks)
            validation_result = await self._validate_token(token_data)
            
            if validation_result and validation_result.is_valid:
                # Cache the validated token
                await self._cache_token(token_data, validation_result)
                
                # Prepare result with validation data
                result = {**token_data, 'validation': asdict(validation_result)}
                return result
                
            return None
            
        except Exception as e:
            logger.error(f"Error validating token {token_data.get('address')}: {e}")
            return None

    async def _quick_validation(self, token_data: Dict) -> bool:
        """Perform quick inexpensive validation checks"""
        address = token_data.get('address')
        
        # Check address format
        if not Web3.is_address(address):
            return False
            
        # Check if token already exists in cache (but not in bloom filter - rare case)
        if self.redis:
            cached = await self.redis.exists(f"token:{address}")
            if cached:
                    return False
                    
        # Check basic contract existence
        try:
            code = await self.async_web3.eth.get_code(address)
            if code == b'' or code == HexBytes('0x'):
                return False
        except Exception:
                return False
                
            return True
            
    async def _medium_validation(self, token_data: Dict) -> bool:
        """Perform medium-cost validation checks"""
        address = token_data.get('address')
        
        # Check if it's an ERC20 token by testing basic interface
        try:
            # Check if contract has standard ERC20 functions
            contract = self.web3.eth.contract(address=address, abi=[
                {"constant":True,"inputs":[],"name":"name","outputs":[{"name":"","type":"string"}],"type":"function"},
                {"constant":True,"inputs":[],"name":"symbol","outputs":[{"name":"","type":"string"}],"type":"function"},
                {"constant":True,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"type":"function"}
            ])
            
            # Try to call basic methods (will fail if not ERC20)
            symbol = await asyncio.to_thread(lambda: contract.functions.symbol().call())
            decimals = await asyncio.to_thread(lambda: contract.functions.decimals().call())
            
            # Skip tokens with suspicious characteristics
            if not symbol or len(symbol) > 30:
                return False
                
            if decimals > 18:
                return False
                
            # Add metadata
            token_data['metadata'] = {
                'symbol': symbol,
                'decimals': decimals
                        }
            
            return True
                    
        except Exception as e:
            logger.debug(f"Medium validation failed for {address}: {e}")
            return False
            
    async def _scan_token_transfers(self, chain_id: int, queue: asyncio.Queue) -> None:
        """Scan for new tokens from transfer events with backpressure"""
        try:
            # Get latest block
            latest_block = await self.async_web3.eth.block_number
            
            # Create filter for token transfers
            transfer_topic = self.web3.keccak(
                text='Transfer(address,address,uint256)'
            ).hex()
            
            # Get transfer events in chunks to manage memory
            chunk_size = 500
            from_block = latest_block - 5000  # Larger range for better discovery
            
            for chunk_start in range(from_block, latest_block, chunk_size):
                chunk_end = min(chunk_start + chunk_size - 1, latest_block)
                
                # Apply backpressure if queue is getting full
                while queue.qsize() > 100:
                    await asyncio.sleep(0.1)
                
                # Get logs for this chunk
                filter_params: FilterParams = {
                    'fromBlock': chunk_start,
                    'toBlock': chunk_end,
                    'topics': [cast(_Hash32, HexBytes(transfer_topic))]
                }
                
                logs = await self.async_web3.eth.get_logs(filter_params)
            
                # Process transfer events
                seen_addresses = set()
                
                for log in logs:
                    token_address = log['address']
                    if token_address in seen_addresses:
                    continue
                    
                    seen_addresses.add(token_address)
                    
                    # Push to queue for further processing
                    await queue.put({
                        'address': token_address,
                        'source': 'transfer',
                        'block_number': log['blockNumber'],
                        'timestamp': time.time(),
                        'chain_id': chain_id
                    })
                
                # Prevent overwhelming the network
                await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error scanning token transfers: {e}")
            self.metrics.record_api_error('transfers', str(e))

    async def _cache_token(self, token_data: Dict, validation_result: ValidationResult) -> None:
        """Cache validated token with appropriate TTL based on priority"""
        if not self.redis or not self.cache:
            return
            
        address = token_data.get('address', '')
        
        try:
            # Determine token priority based on validation score
            priority = 'low_priority'
            if validation_result.security_score > 0.8:
                priority = 'high_priority'
            elif validation_result.security_score > 0.5:
                priority = 'medium_priority'
                
            # Prepare data for caching
            cache_data = {
                'token': token_data,
                'validation': asdict(validation_result),
                'timestamp': time.time()
            }
            
            # Get appropriate cache config
            config = self.cache_config[priority]
            
            # Cache with TTL based on priority
            await self.redis.setex(
                f"token:{address}",
                config.duration,
                json.dumps(cache_data)
            )
            
            # Track validation in stats
            self.validation_stats[address] = {
                'timestamp': time.time(),
                'score': validation_result.security_score,
                'priority': priority
            }
                
        except Exception as e:
            logger.error(f"Error caching token: {e}")

    # The remaining methods would be improved similarly but I will leave
    # them as they are for brevity in this example
    # Methods like _scan_dex_listings, _scan_social_media, etc. would follow 
    # the same pattern of improvements as _scan_token_transfers
    
    # Error handling with circuit breaker
    async def _call_external_api(self, api_name: str, url: str) -> Optional[Dict]:
        """Call external API with circuit breaker pattern"""
        # Check if circuit is open (too many failures)
        breaker = self.circuit_breakers.get(api_name)
        if not breaker:
            return None
            
        if breaker['open']:
            # Circuit is open, check if we should retry
            if time.time() - breaker['last_attempt'] < 60:  # Wait at least 60s
            return None

            # Try to reset circuit
            breaker['open'] = False
            breaker['failures'] = 0
            
        # Get rate limiter
        try:
            rate_limiter = self.rate_limiter_registry.get_limiter(api_name)
        except KeyError:
            # Fallback if no limiter defined
            rate_limiter = self.rate_limiter_registry.get_limiter(
                api_name, 
                max_requests=1, 
                requests_per_second=0.2
            )
        
        try:
            # Use rate limiter as context manager
            async with rate_limiter:
                # Make API call
            async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=10) as response:
                        if response.status == 200:
                            # Reset failures on success
                            breaker['failures'] = 0
                            return await response.json()
            else:
                            # Increment failures
                            breaker['failures'] += 1
                            breaker['last_attempt'] = time.time()
                            
                            # Open circuit if threshold reached
                            if breaker['failures'] >= breaker['threshold']:
                                breaker['open'] = True
                                logger.warning(f"Circuit breaker opened for {api_name}")
                                
                            return None
        except Exception as e:
            # Increment failures
            breaker['failures'] += 1
            breaker['last_attempt'] = time.time()
            
            # Open circuit if threshold reached
            if breaker['failures'] >= breaker['threshold']:
                breaker['open'] = True
                logger.warning(f"Circuit breaker opened for {api_name} due to {str(e)}")
                
            return None

# ... remainder of the file omitted for brevity ... 