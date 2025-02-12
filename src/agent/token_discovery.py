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
from src.utils.rate_limiter import RateLimiterRegistry
from src.utils.metrics import MetricsManager
from src.utils.cache import AsyncCache
from transformers import pipeline
import torch
from asyncio import Semaphore

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Cache configuration"""
    duration: int
    max_size: int
    refresh_threshold: int = field(default=300)  # 5 minutes

@dataclass
class TokenData:
    """Token data structure"""
    address: str
    source: str
    block_number: Optional[int] = None
    timestamp: Optional[float] = None

@dataclass
class ValidationResult:
    """Validation result structure"""
    is_valid: bool
    security_score: float
    social_sentiment: float
    timestamp: float
    metadata: Dict[str, Any]

class TokenDiscovery:
    def __init__(self, config: Dict[str, Any], web3: Web3):
        self.config = config
        self.web3 = web3
        self.async_web3 = get_async_web3()
        self.discovered_tokens: Set[str] = set()
        self.metrics = MetricsManager()
        
        # Initialize components
        self._init_nlp()
        self._init_rate_limiters()
        self._init_worker_pool()
        
        # Enhanced caching with TTL and validation
        self.cache_config = CacheConfig(
            duration=3600,  # 1 hour
            max_size=1000,
            refresh_threshold=300  # 5 minutes
        )
        
        # Initialize Redis and cache
        try:
            self.redis = aioredis.from_url(
                self.config['redis_url'],
                encoding='utf-8',
                decode_responses=True
            )
            self.cache = AsyncCache(str(self.config['redis_url']))
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            self.redis = None
            self.cache = None
        
        # Enhanced validation tracking
        self.validation_stats: Dict[str, Dict[str, Any]] = {}
        self.validation_threshold = 0.7  # 70% confidence required
        
        # Initialize rate limiters
        self.rate_limiters: Dict[str, Semaphore] = {}
        self._setup_rate_limiters()

    def _init_nlp(self) -> None:
        """Initialize NLP components"""
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )
        except Exception as e:
            logger.error(f"Failed to initialize sentiment analyzer: {e}")
            self.sentiment_analyzer = None

    def _init_rate_limiters(self) -> None:
        """Initialize rate limiters"""
        self.rate_limiters = {
            'telegram': Semaphore(5),
            'discord': Semaphore(10),
            'defillama': Semaphore(2),
            'etherscan': Semaphore(5)
        }

    def _init_worker_pool(self) -> None:
        """Initialize thread pool for CPU-bound tasks"""
        self.worker_pool = ThreadPoolExecutor(
            max_workers=10,
            thread_name_prefix="TokenDiscovery"
        )

    def _setup_rate_limiters(self) -> None:
        """Configure rate limiters for different APIs"""
        # Rate limiters are already initialized in _init_rate_limiters
        pass

    async def discover_new_tokens(self) -> List[Dict]:
        """
        Discover new tokens from multiple sources with enhanced error handling
        and parallel processing
        
        Returns:
            List of discovered token data
        """
        try:
            # Record start time for metrics
            start_time = time.time()
            
            # Create tasks for different sources
            tasks = [
                self._scan_dex_listings(),
                self._scan_token_events(),
                self._scan_social_media(),
                self._scan_token_transfers()
            ]
            
            # Execute tasks with timeout
            try:
                results = await asyncio.gather(
                    *tasks,
                    return_exceptions=True
                )
            except asyncio.TimeoutError:
                logger.error("Token discovery timed out")
                self.metrics.record_api_error('discovery', 'timeout')
                return []
                
            # Process results with validation
            discovered_tokens = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    source = ['dex', 'events', 'social', 'transfers'][i]
                    logger.error(f"Error in {source} discovery: {result}")
                    self.metrics.record_api_error('discovery', f"{source}_{str(result)}")
                    continue
                
                # Validate and filter results
                if isinstance(result, list):
                    valid_tokens = await self._validate_discovered_tokens(result)
                    discovered_tokens.extend(valid_tokens)
                
            # Update metrics
            self.metrics.record_discovery('total', len(discovered_tokens))
            duration = time.time() - start_time
            self.metrics.validation_duration.observe(duration)
            
            # Cache results if Redis is available
            if self.redis is not None:
                await self._cache_discovered_tokens(discovered_tokens)
            
            return discovered_tokens
            
        except Exception as e:
            logger.error(f"Error in token discovery: {e}", exc_info=True)
            self.metrics.record_api_error('discovery', str(e))
            return []

    async def _scan_token_transfers(self) -> List[Dict]:
        """Scan for new tokens from transfer events"""
        try:
            # Get latest block
            latest_block = await self.async_web3.eth.block_number
            
            # Create filter for token transfers
            transfer_topic = self.web3.keccak(
                text='Transfer(address,address,uint256)'
            ).hex()
            
            # Get transfer events
            filter_params: FilterParams = {
                'fromBlock': latest_block - 1000,
                'toBlock': latest_block,
                'topics': [cast(_Hash32, HexBytes(transfer_topic))]
            }
            
            logs = await self.async_web3.eth.get_logs(filter_params)
            
            # Process transfer events
            tokens = []
            seen_addresses = set()
            
            for log in logs:
                token_address = log['address']
                if token_address in seen_addresses:
                    continue
                    
                seen_addresses.add(token_address)
                tokens.append({
                    'address': token_address,
                    'source': 'transfer',
                    'block_number': log['blockNumber'],
                    'timestamp': datetime.now().isoformat()
                })
                
            return tokens
            
        except Exception as e:
            logger.error(f"Error scanning token transfers: {e}")
            return []

    async def _validate_discovered_tokens(
        self,
        tokens: List[Dict]
    ) -> List[Dict]:
        """
        Validate discovered tokens with enhanced checks
        
        Args:
            tokens: List of token data to validate
            
        Returns:
            List of validated token data
        """
        valid_tokens = []
        
        for token in tokens:
            try:
                # Skip if already discovered
                if token['address'] in self.discovered_tokens:
                    continue
                    
                # Basic validation
                if not self._is_valid_address(token['address']):
                    continue
                    
                # Check contract code
                if not await self._has_contract_code(token['address']):
                    continue
                    
                # Validate token interface
                if not await self._validate_token_interface(token['address']):
                    continue
                    
                # Check for suspicious patterns
                if await self._check_suspicious_patterns(token['address']):
                    logger.warning(f"Suspicious patterns detected for {token['address']}")
                    continue
                    
                # Validate liquidity
                liquidity = await self._check_liquidity(token['address'])
                if liquidity < self.config['min_liquidity']:
                    continue
                    
                # Add validation metadata
                token['validation'] = {
                    'timestamp': datetime.now().isoformat(),
                    'liquidity': liquidity,
                    'source_reliability': self._get_source_reliability(token['source']),
                    'confidence': self._calculate_confidence(token)
                }
                
                valid_tokens.append(token)
                self.discovered_tokens.add(token['address'])
                
            except Exception as e:
                logger.error(
                    f"Error validating token {token.get('address')}: {e}",
                    extra={'token': token},
                    exc_info=True
                )
                continue
                
        return valid_tokens

    async def _check_suspicious_patterns(self, address: str) -> bool:
        """
        Check for suspicious patterns in token contract
        
        Args:
            address: Token address to check
            
        Returns:
            True if suspicious patterns found, False otherwise
        """
        try:
            # Get contract code
            code = await self.get_code(address)
            
            # Check for known malicious patterns
            suspicious_patterns = [
                b'selfdestruct',
                b'delegatecall',
                b'transfer.ownership',
                b'mint',  # Potential unlimited minting
                b'blacklist'  # Potential blacklisting
            ]
            
            for pattern in suspicious_patterns:
                if pattern in code:
                    logger.warning(
                        f"Suspicious pattern {pattern} found in {address}",
                        extra={'address': address, 'pattern': pattern}
                    )
                    return True
                    
            # Check for honeypot characteristics
            if await self._check_honeypot(address):
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking suspicious patterns: {e}")
            return True  # Fail safe

    def _calculate_confidence(self, token: Dict) -> float:
        """Calculate confidence score for token validation"""
        try:
            scores = [
                self._get_source_reliability(token['source']),
                token.get('liquidity', 0) / self.config['min_liquidity'],
                1.0 if token.get('verified_contract') else 0.5,
                token.get('social_sentiment', 0.5)
            ]
            return statistics.mean(scores)
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0

    async def _cache_discovered_tokens(self, tokens: List[Dict]) -> None:
        """Cache discovered tokens with validation data"""
        if not self.redis:
            return
            
        try:
            pipe = self.redis.pipeline()
            for token in tokens:
                key = f"token:{token['address']}"
                pipe.setex(
                    key,
                    self.cache_config.duration,
                    json.dumps(token)
                )
            await pipe.execute()
        except Exception as e:
            logger.error(f"Error caching tokens: {e}")

    async def _check_liquidity(self, token_address: str) -> float:
        """
        Check token liquidity across multiple DEXes
        
        Args:
            token_address: Token address to check
            
        Returns:
            Total liquidity in USD
        """
        try:
            # Check liquidity in parallel
            tasks = [
                self._check_uniswap_liquidity(token_address),
                self._check_sushiswap_liquidity(token_address),
                self._check_curve_liquidity(token_address)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Sum up valid liquidity values
            total_liquidity = 0.0
            for result in results:
                if isinstance(result, (int, float)):
                    total_liquidity += float(result)
                    
            return total_liquidity
            
        except Exception as e:
            logger.error(f"Error checking liquidity: {e}")
            return 0.0

    async def _validate_token_interface(self, address: str) -> bool:
        """
        Validate token implements required interfaces
        
        Args:
            address: Token address to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            contract = await self.get_contract(address)
            
            # Check ERC20 interface
            required_functions = [
                'totalSupply',
                'balanceOf',
                'transfer',
                'allowance',
                'approve',
                'transferFrom'
            ]
            
            for func in required_functions:
                if not hasattr(contract.functions, func):
                    logger.warning(f"Missing {func} function in {address}")
                    return False
                    
            # Verify function signatures
            try:
                await contract.functions.totalSupply().call()
                await contract.functions.decimals().call()
                await contract.functions.symbol().call()
            except Exception as e:
                logger.error(f"Error calling token functions: {e}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating token interface: {e}")
            return False

    def _get_source_reliability(self, source: str) -> float:
        """Get reliability score for discovery source"""
        reliability_scores = {
            'dex': 0.8,
            'event': 0.7,
            'social': 0.5,
            'transfer': 0.6
        }
        return reliability_scores.get(source, 0.5)

    async def _scan_dex_listings(self) -> List[Dict]:
        """Scan DEX listings for new tokens"""
        try:
            async with aiohttp.ClientSession() as session:
                # Scan multiple DEXes in parallel
                tasks = [
                    self._scan_single_dex(session, dex)
                    for dex in ['uniswap', 'sushiswap', 'pancakeswap']
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                tokens = []
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Error scanning DEX: {result}")
                        self.metrics.record_api_error('dex_scan', str(result))
                        continue
                        
                    tokens.extend(result)
                    self.metrics.record_discovery('dex')
                    
                return tokens
                
        except Exception as e:
            logger.error(f"Error scanning DEX listings: {e}")
            return []
            
    async def _scan_single_dex(
        self,
        session: aiohttp.ClientSession,
        dex: str
    ) -> List[Dict]:
        """
        Scan a single DEX for new token listings
        
        Args:
            session: aiohttp session
            dex: DEX name
            
        Returns:
            List of discovered tokens
        """
        try:
            async with self.rate_limiters['defillama']:
                # Get new listings from DEX
                async with session.get(
                    f"{self.config['defillama_api']}/dex/{dex}/new-tokens"
                ) as response:
                    if response.status != 200:
                        logger.error(
                            f"Error response from {dex}: "
                            f"{response.status}"
                        )
                        return []
                        
                    data = await response.json()
                    return [
                        {
                            'address': token['address'],
                            'source': f'dex_{dex}',
                            'timestamp': time.time()
                        }
                        for token in data.get('tokens', [])
                    ]
                    
        except Exception as e:
            logger.error(f"Error scanning {dex}: {e}")
            return []
            
    async def _scan_token_events(self) -> List[TokenData]:
        """Scan blockchain events for new token deployments"""
        try:
            # Get latest block
            latest_block = await self.async_web3.eth.block_number
            
            # Create filter for token creation events
            token_filter = {
                'fromBlock': latest_block - 1000,  # Last ~1000 blocks
                'toBlock': latest_block,
                'topics': [
                    # ERC20 Transfer event signature
                    '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef',
                    # From zero address (token creation)
                    '0x0000000000000000000000000000000000000000000000000000000000000000'
                ]
            }
            
            # Get logs
            logs = await self.async_web3.eth.get_logs(token_filter)
            
            # Process logs
            tokens = []
            for log in logs:
                token = await self._process_token_event(log)
                if token:
                    tokens.append(token)
                    self.metrics.record_discovery('blockchain')
                    
            return tokens
            
        except Exception as e:
            logger.error(f"Error scanning token events: {e}")
            return []
            
    async def _scan_social_media(self) -> List[Dict]:
        """Scan social media for token mentions"""
        try:
            # Scan Telegram and Discord in parallel
            tasks = [
                self._scan_telegram(),
                self._scan_discord()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            tokens = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error in social media scan: {result}")
                    self.metrics.record_api_error('social_scan', str(result))
                    continue
                    
                tokens.extend(result)
                self.metrics.record_discovery('social')
                
            return tokens
            
        except Exception as e:
            logger.error(f"Error scanning social media: {e}")
            return []
            
    async def _scan_telegram(self) -> List[Dict]:
        """Scan Telegram channels for token mentions"""
        try:
            async with self.rate_limiters['telegram']:
                # Implementation depends on your Telegram API integration
                return []
            
        except Exception as e:
            logger.error(f"Error scanning Telegram: {e}")
            return []
            
    async def _scan_discord(self) -> List[Dict]:
        """Scan Discord channels for token mentions"""
        try:
            async with self.rate_limiters['discord']:
                # Implementation depends on your Discord API integration
            return []
            
        except Exception as e:
            logger.error(f"Error scanning Discord: {e}")
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

    async def _process_token_event(self, event: LogReceipt) -> Optional[TokenData]:
        """Process a token transfer event to extract token data"""
        try:
            # Convert LogReceipt to dict for easier handling
            event_dict = {
                'address': event.get('address'),
                'blockNumber': event.get('blockNumber'),
                'transactionHash': event.get('transactionHash'),
                'topics': event.get('topics', [])
            }
            
            token_address = event_dict['address']
            if not token_address or token_address in self.discovered_tokens:
                return None
            
            if await self._verify_contract(token_address):
                self.discovered_tokens.add(token_address)
                return TokenData(
                    address=token_address,
                    source='event',
                    block_number=int(event_dict['blockNumber'])
                )
            return None
            
        except Exception as e:
            logger.error(f"Error processing token event: {str(e)}")
            return None
        
    async def _get_token_holders(self, session: aiohttp.ClientSession, token_address: str) -> List[HolderInfo]:
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
            transfer_filter = cast(FilterParams, {
                'fromBlock': BlockNumber(from_block),
                'toBlock': BlockNumber(current_block),
                'address': Web3.to_checksum_address(token_address),
                'topics': [self.web3.keccak(text='Transfer(address,address,uint256)').hex()]
            })
            
            events = await self.async_web3.eth.get_logs(transfer_filter)
            
            # Process events to get current holders
            holders: Dict[str, float] = {}
            for event in events:
                try:
                    decoded_event = contract.events.Transfer().process_log(event)
                    from_addr = decoded_event.args.get('from')
                    to_addr = decoded_event.args.get('to')
                    value = float(decoded_event.args.get('value', 0))
                    
                    if from_addr in holders:
                        holders[from_addr] = max(0.0, holders[from_addr] - value)
                    if to_addr not in holders:
                        holders[to_addr] = 0.0
                    holders[to_addr] += value
                except Exception as e:
                    logger.warning(f"Error processing transfer event: {str(e)}")
                    continue
            
            # Convert to HolderInfo objects
            return [
                HolderInfo(address=addr, balance=bal)
                for addr, bal in holders.items()
                if bal > 0
            ]
            
        except Exception as e:
            logger.error(f"Error getting token holders: {str(e)}")
            return []

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
            
    async def _check_honeypot(self, address: str) -> bool:
        """
        Check if token contract has honeypot characteristics
        
        Args:
            address: Token address to check
            
        Returns:
            True if honeypot characteristics found, False otherwise
        """
        try:
            # Get contract code and check for honeypot patterns
            code = await self.get_code(address)
            
            # Check for common honeypot patterns
            honeypot_patterns = [
                b'blacklist',
                b'maxTxAmount',
                b'maxSellAmount',
                b'_maxTxAmount',
                b'_maxSellAmount'
            ]
            
            for pattern in honeypot_patterns:
                if pattern in code:
                    logger.warning(
                        f"Honeypot pattern {pattern} found in {address}",
                        extra={'address': address, 'pattern': pattern}
                    )
                    return True
            
            # Check for suspicious function modifiers
            suspicious_modifiers = [
                b'onlyOwner',
                b'onlyWhitelisted',
                b'onlyAuthorized'
            ]
            
            modifier_count = sum(1 for pattern in suspicious_modifiers if pattern in code)
            if modifier_count >= 2:
                logger.warning(
                    f"Multiple suspicious modifiers found in {address}",
                    extra={'address': address, 'modifier_count': modifier_count}
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking for honeypot: {e}")
            return True  # Fail safe

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
            
    async def _get_dex_liquidity(self, session: aiohttp.ClientSession, token_address: str, dex: str) -> DexLiquidity:
        """Get token liquidity information from a specific DEX"""
        try:
            async with session.get(
                f"https://api.dexscreener.com/latest/dex/tokens/{token_address}",
                params={'dex': dex}
            ) as response:
                data = await response.json()
                
                if 'pairs' not in data:
                    return DexLiquidity(liquidity=0.0, locked_liquidity=0.0, pairs=0)
                    
                total_liquidity = sum(
                    float(pair.get('liquidity', {}).get('usd', 0))
                    for pair in data['pairs']
                )
                
                locked_liquidity = await self._get_locked_liquidity(
                    session, token_address, dex
                )
                
                return DexLiquidity(
                    liquidity=total_liquidity,
                    locked_liquidity=locked_liquidity,
                    pairs=len(data['pairs'])
                )
            
        except Exception as e:
            logger.error(f"Error getting DEX liquidity for {token_address} on {dex}: {str(e)}")
            return DexLiquidity(liquidity=0.0, locked_liquidity=0.0, pairs=0)
            
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
                locked = await self._check_pinksale_lock(token_address)
                return float(locked if locked else 0)
            elif locker == 'dxsale':
                locked = await self._check_dxsale_lock(token_address)
                return float(locked if locked else 0)
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
        
    async def _check_pinksale_lock(self, token_address: str) -> Optional[float]:
        """Check if token has locked liquidity on PinkSale"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config['pinksale_api']}/presales",
                    params={'token': Web3.to_checksum_address(token_address)}
                ) as response:
                    data = await response.json()
                    if not data.get('success'):
                        return None
                    return float(data.get('locked_amount', 0))
        except Exception as e:
            logger.error(f"Error checking PinkSale lock: {str(e)}")
            return None

    async def _check_dxsale_lock(self, token_address: str) -> Optional[float]:
        """Check if token has locked liquidity on DxSale"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config['dxsale_api']}/presales",
                    params={'token': Web3.to_checksum_address(token_address)}
                ) as response:
                    data = await response.json()
                    if not data.get('success'):
                        return None
                    return float(data.get('locked_amount', 0))
        except Exception as e:
            logger.error(f"Error checking DxSale lock: {str(e)}")
            return None

    async def _check_uniswap_liquidity(self, token_address: str) -> float:
        """Check token liquidity on Uniswap"""
        try:
            # Implementation for checking Uniswap liquidity
            return 0.0
        except Exception as e:
            logger.error(f"Error checking Uniswap liquidity: {e}")
            return 0.0

    async def _check_sushiswap_liquidity(self, token_address: str) -> float:
        """Check token liquidity on SushiSwap"""
        try:
            # Implementation for checking SushiSwap liquidity
            return 0.0
        except Exception as e:
            logger.error(f"Error checking SushiSwap liquidity: {e}")
            return 0.0

    async def _check_curve_liquidity(self, token_address: str) -> float:
        """Check token liquidity on Curve"""
        try:
            # Implementation for checking Curve liquidity
            return 0.0
        except Exception as e:
            logger.error(f"Error checking Curve liquidity: {e}")
            return 0.0

    def _is_valid_address(self, address: str) -> bool:
        """Check if address is valid"""
        try:
            return Web3.is_address(address) and Web3.is_checksum_address(address)
        except Exception as e:
            logger.error(f"Error validating address: {e}")
            return False

    async def _has_contract_code(self, address: str) -> bool:
        """Check if address has contract code"""
        try:
            code = await self.get_code(address)
            return len(code) > 0
        except Exception as e:
            logger.error(f"Error checking contract code: {e}")
            return False 