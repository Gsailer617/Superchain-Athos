from typing import Dict, Optional, Any, Union
import logging
from web3 import Web3
from decimal import Decimal
import aiohttp
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class PriceFeed(ABC):
    """Abstract base class for price feeds"""
    
    @abstractmethod
    async def get_price(
        self,
        token_address: str,
        chain: str,
        web3: Web3
    ) -> Optional[float]:
        """Get price for token on specific chain"""
        pass

class ChainlinkPriceFeed(PriceFeed):
    """Chainlink price feed implementation"""
    
    def __init__(self):
        self.feed_addresses = {
            'ethereum': {
                'WETH': '0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419',
                'USDC': '0x8fFfFfd4AfB6115b954Bd326cbe7B4BA576818f6',
                'USDT': '0x3E7d1eAB13ad0104d2750B8863b489D65364e32D'
            },
            'polygon': {
                'WMATIC': '0xAB594600376Ec9fD91F8e885dADF0CE036862dE0',
                'WETH': '0xF9680D99D6C9589e2a93a78A04A279e509205945',
                'USDC': '0xfE4A8cc5b5B2366C1B58Bea3858e81843581b2F7'
            },
            'base': {
                'WETH': '0x71041dddad3595F9CEd3DcCFBe3D1F4b0a16Bb70',
                'USDbC': '0x7e860098F58bBFC8648a4311b374B1D669a2bc6B',
                'USDC': '0x7e860098F58bBFC8648a4311b374B1D669a2bc6B'
            }
        }
        
        self.abi = [
            {
                "inputs": [],
                "name": "latestRoundData",
                "outputs": [
                    {"internalType": "uint80", "name": "roundId", "type": "uint80"},
                    {"internalType": "int256", "name": "answer", "type": "int256"},
                    {"internalType": "uint256", "name": "startedAt", "type": "uint256"},
                    {"internalType": "uint256", "name": "updatedAt", "type": "uint256"},
                    {"internalType": "uint80", "name": "answeredInRound", "type": "uint80"}
                ],
                "stateMutability": "view",
                "type": "function"
            }
        ]
    
    async def get_price(
        self,
        token_address: str,
        chain: str,
        web3: Web3
    ) -> Optional[float]:
        """Get price from Chainlink feed"""
        try:
            feed_address = self.feed_addresses.get(chain, {}).get(token_address)
            if not feed_address:
                return None
                
            contract = web3.eth.contract(
                address=web3.to_checksum_address(feed_address),
                abi=self.abi
            )
            
            latest_data = contract.functions.latestRoundData().call()
            price = Decimal(latest_data[1]) / Decimal(10**8)  # Chainlink uses 8 decimals
            
            return float(price)
            
        except Exception as e:
            logger.error(f"Error getting Chainlink price for {token_address} on {chain}: {str(e)}")
            return None

class UniswapV3PriceFeed(PriceFeed):
    """Uniswap V3 TWAP price feed implementation"""
    
    def __init__(self):
        self.factory_addresses = {
            'ethereum': '0x1F98431c8aD98523631AE4a59f267346ea31F984',
            'polygon': '0x1F98431c8aD98523631AE4a59f267346ea31F984',
            'base': '0x33128a8fC17869897dcE68Ed026d694621f6FDfD'
        }
        
        self.pool_addresses = {}  # Cache for pool addresses
    
    async def get_price(
        self,
        token_address: str,
        chain: str,
        web3: Web3
    ) -> Optional[float]:
        """Get price from Uniswap V3 TWAP"""
        try:
            # Implementation would use Uniswap V3 pool contract to get TWAP
            # This is a placeholder
            return None
            
        except Exception as e:
            logger.error(f"Error getting Uniswap V3 price for {token_address} on {chain}: {str(e)}")
            return None

class SubgraphPriceFeed(PriceFeed):
    """TheGraph subgraph price feed implementation"""
    
    def __init__(self):
        self.subgraph_urls = {
            'ethereum': 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3',
            'polygon': 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3-polygon',
            'base': 'https://api.studio.thegraph.com/query/48211/baseswap/version/latest'
        }
    
    async def get_price(
        self,
        token_address: str,
        chain: str,
        web3: Web3
    ) -> Optional[float]:
        """Get price from subgraph"""
        try:
            subgraph_url = self.subgraph_urls.get(chain)
            if not subgraph_url:
                return None
                
            query = """
            {
                token(id: "%s") {
                    derivedETH
                    totalValueLockedUSD
                }
            }
            """ % token_address.lower()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(subgraph_url, json={'query': query}) as response:
                    if response.status != 200:
                        return None
                        
                    data = await response.json()
                    token_data = data.get('data', {}).get('token')
                    if not token_data:
                        return None
                        
                    return float(token_data['derivedETH'])
                    
        except Exception as e:
            logger.error(f"Error getting subgraph price for {token_address} on {chain}: {str(e)}")
            return None

class PriceFeedRegistry:
    """Registry for managing multiple price feeds"""
    
    def __init__(self):
        """Initialize price feed registry"""
        self.feeds = {
            'chainlink': ChainlinkPriceFeed(),
            'uniswap': UniswapV3PriceFeed(),
            'subgraph': SubgraphPriceFeed()
        }
        
        # Define feed priority for each chain
        self.chain_priorities = {
            'ethereum': ['chainlink', 'uniswap', 'subgraph'],
            'polygon': ['chainlink', 'uniswap', 'subgraph'],
            'base': ['chainlink', 'uniswap', 'subgraph']
        }
        
        self.price_cache = {}
    
    async def get_price(
        self,
        token_address: str,
        chain: str,
        web3: Web3
    ) -> Optional[float]:
        """Get token price using available price feeds
        
        Args:
            token_address: Token address or symbol
            chain: Chain name
            web3: Web3 connection for the chain
            
        Returns:
            Token price if available, None otherwise
        """
        # Check cache first
        cache_key = f"{chain}:{token_address}"
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]
        
        # Try feeds in priority order
        feed_priority = self.chain_priorities.get(chain, [])
        for feed_name in feed_priority:
            feed = self.feeds.get(feed_name)
            if not feed:
                continue
                
            price = await feed.get_price(token_address, chain, web3)
            if price is not None:
                # Cache the price
                self.price_cache[cache_key] = price
                return price
        
        return None
    
    def clear_cache(self):
        """Clear price cache"""
        self.price_cache.clear() 