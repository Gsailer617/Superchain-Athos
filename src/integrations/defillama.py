"""DeFi Llama API integration"""

import aiohttp
from typing import Dict, Any, Optional, List
import logging
from dataclasses import dataclass
import time
import asyncio
from ..utils.rate_limiter import AsyncRateLimiter
from ..utils.metrics import MetricsManager

logger = logging.getLogger(__name__)

@dataclass
class ProtocolData:
    """Protocol data from DeFi Llama"""
    name: str
    tvl: float
    volume_24h: float
    fees_24h: float
    mcap: float
    chain: str
    category: str
    timestamp: float

class DefiLlamaIntegration:
    """Integration with DeFi Llama API"""
    
    def __init__(
        self,
        api_url: str = "https://api.llama.fi",
        rate_limit: int = 10
    ):
        self.api_url = api_url
        self.rate_limiter = AsyncRateLimiter(
            'defillama',
            max_requests=rate_limit,
            requests_per_second=1.0
        )
        self.metrics = MetricsManager()
        
    async def get_protocol_data(
        self,
        protocol_id: str,
        chain: Optional[str] = None
    ) -> Optional[ProtocolData]:
        """Get detailed protocol data"""
        try:
            async with self.rate_limiter:
                async with aiohttp.ClientSession() as session:
                    # Get protocol info
                    info_url = f"{self.api_url}/protocol/{protocol_id}"
                    async with session.get(info_url) as response:
                        if response.status != 200:
                            self.metrics.record_api_error(
                                'defillama',
                                f"Status {response.status}"
                            )
                            return None
                            
                        info = await response.json()
                        
                    # Get volume data
                    volume_url = f"{self.api_url}/summary/protocols/{protocol_id}/volume"
                    async with session.get(volume_url) as response:
                        if response.status != 200:
                            volume_data = {'total24h': 0}
                        else:
                            volume_data = await response.json()
                            
                    # Get fees data
                    fees_url = f"{self.api_url}/summary/protocols/{protocol_id}/fees"
                    async with session.get(fees_url) as response:
                        if response.status != 200:
                            fees_data = {'total24h': 0}
                        else:
                            fees_data = await response.json()
                            
                    # Filter by chain if specified
                    if chain:
                        tvl = info['chainTvls'].get(chain, 0)
                        volume = volume_data.get('chainVolumes', {}).get(chain, 0)
                        fees = fees_data.get('chainFees', {}).get(chain, 0)
                    else:
                        tvl = info['tvl']
                        volume = volume_data.get('total24h', 0)
                        fees = fees_data.get('total24h', 0)
                        
                    # Record metrics
                    self.metrics.observe(
                        'protocol_tvl',
                        tvl,
                        {'protocol': protocol_id}
                    )
                    self.metrics.observe(
                        'protocol_volume',
                        volume,
                        {'protocol': protocol_id}
                    )
                    self.metrics.observe(
                        'protocol_fees',
                        fees,
                        {'protocol': protocol_id}
                    )
                    
                    return ProtocolData(
                        name=info['name'],
                        tvl=float(tvl),
                        volume_24h=float(volume),
                        fees_24h=float(fees),
                        mcap=float(info.get('mcap', 0)),
                        chain=chain or info['chain'],
                        category=info['category'],
                        timestamp=time.time()
                    )
                    
        except Exception as e:
            self.metrics.record_api_error('defillama', str(e))
            logger.error(f"Error getting protocol data: {str(e)}")
            return None
            
    async def get_chain_tvl(self, chain: str) -> float:
        """Get total TVL for a chain"""
        try:
            async with self.rate_limiter:
                async with aiohttp.ClientSession() as session:
                    url = f"{self.api_url}/v2/chains"
                    async with session.get(url) as response:
                        if response.status != 200:
                            self.metrics.record_api_error(
                                'defillama',
                                f"Status {response.status}"
                            )
                            return 0.0
                            
                        data = await response.json()
                        
                    for chain_data in data:
                        if chain_data['name'].lower() == chain.lower():
                            tvl = float(chain_data['tvl'])
                            self.metrics.observe(
                                'chain_tvl',
                                tvl,
                                {'chain': chain}
                            )
                            return tvl
                            
                    return 0.0
                    
        except Exception as e:
            self.metrics.record_api_error('defillama', str(e))
            logger.error(f"Error getting chain TVL: {str(e)}")
            return 0.0
            
    async def get_token_price(
        self,
        token_address: str,
        chain: str = 'ethereum'
    ) -> float:
        """Get token price from DeFi Llama"""
        try:
            async with self.rate_limiter:
                async with aiohttp.ClientSession() as session:
                    url = f"{self.api_url}/prices/current/{chain}:{token_address}"
                    async with session.get(url) as response:
                        if response.status != 200:
                            self.metrics.record_api_error(
                                'defillama',
                                f"Status {response.status}"
                            )
                            return 0.0
                            
                        data = await response.json()
                        coins = data.get('coins', {})
                        key = f"{chain}:{token_address}"
                        
                        if key not in coins:
                            return 0.0
                            
                        price = float(coins[key]['price'])
                        self.metrics.observe(
                            'token_price',
                            price,
                            {'token': token_address}
                        )
                        return price
                        
        except Exception as e:
            self.metrics.record_api_error('defillama', str(e))
            logger.error(f"Error getting token price: {str(e)}")
            return 0.0
            
    async def get_dex_volume(
        self,
        dex_id: str,
        chain: Optional[str] = None
    ) -> float:
        """Get 24h DEX volume"""
        try:
            async with self.rate_limiter:
                async with aiohttp.ClientSession() as session:
                    url = f"{self.api_url}/summary/dexs/{dex_id}/volume"
                    async with session.get(url) as response:
                        if response.status != 200:
                            self.metrics.record_api_error(
                                'defillama',
                                f"Status {response.status}"
                            )
                            return 0.0
                            
                        data = await response.json()
                        
                        if chain:
                            volume = data.get('chainVolumes', {}).get(chain, 0)
                        else:
                            volume = data.get('total24h', 0)
                            
                        self.metrics.observe(
                            'dex_volume',
                            volume,
                            {'dex': dex_id}
                        )
                        return float(volume)
                        
        except Exception as e:
            self.metrics.record_api_error('defillama', str(e))
            logger.error(f"Error getting DEX volume: {str(e)}")
            return 0.0
            
    async def get_bridge_stats(
        self,
        bridge_id: str
    ) -> Dict[str, float]:
        """Get bridge statistics"""
        try:
            async with self.rate_limiter:
                async with aiohttp.ClientSession() as session:
                    url = f"{self.api_url}/bridge/{bridge_id}"
                    async with session.get(url) as response:
                        if response.status != 200:
                            self.metrics.record_api_error(
                                'defillama',
                                f"Status {response.status}"
                            )
                            return {}
                            
                        data = await response.json()
                        
                        stats = {
                            'tvl': float(data.get('tvl', 0)),
                            'volume_24h': float(data.get('volume24h', 0)),
                            'volume_7d': float(data.get('volume7d', 0)),
                            'deposits_24h': float(data.get('deposits24h', 0)),
                            'withdrawals_24h': float(data.get('withdrawals24h', 0))
                        }
                        
                        # Record metrics
                        for key, value in stats.items():
                            self.metrics.observe(
                                f'bridge_{key}',
                                value,
                                {'bridge': bridge_id}
                            )
                            
                        return stats
                        
        except Exception as e:
            self.metrics.record_api_error('defillama', str(e))
            logger.error(f"Error getting bridge stats: {str(e)}")
            return {} 