"""Cross-chain analysis and monitoring"""

import asyncio
from typing import Dict, Any, Optional, List, Tuple
import logging
from dataclasses import dataclass
import time
from ..utils.cache import AsyncCache
from ..utils.metrics import MetricsManager
from ..integrations.defillama import DefiLlamaIntegration
# Import the shared metrics
from .cross_chain_analyzer import ChainMetrics as BaseChainMetrics

logger = logging.getLogger(__name__)

@dataclass
class BridgeMetrics:
    """Bridge performance metrics"""
    tvl: float
    volume_24h: float
    deposits_24h: float
    withdrawals_24h: float
    avg_transfer_time: float
    success_rate: float
    timestamp: float

# Using the extended metrics from cross_chain_analyzer.py
ChainMetrics = BaseChainMetrics

class CrossChainAnalyzer:
    """Analysis of cross-chain metrics and opportunities"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        cache: AsyncCache,
        metrics: MetricsManager
    ):
        self.config = config
        self.cache = cache
        self.metrics = metrics
        self.defillama = DefiLlamaIntegration()
        
    async def _get_bridge_liquidity(
        self,
        bridge_id: str,
        chain: Optional[str] = None
    ) -> float:
        """Get bridge liquidity (TVL)"""
        try:
            # Check cache first
            cache_key = f"bridge_liquidity:{bridge_id}"
            if chain:
                cache_key += f":{chain}"
                
            cached = await self.cache.get(cache_key)
            if cached:
                self.metrics.record_cache_hit()
                return float(cached['tvl'])
                
            # Get bridge stats from DeFi Llama
            stats = await self.defillama.get_bridge_stats(bridge_id)
            
            if not stats:
                return 0.0
                
            tvl = stats['tvl']
            
            # Cache the result
            await self.cache.set(cache_key, {'tvl': tvl})
            self.metrics.record_cache_miss()
            
            # Record metrics
            self.metrics.observe(
                'bridge_liquidity',
                tvl,
                {'bridge': bridge_id}
            )
            
            return float(tvl)
            
        except Exception as e:
            self.metrics.record_api_error('bridge_liquidity', str(e))
            logger.error(f"Error getting bridge liquidity: {str(e)}")
            return 0.0
    
    async def _get_cross_chain_volume(
        self,
        bridge_id: str,
        period: str = '24h'
    ) -> float:
        """Get cross-chain transfer volume"""
        try:
            # Check cache first
            cache_key = f"bridge_volume:{bridge_id}:{period}"
            cached = await self.cache.get(cache_key)
            if cached:
                self.metrics.record_cache_hit()
                return float(cached['volume'])
                
            # Get bridge stats
            stats = await self.defillama.get_bridge_stats(bridge_id)
            
            if not stats:
                return 0.0
                
            # Get volume for requested period
            volume_key = f"volume_{period}"
            volume = stats.get(volume_key, 0)
            
            # Cache the result
            await self.cache.set(cache_key, {'volume': volume})
            self.metrics.record_cache_miss()
            
            # Record metrics
            self.metrics.observe(
                'bridge_volume',
                volume,
                {'bridge': bridge_id, 'period': period}
            )
            
            return float(volume)
            
        except Exception as e:
            self.metrics.record_api_error('bridge_volume', str(e))
            logger.error(f"Error getting bridge volume: {str(e)}")
            return 0.0
            
    async def get_bridge_metrics(
        self,
        bridge_id: str
    ) -> Optional[BridgeMetrics]:
        """Get comprehensive bridge metrics"""
        try:
            # Get bridge stats
            stats = await self.defillama.get_bridge_stats(bridge_id)
            
            if not stats:
                return None
                
            # Get additional metrics from bridge API
            transfer_time = await self._get_average_transfer_time(bridge_id)
            success_rate = await self._get_bridge_success_rate(bridge_id)
            
            metrics = BridgeMetrics(
                tvl=stats['tvl'],
                volume_24h=stats['volume_24h'],
                deposits_24h=stats['deposits_24h'],
                withdrawals_24h=stats['withdrawals_24h'],
                avg_transfer_time=transfer_time,
                success_rate=success_rate,
                timestamp=time.time()
            )
            
            # Record all metrics
            for key, value in stats.items():
                self.metrics.observe(
                    f'bridge_{key}',
                    value,
                    {'bridge': bridge_id}
                )
                
            return metrics
            
        except Exception as e:
            self.metrics.record_api_error('bridge_metrics', str(e))
            logger.error(f"Error getting bridge metrics: {str(e)}")
            return None
            
    async def get_chain_metrics(
        self,
        chain: str
    ) -> Optional[ChainMetrics]:
        """Get comprehensive chain metrics"""
        try:
            # Get chain TVL
            tvl = await self.defillama.get_chain_tvl(chain)
            
            # Get chain volume
            volume = await self._get_chain_volume(chain)
            
            # Get chain stats
            stats = await self._get_chain_stats(chain)
            
            if not stats:
                return None
                
            metrics = ChainMetrics(
                tvl=tvl,
                volume_24h=volume,
                transactions_24h=stats['transactions_24h'],
                avg_block_time=stats['avg_block_time'],
                avg_gas_price=stats['avg_gas_price'],
                timestamp=time.time()
            )
            
            # Record metrics
            self.metrics.observe('chain_tvl', tvl, {'chain': chain})
            self.metrics.observe('chain_volume', volume, {'chain': chain})
            
            for key, value in stats.items():
                self.metrics.observe(
                    f'chain_{key}',
                    value,
                    {'chain': chain}
                )
                
            return metrics
            
        except Exception as e:
            self.metrics.record_api_error('chain_metrics', str(e))
            logger.error(f"Error getting chain metrics: {str(e)}")
            return None
            
    async def _get_average_transfer_time(
        self,
        bridge_id: str
    ) -> float:
        """Get average transfer time for bridge"""
        try:
            # Get recent transfers
            transfers = await self._get_recent_transfers(bridge_id)
            
            if not transfers:
                return 0.0
                
            # Calculate average time
            times = [
                t['completion_time'] - t['initiation_time']
                for t in transfers
                if t['status'] == 'completed'
            ]
            
            if not times:
                return 0.0
                
            avg_time = sum(times) / len(times)
            
            # Record metrics
            self.metrics.observe(
                'bridge_transfer_time',
                avg_time,
                {'bridge': bridge_id}
            )
            
            return float(avg_time)
            
        except Exception as e:
            logger.error(f"Error getting transfer time: {str(e)}")
            return 0.0
            
    async def _get_bridge_success_rate(
        self,
        bridge_id: str
    ) -> float:
        """Get bridge transfer success rate"""
        try:
            # Get recent transfers
            transfers = await self._get_recent_transfers(bridge_id)
            
            if not transfers:
                return 0.0
                
            # Calculate success rate
            total = len(transfers)
            successful = len([
                t for t in transfers
                if t['status'] == 'completed'
            ])
            
            rate = successful / total if total > 0 else 0.0
            
            # Record metrics
            self.metrics.observe(
                'bridge_success_rate',
                rate,
                {'bridge': bridge_id}
            )
            
            return float(rate)
            
        except Exception as e:
            logger.error(f"Error getting success rate: {str(e)}")
            return 0.0
            
    async def _get_recent_transfers(
        self,
        bridge_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent bridge transfers"""
        try:
            # Check cache first
            cache_key = f"bridge_transfers:{bridge_id}"
            cached = await self.cache.get(cache_key)
            if cached:
                self.metrics.record_cache_hit()
                return cached['transfers']
                
            # Get transfers from bridge API
            async with aiohttp.ClientSession() as session:
                url = f"{self.config['bridge_api_url']}/transfers"
                params = {
                    'bridge_id': bridge_id,
                    'limit': limit
                }
                
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        return []
                        
                    data = await response.json()
                    transfers = data.get('transfers', [])
                    
            # Cache the results
            await self.cache.set(
                cache_key,
                {'transfers': transfers}
            )
            self.metrics.record_cache_miss()
            
            return transfers
            
        except Exception as e:
            logger.error(f"Error getting transfers: {str(e)}")
            return []
            
    async def _get_chain_volume(self, chain: str) -> float:
        """Get chain volume from DeFi Llama"""
        try:
            # Check cache first
            cache_key = f"chain_volume:{chain}"
            cached = await self.cache.get(cache_key)
            if cached:
                self.metrics.record_cache_hit()
                return float(cached['volume'])
                
            # Get chain volume
            async with aiohttp.ClientSession() as session:
                url = f"{self.config['defillama_api_url']}/v2/chains/{chain}"
                async with session.get(url) as response:
                    if response.status != 200:
                        return 0.0
                        
                    data = await response.json()
                    volume = float(data.get('volume24h', 0))
                    
            # Cache the result
            await self.cache.set(cache_key, {'volume': volume})
            self.metrics.record_cache_miss()
            
            return volume
            
        except Exception as e:
            logger.error(f"Error getting chain volume: {str(e)}")
            return 0.0
            
    async def _get_chain_stats(self, chain: str) -> Dict[str, Any]:
        """Get chain statistics"""
        try:
            # Check cache first
            cache_key = f"chain_stats:{chain}"
            cached = await self.cache.get(cache_key)
            if cached:
                self.metrics.record_cache_hit()
                return cached
                
            # Get chain stats from API
            async with aiohttp.ClientSession() as session:
                url = f"{self.config['chain_api_url']}/stats/{chain}"
                async with session.get(url) as response:
                    if response.status != 200:
                        return {}
                        
                    stats = await response.json()
                    
            # Cache the results
            await self.cache.set(cache_key, stats)
            self.metrics.record_cache_miss()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting chain stats: {str(e)}")
            return {} 