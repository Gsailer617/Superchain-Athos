from typing import Dict, Optional, Any, List, Tuple
from web3 import Web3, AsyncWeb3
from web3.middleware import geth_poa_middleware, async_geth_poa_middleware
from web3.types import Middleware, AsyncMiddleware
from web3.providers import HTTPProvider, WebsocketProvider
from dataclasses import dataclass, field
import logging
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import backoff

from ..config.chain_specs import ChainSpec, GasModel
from ..utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)

@dataclass
class ChainMetrics:
    """Enhanced metrics for chain monitoring"""
    avg_response_time: float = 0.0
    success_rate: float = 1.0
    failed_requests: int = 0
    total_requests: int = 0
    last_block: int = 0
    last_error: Optional[str] = None
    last_updated: float = 0.0
    congestion_level: float = 0.0  # 0-1 scale
    rpc_latency: Dict[str, float] = field(default_factory=dict)
    gas_prices: List[float] = field(default_factory=list)
    block_times: List[float] = field(default_factory=list)

class EnhancedChainConnector:
    """Enhanced chain connector with advanced monitoring and error handling"""
    
    def __init__(self, max_workers: int = 10):
        self._connections: Dict[str, Web3] = {}
        self._async_connections: Dict[str, AsyncWeb3] = {}
        self._metrics: Dict[str, ChainMetrics] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._session: Optional[aiohttp.ClientSession] = None
        self._fallback_rpcs: Dict[str, List[str]] = self._load_fallback_rpcs()
    
    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
        self.close()
    
    def _load_fallback_rpcs(self) -> Dict[str, List[str]]:
        """Load fallback RPC endpoints for each chain"""
        return {
            'ethereum': [
                'https://eth-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}',
                'https://mainnet.infura.io/v3/${INFURA_KEY}',
                'https://rpc.ankr.com/eth'
            ],
            'base': [
                'https://mainnet.base.org',
                'https://base.blockpi.network/v1/rpc/public',
                'https://base.meowrpc.com'
            ],
            'polygon': [
                'https://polygon-rpc.com',
                'https://rpc-mainnet.matic.network',
                'https://matic-mainnet.chainstacklabs.com'
            ],
            # Add fallbacks for other chains...
        }
    
    @retry_with_backoff(max_retries=3)
    async def get_async_web3(self, chain_name: str) -> AsyncWeb3:
        """Get async Web3 connection with enhanced error handling"""
        # Return cached connection if healthy
        if chain_name in self._async_connections:
            web3 = self._async_connections[chain_name]
            if await web3.is_connected():
                return web3
            else:
                del self._async_connections[chain_name]
        
        # Get chain configuration
        chain_spec = self._get_chain_spec(chain_name)
        
        # Try primary and fallback endpoints
        last_error = None
        endpoints = [chain_spec.rpc_url] + self._fallback_rpcs.get(chain_name, [])
        
        for endpoint in endpoints:
            try:
                web3 = await self._create_async_web3_connection(endpoint, chain_spec)
                if await web3.is_connected():
                    self._async_connections[chain_name] = web3
                    return web3
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to connect to {chain_name} using {endpoint}: {str(e)}")
        
        raise ConnectionError(f"Unable to connect to {chain_name}. Last error: {str(last_error)}")
    
    @retry_with_backoff(max_retries=3)
    def get_web3(self, chain_name: str) -> Web3:
        """Get Web3 connection with enhanced error handling"""
        # Return cached connection if healthy
        if chain_name in self._connections:
            web3 = self._connections[chain_name]
            if web3.is_connected():
                return web3
            else:
                del self._connections[chain_name]
        
        # Get chain configuration
        chain_spec = self._get_chain_spec(chain_name)
        
        # Try primary and fallback endpoints
        last_error = None
        endpoints = [chain_spec.rpc_url] + self._fallback_rpcs.get(chain_name, [])
        
        for endpoint in endpoints:
            try:
                web3 = self._create_web3_connection(endpoint, chain_spec)
                if web3.is_connected():
                    self._connections[chain_name] = web3
                    return web3
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to connect to {chain_name} using {endpoint}: {str(e)}")
        
        raise ConnectionError(f"Unable to connect to {chain_name}. Last error: {str(last_error)}")
    
    async def _create_async_web3_connection(
        self,
        endpoint: str,
        chain_spec: ChainSpec
    ) -> AsyncWeb3:
        """Create async Web3 connection with chain-specific configuration"""
        # Create provider with timeout and custom headers
        if endpoint.startswith('ws'):
            provider = WebsocketProvider(
                endpoint,
                websocket_kwargs={'timeout': chain_spec.rpc.timeout}
            )
        else:
            provider = HTTPProvider(
                endpoint,
                request_kwargs={
                    'timeout': chain_spec.rpc.timeout,
                    'headers': {'User-Agent': 'FlashingBase/1.0.0'}
                }
            )
        
        web3 = AsyncWeb3(provider)
        
        # Add chain-specific middleware
        if not chain_spec.is_l2:
            web3.middleware_onion.inject(async_geth_poa_middleware, layer=0)
        
        # Add custom middleware for metrics
        web3.middleware_onion.add(self._metrics_middleware)
        
        # Verify chain ID
        chain_id = await web3.eth.chain_id
        if chain_id != chain_spec.chain_id:
            raise ValueError(
                f"Chain ID mismatch. Expected {chain_spec.chain_id}, got {chain_id}"
            )
        
        return web3
    
    def _create_web3_connection(
        self,
        endpoint: str,
        chain_spec: ChainSpec
    ) -> Web3:
        """Create Web3 connection with chain-specific configuration"""
        # Create provider with timeout and custom headers
        if endpoint.startswith('ws'):
            provider = WebsocketProvider(
                endpoint,
                websocket_kwargs={'timeout': chain_spec.rpc.timeout}
            )
        else:
            provider = HTTPProvider(
                endpoint,
                request_kwargs={
                    'timeout': chain_spec.rpc.timeout,
                    'headers': {'User-Agent': 'FlashingBase/1.0.0'}
                }
            )
        
        web3 = Web3(provider)
        
        # Add chain-specific middleware
        if not chain_spec.is_l2:
            web3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        # Add custom middleware for metrics
        web3.middleware_onion.add(self._metrics_middleware)
        
        # Verify chain ID
        chain_id = web3.eth.chain_id
        if chain_id != chain_spec.chain_id:
            raise ValueError(
                f"Chain ID mismatch. Expected {chain_spec.chain_id}, got {chain_id}"
            )
        
        return web3
    
    def _metrics_middleware(self, make_request: Callable, w3: Web3) -> Callable:
        """Middleware for collecting metrics"""
        def middleware(method: RPCEndpoint, params: Any) -> RPCResponse:
            start = time.time()
            try:
                response = make_request(method, params)
                duration = time.time() - start
                self._update_metrics(w3.eth.chain_id, duration, True)
                return response
            except Exception as e:
                duration = time.time() - start
                self._update_metrics(w3.eth.chain_id, duration, False, str(e))
                raise
        return middleware
    
    def _update_metrics(
        self,
        chain_id: int,
        duration: float,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """Update chain metrics"""
        chain_name = self._get_chain_name(chain_id)
        metrics = self._metrics.get(chain_name)
        if not metrics:
            metrics = ChainMetrics()
            self._metrics[chain_name] = metrics
        
        metrics.total_requests += 1
        if not success:
            metrics.failed_requests += 1
            metrics.last_error = error
        
        metrics.success_rate = (
            (metrics.total_requests - metrics.failed_requests) /
            metrics.total_requests
        )
        
        # Update response time with exponential moving average
        alpha = 0.1
        metrics.avg_response_time = (
            (1 - alpha) * metrics.avg_response_time +
            alpha * duration
        )
        
        metrics.last_updated = time.time()
    
    async def monitor_chain_health(self, chain_name: str) -> Dict[str, Any]:
        """Monitor chain health metrics"""
        try:
            web3 = await self.get_async_web3(chain_name)
            
            # Get latest block and gas prices
            start = time.time()
            latest_block = await web3.eth.block_number
            block_time = time.time() - start
            
            # Get gas prices based on chain's gas model
            chain_spec = self._get_chain_spec(chain_name)
            gas_prices = await self._get_chain_gas_prices(web3, chain_spec)
            
            # Update metrics
            metrics = self._metrics.get(chain_name)
            if metrics:
                metrics.last_block = latest_block
                metrics.block_times.append(block_time)
                metrics.gas_prices.append(gas_prices['average'])
                
                # Calculate congestion level
                metrics.congestion_level = self._calculate_congestion_level(
                    metrics.block_times[-10:],
                    metrics.gas_prices[-10:],
                    chain_spec
                )
            
            return {
                'status': 'healthy',
                'latest_block': latest_block,
                'gas_prices': gas_prices,
                'block_time': block_time,
                'congestion_level': metrics.congestion_level if metrics else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error monitoring {chain_name}: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def _get_chain_gas_prices(
        self,
        web3: AsyncWeb3,
        chain_spec: ChainSpec
    ) -> Dict[str, float]:
        """Get gas prices based on chain's gas model"""
        if chain_spec.gas.model == GasModel.EIP1559:
            # Get EIP-1559 fees
            block = await web3.eth.get_block('latest')
            base_fee = block['baseFeePerGas']
            max_priority_fee = await web3.eth.max_priority_fee
            
            return {
                'base_fee': base_fee,
                'max_priority_fee': max_priority_fee,
                'average': base_fee + max_priority_fee,
                'fast': base_fee + (max_priority_fee * 1.5),
                'fastest': base_fee + (max_priority_fee * 2)
            }
        else:
            # Legacy gas price
            gas_price = await web3.eth.gas_price
            return {
                'average': gas_price,
                'fast': gas_price * 1.2,
                'fastest': gas_price * 1.5
            }
    
    def _calculate_congestion_level(
        self,
        block_times: List[float],
        gas_prices: List[float],
        chain_spec: ChainSpec
    ) -> float:
        """Calculate chain congestion level (0-1)"""
        if not block_times or not gas_prices:
            return 0.0
        
        # Calculate block time deviation
        avg_block_time = sum(block_times) / len(block_times)
        block_time_factor = max(0, min(1, (
            avg_block_time - chain_spec.block.target_block_time
        ) / chain_spec.block.target_block_time))
        
        # Calculate gas price trend
        gas_price_factor = 0.0
        if len(gas_prices) > 1:
            gas_trend = (gas_prices[-1] - gas_prices[0]) / gas_prices[0]
            gas_price_factor = max(0, min(1, gas_trend))
        
        # Combine factors (can be adjusted based on chain characteristics)
        return (block_time_factor * 0.4) + (gas_price_factor * 0.6)
    
    def get_metrics(self, chain_name: str) -> Optional[ChainMetrics]:
        """Get metrics for specific chain"""
        return self._metrics.get(chain_name)
    
    def get_all_metrics(self) -> Dict[str, ChainMetrics]:
        """Get all chain metrics"""
        return self._metrics.copy()
    
    async def check_all_connections(self) -> Dict[str, bool]:
        """Check health of all connections"""
        results = {}
        
        async def check_connection(chain_name: str) -> None:
            try:
                web3 = await self.get_async_web3(chain_name)
                await web3.eth.get_block_number()
                results[chain_name] = True
            except Exception as e:
                logger.error(f"Connection check failed for {chain_name}: {str(e)}")
                results[chain_name] = False
        
        # Check all chains concurrently
        await asyncio.gather(*[
            check_connection(chain)
            for chain in self._connections.keys()
        ])
        
        return results
    
    def close(self) -> None:
        """Close all connections"""
        # Close sync connections
        for web3 in self._connections.values():
            if isinstance(web3.provider, (HTTPProvider, WebsocketProvider)):
                web3.provider.close()
        self._connections.clear()
        
        # Close async connections
        for web3 in self._async_connections.values():
            if isinstance(web3.provider, (HTTPProvider, WebsocketProvider)):
                web3.provider.close()
        self._async_connections.clear()
        
        # Clear metrics
        self._metrics.clear()
        
        # Shutdown thread pool
        self._executor.shutdown() 