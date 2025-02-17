from typing import Dict, Optional, Any
from web3 import Web3, AsyncWeb3
from web3.middleware.geth_poa import geth_poa_middleware
from web3.providers.rpc import HTTPProvider
from web3.providers.websocket.websocket import WebsocketProvider
from web3.types import AsyncMiddleware, AsyncBaseProvider, BaseProvider
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from dataclasses import dataclass

from ..config.chain_specs import CHAIN_SPECS, ChainSpec

logger = logging.getLogger(__name__)

@dataclass
class ConnectionMetrics:
    """Metrics for connection monitoring"""
    latency: float = 0.0
    success_rate: float = 1.0
    failed_requests: int = 0
    total_requests: int = 0
    last_error: Optional[str] = None
    last_updated: float = 0.0

class ChainConnector:
    """Manages Web3 connections to multiple blockchains"""
    
    def __init__(self):
        self._connections: Dict[str, Web3] = {}
        self._async_connections: Dict[str, AsyncWeb3] = {}
        self._metrics: Dict[str, ConnectionMetrics] = {}
        self._executor = ThreadPoolExecutor(max_workers=10)
        
    def get_web3(self, chain_name: str) -> Web3:
        """Get Web3 connection for specified chain
        
        Args:
            chain_name: Name of the chain to connect to
            
        Returns:
            Web3: Connected Web3 instance
            
        Raises:
            ValueError: If chain configuration not found
            ConnectionError: If unable to connect to chain
        """
        # Return cached connection if available and healthy
        if chain_name in self._connections:
            web3 = self._connections[chain_name]
            if web3.is_connected():
                return web3
            else:
                # Remove stale connection
                del self._connections[chain_name]
        
        # Get chain configuration
        chain_spec = CHAIN_SPECS.get(chain_name)
        if not chain_spec:
            raise ValueError(f"Chain configuration for {chain_name} not found")
        
        # Create new connection
        web3 = self._create_web3_connection(chain_spec)
        
        # Cache connection
        self._connections[chain_name] = web3
        
        return web3
    
    async def get_async_web3(self, chain_name: str) -> AsyncWeb3:
        """Get async Web3 connection for specified chain
        
        Args:
            chain_name: Name of the chain to connect to
            
        Returns:
            AsyncWeb3: Connected AsyncWeb3 instance
            
        Raises:
            ValueError: If chain configuration not found
            ConnectionError: If unable to connect to chain
        """
        # Return cached connection if available and healthy
        if chain_name in self._async_connections:
            web3 = self._async_connections[chain_name]
            if await web3.is_connected():
                return web3
            else:
                # Remove stale connection
                del self._async_connections[chain_name]
        
        # Get chain configuration
        chain_spec = CHAIN_SPECS.get(chain_name)
        if not chain_spec:
            raise ValueError(f"Chain configuration for {chain_name} not found")
        
        # Create new connection
        web3 = await self._create_async_web3_connection(chain_spec)
        
        # Cache connection
        self._async_connections[chain_name] = web3
        
        return web3
    
    def _create_web3_connection(self, chain_spec: ChainSpec) -> Web3:
        """Create new Web3 connection from chain specification"""
        # Try each RPC URL until successful connection
        last_error = None
        for rpc_url in chain_spec.rpc_urls:
            try:
                # Create provider with timeout and custom headers
                provider = HTTPProvider(
                    rpc_url,
                    request_kwargs={
                        'timeout': chain_spec.rpc.timeout,
                        'headers': {'User-Agent': 'FlashingBase/1.0.0'}
                    }
                )
                
                # Create Web3 instance
                web3 = Web3(provider)
                
                # Add PoA middleware if needed
                if not chain_spec.is_l2:
                    web3.middleware_onion.inject(geth_poa_middleware, layer=0)
                
                # Test connection
                if web3.is_connected():
                    # Verify chain ID
                    chain_id = web3.eth.chain_id
                    if chain_id != chain_spec.chain_id:
                        raise ValueError(
                            f"Chain ID mismatch. Expected {chain_spec.chain_id}, got {chain_id}"
                        )
                    
                    # Initialize metrics
                    self._metrics[chain_spec.name] = ConnectionMetrics()
                    
                    return web3
                    
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Failed to connect to {chain_spec.name} using {rpc_url}: {str(e)}"
                )
                continue
        
        # All connection attempts failed
        raise ConnectionError(
            f"Unable to connect to {chain_spec.name}. Last error: {str(last_error)}"
        )
    
    async def _create_async_web3_connection(self, chain_spec: ChainSpec) -> AsyncWeb3:
        """Create new async Web3 connection from chain specification"""
        # Try each WebSocket URL first (preferred for async)
        if chain_spec.ws_urls:
            for ws_url in chain_spec.ws_urls:
                try:
                    provider = WebsocketProvider(ws_url)
                    web3 = AsyncWeb3(provider)
                    
                    # Add PoA middleware if needed
                    if not chain_spec.is_l2:
                        web3.middleware_onion.inject(
                            cast(AsyncMiddleware, geth_poa_middleware),
                            layer=0
                        )
                    
                    # Test connection
                    if await web3.is_connected():
                        # Verify chain ID
                        chain_id = await web3.eth.chain_id
                        if chain_id != chain_spec.chain_id:
                            raise ValueError(
                                f"Chain ID mismatch. Expected {chain_spec.chain_id}, got {chain_id}"
                            )
                        
                        return web3
                        
                except Exception as e:
                    logger.warning(
                        f"Failed to connect to {chain_spec.name} using WebSocket {ws_url}: {str(e)}"
                    )
                    continue
        
        # Fallback to HTTP URLs
        for rpc_url in chain_spec.rpc_urls:
            try:
                provider = HTTPProvider(
                    rpc_url,
                    request_kwargs={
                        'timeout': chain_spec.rpc.timeout,
                        'headers': {'User-Agent': 'FlashingBase/1.0.0'}
                    }
                )
                
                web3 = AsyncWeb3(provider)
                
                # Add PoA middleware if needed
                if not chain_spec.is_l2:
                    web3.middleware_onion.inject(
                        cast(AsyncMiddleware, geth_poa_middleware),
                        layer=0
                    )
                
                # Test connection
                if await web3.is_connected():
                    # Verify chain ID
                    chain_id = await web3.eth.chain_id
                    if chain_id != chain_spec.chain_id:
                        raise ValueError(
                            f"Chain ID mismatch. Expected {chain_spec.chain_id}, got {chain_id}"
                        )
                    
                    return web3
                    
            except Exception as e:
                logger.warning(
                    f"Failed to connect to {chain_spec.name} using {rpc_url}: {str(e)}"
                )
                continue
        
        raise ConnectionError(f"Unable to connect to {chain_spec.name}")
    
    def update_metrics(
        self,
        chain_name: str,
        latency: Optional[float] = None,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """Update connection metrics"""
        metrics = self._metrics.get(chain_name)
        if not metrics:
            metrics = ConnectionMetrics()
            self._metrics[chain_name] = metrics
        
        metrics.total_requests += 1
        if not success:
            metrics.failed_requests += 1
            metrics.last_error = error
        
        metrics.success_rate = (
            (metrics.total_requests - metrics.failed_requests) /
            metrics.total_requests
        )
        
        if latency is not None:
            metrics.latency = latency
        
        metrics.last_updated = time.time()
    
    def get_metrics(self, chain_name: str) -> Optional[ConnectionMetrics]:
        """Get connection metrics for chain"""
        return self._metrics.get(chain_name)
    
    def get_all_metrics(self) -> Dict[str, ConnectionMetrics]:
        """Get all connection metrics"""
        return self._metrics.copy()
    
    async def check_all_connections(self) -> Dict[str, bool]:
        """Check health of all connections
        
        Returns:
            Dict mapping chain names to connection status (True if healthy)
        """
        results = {}
        
        async def check_connection(chain_name: str) -> None:
            try:
                web3 = await self.get_async_web3(chain_name)
                # Test connection by getting latest block
                start = time.time()
                await web3.eth.get_block_number()
                latency = time.time() - start
                
                self.update_metrics(chain_name, latency=latency)
                results[chain_name] = True
                
            except Exception as e:
                self.update_metrics(chain_name, success=False, error=str(e))
                results[chain_name] = False
        
        # Check all chains concurrently
        await asyncio.gather(*[
            check_connection(chain)
            for chain in CHAIN_SPECS.keys()
        ])
        
        return results
    
    def close(self) -> None:
        """Close all connections"""
        # Close sync connections
        for web3 in self._connections.values():
            if isinstance(web3.provider, (HTTPProvider, WebsocketProvider)):
                web3.provider.close()  # type: ignore
        self._connections.clear()
        
        # Close async connections
        for web3 in self._async_connections.values():
            if isinstance(web3.provider, (HTTPProvider, WebsocketProvider)):
                web3.provider.close()  # type: ignore
        self._async_connections.clear()
        
        # Clear metrics
        self._metrics.clear()
        
        # Shutdown thread pool
        self._executor.shutdown()

# Global chain connector instance
chain_connector = ChainConnector()

def get_chain_connector() -> ChainConnector:
    """Get the global chain connector instance"""
    return chain_connector 