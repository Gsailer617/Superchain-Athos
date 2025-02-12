import asyncio
import aiohttp
import logging
from typing import Dict, Optional, Set
from web3 import Web3
from web3.providers import HTTPProvider
import weakref

logger = logging.getLogger(__name__)

class ResourceManager:
    """Manages system resources and connections"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self.sessions: Dict[str, aiohttp.ClientSession] = {}
        self._web3_connections: Dict[str, Web3] = {}
        self._active_tasks: Set[asyncio.Task] = set()
        self._cleanup_callbacks = []
        
    async def get_http_session(self, name: str) -> aiohttp.ClientSession:
        """Get or create an HTTP session"""
        if name not in self.sessions:
            self.sessions[name] = aiohttp.ClientSession()
        return self.sessions[name]
        
    def get_web3(self, provider_url: str) -> Web3:
        """Get or create a Web3 connection"""
        if provider_url not in self._web3_connections:
            provider = HTTPProvider(provider_url)
            web3 = Web3(provider)
            self._web3_connections[provider_url] = web3
        return self._web3_connections[provider_url]
        
    def create_task(self, coro, name: Optional[str] = None) -> asyncio.Task:
        """Create and track an asyncio task"""
        task = asyncio.create_task(coro, name=name)
        self._active_tasks.add(task)
        task.add_done_callback(self._remove_task)
        return task
        
    def _remove_task(self, task):
        """Remove completed task from tracking"""
        self._active_tasks.discard(task)
        
    def register_cleanup(self, callback):
        """Register a cleanup callback"""
        self._cleanup_callbacks.append(weakref.ref(callback))
        
    async def cleanup(self):
        """Cleanup all resources"""
        # Cancel all active tasks
        for task in self._active_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
        # Close HTTP sessions
        for session in self.sessions.values():
            if not session.closed:
                await session.close()
        self.sessions.clear()
        
        # Close Web3 connections
        for web3 in self._web3_connections.values():
            if hasattr(web3.provider, 'close'):
                await web3.provider.close()
        self._web3_connections.clear()
        
        # Run cleanup callbacks
        for callback_ref in self._cleanup_callbacks[:]:
            callback = callback_ref()
            if callback is not None:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback()
                    else:
                        callback()
                except Exception as e:
                    logger.error(f"Error in cleanup callback: {str(e)}")
                    
        self._cleanup_callbacks.clear()
        self._active_tasks.clear()
        
    async def __aenter__(self):
        """Context manager entry"""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.cleanup()
        
# Global instance
resource_manager = ResourceManager() 