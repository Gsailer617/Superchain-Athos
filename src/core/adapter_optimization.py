"""
Adapter Optimization Module

This module provides optimizations for bridge adapters:
- Shared caching mechanisms
- Common pattern implementations
- Performance monitoring
- Resource usage optimization
"""

from typing import Dict, Any, Optional, Type, List, Callable, TypeVar, Generic, Union
from dataclasses import dataclass, field
import time
import asyncio
import functools
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import datetime, timedelta
from web3 import Web3
from web3.types import TxParams, Wei
from enum import Enum
import json

from .bridge_adapter import BridgeAdapter, BridgeConfig, BridgeState, BridgeMetrics
from .error_handling import ErrorSeverity
from .circuit_breaker import CircuitBreaker, CircuitConfig

logger = logging.getLogger(__name__)
T = TypeVar('T')

# Cache configuration
@dataclass
class CacheConfig:
    """Configuration for adapter caching"""
    enable_fee_cache: bool = True
    enable_state_cache: bool = True
    enable_liquidity_cache: bool = True
    fee_cache_ttl: int = 60  # seconds
    state_cache_ttl: int = 300  # seconds
    liquidity_cache_ttl: int = 120  # seconds
    max_cache_size: int = 1000
    monitor_cache_hits: bool = True

# Enhanced metrics tracking
@dataclass
class AdapterMetrics:
    """Extended metrics for adapter performance monitoring"""
    cache_hits: Dict[str, int] = field(default_factory=lambda: {"fee": 0, "state": 0, "liquidity": 0})
    cache_misses: Dict[str, int] = field(default_factory=lambda: {"fee": 0, "state": 0, "liquidity": 0})
    average_response_time: Dict[str, float] = field(default_factory=lambda: {"fee": 0, "state": 0, "liquidity": 0, "transfer": 0})
    calls_count: Dict[str, int] = field(default_factory=lambda: {"fee": 0, "state": 0, "liquidity": 0, "transfer": 0})
    error_count: Dict[str, int] = field(default_factory=dict)
    last_reset_time: datetime = field(default_factory=datetime.now)

@dataclass
class CacheEntry(Generic[T]):
    """Generic cache entry with timestamp"""
    value: T
    timestamp: float

class AdapterOptimizer:
    """Provides optimization capabilities for bridge adapters"""
    
    def __init__(self, adapter: BridgeAdapter, cache_config: Optional[CacheConfig] = None):
        """Initialize optimizer for a specific adapter
        
        Args:
            adapter: The bridge adapter to optimize
            cache_config: Optional cache configuration
        """
        self.adapter = adapter
        self.cache_config = cache_config or CacheConfig()
        self.metrics = AdapterMetrics()
        self._fee_cache: Dict[str, CacheEntry[Dict[str, float]]] = {}
        self._state_cache: Dict[str, CacheEntry[BridgeState]] = {}
        self._liquidity_cache: Dict[str, CacheEntry[float]] = {}
        self._executor = ThreadPoolExecutor(max_workers=5)
        self._lock = threading.RLock()
        self._circuit_breaker = CircuitBreaker(
            name=f"{type(adapter).__name__}_breaker",
            config=CircuitConfig(
                failure_threshold=5,
                recovery_timeout=300.0,  # 5 minutes
                half_open_max_calls=2,
                success_threshold=3
            )
        )
    
    def estimate_fees_with_cache(
        self,
        source_chain: str,
        target_chain: str,
        token: str,
        amount: float
    ) -> Dict[str, float]:
        """Cached fee estimation
        
        Args:
            source_chain: Source chain ID
            target_chain: Target chain ID
            token: Token address or symbol
            amount: Amount to transfer
            
        Returns:
            Dict with fee components
        """
        if not self.cache_config.enable_fee_cache:
            return self._measure_performance(
                "fee",
                lambda: self.adapter.estimate_fees(source_chain, target_chain, token, amount)
            )
            
        # Create cache key
        cache_key = f"{source_chain}:{target_chain}:{token}:{amount}"
        
        with self._lock:
            # Check cache
            if cache_key in self._fee_cache:
                entry = self._fee_cache[cache_key]
                # Check if valid
                if time.time() - entry.timestamp < self.cache_config.fee_cache_ttl:
                    self.metrics.cache_hits["fee"] += 1
                    return entry.value.copy()
            
            # Cache miss or expired
            self.metrics.cache_misses["fee"] += 1
            result = self._measure_performance(
                "fee",
                lambda: self.adapter.estimate_fees(source_chain, target_chain, token, amount)
            )
            
            # Update cache
            self._fee_cache[cache_key] = CacheEntry(value=result.copy(), timestamp=time.time())
            
            # Trim cache if needed
            if len(self._fee_cache) > self.cache_config.max_cache_size:
                self._trim_cache(self._fee_cache)
                
            return result
    
    def get_bridge_state_with_cache(
        self,
        source_chain: str,
        target_chain: str
    ) -> BridgeState:
        """Cached bridge state retrieval
        
        Args:
            source_chain: Source chain ID
            target_chain: Target chain ID
            
        Returns:
            Current bridge state
        """
        if not self.cache_config.enable_state_cache:
            return self._measure_performance(
                "state",
                lambda: self.adapter.get_bridge_state(source_chain, target_chain)
            )
            
        # Create cache key
        cache_key = f"{source_chain}:{target_chain}"
        
        with self._lock:
            # Check cache
            if cache_key in self._state_cache:
                entry = self._state_cache[cache_key]
                # Check if valid
                if time.time() - entry.timestamp < self.cache_config.state_cache_ttl:
                    self.metrics.cache_hits["state"] += 1
                    return entry.value
            
            # Cache miss or expired
            self.metrics.cache_misses["state"] += 1
            result = self._measure_performance(
                "state",
                lambda: self.adapter.get_bridge_state(source_chain, target_chain)
            )
            
            # Update cache
            self._state_cache[cache_key] = CacheEntry(value=result, timestamp=time.time())
                
            return result
    
    def monitor_liquidity_with_cache(
        self,
        chain: str,
        token: str
    ) -> float:
        """Cached liquidity monitoring
        
        Args:
            chain: Chain ID
            token: Token address or symbol
            
        Returns:
            Current liquidity
        """
        if not self.cache_config.enable_liquidity_cache:
            return self._measure_performance(
                "liquidity",
                lambda: self.adapter.monitor_liquidity(chain, token)
            )
            
        # Create cache key
        cache_key = f"{chain}:{token}"
        
        with self._lock:
            # Check cache
            if cache_key in self._liquidity_cache:
                entry = self._liquidity_cache[cache_key]
                # Check if valid
                if time.time() - entry.timestamp < self.cache_config.liquidity_cache_ttl:
                    self.metrics.cache_hits["liquidity"] += 1
                    return entry.value
            
            # Cache miss or expired
            self.metrics.cache_misses["liquidity"] += 1
            result = self._measure_performance(
                "liquidity",
                lambda: self.adapter.monitor_liquidity(chain, token)
            )
            
            # Update cache
            self._liquidity_cache[cache_key] = CacheEntry(value=result, timestamp=time.time())
            
            # Trim cache if needed
            if len(self._liquidity_cache) > self.cache_config.max_cache_size:
                self._trim_cache(self._liquidity_cache)
                
            return result
    
    async def prepare_transfer_with_circuit_breaker(
        self,
        source_chain: str,
        target_chain: str,
        token: str,
        amount: float,
        recipient: str
    ) -> TxParams:
        """Prepare transfer with circuit breaker protection
        
        Args:
            source_chain: Source chain ID
            target_chain: Target chain ID
            token: Token address or symbol
            amount: Amount to transfer
            recipient: Recipient address
            
        Returns:
            Transaction parameters
        """
        async def _prepare():
            start_time = time.time()
            try:
                result = self.adapter.prepare_transfer(source_chain, target_chain, token, amount, recipient)
                self._update_metrics("transfer", time.time() - start_time)
                return result
            except Exception as e:
                self.metrics.error_count[str(type(e).__name__)] = (
                    self.metrics.error_count.get(str(type(e).__name__), 0) + 1
                )
                raise
                
        return await self._circuit_breaker.execute(_prepare)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the adapter
        
        Returns:
            Dictionary with metrics
        """
        with self._lock:
            total_calls = sum(self.metrics.calls_count.values())
            total_cache_hits = sum(self.metrics.cache_hits.values())
            total_cache_misses = sum(self.metrics.cache_misses.values())
            
            cache_hit_ratio = 0.0
            if total_cache_hits + total_cache_misses > 0:
                cache_hit_ratio = total_cache_hits / (total_cache_hits + total_cache_misses)
                
            return {
                "adapter_name": type(self.adapter).__name__,
                "total_calls": total_calls,
                "average_response_times": self.metrics.average_response_time,
                "cache_stats": {
                    "hits": self.metrics.cache_hits,
                    "misses": self.metrics.cache_misses,
                    "hit_ratio": cache_hit_ratio
                },
                "errors": self.metrics.error_count,
                "circuit_breaker": self._circuit_breaker.get_metrics(),
                "cache_sizes": {
                    "fee_cache": len(self._fee_cache),
                    "state_cache": len(self._state_cache),
                    "liquidity_cache": len(self._liquidity_cache)
                }
            }
    
    def clear_caches(self) -> None:
        """Clear all caches"""
        with self._lock:
            self._fee_cache.clear()
            self._state_cache.clear()
            self._liquidity_cache.clear()
            logger.info(f"Cleared all caches for {type(self.adapter).__name__}")
    
    def reset_metrics(self) -> None:
        """Reset all metrics"""
        with self._lock:
            self.metrics = AdapterMetrics()
            logger.info(f"Reset metrics for {type(self.adapter).__name__}")
    
    def _trim_cache(self, cache: Dict[str, CacheEntry]) -> None:
        """Trim cache to maximum size
        
        Strategy: Remove oldest entries first
        """
        if len(cache) <= self.cache_config.max_cache_size:
            return
            
        # Sort by timestamp (oldest first)
        sorted_keys = sorted(cache.keys(), key=lambda k: cache[k].timestamp)
        
        # Remove oldest entries
        keys_to_remove = sorted_keys[:len(cache) - self.cache_config.max_cache_size]
        for key in keys_to_remove:
            del cache[key]
    
    def _measure_performance(self, operation: str, func: Callable[[], T]) -> T:
        """Measure performance of an operation
        
        Args:
            operation: Operation name
            func: Function to measure
            
        Returns:
            Result of the function
        """
        start_time = time.time()
        try:
            result = func()
            execution_time = time.time() - start_time
            self._update_metrics(operation, execution_time)
            return result
        except Exception as e:
            self.metrics.error_count[str(type(e).__name__)] = (
                self.metrics.error_count.get(str(type(e).__name__), 0) + 1
            )
            raise
    
    def _update_metrics(self, operation: str, execution_time: float) -> None:
        """Update performance metrics
        
        Args:
            operation: Operation name
            execution_time: Execution time in seconds
        """
        with self._lock:
            calls = self.metrics.calls_count.get(operation, 0)
            avg_time = self.metrics.average_response_time.get(operation, 0.0)
            
            # Update average (weighted)
            if calls == 0:
                new_avg = execution_time
            else:
                new_avg = (avg_time * calls + execution_time) / (calls + 1)
                
            self.metrics.calls_count[operation] = calls + 1
            self.metrics.average_response_time[operation] = new_avg


class AdapterOptimizerRegistry:
    """Registry for adapter optimizers"""
    
    _optimizers: Dict[str, AdapterOptimizer] = {}
    
    @classmethod
    def get_or_create(
        cls,
        adapter: BridgeAdapter,
        cache_config: Optional[CacheConfig] = None
    ) -> AdapterOptimizer:
        """Get or create an optimizer for an adapter
        
        Args:
            adapter: Bridge adapter
            cache_config: Optional cache configuration
            
        Returns:
            AdapterOptimizer for the adapter
        """
        adapter_id = id(adapter)
        
        if adapter_id not in cls._optimizers:
            cls._optimizers[adapter_id] = AdapterOptimizer(adapter, cache_config)
            
        return cls._optimizers[adapter_id]
    
    @classmethod
    def get_all_metrics(cls) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all optimizers
        
        Returns:
            Dictionary of metrics by adapter
        """
        return {
            str(idx): optimizer.get_performance_metrics()
            for idx, optimizer in cls._optimizers.items()
        }
    
    @classmethod
    def clear_all_caches(cls) -> None:
        """Clear caches for all optimizers"""
        for optimizer in cls._optimizers.values():
            optimizer.clear_caches()
            
    @classmethod
    def reset_all_metrics(cls) -> None:
        """Reset metrics for all optimizers"""
        for optimizer in cls._optimizers.values():
            optimizer.reset_metrics() 