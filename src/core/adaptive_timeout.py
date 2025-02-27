"""
Adaptive Timeout Module

This module provides adaptive timeout mechanisms that automatically adjust
timeout durations based on network conditions and historical performance.
Features:
- Dynamic timeout calculation based on historical latency
- Chain-specific timeout profiles
- Congestion detection and adaptation
- Metrics and monitoring
"""

import time
from typing import Dict, List, Optional, Any, Callable, TypeVar, Union
from dataclasses import dataclass, field
import logging
import math
import statistics
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import functools

logger = logging.getLogger(__name__)
T = TypeVar('T')

class NetworkState(Enum):
    """Network state classifications"""
    NORMAL = "normal"
    CONGESTED = "congested"
    DEGRADED = "degraded"
    UNSTABLE = "unstable"

@dataclass
class TimeoutConfig:
    """Configuration for adaptive timeouts"""
    # Base timeouts (seconds)
    base_timeout: float = 30.0
    min_timeout: float = 10.0
    max_timeout: float = 300.0  # 5 minutes
    
    # Adaptation factors
    latency_multiplier: float = 2.0
    congestion_multiplier: float = 1.5
    retry_multiplier: float = 1.2
    
    # Statistical window
    window_size: int = 20  # Number of data points to keep
    percentile_threshold: float = 95.0  # Use 95th percentile for calculation

@dataclass
class TimeoutStats:
    """Statistics for adaptive timeout calculations"""
    # Raw data
    latencies: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    
    # Derived statistics
    avg_latency: float = 0.0
    std_dev: float = 0.0
    percentile_latency: float = 0.0
    current_timeout: float = 30.0  # Default starting point
    network_state: NetworkState = NetworkState.NORMAL
    
    # Last update
    last_updated: float = field(default_factory=time.time)

class AdaptiveTimeout:
    """Adaptive timeout manager"""
    
    def __init__(self, chain_id: str, config: Optional[TimeoutConfig] = None):
        """Initialize adaptive timeout manager
        
        Args:
            chain_id: Chain identifier
            config: Optional timeout configuration
        """
        self.chain_id = chain_id
        self.config = config or TimeoutConfig()
        self.stats = TimeoutStats()
        self._initialize_timeout()
        
    def _initialize_timeout(self) -> None:
        """Initialize timeout value"""
        self.stats.current_timeout = self.config.base_timeout
        
    def record_latency(self, latency: float, success: bool = True, error_type: Optional[str] = None) -> None:
        """Record a latency observation
        
        Args:
            latency: Observed latency in seconds
            success: Whether the operation succeeded
            error_type: Type of error if any
        """
        # Record data
        self.stats.latencies.append(latency)
        self.stats.timestamps.append(time.time())
        
        if not success and error_type:
            self.stats.errors.append(error_type)
            
        # Trim data if needed
        if len(self.stats.latencies) > self.config.window_size:
            self.stats.latencies = self.stats.latencies[-self.config.window_size:]
            self.stats.timestamps = self.stats.timestamps[-self.config.window_size:]
            self.stats.errors = self.stats.errors[-self.config.window_size:]
            
        # Update statistics
        self._update_statistics()
        
        # Update timeout value
        self._update_timeout()
        
    def get_timeout(self) -> float:
        """Get current recommended timeout
        
        Returns:
            Current adaptive timeout in seconds
        """
        return self.stats.current_timeout
    
    def _update_statistics(self) -> None:
        """Update derived statistics"""
        if not self.stats.latencies:
            return
            
        try:
            # Calculate basic statistics
            self.stats.avg_latency = statistics.mean(self.stats.latencies)
            
            if len(self.stats.latencies) > 1:
                self.stats.std_dev = statistics.stdev(self.stats.latencies)
            
            # Calculate percentile latency
            self.stats.percentile_latency = self._calculate_percentile(
                self.stats.latencies, 
                self.config.percentile_threshold
            )
            
            # Determine network state
            self._determine_network_state()
            
        except Exception as e:
            logger.error(f"Error updating statistics: {str(e)}")
            
        self.stats.last_updated = time.time()
        
    def _determine_network_state(self) -> None:
        """Determine current network state"""
        # Count recent errors
        recent_errors = len(self.stats.errors)
        
        # Calculate coefficient of variation (normalized std dev)
        cv = 0.0
        if self.stats.avg_latency > 0:
            cv = self.stats.std_dev / self.stats.avg_latency
            
        # Determine state based on errors and latency variation
        if recent_errors > len(self.stats.latencies) * 0.5:
            self.stats.network_state = NetworkState.UNSTABLE
        elif recent_errors > len(self.stats.latencies) * 0.25:
            self.stats.network_state = NetworkState.DEGRADED
        elif cv > 0.5:  # High variation in latency
            self.stats.network_state = NetworkState.CONGESTED
        else:
            self.stats.network_state = NetworkState.NORMAL
            
    def _update_timeout(self) -> None:
        """Update timeout based on statistics and network state"""
        if not self.stats.latencies:
            return
            
        # Base timeout on percentile latency
        base = self.stats.percentile_latency * self.config.latency_multiplier
        
        # Apply network state multiplier
        if self.stats.network_state == NetworkState.CONGESTED:
            base *= self.config.congestion_multiplier
        elif self.stats.network_state == NetworkState.DEGRADED:
            base *= self.config.congestion_multiplier * 1.2
        elif self.stats.network_state == NetworkState.UNSTABLE:
            base *= self.config.congestion_multiplier * 1.5
            
        # Ensure timeout is within min/max bounds
        self.stats.current_timeout = max(
            self.config.min_timeout,
            min(base, self.config.max_timeout)
        )
        
    def _calculate_percentile(self, data: List[float], percentile: float) -> float:
        """Calculate a percentile value from a list of data
        
        Args:
            data: List of float values
            percentile: Percentile to calculate (0-100)
            
        Returns:
            Percentile value
        """
        if not data:
            return 0.0
            
        sorted_data = sorted(data)
        n = len(sorted_data)
        
        if n == 1:
            return sorted_data[0]
            
        rank = percentile / 100.0 * (n - 1)
        
        if rank == int(rank):
            # Exact percentile exists in data
            return sorted_data[int(rank)]
        else:
            # Interpolate between two values
            lower_rank = int(rank)
            rank_fraction = rank - lower_rank
            return (sorted_data[lower_rank] * (1 - rank_fraction) + 
                    sorted_data[lower_rank + 1] * rank_fraction)

class AdaptiveTimeoutRegistry:
    """Registry for timeout managers"""
    
    _timeout_managers: Dict[str, AdaptiveTimeout] = {}
    
    @classmethod
    def get_or_create(
        cls,
        chain_id: str,
        config: Optional[TimeoutConfig] = None
    ) -> AdaptiveTimeout:
        """Get or create a timeout manager for a chain
        
        Args:
            chain_id: Chain identifier
            config: Optional timeout configuration
            
        Returns:
            AdaptiveTimeout manager
        """
        if chain_id not in cls._timeout_managers:
            cls._timeout_managers[chain_id] = AdaptiveTimeout(chain_id, config)
            
        return cls._timeout_managers[chain_id]
    
    @classmethod
    def get_all_timeouts(cls) -> Dict[str, float]:
        """Get all current timeout values
        
        Returns:
            Dictionary of chain_id to timeout value
        """
        return {
            chain_id: manager.get_timeout()
            for chain_id, manager in cls._timeout_managers.items()
        }
    
    @classmethod
    def get_all_states(cls) -> Dict[str, NetworkState]:
        """Get all current network states
        
        Returns:
            Dictionary of chain_id to network state
        """
        return {
            chain_id: manager.stats.network_state
            for chain_id, manager in cls._timeout_managers.items()
        }
        
    @classmethod
    def reset_all(cls) -> None:
        """Reset all timeout managers"""
        cls._timeout_managers.clear()
        
# Utility decorator for functions that need adaptive timeouts
def with_adaptive_timeout(chain_id: str):
    """Decorator for functions that need adaptive timeouts
    
    Args:
        chain_id: Chain identifier
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            timeout_manager = AdaptiveTimeoutRegistry.get_or_create(chain_id)
            timeout = timeout_manager.get_timeout()
            
            start_time = time.time()
            success = True
            error_type = None
            
            try:
                # Execute with timeout
                async with asyncio.timeout(timeout):
                    result = await func(*args, **kwargs)
                    
                latency = time.time() - start_time
                timeout_manager.record_latency(latency, success=True)
                return result
                
            except asyncio.TimeoutError:
                latency = time.time() - start_time
                success = False
                error_type = "TimeoutError"
                timeout_manager.record_latency(latency, success=False, error_type=error_type)
                raise
                
            except Exception as e:
                latency = time.time() - start_time
                success = False
                error_type = type(e).__name__
                timeout_manager.record_latency(latency, success=False, error_type=error_type)
                raise
                
        return wrapper
    return decorator 