"""
Circuit Breaker Module

This module provides a comprehensive circuit breaker implementation:
- State management (CLOSED, OPEN, HALF_OPEN)
- Failure threshold monitoring
- Automatic recovery
- Metrics tracking
- Integration with monitoring
"""

import asyncio
from typing import Dict, Optional, Any, Callable, TypeVar, List, Awaitable
from datetime import datetime, timedelta
import structlog
from dataclasses import dataclass
from enum import Enum
from prometheus_client import Counter, Gauge, Histogram

logger = structlog.get_logger(__name__)

T = TypeVar('T')  # Return type for protected functions

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"     # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery

@dataclass
class CircuitStats:
    """Circuit breaker statistics"""
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_calls: int = 0

@dataclass
class CircuitConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    half_open_max_calls: int = 3
    success_threshold: int = 2
    call_timeout: float = 30.0  # seconds
    metric_window: float = 300.0  # 5 minutes

class CircuitBreaker:
    """Circuit breaker implementation"""
    
    def __init__(self, name: str, config: Optional[CircuitConfig] = None):
        self.name = name
        self.config = config or CircuitConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitStats()
        self._state_change_time = datetime.now()
        self._half_open_calls = 0
        self._setup_metrics()

    def _setup_metrics(self):
        """Initialize Prometheus metrics"""
        self._state_gauge = Gauge(
            'circuit_breaker_state',
            'Current circuit breaker state (0=closed, 1=open, 2=half-open)',
            ['name']
        )
        self._failure_counter = Counter(
            'circuit_breaker_failures_total',
            'Number of circuit breaker failures',
            ['name']
        )
        self._success_counter = Counter(
            'circuit_breaker_successes_total',
            'Number of circuit breaker successes',
            ['name']
        )
        self._execution_time = Histogram(
            'circuit_breaker_execution_seconds',
            'Time spent executing protected functions',
            ['name', 'state']
        )

    async def execute(self, func: Callable[[], Awaitable[T]]) -> T:
        """Execute a function with circuit breaker protection"""
        if not self._can_execute():
            raise RuntimeError(f"Circuit breaker {self.name} is OPEN")
        
        start_time = asyncio.get_event_loop().time()
        try:
            # Set timeout for the function
            async with asyncio.timeout(self.config.call_timeout):
                result = await func()
            
            self._record_success()
            return result
            
        except Exception as e:
            self._record_failure()
            raise
        finally:
            duration = asyncio.get_event_loop().time() - start_time
            self._execution_time.labels(
                name=self.name,
                state=self.state.value
            ).observe(duration)

    def _can_execute(self) -> bool:
        """Check if execution is allowed"""
        now = datetime.now()
        
        if self.state == CircuitState.CLOSED:
            return True
            
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if (now - self._state_change_time).total_seconds() >= self.config.recovery_timeout:
                self._transition_to_half_open()
                return True
            return False
            
        if self.state == CircuitState.HALF_OPEN:
            # Allow limited calls in HALF_OPEN state
            return self._half_open_calls < self.config.half_open_max_calls
            
        return False

    def _record_success(self):
        """Record a successful execution"""
        self.stats.success_count += 1
        self.stats.consecutive_successes += 1
        self.stats.consecutive_failures = 0
        self.stats.last_success_time = datetime.now()
        self.stats.total_calls += 1
        
        self._success_counter.labels(name=self.name).inc()
        
        # Update state based on success
        if self.state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1
            if self.stats.consecutive_successes >= self.config.success_threshold:
                self._transition_to_closed()

    def _record_failure(self):
        """Record a failed execution"""
        self.stats.failure_count += 1
        self.stats.consecutive_failures += 1
        self.stats.consecutive_successes = 0
        self.stats.last_failure_time = datetime.now()
        self.stats.total_calls += 1
        
        self._failure_counter.labels(name=self.name).inc()
        
        # Update state based on failure
        if self.state == CircuitState.CLOSED:
            if self.stats.consecutive_failures >= self.config.failure_threshold:
                self._transition_to_open()
        elif self.state == CircuitState.HALF_OPEN:
            self._transition_to_open()

    def _transition_to_open(self):
        """Transition to OPEN state"""
        self.state = CircuitState.OPEN
        self._state_change_time = datetime.now()
        self._state_gauge.labels(name=self.name).set(1)
        logger.warning(f"Circuit breaker {self.name} OPENED")

    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state"""
        self.state = CircuitState.HALF_OPEN
        self._state_change_time = datetime.now()
        self._half_open_calls = 0
        self._state_gauge.labels(name=self.name).set(2)
        logger.info(f"Circuit breaker {self.name} HALF-OPEN")

    def _transition_to_closed(self):
        """Transition to CLOSED state"""
        self.state = CircuitState.CLOSED
        self._state_change_time = datetime.now()
        self._state_gauge.labels(name=self.name).set(0)
        logger.info(f"Circuit breaker {self.name} CLOSED")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current circuit breaker metrics"""
        now = datetime.now()
        window_start = now - timedelta(seconds=self.config.metric_window)
        
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_rate': (
                self.stats.failure_count / self.stats.total_calls
                if self.stats.total_calls > 0 else 0.0
            ),
            'success_rate': (
                self.stats.success_count / self.stats.total_calls
                if self.stats.total_calls > 0 else 1.0
            ),
            'total_calls': self.stats.total_calls,
            'consecutive_failures': self.stats.consecutive_failures,
            'consecutive_successes': self.stats.consecutive_successes,
            'time_in_current_state': (
                now - self._state_change_time
            ).total_seconds()
        }

class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers"""
    
    def __init__(self):
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        
    def get_or_create(self, 
                      name: str,
                      config: Optional[CircuitConfig] = None) -> CircuitBreaker:
        """Get existing or create new circuit breaker"""
        if name not in self._circuit_breakers:
            self._circuit_breakers[name] = CircuitBreaker(name, config)
        return self._circuit_breakers[name]
        
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers"""
        return {
            name: cb.get_metrics()
            for name, cb in self._circuit_breakers.items()
        }
        
    def reset_all(self):
        """Reset all circuit breakers to CLOSED state"""
        for cb in self._circuit_breakers.values():
            cb._transition_to_closed()

# Global circuit breaker registry
circuit_breaker_registry = CircuitBreakerRegistry() 