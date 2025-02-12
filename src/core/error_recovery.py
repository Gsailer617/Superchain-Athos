"""
Error Recovery System

This module provides a comprehensive error recovery system with:
- Automatic retry mechanisms
- Error classification and handling
- Recovery strategies
- Error tracking and analysis
- Integration with monitoring
"""

import asyncio
from typing import TypeVar, Callable, Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import structlog
from dataclasses import dataclass
from enum import Enum
from prometheus_client import Counter, Histogram
from tenacity import retry, stop_after_attempt, wait_exponential
import traceback

logger = structlog.get_logger(__name__)

T = TypeVar('T')  # Return type for recovery functions

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for classification"""
    NETWORK = "network"
    VALIDATION = "validation"
    EXECUTION = "execution"
    SYSTEM = "system"
    EXTERNAL = "external"

@dataclass
class ErrorContext:
    """Error context information"""
    error_type: str
    message: str
    severity: ErrorSeverity
    category: ErrorCategory
    timestamp: datetime
    stack_trace: str
    context: Dict[str, Any]
    recovery_attempts: int = 0

class RecoveryStrategy(Enum):
    """Available recovery strategies"""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    RESET = "reset"
    NOTIFY = "notify"

class ErrorRecoverySystem:
    """Main error recovery system"""
    
    def __init__(self):
        self._error_history: List[ErrorContext] = []
        self._setup_metrics()
        
    def _setup_metrics(self):
        """Initialize Prometheus metrics"""
        self._error_counter = Counter(
            'error_recovery_total',
            'Number of errors handled',
            ['severity', 'category', 'strategy']
        )
        self._recovery_time = Histogram(
            'error_recovery_seconds',
            'Time spent in error recovery',
            ['severity', 'category']
        )
        self._recovery_success = Counter(
            'error_recovery_success_total',
            'Number of successful recoveries',
            ['severity', 'category']
        )

    def classify_error(self, error: Exception) -> ErrorContext:
        """Classify an error and create context"""
        error_type = type(error).__name__
        
        # Determine severity and category based on error type
        severity = ErrorSeverity.MEDIUM
        category = ErrorCategory.SYSTEM
        
        if "Timeout" in error_type or "Connection" in error_type:
            category = ErrorCategory.NETWORK
            severity = ErrorSeverity.LOW
        elif "Validation" in error_type:
            category = ErrorCategory.VALIDATION
            severity = ErrorSeverity.MEDIUM
        elif "Execution" in error_type:
            category = ErrorCategory.EXECUTION
            severity = ErrorSeverity.HIGH
        elif hasattr(error, 'severity'):
            severity = getattr(error, 'severity')
        
        context = ErrorContext(
            error_type=error_type,
            message=str(error),
            severity=severity,
            category=category,
            timestamp=datetime.now(),
            stack_trace=traceback.format_exc(),
            context={
                'error_args': getattr(error, 'args', ()),
                'error_attrs': {k: v for k, v in error.__dict__.items() if not k.startswith('_')}
            }
        )
        
        self._error_history.append(context)
        return context

    async def handle_error(self, 
                          error: Exception,
                          recovery_func: Optional[Callable[[], T]] = None,
                          max_retries: int = 3,
                          fallback_value: Optional[T] = None) -> Optional[T]:
        """Handle an error with recovery strategy"""
        start_time = asyncio.get_event_loop().time()
        context = self.classify_error(error)
        
        try:
            # Log error
            logger.error(
                "Error occurred",
                error_type=context.error_type,
                severity=context.severity.value,
                category=context.category.value,
                message=context.message
            )
            
            # Determine recovery strategy
            strategy = self._determine_strategy(context)
            
            # Track metrics
            self._error_counter.labels(
                severity=context.severity.value,
                category=context.category.value,
                strategy=strategy.value
            ).inc()
            
            # Execute recovery strategy
            if strategy == RecoveryStrategy.RETRY and recovery_func:
                result = await self._execute_with_retry(
                    recovery_func,
                    max_retries,
                    context
                )
                if result is not None:
                    self._recovery_success.labels(
                        severity=context.severity.value,
                        category=context.category.value
                    ).inc()
                return result
                
            elif strategy == RecoveryStrategy.FALLBACK:
                return fallback_value
                
            elif strategy == RecoveryStrategy.CIRCUIT_BREAK:
                await self._handle_circuit_break(context)
                return None
                
            elif strategy == RecoveryStrategy.RESET:
                await self._handle_reset(context)
                return None
                
            # Default to notification
            await self._notify_error(context)
            return None
            
        finally:
            duration = asyncio.get_event_loop().time() - start_time
            self._recovery_time.labels(
                severity=context.severity.value,
                category=context.category.value
            ).observe(duration)

    def _determine_strategy(self, context: ErrorContext) -> RecoveryStrategy:
        """Determine the best recovery strategy"""
        if context.severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.CIRCUIT_BREAK
            
        if context.category == ErrorCategory.NETWORK:
            return RecoveryStrategy.RETRY
            
        if context.recovery_attempts >= 3:
            return RecoveryStrategy.FALLBACK
            
        if context.severity == ErrorSeverity.HIGH:
            return RecoveryStrategy.NOTIFY
            
        return RecoveryStrategy.RETRY

    async def _execute_with_retry(self,
                                func: Callable[[], T],
                                max_retries: int,
                                context: ErrorContext) -> Optional[T]:
        """Execute function with retry logic"""
        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=4, max=10)
        )
        async def _retry():
            try:
                context.recovery_attempts += 1
                return await func()
            except Exception as e:
                logger.warning(
                    f"Retry {context.recovery_attempts} failed",
                    error=str(e)
                )
                raise
        
        try:
            return await _retry()
        except Exception:
            return None

    async def _handle_circuit_break(self, context: ErrorContext):
        """Handle circuit breaker activation"""
        logger.warning(
            "Circuit breaker activated",
            error_type=context.error_type,
            severity=context.severity.value
        )
        # Circuit breaker logic will be implemented in the CircuitBreaker module

    async def _handle_reset(self, context: ErrorContext):
        """Handle system reset"""
        logger.warning(
            "System reset initiated",
            error_type=context.error_type,
            severity=context.severity.value
        )
        # Reset logic specific to the error context

    async def _notify_error(self, context: ErrorContext):
        """Notify about error"""
        logger.error(
            "Error notification",
            error_type=context.error_type,
            severity=context.severity.value,
            category=context.category.value,
            message=context.message
        )
        # Notification logic will be implemented separately

    def get_error_history(self,
                         severity: Optional[ErrorSeverity] = None,
                         category: Optional[ErrorCategory] = None,
                         since: Optional[datetime] = None) -> List[ErrorContext]:
        """Get filtered error history"""
        filtered = self._error_history
        
        if severity:
            filtered = [e for e in filtered if e.severity == severity]
        if category:
            filtered = [e for e in filtered if e.category == category]
        if since:
            filtered = [e for e in filtered if e.timestamp >= since]
            
        return filtered

    async def analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze error patterns for insights"""
        analysis = {
            'total_errors': len(self._error_history),
            'errors_by_severity': {},
            'errors_by_category': {},
            'recent_errors': len([
                e for e in self._error_history
                if (datetime.now() - e.timestamp).total_seconds() < 3600
            ]),
            'recovery_success_rate': self._calculate_recovery_rate()
        }
        
        # Analyze by severity
        for severity in ErrorSeverity:
            analysis['errors_by_severity'][severity.value] = len([
                e for e in self._error_history if e.severity == severity
            ])
            
        # Analyze by category
        for category in ErrorCategory:
            analysis['errors_by_category'][category.value] = len([
                e for e in self._error_history if e.category == category
            ])
            
        return analysis

    def _calculate_recovery_rate(self) -> float:
        """Calculate the success rate of recovery attempts"""
        if not self._error_history:
            return 1.0
            
        successful = sum(
            1 for e in self._error_history
            if e.recovery_attempts > 0 and e.recovery_attempts < 3
        )
        return successful / len(self._error_history)

# Global error recovery system instance
error_recovery = ErrorRecoverySystem() 