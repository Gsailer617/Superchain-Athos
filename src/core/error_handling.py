"""
Enhanced error handling system with detailed tracking and fallbacks
"""

import logging
import traceback
from typing import Dict, Any, Optional, Type, TypeVar, Callable
from dataclasses import dataclass
from datetime import datetime
import functools
import json
from enum import Enum
import sys
import asyncio
from prometheus_client import Counter, Histogram

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type variable for generic error handling
T = TypeVar('T')

class ErrorSeverity(Enum):
    """Error severity levels"""
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4
    CRITICAL = 5

@dataclass
class ErrorContext:
    """Detailed error context"""
    timestamp: datetime
    severity: ErrorSeverity
    error_type: str
    message: str
    stack_trace: str
    component: str
    operation: str
    input_data: Optional[Dict[str, Any]] = None
    user_id: Optional[int] = None
    additional_context: Optional[Dict[str, Any]] = None

class ErrorMetrics:
    """Error metrics tracking"""
    
    def __init__(self):
        self.error_counter = Counter(
            'application_errors_total',
            'Total number of errors',
            ['component', 'error_type', 'severity']
        )
        self.error_duration = Histogram(
            'error_handling_duration_seconds',
            'Time spent handling errors',
            ['component', 'error_type']
        )

class ErrorHandler:
    """Enhanced error handler with fallbacks and recovery"""
    
    def __init__(self):
        self.error_history: Dict[str, list[ErrorContext]] = {}
        self.metrics = ErrorMetrics()
        self.fallback_handlers: Dict[Type[Exception], Callable] = {}
        self.recovery_strategies: Dict[str, Callable] = {}
        
    def record_error(self, context: ErrorContext) -> None:
        """Record error with full context"""
        # Update error history
        component_errors = self.error_history.setdefault(context.component, [])
        component_errors.append(context)
        
        # Update metrics
        self.metrics.error_counter.labels(
            component=context.component,
            error_type=context.error_type,
            severity=context.severity.name
        ).inc()
        
        # Log error with context
        log_message = (
            f"{context.severity.name} in {context.component}.{context.operation}: "
            f"{context.message}"
        )
        
        if context.input_data:
            log_message += f"\nInput data: {json.dumps(context.input_data, indent=2)}"
            
        if context.additional_context:
            log_message += f"\nAdditional context: {json.dumps(context.additional_context, indent=2)}"
            
        logger.error(
            log_message,
            extra={
                'error_context': asdict(context),
                'stack_trace': context.stack_trace
            }
        )
        
    def register_fallback(
        self,
        exception_type: Type[Exception],
        handler: Callable
    ) -> None:
        """Register fallback handler for specific exception type"""
        self.fallback_handlers[exception_type] = handler
        
    def register_recovery(
        self,
        component: str,
        strategy: Callable
    ) -> None:
        """Register recovery strategy for component"""
        self.recovery_strategies[component] = strategy
        
    async def handle_error(
        self,
        error: Exception,
        component: str,
        operation: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        **kwargs
    ) -> Any:
        """Handle error with fallback and recovery"""
        start_time = datetime.now()
        
        try:
            # Create error context
            context = ErrorContext(
                timestamp=start_time,
                severity=severity,
                error_type=error.__class__.__name__,
                message=str(error),
                stack_trace=traceback.format_exc(),
                component=component,
                operation=operation,
                input_data=kwargs.get('input_data'),
                user_id=kwargs.get('user_id'),
                additional_context=kwargs.get('context')
            )
            
            # Record error
            self.record_error(context)
            
            # Try fallback handler
            if type(error) in self.fallback_handlers:
                logger.info(f"Attempting fallback for {error.__class__.__name__}")
                return await self.fallback_handlers[type(error)](error, **kwargs)
                
            # Try recovery strategy
            if component in self.recovery_strategies:
                logger.info(f"Attempting recovery for {component}")
                return await self.recovery_strategies[component](error, **kwargs)
                
            # No fallback or recovery available
            raise error
            
        finally:
            duration = (datetime.now() - start_time).total_seconds()
            self.metrics.error_duration.labels(
                component=component,
                error_type=error.__class__.__name__
            ).observe(duration)
            
    def get_error_summary(self, component: Optional[str] = None) -> Dict[str, Any]:
        """Get error summary statistics"""
        if component:
            errors = self.error_history.get(component, [])
        else:
            errors = [e for errs in self.error_history.values() for e in errs]
            
        return {
            'total_errors': len(errors),
            'by_severity': {
                severity.name: len([e for e in errors if e.severity == severity])
                for severity in ErrorSeverity
            },
            'by_type': {
                error_type: len([e for e in errors if e.error_type == error_type])
                for error_type in set(e.error_type for e in errors)
            },
            'recent_errors': [
                {
                    'timestamp': e.timestamp.isoformat(),
                    'component': e.component,
                    'operation': e.operation,
                    'message': e.message,
                    'severity': e.severity.name
                }
                for e in sorted(errors, key=lambda x: x.timestamp, reverse=True)[:10]
            ]
        }

# Global error handler instance
error_handler = ErrorHandler()

def with_error_handling(
    component: str,
    operation: str,
    severity: ErrorSeverity = ErrorSeverity.ERROR
):
    """Decorator for error handling"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                return await error_handler.handle_error(
                    error=e,
                    component=component,
                    operation=operation,
                    severity=severity,
                    input_data={'args': args, 'kwargs': kwargs}
                )
        return wrapper
    return decorator

# Example usage:
"""
@with_error_handling('market_analyzer', 'analyze_opportunity')
async def analyze_opportunity(opportunity_data: Dict[str, Any]) -> Dict[str, Any]:
    # Function implementation
    pass
""" 