import functools
import logging
import time
from typing import TypeVar, Callable, Any, Optional
import backoff

logger = logging.getLogger(__name__)

T = TypeVar('T')

def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """Retry decorator with exponential backoff
    
    Args:
        max_retries: Maximum number of retries
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None
            delay = initial_delay
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    
                    # Calculate next delay
                    delay = min(delay * exponential_base, max_delay)
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: "
                        f"{str(e)}. Retrying in {delay:.2f}s..."
                    )
                    
                    # Wait before next attempt
                    await asyncio.sleep(delay)
            
            # If we get here, all retries failed
            raise last_exception or Exception(f"All {max_retries} retries failed")
        
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None
            delay = initial_delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    
                    # Calculate next delay
                    delay = min(delay * exponential_base, max_delay)
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: "
                        f"{str(e)}. Retrying in {delay:.2f}s..."
                    )
                    
                    # Wait before next attempt
                    time.sleep(delay)
            
            # If we get here, all retries failed
            raise last_exception or Exception(f"All {max_retries} retries failed")
        
        # Return appropriate wrapper based on if function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator

def backoff_hdlr(details: Dict[str, Any]) -> None:
    """Handler for backoff events"""
    logger.warning(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with args {args} and kwargs "
        "{kwargs}".format(**details)
    )

# Predefined retry decorators
retry_connection = retry_with_backoff(
    max_retries=3,
    initial_delay=1.0,
    max_delay=10.0,
    exceptions=(ConnectionError, TimeoutError)
)

retry_transaction = retry_with_backoff(
    max_retries=5,
    initial_delay=2.0,
    max_delay=30.0,
    exceptions=(
        ValueError,
        TimeoutError,
        ConnectionError
    )
)

# Backoff decorators for specific use cases
on_chain_exception = backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=5,
    max_time=30,
    on_backoff=backoff_hdlr
)

on_network_exception = backoff.on_exception(
    backoff.expo,
    (ConnectionError, TimeoutError),
    max_tries=3,
    max_time=15,
    on_backoff=backoff_hdlr
)

on_validation_error = backoff.on_exception(
    backoff.expo,
    ValueError,
    max_tries=3,
    max_time=10,
    on_backoff=backoff_hdlr
) 