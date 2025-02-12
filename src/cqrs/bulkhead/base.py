from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)

@dataclass
class BulkheadConfig:
    """Configuration for a bulkhead"""
    max_concurrent_calls: int
    max_queue_size: int
    timeout_seconds: float

class Bulkhead:
    """Implements the bulkhead pattern for isolating components"""
    
    def __init__(self, name: str, config: BulkheadConfig):
        self.name = name
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrent_calls)
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.metrics: Dict[str, int] = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "rejected_calls": 0,
            "timeout_calls": 0
        }
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function within the bulkhead"""
        self.metrics["total_calls"] += 1
        
        try:
            # Check if queue is full
            if self.queue.full():
                self.metrics["rejected_calls"] += 1
                raise RuntimeError(f"Bulkhead {self.name} queue is full")
            
            # Add to queue
            await self.queue.put(None)
            
            try:
                # Execute with timeout and concurrency control
                async with self.semaphore:
                    try:
                        result = await asyncio.wait_for(
                            func(*args, **kwargs),
                            timeout=self.config.timeout_seconds
                        )
                        self.metrics["successful_calls"] += 1
                        return result
                        
                    except asyncio.TimeoutError:
                        self.metrics["timeout_calls"] += 1
                        raise RuntimeError(
                            f"Operation in bulkhead {self.name} timed out"
                        )
                        
            finally:
                # Remove from queue
                await self.queue.get()
                self.queue.task_done()
                
        except Exception as e:
            self.metrics["failed_calls"] += 1
            logger.error(
                f"Error in bulkhead {self.name}: {str(e)}",
                exc_info=True
            )
            raise

class BulkheadRegistry:
    """Registry for managing multiple bulkheads"""
    
    def __init__(self):
        self.bulkheads: Dict[str, Bulkhead] = {}
    
    def register(
        self,
        name: str,
        max_concurrent_calls: int = 10,
        max_queue_size: int = 20,
        timeout_seconds: float = 30.0
    ) -> Bulkhead:
        """Register a new bulkhead"""
        config = BulkheadConfig(
            max_concurrent_calls=max_concurrent_calls,
            max_queue_size=max_queue_size,
            timeout_seconds=timeout_seconds
        )
        bulkhead = Bulkhead(name, config)
        self.bulkheads[name] = bulkhead
        return bulkhead
    
    def get(self, name: str) -> Optional[Bulkhead]:
        """Get a registered bulkhead"""
        return self.bulkheads.get(name)
    
    def get_metrics(self) -> Dict[str, Dict[str, int]]:
        """Get metrics for all bulkheads"""
        return {
            name: bulkhead.metrics
            for name, bulkhead in self.bulkheads.items()
        } 