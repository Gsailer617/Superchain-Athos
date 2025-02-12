"""Advanced asynchronous processing and concurrency management"""

import anyio
from anyio import create_task_group, move_on_after, fail_after
from anyio.abc import TaskGroup
from typing import List, Dict, Any, Callable, Coroutine, TypeVar, Optional
import logging
from dataclasses import dataclass
from contextlib import asynccontextmanager
import time
import ray

logger = logging.getLogger(__name__)
T = TypeVar('T')

@dataclass
class TaskResult:
    """Result of an async task execution"""
    success: bool
    result: Optional[Any] = None
    error: Optional[Exception] = None
    duration: float = 0.0

class AsyncWorkflowManager:
    """Manages complex async workflows and parallel processing"""
    
    def __init__(self, max_concurrency: int = 10, timeout: float = 30.0):
        """Initialize workflow manager"""
        self.max_concurrency = max_concurrency
        self.timeout = timeout
        self._task_group: Optional[TaskGroup] = None
        
    @asynccontextmanager
    async def workflow(self):
        """Context manager for task group management"""
        async with create_task_group() as tg:
            self._task_group = tg
            try:
                yield self
            finally:
                self._task_group = None
    
    async def run_concurrent(
        self,
        tasks: List[Callable[..., Coroutine]],
        *args,
        timeout: Optional[float] = None,
        return_exceptions: bool = False
    ) -> List[TaskResult]:
        """Run multiple coroutines concurrently with timeout"""
        results: List[TaskResult] = []
        timeout = timeout or self.timeout
        
        async def _run_task(task: Callable, *task_args) -> TaskResult:
            start_time = time.time()
            try:
                async with fail_after(timeout):
                    result = await task(*task_args)
                    return TaskResult(
                        success=True,
                        result=result,
                        duration=time.time() - start_time
                    )
            except Exception as e:
                logger.error(f"Task error: {str(e)}", exc_info=True)
                return TaskResult(
                    success=False,
                    error=e,
                    duration=time.time() - start_time
                )
        
        async with create_task_group() as tg:
            for task in tasks:
                tg.start_soon(_run_task, task, *args)
        
        if not return_exceptions:
            results = [r for r in results if r.success]
        
        return results
    
    async def run_with_retry(
        self,
        task: Callable[..., Coroutine],
        *args,
        max_retries: int = 3,
        backoff_factor: float = 1.5
    ) -> TaskResult:
        """Run a task with exponential backoff retry"""
        for attempt in range(max_retries):
            start_time = time.time()
            try:
                async with fail_after(self.timeout):
                    result = await task(*args)
                    return TaskResult(
                        success=True,
                        result=result,
                        duration=time.time() - start_time
                    )
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}",
                    exc_info=True
                )
                if attempt < max_retries - 1:
                    delay = backoff_factor ** attempt
                    await anyio.sleep(delay)
                else:
                    return TaskResult(
                        success=False,
                        error=e,
                        duration=time.time() - start_time
                    )
    
    @staticmethod
    def run_parallel(
        func: Callable,
        data: List[Any],
        num_cpus: Optional[int] = None
    ) -> List[Any]:
        """Run CPU-intensive tasks in parallel using Ray"""
        if not ray.is_initialized():
            ray.init(num_cpus=num_cpus)
        
        @ray.remote
        def _remote_func(item):
            return func(item)
        
        futures = [_remote_func.remote(item) for item in data]
        return ray.get(futures)
    
    async def run_periodic(
        self,
        task: Callable[..., Coroutine],
        interval: float,
        *args,
        max_iterations: Optional[int] = None
    ):
        """Run a task periodically with a fixed interval"""
        iteration = 0
        while True:
            if max_iterations and iteration >= max_iterations:
                break
                
            start_time = time.time()
            try:
                await task(*args)
            except Exception as e:
                logger.error(f"Periodic task error: {str(e)}", exc_info=True)
            
            # Calculate sleep time accounting for task duration
            elapsed = time.time() - start_time
            sleep_time = max(0, interval - elapsed)
            await anyio.sleep(sleep_time)
            
            iteration += 1
    
    async def run_throttled(
        self,
        tasks: List[Callable[..., Coroutine]],
        *args,
        max_concurrent: Optional[int] = None,
        interval: Optional[float] = None
    ) -> List[TaskResult]:
        """Run tasks with throttling for rate limiting"""
        max_concurrent = max_concurrent or self.max_concurrency
        results: List[TaskResult] = []
        
        async def _throttled_task(task: Callable, *task_args) -> TaskResult:
            start_time = time.time()
            try:
                result = await task(*task_args)
                return TaskResult(
                    success=True,
                    result=result,
                    duration=time.time() - start_time
                )
            except Exception as e:
                return TaskResult(
                    success=False,
                    error=e,
                    duration=time.time() - start_time
                )
        
        # Process tasks in batches
        for i in range(0, len(tasks), max_concurrent):
            batch = tasks[i:i + max_concurrent]
            batch_results = await self.run_concurrent(
                [lambda t=t: _throttled_task(t, *args) for t in batch]
            )
            results.extend(batch_results)
            
            if interval and i + max_concurrent < len(tasks):
                await anyio.sleep(interval)
        
        return results 