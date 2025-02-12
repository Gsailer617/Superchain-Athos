"""Parallel processing management for heavy computations"""

import ray
from typing import List, Dict, Any, Callable, Optional, TypeVar, Generic
import numpy as np
from dataclasses import dataclass
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

logger = logging.getLogger(__name__)
T = TypeVar('T')
U = TypeVar('U')

@dataclass
class ComputeResult(Generic[T]):
    """Result of a parallel computation"""
    result: T
    duration: float
    resources_used: Dict[str, float]

class ParallelManager:
    """Manages parallel and distributed computations"""
    
    def __init__(
        self,
        num_cpus: Optional[int] = None,
        num_gpus: Optional[int] = None,
        memory_limit: Optional[float] = None
    ):
        """Initialize parallel manager"""
        self.num_cpus = num_cpus or mp.cpu_count()
        self.num_gpus = num_gpus
        self.memory_limit = memory_limit
        self._init_ray()
        
    def _init_ray(self):
        """Initialize Ray if not already initialized"""
        if not ray.is_initialized():
            ray.init(
                num_cpus=self.num_cpus,
                num_gpus=self.num_gpus,
                _memory=self.memory_limit
            )
    
    def map_parallel(
        self,
        func: Callable[[T], U],
        data: List[T],
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None
    ) -> List[ComputeResult[U]]:
        """Execute function on data items in parallel"""
        
        @ray.remote
        def _process_batch(batch: List[T]) -> List[ComputeResult[U]]:
            results = []
            for item in batch:
                start_time = time.time()
                try:
                    result = func(item)
                    results.append(ComputeResult(
                        result=result,
                        duration=time.time() - start_time,
                        resources_used=self._get_resource_usage()
                    ))
                except Exception as e:
                    logger.error(f"Error processing item: {str(e)}", exc_info=True)
            return results
        
        # Split data into batches
        batch_size = batch_size or max(1, len(data) // (self.num_cpus * 2))
        batches = [
            data[i:i + batch_size]
            for i in range(0, len(data), batch_size)
        ]
        
        # Process batches in parallel
        futures = [_process_batch.remote(batch) for batch in batches]
        results = []
        for batch_results in ray.get(futures):
            results.extend(batch_results)
        
        return results
    
    def process_matrix(
        self,
        matrix: np.ndarray,
        func: Callable[[np.ndarray], np.ndarray],
        split_axis: int = 0
    ) -> ComputeResult[np.ndarray]:
        """Process large matrices in parallel"""
        
        @ray.remote
        def _process_slice(slice_data: np.ndarray) -> np.ndarray:
            return func(slice_data)
        
        start_time = time.time()
        
        # Split matrix along specified axis
        splits = np.array_split(matrix, self.num_cpus, axis=split_axis)
        
        # Process splits in parallel
        futures = [_process_slice.remote(split) for split in splits]
        results = ray.get(futures)
        
        # Combine results
        combined = np.concatenate(results, axis=split_axis)
        
        return ComputeResult(
            result=combined,
            duration=time.time() - start_time,
            resources_used=self._get_resource_usage()
        )
    
    def parallel_optimization(
        self,
        objective_func: Callable[[np.ndarray], float],
        initial_points: List[np.ndarray],
        num_iterations: int,
        num_workers: Optional[int] = None
    ) -> ComputeResult[Dict[str, Any]]:
        """Run parallel optimization with multiple starting points"""
        
        @ray.remote
        def _optimize_from_point(start_point: np.ndarray) -> Dict[str, Any]:
            best_value = float('inf')
            best_point = None
            
            for _ in range(num_iterations):
                try:
                    value = objective_func(start_point)
                    if value < best_value:
                        best_value = value
                        best_point = start_point.copy()
                    
                    # Simple gradient descent (replace with your optimization logic)
                    gradient = self._numerical_gradient(objective_func, start_point)
                    start_point = start_point - 0.01 * gradient
                    
                except Exception as e:
                    logger.error(f"Optimization error: {str(e)}", exc_info=True)
                    break
            
            return {
                'best_value': best_value,
                'best_point': best_point
            }
        
        start_time = time.time()
        
        # Run optimization from each starting point in parallel
        futures = [_optimize_from_point.remote(point) for point in initial_points]
        results = ray.get(futures)
        
        # Find best result across all runs
        best_result = min(results, key=lambda x: x['best_value'])
        
        return ComputeResult(
            result=best_result,
            duration=time.time() - start_time,
            resources_used=self._get_resource_usage()
        )
    
    def _numerical_gradient(
        self,
        func: Callable[[np.ndarray], float],
        point: np.ndarray,
        epsilon: float = 1e-7
    ) -> np.ndarray:
        """Calculate numerical gradient"""
        gradient = np.zeros_like(point)
        for i in range(len(point)):
            point[i] += epsilon
            plus_value = func(point)
            point[i] -= 2 * epsilon
            minus_value = func(point)
            point[i] += epsilon
            gradient[i] = (plus_value - minus_value) / (2 * epsilon)
        return gradient
    
    def _get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        try:
            import psutil
            process = psutil.Process()
            return {
                'cpu_percent': process.cpu_percent(),
                'memory_percent': process.memory_percent(),
                'num_threads': process.num_threads()
            }
        except ImportError:
            return {} 