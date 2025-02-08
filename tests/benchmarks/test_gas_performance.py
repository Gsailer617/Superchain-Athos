import pytest
import asyncio
import time
import torch
from typing import Dict, Any, List
from datetime import datetime
from unittest.mock import patch
import statistics

@pytest.fixture
async def benchmark_system():
    """Fixture for benchmarking system setup."""
    from gas.optimizer import AsyncGasOptimizer
    from gas.implementation import (
        NetworkMonitor,
        TransactionManager,
        GasOptimizationModel
    )
    
    optimizer = AsyncGasOptimizer(mode='performance')
    monitor = NetworkMonitor(rpc_config={'endpoint': 'http://localhost:8545'})
    model = GasOptimizationModel(
        input_dim=10,
        hidden_dims=[128, 64],
        output_dim=2
    )
    
    return {
        'optimizer': optimizer,
        'monitor': monitor,
        'model': model
    }

async def measure_execution_time(func: callable, *args, **kwargs) -> float:
    """Measure execution time of an async function."""
    start_time = time.perf_counter()
    await func(*args, **kwargs)
    end_time = time.perf_counter()
    return end_time - start_time

@pytest.mark.benchmark
async def test_optimization_latency(benchmark_system: Dict[str, Any]):
    """Benchmark gas optimization latency."""
    optimizer = benchmark_system['optimizer']
    latencies = []
    
    # Measure optimization latency under different loads
    for urgency in ['low', 'normal', 'high']:
        tx_data = {
            'to': '0x123...',
            'value': 1000000,
            'urgency': urgency
        }
        
        # Measure 100 optimizations for each urgency level
        for _ in range(100):
            latency = await measure_execution_time(
                optimizer.optimize_gas_params,
                tx_data
            )
            latencies.append(latency)
    
    # Analyze latency distribution
    avg_latency = statistics.mean(latencies)
    p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
    p99_latency = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
    
    # Assert performance requirements
    assert avg_latency < 0.1  # Average latency under 100ms
    assert p95_latency < 0.2  # 95th percentile under 200ms
    assert p99_latency < 0.5  # 99th percentile under 500ms

@pytest.mark.benchmark
async def test_concurrent_performance(benchmark_system: Dict[str, Any]):
    """Benchmark performance under concurrent load."""
    optimizer = benchmark_system['optimizer']
    
    async def optimization_worker() -> List[float]:
        latencies = []
        for _ in range(10):
            start_time = time.perf_counter()
            await optimizer.optimize_gas_params({'urgency': 'normal'})
            latencies.append(time.perf_counter() - start_time)
        return latencies
    
    # Launch concurrent workers
    num_workers = [1, 5, 10, 20, 50]
    results: Dict[int, Dict[str, float]] = {}
    
    for n in num_workers:
        workers = [optimization_worker() for _ in range(n)]
        worker_latencies = await asyncio.gather(*workers)
        
        # Flatten latencies and compute statistics
        latencies = [l for worker in worker_latencies for l in worker]
        results[n] = {
            'avg': statistics.mean(latencies),
            'p95': statistics.quantiles(latencies, n=20)[18],
            'throughput': len(latencies) / sum(latencies)
        }
    
    # Verify scaling behavior
    for n in num_workers[1:]:
        # Throughput should scale sub-linearly
        assert results[n].get('throughput', 0) > results[1].get('throughput', 0)
        # Average latency shouldn't degrade more than 5x
        assert results[n].get('avg', 0) < results[1].get('avg', 0) * 5

@pytest.mark.benchmark
async def test_memory_usage(benchmark_system: Dict[str, Any]):
    """Benchmark memory usage under load."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Generate load
    optimizer = benchmark_system['optimizer']
    model = benchmark_system['model']
    
    # Train model with increasing data
    for batch_size in [100, 1000, 10000]:
        features = torch.randn(batch_size, 10)
        targets = torch.randn(batch_size, 2)
        await model.update(features, targets)
        
        # Run optimizations
        for _ in range(100):
            await optimizer.optimize_gas_params({'urgency': 'normal'})
        
        current_memory = process.memory_info().rss
        memory_increase = current_memory - initial_memory
        
        # Memory shouldn't grow more than 10x
        assert memory_increase < initial_memory * 10

@pytest.mark.benchmark
async def test_model_inference_speed(benchmark_system: Dict[str, Any]):
    """Benchmark model inference speed."""
    model = benchmark_system['model']
    batch_sizes = [1, 10, 100, 1000]
    inference_times: Dict[int, Dict[str, float]] = {}
    
    for batch_size in batch_sizes:
        features = torch.randn(batch_size, 10)
        
        # Warm-up
        for _ in range(10):
            await model.predict(features)
        
        # Measure inference time
        latencies = []
        for _ in range(100):
            latency = await measure_execution_time(
                model.predict,
                features
            )
            latencies.append(latency)
        
        inference_times[batch_size] = {
            'avg': statistics.mean(latencies),
            'p95': statistics.quantiles(latencies, n=20)[18],
            'throughput': batch_size / statistics.mean(latencies)
        }
    
    # Verify batch processing efficiency
    for batch_size in batch_sizes[1:]:
        # Batch processing should be more efficient
        assert inference_times[batch_size].get('throughput', 0) > inference_times[1].get('throughput', 0) * batch_size * 0.5

@pytest.mark.benchmark
async def test_cache_performance(benchmark_system: Dict[str, Any]):
    """Benchmark cache performance."""
    optimizer = benchmark_system['optimizer']
    
    # Measure cache hit vs miss latency
    cache_latencies = {'hit': [], 'miss': []}
    
    # First request (cache miss)
    start_time = time.perf_counter()
    await optimizer.optimize_gas_params({'urgency': 'normal'})
    cache_latencies['miss'].append(time.perf_counter() - start_time)
    
    # Immediate second request (cache hit)
    start_time = time.perf_counter()
    await optimizer.optimize_gas_params({'urgency': 'normal'})
    cache_latencies['hit'].append(time.perf_counter() - start_time)
    
    # Repeat for statistics
    for _ in range(99):
        # Clear cache and measure miss
        optimizer.clear_cache()
        start_time = time.perf_counter()
        await optimizer.optimize_gas_params({'urgency': 'normal'})
        cache_latencies['miss'].append(time.perf_counter() - start_time)
        
        # Measure hit
        start_time = time.perf_counter()
        await optimizer.optimize_gas_params({'urgency': 'normal'})
        cache_latencies['hit'].append(time.perf_counter() - start_time)
    
    # Cache hits should be significantly faster
    avg_hit = statistics.mean(cache_latencies['hit'])
    avg_miss = statistics.mean(cache_latencies['miss'])
    assert avg_hit < avg_miss * 0.2  # Cache hits should be 5x faster 