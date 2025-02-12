import pytest
import asyncio
import aiohttp
import time
import psutil
import statistics
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from src.monitoring.performance_monitor import performance_monitor
from src.core.circuit_breaker import circuit_breaker_registry
from src.core.lock_manager import distributed_lock_manager

@dataclass
class LoadTestMetrics:
    """Metrics collected during load testing"""
    requests_per_second: float
    average_response_time: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    concurrent_connections: int
    success_rate: float

class LoadGenerator:
    """Generates load for testing"""
    
    def __init__(self, base_url: str, num_users: int = 100):
        self.base_url = base_url
        self.num_users = num_users
        self.session: Optional[aiohttp.ClientSession] = None
        self.metrics: List[Dict[str, Any]] = []
    
    async def setup(self) -> None:
        """Setup load generator"""
        self.session = aiohttp.ClientSession()
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        if self.session:
            await self.session.close()
    
    async def generate_load(self, duration_seconds: int = 60) -> LoadTestMetrics:
        """Generate load for specified duration"""
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        tasks = []
        total_requests = 0
        successful_requests = 0
        response_times = []
        
        while time.time() < end_time:
            # Create concurrent requests
            batch_tasks = [
                self._make_request()
                for _ in range(self.num_users)
            ]
            
            # Execute batch
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process results
            for result in batch_results:
                total_requests += 1
                if isinstance(result, Exception):
                    continue
                    
                successful_requests += 1
                response_times.append(result)
            
            # Brief pause between batches
            await asyncio.sleep(0.1)
        
        # Calculate metrics
        actual_duration = time.time() - start_time
        rps = total_requests / actual_duration
        avg_response_time = statistics.mean(response_times) if response_times else 0
        error_rate = (total_requests - successful_requests) / total_requests if total_requests > 0 else 0
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        return LoadTestMetrics(
            requests_per_second=rps,
            average_response_time=avg_response_time,
            error_rate=error_rate,
            cpu_usage=cpu_percent,
            memory_usage=memory_percent,
            concurrent_connections=self.num_users,
            success_rate=success_rate
        )
    
    async def _make_request(self) -> float:
        """Make a single request and return response time"""
        if not self.session:
            raise RuntimeError("Session not initialized")
            
        start_time = time.time()
        async with self.session.get(f"{self.base_url}/api/v1/status") as response:
            await response.text()
            return time.time() - start_time

@pytest.mark.load
class TestLoadPerformance:
    """Load testing suite"""
    
    @pytest.fixture
    async def load_generator(self):
        """Create load generator instance"""
        generator = LoadGenerator("http://localhost:8000")
        await generator.setup()
        yield generator
        await generator.cleanup()
    
    @pytest.mark.asyncio
    async def test_baseline_performance(self, load_generator):
        """Test baseline performance with moderate load"""
        metrics = await load_generator.generate_load(duration_seconds=30)
        
        # Verify baseline performance
        assert metrics.requests_per_second >= 10
        assert metrics.average_response_time <= 0.1
        assert metrics.error_rate <= 0.01
        assert metrics.success_rate >= 0.99
    
    @pytest.mark.asyncio
    async def test_high_concurrency(self, load_generator):
        """Test performance under high concurrency"""
        # Increase number of concurrent users
        load_generator.num_users = 500
        
        metrics = await load_generator.generate_load(duration_seconds=30)
        
        # Verify system handles high concurrency
        assert metrics.requests_per_second >= 50
        assert metrics.average_response_time <= 0.2
        assert metrics.error_rate <= 0.05
        assert metrics.success_rate >= 0.95
    
    @pytest.mark.asyncio
    async def test_sustained_load(self, load_generator):
        """Test performance under sustained load"""
        # Run for longer duration
        metrics = await load_generator.generate_load(duration_seconds=300)
        
        # Verify system maintains performance
        assert metrics.requests_per_second >= 10
        assert metrics.average_response_time <= 0.15
        assert metrics.error_rate <= 0.02
        assert metrics.success_rate >= 0.98
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_activation(self, load_generator):
        """Test circuit breaker activation under load"""
        # Create extreme load
        load_generator.num_users = 1000
        
        metrics = await load_generator.generate_load(duration_seconds=30)
        
        # Verify circuit breakers activated appropriately
        breakers = circuit_breaker_registry.get_all_breakers()
        assert any(breaker.is_open() for breaker in breakers)
    
    @pytest.mark.asyncio
    async def test_resource_limits(self, load_generator):
        """Test system behavior near resource limits"""
        initial_metrics = await load_generator.generate_load(duration_seconds=10)
        
        # Gradually increase load
        for num_users in [200, 400, 600, 800, 1000]:
            load_generator.num_users = num_users
            metrics = await load_generator.generate_load(duration_seconds=10)
            
            # Record metrics for analysis
            performance_monitor.record_load_test_metrics(metrics)
            
            # Verify graceful degradation
            assert metrics.success_rate > 0.5  # System should maintain at least 50% success rate
            
            # Brief cooldown between tests
            await asyncio.sleep(5)
    
    @pytest.mark.asyncio
    async def test_distributed_lock_performance(self, load_generator):
        """Test distributed lock performance under load"""
        async def acquire_and_release_lock():
            lock_name = "test_lock"
            acquired = await distributed_lock_manager.acquire_lock(lock_name)
            if acquired:
                await asyncio.sleep(0.1)  # Simulate work
                await distributed_lock_manager.release_lock(lock_name)
            return acquired
        
        # Run concurrent lock operations
        tasks = [acquire_and_release_lock() for _ in range(100)]
        results = await asyncio.gather(*tasks)
        
        # Verify lock behavior
        assert sum(results) > 0  # Some locks should be acquired
        assert sum(results) < len(results)  # But not all (due to contention) 