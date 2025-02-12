"""Performance benchmarking test suite"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from src.agent.token_discovery import TokenDiscovery
from src.utils.rate_limiter import AsyncRateLimiter
from src.services.sentiment import SentimentAnalyzer

def measure_time(func):
    """Decorator to measure execution time"""
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        end = time.perf_counter()
        duration = end - start
        print(f"{func.__name__} took {duration:.2f} seconds")
        return result, duration
    return wrapper

@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmarking tests"""
    
    @pytest.fixture
    async def setup(self):
        """Setup test environment"""
        config = {
            'redis_url': 'redis://localhost',
            'etherscan_api_key': 'test_key',
            'max_concurrent_requests': 10,
            'batch_size': 100
        }
        self.discovery = TokenDiscovery(config)
        yield
        await self.discovery.cleanup()
    
    @measure_time
    async def test_token_discovery_performance(self, setup):
        """Benchmark token discovery performance"""
        # Test discovering multiple tokens in parallel
        tokens = await self.discovery.discover_new_tokens(
            max_tokens=100,
            concurrent_limit=10
        )
        return len(tokens)
    
    @measure_time
    async def test_validation_performance(self, setup):
        """Benchmark token validation performance"""
        # Generate test addresses
        addresses = [f"0x{i:040x}" for i in range(100)]
        
        # Validate tokens in parallel
        tasks = [
            self.discovery.validate_token(addr)
            for addr in addresses
        ]
        results = await asyncio.gather(*tasks)
        return len([r for r in results if r])
    
    @measure_time
    async def test_sentiment_analysis_performance(self):
        """Benchmark sentiment analysis performance"""
        analyzer = SentimentAnalyzer()
        
        # Generate test messages
        messages = {
            'telegram': ["Great project!"] * 50,
            'discord': ["Amazing team!"] * 50
        }
        
        # Analyze sentiment in parallel
        tasks = []
        for _ in range(10):  # Test 10 different tokens
            tasks.append(
                analyzer.get_token_sentiment(
                    f"0x{_:040x}",
                    messages.copy()
                )
            )
        results = await asyncio.gather(*tasks)
        return len(results)
    
    @measure_time
    async def test_rate_limiter_performance(self):
        """Benchmark rate limiter performance"""
        limiter = AsyncRateLimiter(
            'test',
            max_requests=100,
            requests_per_second=50
        )
        
        async def make_request():
            async with limiter:
                await asyncio.sleep(0.01)  # Simulate API call
        
        # Make concurrent requests
        tasks = [make_request() for _ in range(200)]
        await asyncio.gather(*tasks)
        return 200
    
    @measure_time
    async def test_cache_performance(self, setup):
        """Benchmark cache performance"""
        # Generate test data
        data = {f"key_{i}": f"value_{i}" for i in range(1000)}
        
        # Test cache write performance
        write_start = time.perf_counter()
        tasks = [
            self.discovery.cache.set(k, v)
            for k, v in data.items()
        ]
        await asyncio.gather(*tasks)
        write_duration = time.perf_counter() - write_start
        
        # Test cache read performance
        read_start = time.perf_counter()
        tasks = [
            self.discovery.cache.get(k)
            for k in data.keys()
        ]
        results = await asyncio.gather(*tasks)
        read_duration = time.perf_counter() - read_start
        
        return {
            'write_duration': write_duration,
            'read_duration': read_duration,
            'total_operations': len(data) * 2
        }

@pytest.mark.benchmark
class TestNetworkSimulation:
    """Network delay simulation tests"""
    
    @pytest.fixture
    def network_delay():
        """Simulate network delay"""
        async def delay_middleware(request):
            await asyncio.sleep(0.1)  # 100ms delay
            return await request
        return delay_middleware
    
    @measure_time
    async def test_dex_interaction_with_delay(self, network_delay):
        """Test DEX interactions with network delay"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.get = AsyncMock(side_effect=network_delay)
            
            # Simulate multiple DEX API calls
            tasks = [
                mock_session.get(f"http://api.dex.com/token/{i}")
                for i in range(50)
            ]
            results = await asyncio.gather(*tasks)
            return len(results)
    
    @measure_time
    async def test_blockchain_interaction_with_delay(self, network_delay):
        """Test blockchain interactions with network delay"""
        with patch('web3.eth.Eth') as mock_eth:
            mock_eth.get_block = AsyncMock(side_effect=network_delay)
            
            # Simulate multiple blockchain calls
            tasks = [
                mock_eth.get_block(i)
                for i in range(50)
            ]
            results = await asyncio.gather(*tasks)
            return len(results)

@pytest.mark.benchmark
class TestGasOptimization:
    """Gas optimization benchmarking"""
    
    @measure_time
    async def test_gas_estimation_accuracy(self):
        """Test gas estimation accuracy"""
        with patch('web3.eth.Eth') as mock_eth:
            mock_eth.estimate_gas = AsyncMock(return_value=100000)
            mock_eth.get_transaction_receipt = AsyncMock(
                return_value={'gasUsed': 95000}
            )
            
            # Simulate multiple transactions
            estimates = []
            actual = []
            for _ in range(100):
                estimates.append(await mock_eth.estimate_gas())
                receipt = await mock_eth.get_transaction_receipt()
                actual.append(receipt['gasUsed'])
            
            # Calculate accuracy
            accuracy = sum(
                1 for e, a in zip(estimates, actual)
                if abs(e - a) / e < 0.1  # Within 10%
            ) / len(estimates)
            
            return accuracy
    
    @measure_time
    async def test_gas_optimization_strategies(self):
        """Test different gas optimization strategies"""
        strategies = {
            'legacy': lambda: 100000,
            'eip1559': lambda: (50000, 100000),  # base, max
            'dynamic': lambda: min(100000, 80000 + time.time() % 40000)
        }
        
        results = {}
        for name, strategy in strategies.items():
            start = time.perf_counter()
            for _ in range(1000):
                _ = strategy()
            duration = time.perf_counter() - start
            results[name] = duration
        
        return results 