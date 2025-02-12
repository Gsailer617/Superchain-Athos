"""Shared test fixtures that don't affect live code"""

import pytest
from typing import Dict, Any
from web3 import Web3
import redis
import asyncio
from tests.utils.test_utils import create_mock_web3, get_test_config, create_mock_market_data
from src.monitoring.monitor_manager import MonitorManager
from src.monitoring.performance_monitor import PerformanceMonitor

@pytest.fixture
def mock_web3() -> Web3:
    """Fixture for mock Web3 instance"""
    return create_mock_web3()

@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Fixture for test configuration"""
    return get_test_config()

@pytest.fixture
def mock_market_data() -> Dict[str, Any]:
    """Fixture for mock market data"""
    return create_mock_market_data()

@pytest.fixture
def mock_token_pair() -> tuple[str, str]:
    """Fixture for mock token pair"""
    return ('0x1234...', '0x5678...')

@pytest.fixture
def mock_test_amounts() -> list[float]:
    """Fixture for mock test amounts"""
    return [0.1, 0.5, 1.0]

@pytest.fixture
async def redis_client():
    """Fixture for Redis client"""
    client = redis.Redis(
        host='localhost',
        port=6379,
        db=1,  # Use different DB for tests
        decode_responses=True
    )
    yield client
    await client.aclose()

@pytest.fixture
async def monitor_config() -> Dict[str, Any]:
    """Fixture for monitoring configuration"""
    return {
        'monitoring': {
            'storage_path': 'tests/data/monitoring',
            'prometheus_port': 8002,
            'cache_enabled': True,
            'redis_host': 'localhost',
            'redis_port': 6379,
            'redis_db': 1,
            'max_memory_entries': 1000,
            'flush_interval': 10
        },
        'jaeger': {
            'host': 'localhost',
            'port': 6831
        },
        'sentry': {
            'dsn': None,  # Disable Sentry for tests
            'traces_sample_rate': 0.0
        }
    }

@pytest.fixture
async def performance_monitor():
    """Fixture for PerformanceMonitor"""
    monitor = PerformanceMonitor(port=8001)  # Use different port for tests
    yield monitor
    await monitor.stop()

@pytest.fixture
async def monitor_manager(monitor_config):
    """Fixture for MonitorManager"""
    manager = MonitorManager(monitor_config)
    await manager.start()
    yield manager
    await manager.stop()

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close() 