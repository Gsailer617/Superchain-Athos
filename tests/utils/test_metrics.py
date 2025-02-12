"""Unit tests for the metrics module"""

import pytest
from unittest.mock import MagicMock, patch
from prometheus_client import CollectorRegistry
from src.utils.metrics import MetricsManager

@pytest.fixture
def registry():
    """Create a new registry for each test"""
    return CollectorRegistry()

@pytest.fixture
def metrics_manager(registry):
    """Create a metrics manager instance"""
    return MetricsManager(registry)

def test_initialization(metrics_manager):
    """Test metrics manager initialization"""
    assert metrics_manager.registry is not None
    
    # Verify all metrics are initialized
    assert metrics_manager.tokens_discovered is not None
    assert metrics_manager.validation_results is not None
    assert metrics_manager.validation_duration is not None
    assert metrics_manager.api_requests is not None
    assert metrics_manager.api_errors is not None
    assert metrics_manager.api_latency is not None
    assert metrics_manager.cache_operations is not None
    assert metrics_manager.cache_hits is not None
    assert metrics_manager.cache_misses is not None
    assert metrics_manager.sentiment_scores is not None
    assert metrics_manager.security_scores is not None
    assert metrics_manager.liquidity_amount is not None
    assert metrics_manager.rate_limit_hits is not None

def test_record_discovery(metrics_manager):
    """Test recording token discovery"""
    metrics_manager.record_discovery('dex')
    
    # Get the current value
    value = 0
    for metric in metrics_manager.tokens_discovered.collect():
        for sample in metric.samples:
            if sample.labels['source'] == 'dex':
                value = sample.value
                break
    
    assert value == 1.0

def test_record_validation(metrics_manager):
    """Test recording validation results"""
    # Test successful validation
    metrics_manager.record_validation(True)
    success_value = 0
    for metric in metrics_manager.validation_results.collect():
        for sample in metric.samples:
            if sample.labels['result'] == 'success':
                success_value = sample.value
                break
    assert success_value == 1.0
    
    # Test failed validation
    metrics_manager.record_validation(False)
    failure_value = 0
    for metric in metrics_manager.validation_results.collect():
        for sample in metric.samples:
            if sample.labels['result'] == 'failure':
                failure_value = sample.value
                break
    assert failure_value == 1.0

def test_record_api_request(metrics_manager):
    """Test recording API requests"""
    metrics_manager.record_api_request('etherscan', 'getTokenInfo')
    
    value = 0
    for metric in metrics_manager.api_requests.collect():
        for sample in metric.samples:
            if (sample.labels['api'] == 'etherscan' and 
                sample.labels['endpoint'] == 'getTokenInfo'):
                value = sample.value
                break
    
    assert value == 1.0

def test_record_api_error(metrics_manager):
    """Test recording API errors"""
    metrics_manager.record_api_error('etherscan', 'rate_limit')
    
    value = 0
    for metric in metrics_manager.api_errors.collect():
        for sample in metric.samples:
            if (sample.labels['api'] == 'etherscan' and 
                sample.labels['error_type'] == 'rate_limit'):
                value = sample.value
                break
    
    assert value == 1.0

def test_record_cache_operations(metrics_manager):
    """Test recording cache operations"""
    metrics_manager.record_cache_operation('set')
    
    value = 0
    for metric in metrics_manager.cache_operations.collect():
        for sample in metric.samples:
            if sample.labels['operation'] == 'set':
                value = sample.value
                break
    
    assert value == 1.0

def test_record_cache_hits_and_misses(metrics_manager):
    """Test recording cache hits and misses"""
    # Record hits
    metrics_manager.record_cache_hit()
    hits = 0
    for metric in metrics_manager.cache_hits.collect():
        for sample in metric.samples:
            hits = sample.value
            break
    assert hits == 1.0
    
    # Record misses
    metrics_manager.record_cache_miss()
    misses = 0
    for metric in metrics_manager.cache_misses.collect():
        for sample in metric.samples:
            misses = sample.value
            break
    assert misses == 1.0

def test_record_sentiment_score(metrics_manager):
    """Test recording sentiment scores"""
    metrics_manager.record_sentiment_score('telegram', 0.8)
    
    # Verify the score was recorded
    found = False
    for metric in metrics_manager.sentiment_scores.collect():
        for sample in metric.samples:
            if (sample.labels['source'] == 'telegram' and 
                sample.name.endswith('_sum')):
                assert sample.value == 0.8
                found = True
                break
    assert found

def test_record_security_score(metrics_manager):
    """Test recording security scores"""
    metrics_manager.record_security_score(0.9)
    
    # Verify the score was recorded
    found = False
    for metric in metrics_manager.security_scores.collect():
        for sample in metric.samples:
            if sample.name.endswith('_sum'):
                assert sample.value == 0.9
                found = True
                break
    assert found

def test_update_liquidity(metrics_manager):
    """Test updating liquidity amounts"""
    metrics_manager.update_liquidity(
        '0x123...', 'uniswap', 1000000.0
    )
    
    value = 0
    for metric in metrics_manager.liquidity_amount.collect():
        for sample in metric.samples:
            if (sample.labels['token_address'] == '0x123...' and 
                sample.labels['dex'] == 'uniswap'):
                value = sample.value
                break
    
    assert value == 1000000.0

def test_record_rate_limit(metrics_manager):
    """Test recording rate limit hits"""
    metrics_manager.record_rate_limit('etherscan')
    
    value = 0
    for metric in metrics_manager.rate_limit_hits.collect():
        for sample in metric.samples:
            if sample.labels['api'] == 'etherscan':
                value = sample.value
                break
    
    assert value == 1.0

def test_get_metrics(metrics_manager):
    """Test getting aggregated metrics"""
    # Record some test data
    metrics_manager.record_discovery('dex')
    metrics_manager.record_validation(True)
    metrics_manager.record_validation(False)
    metrics_manager.record_cache_hit()
    metrics_manager.record_cache_miss()
    
    # Get metrics
    metrics = metrics_manager.get_metrics()
    
    # Verify metrics
    assert metrics['discoveries'] == 1.0
    assert metrics['validations_success'] == 1.0
    assert metrics['validations_failure'] == 1.0
    assert metrics['cache_hit_ratio'] == 0.5  # 1 hit, 1 miss 