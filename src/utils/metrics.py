"""
Metrics Module

This module provides centralized Prometheus metrics setup and reporting functionality.
"""

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
from typing import Dict, Union, Optional
import logging

logger = logging.getLogger(__name__)

class MetricsManager:
    """
    Centralized metrics management for token discovery and validation
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize metrics manager with optional custom registry
        
        Args:
            registry: Optional custom Prometheus registry
        """
        self.registry = registry or CollectorRegistry()
        self._setup_metrics()
        
    def _setup_metrics(self):
        """Initialize all Prometheus metrics"""
        # Discovery metrics
        self.tokens_discovered = Counter(
            'tokens_discovered_total',
            'Total number of tokens discovered',
            ['source'],
            registry=self.registry
        )
        
        self.validation_results = Counter(
            'token_validation_results_total',
            'Results of token validation',
            ['result'],
            registry=self.registry
        )
        
        self.validation_duration = Histogram(
            'token_validation_duration_seconds',
            'Time spent validating tokens',
            registry=self.registry
        )
        
        # API metrics
        self.api_requests = Counter(
            'api_requests_total',
            'Total API requests made',
            ['api', 'endpoint'],
            registry=self.registry
        )
        
        self.api_errors = Counter(
            'api_errors_total',
            'Total API errors encountered',
            ['api', 'error_type'],
            registry=self.registry
        )
        
        self.api_latency = Histogram(
            'api_request_duration_seconds',
            'API request latency',
            ['api'],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_operations = Counter(
            'cache_operations_total',
            'Total cache operations',
            ['operation'],
            registry=self.registry
        )
        
        self.cache_hits = Counter(
            'cache_hits_total',
            'Total cache hits',
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'cache_misses_total',
            'Total cache misses',
            registry=self.registry
        )
        
        # Social sentiment metrics
        self.sentiment_scores = Histogram(
            'token_sentiment_scores',
            'Distribution of token sentiment scores',
            ['source'],
            registry=self.registry
        )
        
        # Security metrics
        self.security_scores = Histogram(
            'token_security_scores',
            'Distribution of token security scores',
            registry=self.registry
        )
        
        # Liquidity metrics
        self.liquidity_amount = Gauge(
            'token_liquidity_amount',
            'Current token liquidity amount',
            ['token_address', 'dex'],
            registry=self.registry
        )
        
        # Rate limiting metrics
        self.rate_limit_hits = Counter(
            'rate_limit_hits_total',
            'Number of rate limit hits',
            ['api'],
            registry=self.registry
        )
        
    def record_discovery(self, source: str):
        """Record token discovery from a source"""
        self.tokens_discovered.labels(source=source).inc()
        
    def record_validation(self, success: bool):
        """Record token validation result"""
        result = 'success' if success else 'failure'
        self.validation_results.labels(result=result).inc()
        
    def record_api_request(self, api: str, endpoint: str):
        """Record API request"""
        self.api_requests.labels(api=api, endpoint=endpoint).inc()
        
    def record_api_error(self, api: str, error_type: str):
        """Record API error"""
        self.api_errors.labels(api=api, error_type=error_type).inc()
        
    def record_cache_operation(self, operation: str):
        """Record cache operation"""
        self.cache_operations.labels(operation=operation).inc()
        
    def record_cache_hit(self):
        """Record cache hit"""
        self.cache_hits.inc()
        
    def record_cache_miss(self):
        """Record cache miss"""
        self.cache_misses.inc()
        
    def record_sentiment_score(self, source: str, score: float):
        """Record sentiment analysis score"""
        self.sentiment_scores.labels(source=source).observe(score)
        
    def record_security_score(self, score: float):
        """Record security analysis score"""
        self.security_scores.observe(score)
        
    def update_liquidity(self, token_address: str, dex: str, amount: float):
        """Update token liquidity amount"""
        self.liquidity_amount.labels(
            token_address=token_address,
            dex=dex
        ).set(amount)
        
    def record_rate_limit(self, api: str):
        """Record rate limit hit"""
        self.rate_limit_hits.labels(api=api).inc()
        
    def get_metrics(self) -> Dict[str, float]:
        """
        Get current metrics values
        
        Returns:
            Dictionary of metric names and their current values
        """
        return {
            'discoveries': self.tokens_discovered._value.sum(),
            'validations_success': self.validation_results.labels(result='success')._value,
            'validations_failure': self.validation_results.labels(result='failure')._value,
            'cache_hit_ratio': (
                self.cache_hits._value / 
                (self.cache_hits._value + self.cache_misses._value)
                if (self.cache_hits._value + self.cache_misses._value) > 0
                else 0
            )
        } 