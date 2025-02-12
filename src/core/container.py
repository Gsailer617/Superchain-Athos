"""Dependency injection container configuration"""

from dependency_injector import containers, providers
from src.services.interfaces import (
    BlockchainServiceInterface,
    DexServiceInterface,
    DataProviderInterface,
    ValidationServiceInterface,
    MLServiceInterface
)
from src.utils.cache import CacheConfig
from src.utils.rate_limiter import RateLimiterRegistry
from src.monitoring.metrics import MetricsManager
from src.monitoring.monitoring import MonitoringManager, CrossChainMonitor, MLMonitor, LLMMonitor

class Container(containers.DeclarativeContainer):
    """Application container."""
    
    config = providers.Configuration()
    
    # Core utilities
    cache_config = providers.Singleton(
        CacheConfig,
        duration=config.cache.duration,
        refresh_threshold=config.cache.refresh_threshold,
        max_size=config.cache.max_size
    )
    
    rate_limiter_registry = providers.Singleton(
        RateLimiterRegistry
    )
    
    metrics_manager = providers.Singleton(
        MetricsManager
    )
    
    # Monitoring services
    monitoring_manager = providers.Singleton(
        MonitoringManager,
        config=config.monitoring
    )
    
    # Cross-chain monitoring
    cross_chain_monitor = providers.Singleton(
        CrossChainMonitor,
        metrics_manager=metrics_manager,
        config=config.monitoring
    )
    
    # ML monitoring
    ml_monitor = providers.Singleton(
        MLMonitor,
        metrics_manager=metrics_manager,
        config=config.monitoring
    )
    
    # LLM monitoring
    llm_monitor = providers.Singleton(
        LLMMonitor,
        metrics_manager=metrics_manager,
        config=config.monitoring
    )
    
    # External services
    blockchain_service = providers.AbstractSingleton(
        BlockchainServiceInterface
    )
    
    dex_service = providers.AbstractSingleton(
        DexServiceInterface
    )
    
    data_provider = providers.AbstractSingleton(
        DataProviderInterface
    )
    
    validation_service = providers.AbstractSingleton(
        ValidationServiceInterface
    )
    
    ml_service = providers.AbstractSingleton(
        MLServiceInterface
    )
    
    # Service factory methods
    @classmethod
    def configure_blockchain_service(cls, implementation: type[BlockchainServiceInterface]) -> None:
        """Configure blockchain service implementation"""
        cls.blockchain_service.override(providers.Singleton(implementation))
    
    @classmethod
    def configure_dex_service(cls, implementation: type[DexServiceInterface]) -> None:
        """Configure DEX service implementation"""
        cls.dex_service.override(providers.Singleton(implementation))
    
    @classmethod
    def configure_data_provider(cls, implementation: type[DataProviderInterface]) -> None:
        """Configure data provider implementation"""
        cls.data_provider.override(providers.Singleton(implementation))
    
    @classmethod
    def configure_validation_service(cls, implementation: type[ValidationServiceInterface]) -> None:
        """Configure validation service implementation"""
        cls.validation_service.override(providers.Singleton(implementation))
    
    @classmethod
    def configure_ml_service(cls, implementation: type[MLServiceInterface]) -> None:
        """Configure ML service implementation"""
        cls.ml_service.override(providers.Singleton(implementation)) 