"""
Configuration Management System

This module centralizes all configuration-related imports and provides
a unified interface for accessing configuration data across the application.
"""

# Core configuration types and enums
from .chain_config_template import (
    ConsensusType, RollupType, GasModel,
    RPCConfig, GasConfig, BlockConfig,
    SecurityConfig, NetworkConfig, APIConfig,
    PerformanceConfig, ChainFeatures, BridgeConfig,
    ChainConfig
)

# Chain specifications
from .chain_specs import (
    ChainSpec, get_chain_spec, 
    get_all_supported_chains as get_all_chain_specs,
    get_l2_chains as get_l2_chain_specs,
    get_l1_chains as get_l1_chain_specs
)

# Chain configurations
from .chain_configurations import (
    get_chain_config, get_all_supported_chains,
    get_l2_chains, get_l1_chains
)

# Environment management
from .environment import (
    EnvironmentManager, env, get_env_manager
)

# Configuration validation
from .validators import (
    validate_chain_json, validate_chain_config, 
    validate_config_file, validate_all_chain_configs
)

# Configuration loading
from .loader import (
    ConfigLoader, config_loader, get_config_loader
)

# Simplified configuration interface
class ConfigManager:
    """
    Centralized configuration manager that provides access
    to all configuration data through a unified interface.
    """
    
    @staticmethod
    def get_chain_config(chain_name: str) -> ChainConfig:
        """Get full chain configuration by name"""
        return get_chain_config(chain_name)
        
    @staticmethod
    def get_chain_spec(chain_name: str) -> ChainSpec:
        """Get simplified chain specification by name"""
        return get_chain_spec(chain_name)
        
    @staticmethod
    def get_all_chains() -> list[str]:
        """Get all supported chain names"""
        return get_all_supported_chains()
        
    @staticmethod
    def get_l1_chains() -> list[str]:
        """Get L1 chain names"""
        return get_l1_chains()
        
    @staticmethod
    def get_l2_chains() -> list[str]:
        """Get L2 chain names"""
        return get_l2_chains()
        
    @staticmethod
    def load_chain_config(chain_name: str, env_name: str = None):
        """Load chain configuration with environment specifics"""
        loader = get_config_loader(env_name)
        return loader.load_chain_config(chain_name)
        
    @staticmethod
    def validate_chain_config(config: ChainConfig) -> list[str]:
        """Validate chain configuration"""
        return validate_chain_config(config)
        
    @staticmethod
    def get_environment() -> EnvironmentManager:
        """Get current environment manager"""
        return env

# Create a singleton instance for easy imports
config = ConfigManager()

__all__ = [
    # Core types
    'ConsensusType', 'RollupType', 'GasModel',
    'RPCConfig', 'GasConfig', 'BlockConfig',
    'SecurityConfig', 'NetworkConfig', 'APIConfig',
    'PerformanceConfig', 'ChainFeatures', 'BridgeConfig',
    'ChainConfig', 'ChainSpec',
    
    # Chain configuration functions
    'get_chain_config', 'get_chain_spec',
    'get_all_supported_chains', 'get_l2_chains', 'get_l1_chains',
    
    # Environment management
    'EnvironmentManager', 'env', 'get_env_manager',
    
    # Configuration validation
    'validate_chain_json', 'validate_chain_config',
    'validate_config_file', 'validate_all_chain_configs',
    
    # Configuration loading
    'ConfigLoader', 'config_loader', 'get_config_loader',
    
    # ConfigManager
    'ConfigManager', 'config'
] 