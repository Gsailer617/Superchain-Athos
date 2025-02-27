"""
Configuration Loader

This module provides utilities for loading and managing configuration
from various sources including files, environment variables, and defaults.
"""

import os
import json
import logging
import yaml
from typing import Dict, Any, Optional, List, Union, TypeVar, Type, Callable, cast
from pathlib import Path
from dataclasses import asdict

from .environment import env, EnvironmentManager
from .validators import validate_chain_json, validate_config_file
from .chain_config_template import ChainConfig
from .chain_configurations import CHAIN_CONFIGS

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ConfigLoader:
    """
    Configuration loader that supports multiple formats and environments.
    
    Features:
    - Load from JSON, YAML, or Python files
    - Environment variable interpolation
    - Validation against schemas
    - Default values fallback
    """
    
    def __init__(self, env_manager: Optional[EnvironmentManager] = None):
        self.env = env_manager or env
        self.config_cache: Dict[str, Any] = {}
        
    def load_json_config(self, path: str) -> Dict[str, Any]:
        """
        Load and parse a JSON configuration file
        
        Args:
            path: Path to JSON config file
            
        Returns:
            Parsed configuration dictionary
        """
        return self.env.load_json_config(path)
    
    def load_yaml_config(self, path: str) -> Dict[str, Any]:
        """
        Load and parse a YAML configuration file
        
        Args:
            path: Path to YAML config file
            
        Returns:
            Parsed configuration dictionary
        """
        try:
            config_path = Path(path)
            if not config_path.exists():
                logger.warning(f"Config file not found: {path}")
                return {}
                
            with open(config_path, 'r') as f:
                config_str = f.read()
                
            # Replace environment variables
            config_str = self.env.interpolate_config(config_str)
            
            # Parse YAML
            return yaml.safe_load(config_str)
            
        except Exception as e:
            logger.error(f"Error loading YAML config file {path}: {e}")
            return {}
    
    def load_chain_config(self, chain_name: str) -> Optional[ChainConfig]:
        """
        Load chain configuration by name
        
        Args:
            chain_name: Chain name (e.g., 'ethereum', 'polygon')
            
        Returns:
            ChainConfig if found, None otherwise
        """
        # Check predefined configurations first
        if chain_name in CHAIN_CONFIGS:
            chain_config = CHAIN_CONFIGS[chain_name]
            
            # Update with environment-specific overrides
            chain_env_file = f"config/chains/{chain_name}.{self.env.env_name}.json"
            if os.path.exists(chain_env_file):
                try:
                    overrides = self.load_json_config(chain_env_file)
                    
                    # TODO: Apply overrides to chain_config
                    # This would require a deep merge implementation
                    # For now, log that we found overrides
                    logger.info(f"Found environment-specific overrides for {chain_name}")
                    
                except Exception as e:
                    logger.error(f"Error loading chain overrides: {e}")
            
            return chain_config
        
        # Try to load from JSON config file
        chain_file = f"config/chains/{chain_name}.json"
        if os.path.exists(chain_file):
            try:
                chain_data = self.load_json_config(chain_file)
                
                # Validate the configuration
                errors = validate_chain_json(chain_data)
                if errors:
                    for error in errors:
                        logger.error(f"Chain config validation error for {chain_name}: {error}")
                    return None
                
                # TODO: Convert to ChainConfig object
                # This would require implementing a conversion function
                logger.info(f"Loaded chain configuration from file: {chain_name}")
                
                # For now, return None
                return None
                
            except Exception as e:
                logger.error(f"Error loading chain configuration: {e}")
                return None
        
        logger.warning(f"Chain configuration not found for: {chain_name}")
        return None
    
    def load_config_to_dataclass(
        self, 
        config_path: str,
        dataclass_type: Type[T],
        converter: Optional[Callable[[Dict[str, Any]], T]] = None
    ) -> Optional[T]:
        """
        Load configuration file and convert to dataclass instance
        
        Args:
            config_path: Path to configuration file
            dataclass_type: Target dataclass type
            converter: Optional function to convert dict to dataclass
            
        Returns:
            Dataclass instance if successful, None otherwise
        """
        # Determine file format from extension
        if config_path.endswith('.json'):
            config_data = self.load_json_config(config_path)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config_data = self.load_yaml_config(config_path)
        else:
            logger.error(f"Unsupported config file format: {config_path}")
            return None
            
        if not config_data:
            return None
            
        try:
            # Use converter if provided
            if converter:
                return converter(config_data)
                
            # Simple conversion using dataclass constructor
            return dataclass_type(**config_data)
            
        except Exception as e:
            logger.error(f"Error converting config to {dataclass_type.__name__}: {e}")
            return None
    
    def export_config_to_json(self, config: Any, output_path: str) -> bool:
        """
        Export configuration to JSON file
        
        Args:
            config: Configuration object (dataclass or dict)
            output_path: Path to output JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert dataclass to dict if needed
            if hasattr(config, '__dataclass_fields__'):
                config_dict = asdict(config)
            else:
                config_dict = dict(config)
                
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
            # Write to file
            with open(output_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
                
            logger.info(f"Exported configuration to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            return False

# Create a default loader instance
config_loader = ConfigLoader()

def get_config_loader(env_name: Optional[str] = None) -> ConfigLoader:
    """Get configuration loader instance"""
    if not env_name:
        return config_loader
        
    return ConfigLoader(EnvironmentManager(env_name)) 