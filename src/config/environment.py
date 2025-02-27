"""
Environment Configuration

This module provides utilities for loading environment variables and secrets
to be used in configuration files, ensuring sensitive data remains secure
and configuration remains flexible across environments.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class EnvironmentManager:
    """
    Manages environment variables and secrets for application configuration.
    
    Features:
    - Loads from .env files
    - Supports environment-specific configs (.env.development, .env.production)
    - Interpolates environment variables in config values
    - Supports secrets loading from secure storage
    """
    
    def __init__(self, env_name: Optional[str] = None):
        # Load .env files
        self.env_name = env_name or os.getenv("APP_ENV", "development")
        self.load_env_files()
        
        # Cache values
        self._env_cache: Dict[str, Any] = {}
        
    def load_env_files(self) -> None:
        """Load environment variables from .env files"""
        # Base .env file
        load_dotenv()
        
        # Environment specific file (.env.development, .env.production, etc.)
        env_specific_path = f".env.{self.env_name}"
        if os.path.exists(env_specific_path):
            load_dotenv(env_specific_path)
            logger.info(f"Loaded environment specific config from {env_specific_path}")
            
        # Local overrides (not in version control)
        local_env_path = ".env.local"
        if os.path.exists(local_env_path):
            load_dotenv(local_env_path)
            logger.info(f"Loaded local environment overrides from {local_env_path}")
            
    def get(self, key: str, default: Any = None) -> Any:
        """Get environment variable value with optional default"""
        if key in self._env_cache:
            return self._env_cache[key]
            
        value = os.getenv(key, default)
        self._env_cache[key] = value
        return value
        
    def interpolate_config(self, config_str: str) -> str:
        """Interpolate environment variables in config string
        
        Replaces ${VAR_NAME} with the value of environment variable VAR_NAME
        """
        import re
        
        # Pattern to match ${VAR_NAME} format
        pattern = r'\${([A-Za-z0-9_]+)}'
        
        def replace_var(match):
            var_name = match.group(1)
            return self.get(var_name, '')
            
        return re.sub(pattern, replace_var, config_str)
        
    def load_rpc_secrets(self) -> Dict[str, str]:
        """Load RPC API keys and endpoints from secure storage"""
        secrets = {}
        
        # Common API providers
        providers = [
            "INFURA", "ALCHEMY", "ANKR", "MORALIS", 
            "GETBLOCK", "QUICKNODE", "LLAMA_NODES"
        ]
        
        for provider in providers:
            key = f"{provider}_KEY"
            if os.getenv(key):
                secrets[key] = os.getenv(key)
                
        return secrets
        
    def load_json_config(self, path: str) -> Dict[str, Any]:
        """Load JSON config file with environment variable interpolation"""
        try:
            config_path = Path(path)
            if not config_path.exists():
                logger.warning(f"Config file not found: {path}")
                return {}
                
            with open(config_path, 'r') as f:
                config_str = f.read()
                
            # Replace environment variables
            config_str = self.interpolate_config(config_str)
            
            # Parse JSON
            return json.loads(config_str)
            
        except Exception as e:
            logger.error(f"Error loading config file {path}: {e}")
            return {}
            
    def get_chain_api_keys(self) -> Dict[str, Dict[str, str]]:
        """Get API keys for various chain services"""
        api_keys = {}
        
        # Common explorer APIs
        explorers = {
            "etherscan": "ETHERSCAN_API_KEY",
            "bscscan": "BSCSCAN_API_KEY",
            "polygonscan": "POLYGONSCAN_API_KEY",
            "optimistic_etherscan": "OPTIMISM_API_KEY",
            "arbiscan": "ARBITRUM_API_KEY",
            "snowtrace": "AVALANCHE_API_KEY",
            "ftmscan": "FANTOM_API_KEY",
            "basescan": "BASE_API_KEY",
            "modescan": "MODE_API_KEY"
        }
        
        for explorer, env_var in explorers.items():
            key = self.get(env_var)
            if key:
                api_keys[explorer] = {"api_key": key}
                
        return api_keys

# Singleton instance for easy imports
env = EnvironmentManager()

def get_env_manager(env_name: Optional[str] = None) -> EnvironmentManager:
    """Get environment manager instance"""
    global env
    if env_name and env_name != env.env_name:
        env = EnvironmentManager(env_name)
    return env 