import os
from dotenv import load_dotenv
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

class EnvironmentConfig:
    """Centralized environment configuration management"""
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # Load environment variables only once
        load_dotenv()
        
        # Check if we're in test mode
        self.is_test = os.getenv('TEST_MODE') == '1'
        
        # Required environment variables
        self.required_vars = {
            'TELEGRAM_BOT_TOKEN': str,
            'CHAT_ID': int,
            'ADMIN_IDS': str,
            'MAINNET_PRIVATE_KEY': str,
            'WEB3_PROVIDER_URL': str
        }
        
        # Optional environment variables with defaults
        self.optional_vars = {
            'HF_API_KEY': (str, None),
            'DEFILLAMA_API_KEY': (str, None),
            'BASESCAN_API_KEY': (str, None),
            'LOG_LEVEL': (str, 'INFO')
        }
        
        self._config: Dict[str, Any] = {}
        self._load_and_validate()
        self._initialized = True
        
    def _load_and_validate(self):
        """Load and validate environment variables"""
        # In test mode, use mock values
        if self.is_test:
            self._config = {
                'TELEGRAM_BOT_TOKEN': 'test_token',
                'CHAT_ID': 123456789,
                'ADMIN_IDS': '123456789',
                'MAINNET_PRIVATE_KEY': '0x0000000000000000000000000000000000000000000000000000000000000001',
                'WEB3_PROVIDER_URL': 'http://localhost:8545',
                'LOG_LEVEL': 'INFO'
            }
            return

        # For non-test mode, check required variables
        missing_vars = []
        for var_name, var_type in self.required_vars.items():
            value = os.getenv(var_name)
            if value is None:
                missing_vars.append(var_name)
                continue
                
            try:
                if var_type == int:
                    self._config[var_name] = int(value)
                else:
                    self._config[var_name] = var_type(value)
            except ValueError as e:
                logger.error(f"Invalid value for {var_name}: {str(e)}")
                missing_vars.append(var_name)
                
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
            
        # Load optional variables with defaults
        for var_name, (var_type, default) in self.optional_vars.items():
            value = os.getenv(var_name)
            if value is None:
                self._config[var_name] = default
            else:
                try:
                    self._config[var_name] = var_type(value)
                except ValueError:
                    logger.warning(f"Invalid value for {var_name}, using default: {default}")
                    self._config[var_name] = default
                    
    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get environment variable value"""
        return self._config.get(key, default)
        
    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access to config values"""
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
        
# Global instance
env_config = EnvironmentConfig() 