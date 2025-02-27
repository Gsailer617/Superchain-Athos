"""
Configuration Manager Utility - Wrapper for core ConfigurationManager

This module provides a simplified interface to the core configuration manager.
"""

from typing import Dict, Any, Optional, Union
from src.core.config_manager import ConfigurationManager
import os
from pathlib import Path

class ConfigManager:
    """Simplified wrapper for the core ConfigurationManager
    
    This class provides a more straightforward interface to the robust 
    core configuration management system.
    """
    
    _instance: Optional['ConfigManager'] = None
    
    @classmethod
    def get_instance(cls) -> 'ConfigManager':
        """Singleton pattern to get or create config manager instance"""
        if cls._instance is None:
            cls._instance = ConfigManager()
        return cls._instance
    
    def __init__(self):
        """Initialize the config manager with default settings"""
        config_dir = os.environ.get("CONFIG_DIR", str(Path(__file__).parent.parent.parent / "config"))
        schema_dir = os.environ.get("SCHEMA_DIR", str(Path(__file__).parent.parent.parent / "schemas"))
        
        self._manager = ConfigurationManager(
            config_dir=config_dir,
            schema_dir=schema_dir,
            env_prefix="FLASHING_"
        )
        self._manager.init()
    
    def get(self, component: str, key: Optional[str] = None, default: Any = None) -> Any:
        """Get configuration value"""
        return self._manager.get_config(component, key, default)
    
    def set(self, component: str, key: str, value: Any, persist: bool = False) -> None:
        """Set configuration value"""
        self._manager.set_config(component, key, value, persist)
    
    def watch(self, component: str, callback: callable) -> None:
        """Watch for configuration changes"""
        self._manager.watch_config(component, callback)
    
    def validate(self, component: str) -> bool:
        """Validate configuration"""
        errors = self._manager.validate_config(component)
        return len(errors) == 0 