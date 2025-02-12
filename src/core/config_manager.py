"""
Configuration Management Module

This module provides centralized configuration management:
- Configuration validation
- Dynamic updates
- Environment variable integration
- Secure secrets handling
- Configuration versioning
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List, Set, Union, Callable
from dataclasses import dataclass, field
import structlog
from datetime import datetime
from pathlib import Path
import asyncio
from prometheus_client import Counter, Gauge
from functools import wraps
import jsonschema
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading

logger = structlog.get_logger(__name__)

@dataclass
class ConfigMetadata:
    """Configuration metadata"""
    version: str
    last_updated: datetime
    updated_by: str
    description: str = ""
    tags: Set[str] = field(default_factory=set)

class ConfigValidationError(Exception):
    """Configuration validation error"""
    pass

class ConfigurationManager:
    """Centralized configuration management"""
    
    def __init__(self, 
                 config_dir: str,
                 schema_dir: Optional[str] = None,
                 env_prefix: str = "APP_"):
        self.config_dir = Path(config_dir)
        self.schema_dir = Path(schema_dir) if schema_dir else None
        self.env_prefix = env_prefix
        self._config: Dict[str, Any] = {}
        self._schemas: Dict[str, Dict] = {}
        self._metadata: Dict[str, ConfigMetadata] = {}
        self._watchers: Dict[str, List[Callable]] = {}
        self._observer: Optional[Observer] = None
        self._setup_metrics()

    def _setup_metrics(self):
        """Initialize Prometheus metrics"""
        self._config_updates = Counter(
            'config_updates_total',
            'Number of configuration updates',
            ['component']
        )
        self._config_errors = Counter(
            'config_validation_errors_total',
            'Number of configuration validation errors',
            ['component']
        )
        self._config_size = Gauge(
            'config_size_bytes',
            'Size of configuration in bytes',
            ['component']
        )

    def init(self):
        """Initialize configuration manager"""
        # Load schemas
        if self.schema_dir and self.schema_dir.exists():
            for schema_file in self.schema_dir.glob("*.json"):
                with open(schema_file) as f:
                    self._schemas[schema_file.stem] = json.load(f)

        # Load initial configurations
        self._load_all_configs()

        # Start file watcher
        self._start_file_watcher()

    def _start_file_watcher(self):
        """Start watching configuration files for changes"""
        class ConfigFileHandler(FileSystemEventHandler):
            def __init__(self, manager):
                self.manager = manager

            def on_modified(self, event):
                if not event.is_directory and event.src_path.endswith(('.json', '.yaml', '.yml')):
                    component = Path(event.src_path).stem
                    self.manager._load_component_config(component)
                    self.manager._notify_watchers(component)

        self._observer = Observer()
        self._observer.schedule(
            ConfigFileHandler(self),
            str(self.config_dir),
            recursive=False
        )
        self._observer.start()

    def _load_all_configs(self):
        """Load all configuration files"""
        for config_file in self.config_dir.glob("*.*"):
            if config_file.suffix in ('.json', '.yaml', '.yml'):
                self._load_component_config(config_file.stem)

    def _load_component_config(self, component: str):
        """Load configuration for a specific component"""
        try:
            # Find config file
            config_path = None
            for ext in ('.json', '.yaml', '.yml'):
                path = self.config_dir / f"{component}{ext}"
                if path.exists():
                    config_path = path
                    break

            if not config_path:
                logger.warning(f"No configuration file found for {component}")
                return

            # Load config
            with open(config_path) as f:
                if config_path.suffix == '.json':
                    config = json.load(f)
                else:
                    config = yaml.safe_load(f)

            # Validate against schema
            if component in self._schemas:
                try:
                    jsonschema.validate(config, self._schemas[component])
                except jsonschema.exceptions.ValidationError as e:
                    raise ConfigValidationError(f"Schema validation failed: {str(e)}")

            # Override with environment variables
            config = self._apply_env_overrides(component, config)

            # Update config and metadata
            self._config[component] = config
            self._metadata[component] = ConfigMetadata(
                version=str(config.get('version', '1.0.0')),
                last_updated=datetime.now(),
                updated_by='system',
                description=config.get('description', ''),
                tags=set(config.get('tags', []))
            )

            # Update metrics
            self._config_updates.labels(component=component).inc()
            self._config_size.labels(component=component).set(
                len(json.dumps(config))
            )

        except Exception as e:
            logger.error(f"Error loading config for {component}: {str(e)}")
            self._config_errors.labels(component=component).inc()
            raise

    def _apply_env_overrides(self,
                            component: str,
                            config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides"""
        def _override_value(key: str, value: Any) -> Any:
            env_key = f"{self.env_prefix}{component.upper()}_{key.upper()}"
            env_value = os.environ.get(env_key)
            
            if env_value is not None:
                # Convert environment value to appropriate type
                if isinstance(value, bool):
                    return env_value.lower() in ('true', '1', 'yes')
                elif isinstance(value, int):
                    return int(env_value)
                elif isinstance(value, float):
                    return float(env_value)
                elif isinstance(value, list):
                    return env_value.split(',')
                else:
                    return env_value
            return value

        def _process_dict(d: Dict[str, Any]) -> Dict[str, Any]:
            result = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    result[k] = _process_dict(v)
                else:
                    result[k] = _override_value(k, v)
            return result

        return _process_dict(config)

    def get_config(self, 
                  component: str,
                  key: Optional[str] = None,
                  default: Any = None) -> Any:
        """Get configuration value"""
        if component not in self._config:
            return default

        if key is None:
            return self._config[component]

        # Support nested keys (e.g., "database.host")
        value = self._config[component]
        for k in key.split('.'):
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set_config(self,
                   component: str,
                   key: str,
                   value: Any,
                   persist: bool = False):
        """Set configuration value"""
        if component not in self._config:
            self._config[component] = {}

        # Handle nested keys
        keys = key.split('.')
        current = self._config[component]
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

        # Validate if schema exists
        if component in self._schemas:
            try:
                jsonschema.validate(self._config[component], self._schemas[component])
            except jsonschema.exceptions.ValidationError as e:
                # Rollback change
                self._load_component_config(component)
                raise ConfigValidationError(f"Schema validation failed: {str(e)}")

        # Persist to file if requested
        if persist:
            self._save_component_config(component)

        # Update metadata
        self._metadata[component].last_updated = datetime.now()
        self._config_updates.labels(component=component).inc()

        # Notify watchers
        self._notify_watchers(component)

    def _save_component_config(self, component: str):
        """Save component configuration to file"""
        try:
            config_path = self.config_dir / f"{component}.json"
            with open(config_path, 'w') as f:
                json.dump(
                    self._config[component],
                    f,
                    indent=2,
                    sort_keys=True
                )
        except Exception as e:
            logger.error(f"Error saving config for {component}: {str(e)}")
            raise

    def watch_config(self,
                    component: str,
                    callback: Callable[[str, Dict[str, Any]], None]):
        """Register a configuration change watcher"""
        if component not in self._watchers:
            self._watchers[component] = []
        self._watchers[component].append(callback)

    def _notify_watchers(self, component: str):
        """Notify watchers of configuration changes"""
        if component in self._watchers:
            config = self._config.get(component, {})
            for callback in self._watchers[component]:
                try:
                    callback(component, config)
                except Exception as e:
                    logger.error(
                        f"Error in config watcher callback: {str(e)}",
                        component=component
                    )

    def get_metadata(self, component: str) -> Optional[ConfigMetadata]:
        """Get configuration metadata"""
        return self._metadata.get(component)

    def validate_config(self, component: str) -> List[str]:
        """Validate component configuration"""
        errors = []
        
        if component not in self._config:
            errors.append(f"No configuration found for {component}")
            return errors

        # Schema validation
        if component in self._schemas:
            try:
                jsonschema.validate(self._config[component], self._schemas[component])
            except jsonschema.exceptions.ValidationError as e:
                errors.append(f"Schema validation error: {str(e)}")

        # Custom validation logic can be added here
        return errors

    def close(self):
        """Clean up resources"""
        if self._observer:
            self._observer.stop()
            self._observer.join()

# Global configuration manager instance
config_manager: Optional[ConfigurationManager] = None 