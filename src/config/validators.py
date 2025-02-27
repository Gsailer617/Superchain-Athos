"""
Configuration Validators

This module provides validators for ensuring configuration correctness
and detecting errors early at load time rather than during runtime.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
import jsonschema
from pathlib import Path

from .chain_config_template import ChainConfig

logger = logging.getLogger(__name__)

# JSON Schema for chain configuration validation
CHAIN_CONFIG_SCHEMA = {
    "type": "object",
    "required": ["name", "chain_id", "native_currency", "rpc_url"],
    "properties": {
        "name": {"type": "string"},
        "chain_id": {"type": "integer"},
        "native_currency": {"type": "string"},
        "rpc_url": {"type": "string"},
        "chain_type": {"type": "string"},
        "block_time": {"type": "number"},
        "is_enabled": {"type": "boolean"},
        "explorer_url": {"type": "string"},
        "tokens": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "required": ["address", "decimals", "symbol"],
                "properties": {
                    "address": {"type": "string"},
                    "decimals": {"type": "integer"},
                    "symbol": {"type": "string"},
                    "is_native": {"type": "boolean"}
                }
            }
        }
    }
}

def validate_chain_json(chain_data: Dict[str, Any]) -> List[str]:
    """
    Validate chain configuration against JSON schema
    
    Args:
        chain_data: Chain configuration data
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    try:
        jsonschema.validate(instance=chain_data, schema=CHAIN_CONFIG_SCHEMA)
    except jsonschema.exceptions.ValidationError as e:
        errors.append(f"Schema validation error: {e.message}")
    except Exception as e:
        errors.append(f"Validation error: {str(e)}")
        
    # Additional custom validations
    if 'chain_id' in chain_data:
        if chain_data['chain_id'] <= 0:
            errors.append(f"Chain ID must be positive, got {chain_data['chain_id']}")
            
    if 'tokens' in chain_data:
        for token_symbol, token_data in chain_data['tokens'].items():
            if 'address' in token_data:
                # Simple hex check - could be more sophisticated with checksumming
                if not token_data['address'].startswith('0x'):
                    errors.append(f"Token address must be hex string: {token_data['address']}")
                    
            if 'decimals' in token_data:
                if token_data['decimals'] < 0 or token_data['decimals'] > 24:
                    errors.append(f"Token decimals out of reasonable range (0-24): {token_data['decimals']}")
    
    return errors

def validate_chain_config(config: ChainConfig) -> List[str]:
    """
    Validate ChainConfig instance for consistency and correctness
    
    Args:
        config: Chain configuration to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Basic validations
    if config.chain_id <= 0:
        errors.append(f"Chain ID must be positive, got {config.chain_id}")
        
    if not config.rpc.http_urls:
        errors.append("At least one HTTP RPC URL is required")
        
    # L2 validations
    if config.is_l2():
        if not config.parent_chain_id:
            errors.append("L2 chains must specify parent_chain_id")
    
    # Gas model validations
    try:
        # Test get_gas_settings to ensure it doesn't raise exceptions
        config.get_gas_settings()
    except Exception as e:
        errors.append(f"Invalid gas settings: {str(e)}")
        
    return errors

def validate_config_file(file_path: str) -> List[str]:
    """
    Validate a JSON configuration file against schema
    
    Args:
        file_path: Path to JSON config file
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    try:
        path = Path(file_path)
        if not path.exists():
            return [f"Config file not found: {file_path}"]
            
        with open(path, 'r') as f:
            try:
                config_data = json.load(f)
            except json.JSONDecodeError as e:
                return [f"Invalid JSON in {file_path}: {str(e)}"]
                
        # Validate overall file structure
        if not isinstance(config_data, dict):
            return [f"Config file must contain a JSON object/dictionary"]
            
        # Validate each chain configuration
        for chain_name, chain_data in config_data.items():
            chain_errors = validate_chain_json(chain_data)
            for error in chain_errors:
                errors.append(f"{chain_name}: {error}")
                
    except Exception as e:
        errors.append(f"Error validating config file {file_path}: {str(e)}")
        
    return errors

def validate_all_chain_configs(configs: Dict[str, ChainConfig]) -> Dict[str, List[str]]:
    """
    Validate all chain configurations
    
    Args:
        configs: Dictionary of chain configurations
        
    Returns:
        Dictionary mapping chain names to validation errors
    """
    results = {}
    
    for chain_name, config in configs.items():
        errors = validate_chain_config(config)
        if errors:
            results[chain_name] = errors
            
    return results 