"""Application configuration management"""

from pydantic import BaseSettings, Field, validator
from typing import Dict, Optional
from pathlib import Path

class CacheSettings(BaseSettings):
    """Cache configuration settings"""
    duration: int = Field(default=3600, description="Cache duration in seconds")
    refresh_threshold: float = Field(default=0.8, description="Cache refresh threshold")
    max_size: int = Field(default=1000, description="Maximum cache size")
    
    @validator('refresh_threshold')
    def validate_refresh_threshold(cls, v):
        if not 0 < v < 1:
            raise ValueError("refresh_threshold must be between 0 and 1")
        return v

class BlockchainSettings(BaseSettings):
    """Blockchain configuration settings"""
    rpc_url: str = Field(..., description="RPC endpoint URL")
    chain_id: int = Field(..., description="Chain ID")
    max_gas_price: int = Field(default=100, description="Maximum gas price in gwei")
    confirmation_blocks: int = Field(default=1, description="Number of confirmation blocks")

class DexSettings(BaseSettings):
    """DEX configuration settings"""
    router_address: str = Field(..., description="DEX router contract address")
    factory_address: str = Field(..., description="DEX factory contract address")
    base_tokens: Dict[str, str] = Field(default_factory=dict, description="Base token addresses")
    min_liquidity: float = Field(default=10000, description="Minimum liquidity threshold")

class MLSettings(BaseSettings):
    """Machine learning configuration settings"""
    model_path: Path = Field(..., description="Path to ML model")
    batch_size: int = Field(default=32, description="Batch size for inference")
    confidence_threshold: float = Field(default=0.8, description="Confidence threshold")
    
    @validator('confidence_threshold')
    def validate_confidence_threshold(cls, v):
        if not 0 < v < 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        return v

class Settings(BaseSettings):
    """Main application settings"""
    environment: str = Field(default="development", description="Environment (development/production)")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Component settings
    cache: CacheSettings = Field(default_factory=CacheSettings)
    blockchain: BlockchainSettings
    dex: DexSettings
    ml: MLSettings
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @classmethod
    def load_from_yaml(cls, yaml_path: str) -> "Settings":
        """Load settings from YAML file"""
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict) 