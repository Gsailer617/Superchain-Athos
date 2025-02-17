import os
import json
from typing import Dict, Any, Optional, Mapping
from pathlib import Path
from dataclasses import asdict

from .bridge_config import (
    BridgeConfig,
    ChainConfig,
    TokenConfig,
    BridgeGlobalConfig,
    LayerZeroConfig,
    DeBridgeConfig,
    SuperbridgeConfig
)

class ConfigLoader:
    """Configuration loader for bridge settings"""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = config_dir if config_dir is not None else os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'config'
        )
        self._ensure_config_dir()
    
    def _ensure_config_dir(self) -> None:
        """Ensure config directory exists"""
        Path(self.config_dir).mkdir(parents=True, exist_ok=True)
    
    def load_config(self, env: str = 'mainnet') -> BridgeConfig:
        """Load bridge configuration for specified environment"""
        config_path = os.path.join(self.config_dir, f'bridge_config_{env}.json')
        
        # Load environment variables for sensitive data
        rpc_urls: Dict[str, str] = {
            'ethereum': os.getenv('ETH_RPC_URL', ''),
            'base': os.getenv('BASE_RPC_URL', ''),
            'polygon': os.getenv('POLYGON_RPC_URL', '')
        }
        
        # Create default config if file doesn't exist
        if not os.path.exists(config_path):
            config = self._create_default_config(rpc_urls)
            self.save_config(config, env)
            return config
        
        # Load existing config
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Update RPC URLs from environment
        for chain, url in rpc_urls.items():
            if url and chain in config_dict['chains']:
                config_dict['chains'][chain]['rpc_url'] = url
        
        # Convert dict to dataclass instances
        return self._dict_to_config(config_dict)
    
    def save_config(self, config: BridgeConfig, env: str = 'mainnet') -> None:
        """Save bridge configuration to file"""
        config_path = os.path.join(self.config_dir, f'bridge_config_{env}.json')
        
        # Convert dataclass to dict
        config_dict = asdict(config)
        
        # Remove sensitive data before saving
        for chain_config in config_dict['chains'].values():
            chain_config['rpc_url'] = ''  # Don't save RPC URLs
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def _create_default_config(self, rpc_urls: Dict[str, str]) -> BridgeConfig:
        """Create default bridge configuration"""
        gas_settings = self._get_gas_settings()
        bridge_contracts = self._get_bridge_contracts()
        
        return BridgeConfig(
            global_config=BridgeGlobalConfig(),
            chains={
                chain: ChainConfig(
                    chain_id=self._get_chain_id(chain),
                    rpc_url=url or '',
                    confirmation_blocks=self._get_confirmation_blocks(chain)
                )
                for chain, url in rpc_urls.items()
            },
            tokens=self._get_default_tokens(),
            layerzero_config=LayerZeroConfig(),
            debridge_config=DeBridgeConfig(),
            superbridge_config=SuperbridgeConfig(),
            mode_config={
                'l1_bridge': bridge_contracts['mode']['l1_bridge'],
                'l2_bridge': bridge_contracts['mode']['l2_bridge'],
                'message_service': bridge_contracts['mode']['message_service'],
                'max_gas_price': gas_settings['mode']['max_gas_price'],
                'priority_fee': gas_settings['mode']['priority_fee']
            },
            sonic_config={
                'bridge_router': bridge_contracts['sonic']['bridge_router'],
                'token_bridge': bridge_contracts['sonic']['token_bridge'],
                'liquidity_pool': bridge_contracts['sonic']['liquidity_pool'],
                'max_gas_price': gas_settings['sonic']['max_gas_price'],
                'priority_fee': gas_settings['sonic']['priority_fee']
            }
        )
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> BridgeConfig:
        """Convert configuration dictionary to dataclass instances"""
        return BridgeConfig(
            global_config=BridgeGlobalConfig(**config_dict['global_config']),
            chains={
                chain: ChainConfig(**chain_config)
                for chain, chain_config in config_dict['chains'].items()
            },
            tokens={
                token: TokenConfig(**token_config)
                for token, token_config in config_dict['tokens'].items()
            },
            layerzero_config=LayerZeroConfig(**config_dict['layerzero_config']),
            debridge_config=DeBridgeConfig(**config_dict['debridge_config']),
            superbridge_config=SuperbridgeConfig(**config_dict['superbridge_config']),
            mode_config=config_dict['mode_config'],
            sonic_config=config_dict['sonic_config']
        )
    
    @staticmethod
    def _get_chain_id(chain: str) -> int:
        """Get chain ID for supported chains"""
        chain_ids = {
            'ethereum': 1,
            'base': 8453,
            'polygon': 137
        }
        return chain_ids.get(chain, 0)
    
    @staticmethod
    def _get_confirmation_blocks(chain: str) -> int:
        """Get recommended confirmation blocks for each chain"""
        confirmations = {
            'ethereum': 12,
            'base': 5,
            'polygon': 128
        }
        return confirmations.get(chain, 3)
    
    @staticmethod
    def _get_default_tokens() -> Dict[str, TokenConfig]:
        """Get default token configurations"""
        return {
            'USDC': TokenConfig(
                symbol='USDC',
                decimals=6,
                addresses={
                    'ethereum': '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
                    'base': '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913',
                    'polygon': '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174'
                },
                min_amount=100,
                max_amount=1000000
            ),
            'USDT': TokenConfig(
                symbol='USDT',
                decimals=6,
                addresses={
                    'ethereum': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
                    'base': '0x4A3A6Dd60A34bB2Aba60D73B4C88315E9CeB6A3D',
                    'polygon': '0xc2132D05D31c914a87C6611C10748AEb04B58e8F'
                },
                min_amount=100,
                max_amount=1000000
            ),
            'ETH': TokenConfig(
                symbol='ETH',
                decimals=18,
                addresses={
                    'ethereum': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',  # WETH
                    'base': '0x4200000000000000000000000000000000000006',      # WETH
                    'polygon': '0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619'    # WETH
                },
                min_amount=0.1,
                max_amount=1000
            )
        }
    
    def _get_chain_rpc_urls(self) -> Dict[str, str]:
        """Get RPC URLs from environment variables"""
        return {
            'ethereum': os.getenv('ETHEREUM_RPC_URL', ''),
            'base': os.getenv('BASE_RPC_URL', ''),
            'polygon': os.getenv('POLYGON_RPC_URL', ''),
            'arbitrum': os.getenv('ARBITRUM_RPC_URL', ''),
            'optimism': os.getenv('OPTIMISM_RPC_URL', ''),
            'mode': os.getenv('MODE_RPC_URL', ''),
            'sonic': os.getenv('SONIC_RPC_URL', '')
        }
    
    def _get_chain_ws_urls(self) -> Dict[str, Optional[str]]:
        """Get WebSocket URLs from environment variables"""
        return {
            'ethereum': os.getenv('ETHEREUM_WS_URL'),
            'base': os.getenv('BASE_WS_URL'),
            'polygon': os.getenv('POLYGON_WS_URL'),
            'arbitrum': os.getenv('ARBITRUM_WS_URL'),
            'optimism': os.getenv('OPTIMISM_WS_URL'),
            'mode': os.getenv('MODE_WS_URL'),
            'sonic': os.getenv('SONIC_WS_URL')
        }
    
    def _get_chain_api_keys(self) -> Dict[str, Optional[str]]:
        """Get API keys from environment variables"""
        return {
            'ethereum': os.getenv('ETHERSCAN_API_KEY'),
            'base': os.getenv('BASESCAN_API_KEY'),
            'polygon': os.getenv('POLYGONSCAN_API_KEY'),
            'arbitrum': os.getenv('ARBISCAN_API_KEY'),
            'optimism': os.getenv('OPTIMISM_API_KEY'),
            'mode': os.getenv('MODE_API_KEY'),
            'sonic': os.getenv('SONIC_API_KEY')
        }
    
    def _get_bridge_contracts(self) -> Dict[str, Dict[str, str]]:
        """Get bridge contract addresses from environment variables"""
        return {
            'mode': {
                'l1_bridge': os.getenv('MODE_L1_BRIDGE', ''),
                'l2_bridge': os.getenv('MODE_L2_BRIDGE', ''),
                'message_service': os.getenv('MODE_MESSAGE_SERVICE', '')
            },
            'sonic': {
                'bridge_router': os.getenv('SONIC_BRIDGE_ROUTER', ''),
                'token_bridge': os.getenv('SONIC_TOKEN_BRIDGE', ''),
                'liquidity_pool': os.getenv('SONIC_LIQUIDITY_POOL', '')
            }
        }
    
    def _get_gas_settings(self) -> Dict[str, Dict[str, int]]:
        """Get gas settings from environment variables"""
        return {
            'mode': {
                'max_gas_price': int(os.getenv('MODE_MAX_GAS_PRICE', '500')),
                'priority_fee': int(os.getenv('MODE_PRIORITY_FEE', '2'))
            },
            'sonic': {
                'max_gas_price': int(os.getenv('SONIC_MAX_GAS_PRICE', '1000')),
                'priority_fee': int(os.getenv('SONIC_PRIORITY_FEE', '1'))
            }
        } 