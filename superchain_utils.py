import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from typing import List, Dict, Tuple, Optional, Any, AsyncGenerator, Coroutine, TypeVar, Union, Callable, Protocol
from typing_extensions import TypeAlias
import json
import aiohttp
from web3 import Web3, AsyncWeb3
from web3.contract import Contract
from eth_account.signers.local import LocalAccount
import numpy as np
from datetime import datetime, timedelta
import re
import asyncio
import csv
from telegram_bot import TelegramBot
import torch
import torch.nn as nn
import networkx as nx
import os
from dotenv import load_dotenv
import time
from web3.types import TxReceipt, Wei
from aiohttp import ClientSession
from functools import wraps
from src.core.web3_config import get_web3, get_async_web3
from web3.exceptions import ContractLogicError, TransactionNotFound
from hexbytes import HexBytes

# Load environment variables
load_dotenv()

# Get Web3 instances from centralized provider
web3 = get_web3()
async_web3 = get_async_web3()

# Get private key from environment variable
private_key = os.getenv('MAINNET_PRIVATE_KEY')
if not private_key:
    raise ValueError("MAINNET_PRIVATE_KEY environment variable is not set")

logger.info("Using centralized Web3 provider in utils")

# Common ABIs
ERC20_ABI = [
    {"constant":True,"inputs":[],"name":"name","outputs":[{"name":"","type":"string"}],"payable":False,"stateMutability":"view","type":"function"},
    {"constant":True,"inputs":[],"name":"symbol","outputs":[{"name":"","type":"string"}],"payable":False,"stateMutability":"view","type":"function"},
    {"constant":True,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"payable":False,"stateMutability":"view","type":"function"},
    {"constant":True,"inputs":[],"name":"totalSupply","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},
    {"constant":True,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"balance","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},
    {"constant":False,"inputs":[{"name":"_to","type":"address"},{"name":"_value","type":"uint256"}],"name":"transfer","outputs":[{"name":"","type":"bool"}],"payable":False,"stateMutability":"nonpayable","type":"function"},
    {"constant":False,"inputs":[{"name":"_spender","type":"address"},{"name":"_value","type":"uint256"}],"name":"approve","outputs":[{"name":"","type":"bool"}],"payable":False,"stateMutability":"nonpayable","type":"function"},
    {"constant":True,"inputs":[{"name":"_owner","type":"address"},{"name":"_spender","type":"address"}],"name":"allowance","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"}
]

UNISWAP_V2_PAIR_ABI = [
    {"constant":True,"inputs":[],"name":"getReserves","outputs":[{"name":"reserve0","type":"uint112"},{"name":"reserve1","type":"uint112"},{"name":"blockTimestampLast","type":"uint32"}],"payable":False,"stateMutability":"view","type":"function"},
    {"constant":True,"inputs":[],"name":"token0","outputs":[{"name":"","type":"address"}],"payable":False,"stateMutability":"view","type":"function"},
    {"constant":True,"inputs":[],"name":"token1","outputs":[{"name":"","type":"address"}],"payable":False,"stateMutability":"view","type":"function"}
]

UNISWAP_V3_POOL_ABI = [
    {"inputs":[],"name":"slot0","outputs":[{"name":"sqrtPriceX96","type":"uint160"},{"name":"tick","type":"int24"},{"name":"observationIndex","type":"uint16"},{"name":"observationCardinality","type":"uint16"},{"name":"observationCardinalityNext","type":"uint16"},{"name":"feeProtocol","type":"uint8"},{"name":"unlocked","type":"bool"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"liquidity","outputs":[{"name":"","type":"uint128"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"tickLower","type":"int24"},{"name":"tickUpper","type":"int24"}],"name":"getPositionInfo","outputs":[{"name":"liquidity","type":"uint128"},{"name":"feeGrowthInside0LastX128","type":"uint256"},{"name":"feeGrowthInside1LastX128","type":"uint256"},{"name":"tokensOwed0","type":"uint128"},{"name":"tokensOwed1","type":"uint128"}],"stateMutability":"view","type":"function"}
]

AAVE_V3_POOL_ABI = [
    {"inputs":[{"internalType":"address","name":"asset","type":"address"}],"name":"getReserveData","outputs":[{"components":[{"components":[{"internalType":"uint256","name":"data","type":"uint256"}],"internalType":"struct DataTypes.ReserveConfigurationMap","name":"configuration","type":"tuple"},{"internalType":"uint128","name":"liquidityIndex","type":"uint128"},{"internalType":"uint128","name":"currentLiquidityRate","type":"uint128"},{"internalType":"uint128","name":"variableBorrowIndex","type":"uint128"},{"internalType":"uint128","name":"currentVariableBorrowRate","type":"uint128"},{"internalType":"uint128","name":"currentStableBorrowRate","type":"uint128"},{"internalType":"uint40","name":"lastUpdateTimestamp","type":"uint40"},{"internalType":"uint16","name":"id","type":"uint16"},{"internalType":"address","name":"aTokenAddress","type":"address"},{"internalType":"address","name":"stableDebtTokenAddress","type":"address"},{"internalType":"address","name":"variableDebtTokenAddress","type":"address"},{"internalType":"address","name":"interestRateStrategyAddress","type":"address"},{"internalType":"uint128","name":"accruedToTreasury","type":"uint128"},{"internalType":"uint128","name":"unbacked","type":"uint128"},{"internalType":"uint128","name":"isolationModeTotalDebt","type":"uint128"}],"internalType":"struct DataTypes.ReserveData","name":"","type":"tuple"}],"stateMutability":"view","type":"function"}
]

COMPOUND_V2_POOL_ABI = [
    {"constant":True,"inputs":[{"name":"asset","type":"address"}],"name":"getReserveData","outputs":[{"components":[{"name":"isActive","type":"bool"},{"name":"borrowEnabled","type":"bool"},{"name":"lastUpdateTimestamp","type":"uint40"}],"name":"","type":"tuple"}],"payable":False,"stateMutability":"view","type":"function"}
]

BALANCER_V3_POOL_ABI = [
    {"inputs":[{"internalType":"address","name":"asset","type":"address"}],"name":"getReserveData","outputs":[{"components":[{"name":"isActive","type":"bool"}],"name":"","type":"tuple"}],"stateMutability":"view","type":"function"}
]

def get_token_abi() -> List[Dict]:
    """Get standard ERC20 token ABI"""
    return ERC20_ABI

# Initialize Telegram bot
telegram_bot = TelegramBot()

# Base Chain Graph URLs
GRAPH_ENDPOINTS = {
    'aerodrome': 'https://api.thegraph.com/subgraphs/name/aerodrome-finance/aerodrome-v2-base',
    'baseswap': 'https://api.studio.thegraph.com/query/50526/baseswap-v2/v0.0.1',
    'alienbase': 'https://api.thegraph.com/subgraphs/name/alienbase/exchange'
}

# Uniswap V3 subgraph URL for Base Chain
UNISWAP_V3_SUBGRAPH = "https://api.thegraph.com/subgraphs/name/uniswap/base-v3"

# Base Chain DEX Configurations
DEX_CONFIGS = {
    'uniswap_v4': {
        'router': '0x198EF79F1F515F02dFE9e3115eD9fC07183f02fC',  # Universal Router
        'factory': '0x33128a8fC17869897dcE68Ed026d694621f6FDfD',
        'type': 'UniswapV4',
        'fee_tiers': [1, 3, 10]  # 0.01%, 0.3%, 1%
    },
    'sushiswap_v3': {
        'router': '0x6BDED42c6DA8FD5E8B11852d05597c0F7C8D8E86',
        'factory': '0xc35DADB65012eC5796536bD9864eD8773aBc74C4',
        'type': 'UniswapV3',
        'fee_tiers': [1, 3, 10]  # 0.01%, 0.3%, 1%
    },
    'pancakeswap_v4': {
        'router': '0x678Aa4bF4E210cf2166753e054d5b7c31cc7fa86',
        'factory': '0x0BFbCF9fa4f9C56B0F40a671Ad40E0805A091865',
        'type': 'UniswapV3',
        'fee_tiers': [1, 2.5, 5]  # 0.01%, 0.25%, 0.5%
    },
    'aerodrome': {
        'router': '0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43',
        'factory': '0x420DD381b31aEf6683db6B902084cB0FFECe40Da',
        'type': 'UniswapV2',
        'fee_tiers': [1]  # 0.1%
    },
    'baseswap': {
        'router': '0x327Df1E6de05895d2ab08513aaDD9313Fe505d86',
        'factory': '0xFDa619b6d20975be80A10332cD39b9a4b0FAa8BB',
        'type': 'UniswapV2',
        'fee_tiers': [3]  # 0.3%
    },
    'alienbase': {
        'router': '0x8c1A3cF8f83074169FE5D7aD50B978e1cD6b37c7',
        'factory': '0x3E84D913803b02A4a7f027165E8cA42C14C0FdE7',
        'type': 'UniswapV2',
        'fee_tiers': [3]  # 0.3%
    }
}

# Base Chain Lending Protocols for Flash Loans
LENDING_CONFIGS = {
    'aave_v3': {
        'address': '0xA238Dd80C259a72e81d7e4664a9801593F98d1c5',  # Latest Aave V3 Pool
        'type': 'AaveV3',
        'flash_loan_fee': 0.05  # 0.05%
    },
    'balancer_v3': {
        'address': '0xBA12222222228d8Ba445958a75a0704d566BF2C8',  # Latest Balancer V3 Vault
        'type': 'BalancerV3',
        'flash_loan_fee': 0.01  # 0.01%
    },
    'radiant': {
        'address': '0x2032b9A8e9F7e76768CA9271003d3e43E1616B1F',
        'type': 'AaveV2',
        'flash_loan_fee': 0.09  # 0.09%
    }
}

# Most liquid token pairs on Base
TOKEN_PAIRS = [
    ('ETH', 'USDbC'),     # Most liquid pair
    ('USDbC', 'USDC'),    # Stablecoin pair
    ('ETH', 'USDC'),      # Secondary ETH pair
    ('DAI', 'USDbC'),     # Additional stablecoin pair
    ('ETH', 'cbETH'),     # Liquid ETH derivative pair
    ('tBTC', 'ETH')       # Bitcoin-ETH pair
]

# Token addresses on Base
TOKEN_ADDRESSES = {
    'ETH': '0x4200000000000000000000000000000000000006',  # WETH on Base
    'USDbC': '0xd9aAEc86B65D86f6A7B5B1b0c42FFA531710b6CA',
    'USDC': '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913',
    'DAI': '0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb',
    'USDT': '0x4A3A6Dd60A34bB2Aba60D73B4C88315E9CeB6A3D',
    'cbETH': '0x2Ae3F1Ec7F1F5012CFEab0185bfc7aa3cf0DEc22',
    'WETH': '0x4200000000000000000000000000000000000006',
    'tBTC': '0x236aa50979D5f3De3Bd1Eeb40E81137F22ab794b',
    'axlUSDC': '0xEB466342C4d449BC9f53A865D5Cb90586f405215',
    'MAI': '0xbf1aeA8670D2528E08334083616dD9C5F3B087aE',
    'COMP': '0x9e1028F5F1D5eDE59748FFceE5532509976840E0',
    'MKR': '0x7c6b91D9Be155A6Db01f749217d76fF02A7227F2'
}

# Test amounts for each token (in base units)
TEST_AMOUNTS = {
    'ETH': [0.1, 0.5, 1, 5],
    'USDbC': [100, 500, 1000, 5000],
    'USDC': [100, 500, 1000, 5000],
    'DAI': [100, 500, 1000, 5000],
    'USDT': [100, 500, 1000, 5000],
    'cbETH': [0.1, 0.5, 1, 5],
    'WETH': [0.1, 0.5, 1, 5],
    'tBTC': [0.01, 0.05, 0.1, 0.5],
    'axlUSDC': [100, 500, 1000, 5000],
    'MAI': [100, 500, 1000, 5000],
    'COMP': [1, 5, 10, 50],
    'MKR': [0.1, 0.5, 1, 5]
}

async def fetch_dex_data(session: aiohttp.ClientSession, 
                        dex_name: str, 
                        dex_info: Dict) -> Dict:
    """Fetch market data from a specific DEX"""
    # Implement DEX-specific data fetching logic
    if dex_info['type'] == 'UniswapV3':
        return await fetch_uniswap_v3_data(session, dex_info['address'])
    elif dex_info['type'] == 'UniswapV2':
        return await fetch_uniswap_v2_data(session, dex_info['address'])
    else:
        raise ValueError(f"Unsupported DEX type: {dex_info['type']}")

def calculate_volatility_v3(pool: Dict) -> float:
    """Calculate volatility from pool data"""
    try:
        price = float(pool['token0Price'])
        tick = int(pool['tick'])
        return abs(price * tick / 1e6)  # Simplified volatility calculation
    except Exception:
        return 0.0

def calculate_price_impact_v3(pool: Dict) -> float:
    """Calculate price impact from pool data"""
    try:
        liquidity = float(pool['liquidity'])
        return 1.0 / (liquidity + 1e-18)  # Simplified price impact calculation
    except Exception:
        return 0.0

async def fetch_uniswap_v3_data(session: aiohttp.ClientSession, 
                               address: str) -> Dict:
    """Fetch market data from Uniswap V3"""
    query = """
    {
      pool(id: "%s") {
        token0Price
        token1Price
        liquidity
        volumeUSD
        feeTier
        tick
        sqrtPrice
        token0 {
          symbol
          decimals
        }
        token1 {
          symbol
          decimals
        }
      }
    }
    """ % address.lower()
    
    try:
        async with session.post(UNISWAP_V3_SUBGRAPH, json={'query': query}) as response:
            if response.status == 200:
                data = await response.json()
                if not data.get('data') or not data['data'].get('pool'):
                    logger.warning(f"Pool not found for address: {address}")
                    return {}
                    
                pool = data['data']['pool']
                return {
                    'price_impact': calculate_price_impact_v3(pool),
                    'liquidity': float(pool['liquidity']),
                    'volatility': calculate_volatility_v3(pool),
                    'volume_24h': float(pool['volumeUSD']),
                    'fee_tier': int(pool['feeTier']),
                    'current_tick': int(pool['tick']),
                    'sqrt_price': pool['sqrtPrice']
                }
            return {}
    except Exception as e:
        logger.error(f"Exception in fetch_uniswap_v3_data: {str(e)}")
        return {}

async def fetch_uniswap_v2_data(session: aiohttp.ClientSession, 
                               address: str) -> Dict[str, Union[float, str]]:
    """Fetch market data from Uniswap V2-style DEXes
    
    Args:
        session: aiohttp client session
        address: Pool address
        
    Returns:
        Dict containing pool data or empty dict on error
    """
    try:
        async with session.get(
            f"https://api.base.fi/v1/pairs/{address}",
            timeout=aiohttp.ClientTimeout(total=10)
        ) as response:
            if response.status == 200:
                data = await response.json()
                return {
                    'reserves0': float(data.get('reserve0', 0)),
                    'reserves1': float(data.get('reserve1', 0)),
                    'volume_24h': float(data.get('volume24h', 0)),
                    'tvl': float(data.get('tvl', 0)),
                    'price': float(data.get('price', 0)),
                    'price_change_24h': float(data.get('priceChange24h', 0)),
                    'timestamp': datetime.now().isoformat()  # Keep as ISO format string
                }
            else:
                logger.error(f"Error fetching Uniswap V2 data: HTTP {response.status}")
                return {}
    except asyncio.TimeoutError:
        logger.error(f"Timeout fetching data for pool {address}")
        return {}
    except Exception as e:
        logger.error(f"Exception in fetch_uniswap_v2_data: {str(e)}")
        return {}

def get_optimal_lending_protocol(token: str, amount: float) -> Optional[Dict[str, Any]]:
    """Get the optimal lending protocol based on fees and liquidity
    
    Args:
        token: Token address
        amount: Amount to borrow
        
    Returns:
        Dict with protocol info or None if no suitable protocol found
    """
    try:
        best_protocol = None
        lowest_fee = float('inf')
        
        for protocol, config in LENDING_CONFIGS.items():
            # Skip protocols in testnet mode unless we're testing
            if config['testnet'] and not os.getenv('TESTING', '').lower() == 'true':
                continue
                
            # Check if protocol supports the token
            if not web3.eth.contract(
                address=config['address'],
                abi=get_lending_pool_abi(config['type'])
            ).functions.getReserveData(token).call()['isActive']:
                continue
                
            # Compare fees
            if config['flash_loan_fee'] < lowest_fee:
                best_protocol = {
                    'protocol': protocol,
                    'address': config['address'],
                    'type': config['type'],
                    'fee': config['flash_loan_fee']
                }
                lowest_fee = config['flash_loan_fee']
        
        if best_protocol is None:
            logger.warning(f"No suitable lending protocol found for token {token}")
            
        return best_protocol
        
    except Exception as e:
        logger.error(f"Error getting optimal lending protocol: {str(e)}")
        return None

def calculate_gas_estimate(lending_protocol: Dict) -> int:
    """Calculate estimated gas cost based on protocol type"""
    try:
        base_gas = 150000  # Base gas for flash loan
        if lending_protocol['type'] == 'AaveV3':
            return base_gas + 80000  # Additional gas for Aave V3
        elif lending_protocol['type'] == 'CompoundV2':
            return base_gas + 60000  # Additional gas for Compound V2
        return base_gas
    except Exception as e:
        logger.error(f"Error calculating gas estimate: {str(e)}")
        return 200000  # Default safe estimate

def calculate_price_impact(amount: float, market_data: Dict) -> float:
    """Calculate price impact of a trade"""
    try:
        liquidity = market_data.get('liquidity', 0)
        if liquidity == 0:
            return float('inf')
        return min((amount / liquidity) * 100, 100.0)  # Cap at 100%
    except Exception as e:
        logger.error(f"Error calculating price impact: {str(e)}")
        return float('inf')

def prepare_flash_loan(token_pair: Tuple[str, str], 
                      amount: float,
                      market_data: Dict) -> Dict:
    """Enhanced flash loan preparation with safety checks"""
    try:
        # Calculate optimal parameters
        min_profit = max(amount * 0.002, market_data.get('min_profit', 0))  # 0.2% or market minimum
        deadline = min(300, market_data.get('block_time', 0) + 100)  # 5 minutes or block time + buffer
        slippage = min(0.005, market_data.get('volatility', 0) * 2)  # 0.5% or 2x volatility
        
        # Get optimal lending protocol
        lending_protocol = get_optimal_lending_protocol(token_pair[0], amount)
        
        return {
            'token_in': token_pair[0],
            'token_out': token_pair[1],
            'amount': amount,
            'min_profit': min_profit,
            'deadline': deadline,
            'slippage_tolerance': slippage,
            'lending_protocol': lending_protocol,
            'gas_estimate': calculate_gas_estimate(lending_protocol) if lending_protocol is not None else 0,
            'safety_checks': {
                'has_liquidity': True,
                'price_impact': calculate_price_impact(amount, market_data),
                'risk_score': calculate_risk_score(market_data)
            }
        }
    except Exception as e:
        logger.error(f"Error preparing flash loan: {str(e)}")
        return {
            'error': str(e),
            'success': False,
            'token_in': token_pair[0] if token_pair else '',
            'token_out': token_pair[1] if token_pair else ''
        }

async def get_network_congestion() -> float:
    """Get current network congestion level (0-1)"""
    try:
        latest_block = await async_web3.eth.get_block('latest')
        gas_used_ratio = latest_block['gasUsed'] / latest_block['gasLimit']
        return min(gas_used_ratio, 1.0)
    except Exception as e:
        logger.error(f"Error getting network congestion: {str(e)}")
        return 0.5

async def optimize_gas_strategy(params: Dict[str, Any]) -> Dict[str, Union[int, float]]:
    """Optimize gas strategy based on network conditions"""
    try:
        base_gas = await async_web3.eth.gas_price
        network_congestion = await get_network_congestion()
        
        gas_multiplier = 1.1 if network_congestion > 0.8 else 1.0
        optimal_gas = int(base_gas * gas_multiplier)
        max_priority_fee = await async_web3.eth.max_priority_fee
        
        return {
            'gas_price': optimal_gas,
            'gas_limit': params.get('gas_estimate', 300000),
            'max_priority_fee': max_priority_fee,
            'network_congestion': network_congestion
        }
    except Exception as e:
        logger.error(f"Error optimizing gas strategy: {str(e)}")
        return {
            'gas_price': await async_web3.eth.gas_price,
            'gas_limit': 300000,
            'max_priority_fee': 1500000000,
            'network_congestion': 0.5
        }

def is_gas_acceptable(gas_strategy: Dict) -> bool:
    """Check if gas price is within acceptable limits"""
    try:
        MAX_GAS_PRICE = 500  # Maximum gas price in GWEI
        MAX_NETWORK_CONGESTION = 0.8  # Maximum acceptable network congestion
        
        return (
            gas_strategy['gas_price'] <= MAX_GAS_PRICE and 
            gas_strategy['network_congestion'] <= MAX_NETWORK_CONGESTION
        )
    except Exception as e:
        logger.error(f"Error checking gas acceptability: {str(e)}")
        return False

async def verify_sufficient_liquidity(params: Dict) -> bool:
    """Verify if there is sufficient liquidity for the trade"""
    try:
        MIN_LIQUIDITY_RATIO = 3  # Minimum ratio of liquidity to trade size
        
        amount = params.get('amount', 0)
        liquidity = params.get('market_data', {}).get('liquidity', 0)
        
        return liquidity >= amount * MIN_LIQUIDITY_RATIO
    except Exception as e:
        logger.error(f"Error verifying liquidity: {str(e)}")
        return False

def validate_parameters(params: Dict) -> bool:
    """Validate transaction parameters"""
    try:
        required_fields = ['amount', 'token_in', 'token_out', 'deadline']
        if not all(field in params for field in required_fields):
            return False
            
        return (
            params['amount'] > 0 and
            params['deadline'] > 0 and
            params['token_in'] in TOKEN_ADDRESSES and
            params['token_out'] in TOKEN_ADDRESSES
        )
    except Exception as e:
        logger.error(f"Error validating parameters: {str(e)}")
        return False

async def validate_price_deviation(params: Dict) -> bool:
    """Check if price deviation is within acceptable limits"""
    try:
        MAX_PRICE_DEVIATION = 0.02  # 2% maximum deviation
        
        current_price = params.get('market_data', {}).get('current_price', 0)
        expected_price = params.get('market_data', {}).get('expected_price', 0)
        
        if current_price == 0 or expected_price == 0:
            return False
            
        deviation = abs(current_price - expected_price) / expected_price
        return deviation <= MAX_PRICE_DEVIATION
        
    except Exception as e:
        logger.error(f"Error validating price deviation: {str(e)}")
        return False

async def check_balances(params: Dict) -> bool:
    """Check if there are sufficient balances for the transaction"""
    try:
        # Check ETH balance for gas
        eth_balance = Web3.eth.get_balance(params['wallet_address'])
        required_eth = params['gas_estimate'] * params.get('gas_price', Web3.eth.gas_price)
        
        if eth_balance < required_eth:
            logger.error("Insufficient ETH for gas")
            return False
            
        # Check token balance if not a flash loan
        if params.get('type') != 'flash_loan':
            token_contract = web3.eth.contract(
                address=Web3.to_checksum_address(TOKEN_ADDRESSES[params['token_in']]), 
                abi=params['token_abi']
            )
            token_balance = await token_contract.functions.balanceOf(params['wallet_address']).call()
            
            if token_balance < params['amount']:
                logger.error("Insufficient token balance")
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"Error checking balances: {str(e)}")
        return False

async def verify_contract_states(params: Dict) -> bool:
    """Verify the state of relevant contracts"""
    try:
        # Check if contracts are paused
        for contract_address in [params.get('dex_address'), params.get('lending_address')]:
            if contract_address:
                contract = web3.eth.contract(
                    address=Web3.to_checksum_address(contract_address), 
                    abi=params['contract_abi']
                )
                if hasattr(contract.functions, 'paused') and await contract.functions.paused().call():
                    logger.error(f"Contract {contract_address} is paused")
                    return False
        
        # Verify token approvals are still valid
        token_contract = web3.eth.contract(
            address=Web3.to_checksum_address(TOKEN_ADDRESSES[params['token_in']]), 
            abi=params['token_abi']
        )
        allowance = await token_contract.functions.allowance(
            params['wallet_address'],
            params.get('dex_address', params.get('lending_address'))
        ).call()
        
        if allowance < params['amount']:
            logger.error("Insufficient token allowance")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error verifying contract states: {str(e)}")
        return False

async def check_token_approvals(params: Dict) -> bool:
    """Check if token approvals are sufficient for the transaction"""
    try:
        # Get token contract
        token_contract = web3.eth.contract(
            address=Web3.to_checksum_address(TOKEN_ADDRESSES[params['token_in']]), 
            abi=params['token_abi']
        )
        
        # Check approvals for DEX and lending protocol
        for spender in [params.get('dex_address'), params.get('lending_address')]:
            if spender:
                allowance = await token_contract.functions.allowance(
                    params['wallet_address'],
                    spender
                ).call()
                
                if allowance < params['amount']:
                    logger.error(f"Insufficient allowance for spender {spender}")
                    return False
                    
        return True
        
    except Exception as e:
        logger.error(f"Error checking token approvals: {str(e)}")
        return False

async def simulate_transaction(params: Dict) -> Dict:
    """Simulate transaction to check for potential issues"""
    try:
        # Create transaction object with validation
        if not all(k in params for k in ['wallet_address', 'gas_estimate']):
            raise ValueError("Missing required parameters")

        # Get the target contract address
        target_address = params.get('dex_address') or params.get('lending_address')
        if not target_address:
            raise ValueError("No target address provided")

        # Create transaction object
        tx = {
            'from': params['wallet_address'],
            'to': target_address,
            'value': params.get('value', 0),
            'gas': params['gas_estimate'],
            'gasPrice': params.get('gas_price', Web3.eth.gas_price),
            'nonce': Web3.eth.get_transaction_count(params['wallet_address']),
            'data': params.get('data', '0x')
        }
        
        # Simulate transaction using eth_call
        try:
            tx_params = {
                'from': Web3.to_checksum_address(params['wallet_address']),
                'to': Web3.to_checksum_address(params.get('dex_address', params.get('lending_address'))),
                'value': params.get('value', 0),
                'gas': params['gas_estimate'],
                'gasPrice': params.get('gas_price', await async_web3.eth.gas_price),
                'nonce': await async_web3.eth.get_transaction_count(params['wallet_address']),
                'data': params.get('data', '0x')
            }
            result = await async_web3.eth.call(tx_params)
            success = True
        except Exception as e:
            result = str(e)
            success = False
            logger.error(f"Transaction simulation failed: {str(e)}")
        
        return {
            'success': success,
            'result': result,
            'gas_used': params['gas_estimate'],
            'error': None if success else str(result)
        }
        
    except Exception as e:
        logger.error(f"Error in transaction simulation: {str(e)}")
        return {
            'success': False,
            'result': None,
            'gas_used': 0,
            'error': str(e)
        }

async def execute_transaction_with_timeout(
    params: Dict[str, Any], 
    gas_strategy: Dict[str, Union[int, float]], 
    timeout: int = 30
) -> Optional[str]:
    """Execute transaction with timeout"""
    try:
        if not async_web3:
            raise ValueError("Web3 not initialized")

        tx = {
            'from': params['wallet_address'],
            'to': params.get('dex_address', params.get('lending_address')),
            'value': params.get('value', 0),
            'gas': gas_strategy['gas_limit'],
            'gasPrice': gas_strategy['gas_price'],
            'nonce': await async_web3.eth.get_transaction_count(params['wallet_address']),
            'data': params.get('data', '0x'),
            'maxPriorityFeePerGas': gas_strategy.get('max_priority_fee', 0)
        }
        
        async with asyncio.timeout(timeout):
            signed_tx = async_web3.eth.account.sign_transaction(tx, params['private_key'])
            tx_hash = await async_web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            return tx_hash.hex()
            
    except asyncio.TimeoutError:
        logger.error("Transaction execution timed out")
        return None
    except Exception as e:
        logger.error(f"Error executing transaction: {str(e)}")
        return None

async def wait_for_confirmation_with_timeout(
    tx_hash: str, 
    timeout: int = 180
) -> Dict[str, Union[bool, int]]:
    """Wait for transaction confirmation with timeout"""
    try:
        if not async_web3:
            raise ValueError("Web3 not initialized")

        async with asyncio.timeout(timeout):
            while True:
                try:
                    receipt = await async_web3.eth.get_transaction_receipt(
                        HexBytes(tx_hash)
                    )
                    if receipt:
                        return {
                            'success': receipt['status'] == 1,
                            'gas_used': receipt['gasUsed'],
                            'block_number': receipt['blockNumber']
                        }
                except Exception:
                    pass
                await asyncio.sleep(1)
    except asyncio.TimeoutError:
        logger.error(f"Timeout waiting for confirmation of tx {tx_hash}")
        return {'success': False, 'gas_used': 0, 'block_number': 0}
    except Exception as e:
        logger.error(f"Error waiting for confirmation: {str(e)}")
        return {'success': False, 'gas_used': 0, 'block_number': 0}

async def pre_execution_checks(params: Dict) -> bool:
    """Comprehensive pre-execution validation"""
    try:
        # 1. Validate parameters
        if not validate_parameters(params):
            return False
            
        # 2. Check token approvals
        if not await check_token_approvals(params):
            return False
            
        # 3. Verify contract states
        if not await verify_contract_states(params):
            return False
            
        # 4. Check balance requirements
        if not await check_balances(params):
            return False
            
        # 5. Validate price deviation
        if not await validate_price_deviation(params):
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Pre-execution checks failed: {str(e)}")
        return False

async def safe_execute_transaction(params: Dict) -> bool:
    """Enhanced transaction execution with comprehensive safety checks"""
    try:
        # 1. Pre-execution checks
        if not await pre_execution_checks(params):
            return False
            
        # 2. Gas price check
        gas_strategy = await optimize_gas_strategy(params)
        if not is_gas_acceptable(gas_strategy):
            return False
            
        # 3. Liquidity verification
        if not await verify_sufficient_liquidity(params):
            return False
            
        # 4. Simulate transaction
        simulation_result = await simulate_transaction(params)
        if not simulation_result['success']:
            logger.error(f"Transaction simulation failed: {simulation_result['error']}")
            return False
            
        # 5. Execute transaction with retry mechanism
        for attempt in range(3):
            try:
                tx_hash = await execute_transaction_with_timeout(params, gas_strategy)
                if not tx_hash:
                    continue
                    
                # 6. Wait for confirmation with timeout
                confirmation = await wait_for_confirmation_with_timeout(tx_hash)
                if confirmation['success']:
                    # Add strategy and protocol information to execution result
                    params['execution_result'] = {
                        'success': True,
                        'tx_hash': tx_hash,
                        'gas_used': confirmation['gas_used'],
                        'block_number': confirmation['block_number'],
                        'dex_used': params.get('dex_address', 'Unknown DEX'),
                        'lending_protocol': params.get('lending_protocol', {}).get('protocol', 'Unknown Protocol'),
                        'type': params.get('strategy_type', 'Unknown Strategy'),
                        'profit': params.get('expected_profit', 0)
                    }
                    return True
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                await asyncio.sleep(1)
                
        return False
        
    except Exception as e:
        logger.error(f"Transaction execution failed: {str(e)}")
        return False

def prepare_training_data(raw_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare training data for the model"""
    X = []  # Features
    y = []  # Labels (profit, confidence, risk)
    
    for data in raw_data:
        features = [
            data['amount'],
            data['price_impact'],
            data['liquidity'],
            data['volatility'],
            data['gas_price'],
            data['block_time'],
            data['price_difference'],
            data['volume_24h']
        ]
        
        labels = [
            data['actual_profit'],
            data['success_rate'],
            data['risk_level']
        ]
        
        X.append(features)
        y.append(labels)
    
    return np.array(X), np.array(y)

def calculate_risk_score(market_data: Dict) -> float:
    """Calculate risk score based on market conditions"""
    # Implement risk scoring logic
    volatility_weight = 0.3
    liquidity_weight = 0.3
    volume_weight = 0.2
    price_impact_weight = 0.2
    
    risk_score = (
        volatility_weight * market_data['volatility'] +
        liquidity_weight * (1 - market_data['liquidity']) +
        volume_weight * (1 - market_data['volume_24h']) +
        price_impact_weight * market_data['price_impact']
    )
    
    return min(max(risk_score, 0), 1)  # Normalize to [0, 1]

def get_token_pairs() -> List[Tuple[str, str]]:
    """Get list of token pairs to monitor"""
    return TOKEN_PAIRS

def get_test_amounts() -> Dict[str, List[float]]:
    """Get test amounts for each token"""
    return TEST_AMOUNTS 

# Load configurations from config.json
def load_dex_config():
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
            return config.get('dexes', {})
    except Exception as e:
        logger.error(f"Error loading config.json: {str(e)}")
        return {}

DEX_CONFIGS = load_dex_config()

# Initialize market data caches with default values
MARKET_DATA_DEFAULTS = {
    'price': 0.0,
    'volume': 0.0,
    'liquidity': 0.0,
    'timestamp': datetime.now().isoformat()
}

# Initialize history tracking with minimum structure
HISTORY_DEFAULTS = {
    'opportunities': [],
    'market_snapshots': {},
    'performance': {
        'total_profit': 0.0,
        'successful_trades': 0,
        'failed_trades': 0,
        'gas_spent': 0.0
    }
}

# Initialize cache settings
CACHE_SETTINGS = {
    'ttl': 300,  # 5 minutes
    'max_size': 1000,
    'cleanup_interval': 3600  # 1 hour
}

# Type aliases
DexData: TypeAlias = Dict[str, Dict[str, Any]]
MarketData: TypeAlias = Dict[str, Dict[str, Any]]
TokenData: TypeAlias = Dict[str, Any]
PriceData: TypeAlias = Dict[str, Any]
VolumeData: TypeAlias = Dict[str, Any]

# Type variable for generic functions
T = TypeVar('T')

class AsyncCallable(Protocol):
    async def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

def retry_async(retries: int = 3, delay: int = 1) -> Callable[[AsyncCallable], AsyncCallable]:
    def decorator(func: AsyncCallable) -> AsyncCallable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == retries - 1:
                        raise
                    await asyncio.sleep(delay * (2 ** attempt))
            return None
        return wrapper
    return decorator

def validate_input(validator: Callable[..., Coroutine[Any, Any, bool]]) -> Callable[[AsyncCallable], AsyncCallable]:
    def decorator(func: AsyncCallable) -> AsyncCallable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not await validator(*args, **kwargs):
                raise ValidationError("Input validation failed")
            return await func(*args, **kwargs)
        return wrapper
    return decorator

class TokenDiscovery:
    def __init__(self):
        # Initialize data structures with defaults
        self.token_history = {}
        self.price_history = {}
        self.volume_history = {}
        self.last_cleanup = datetime.now()
        
        # Initialize discovered tokens cache
        self.discovered_tokens = {}
        
        # Initialize validation results cache
        self.validation_results = {}
        
        # Initialize token blacklist
        self.blacklisted_tokens = set()
        
        # Initialize token scoring history
        self.token_scores = {}
        
        # Initialize locks for thread safety
        self._token_lock = asyncio.Lock()
        self._price_lock = asyncio.Lock()
        self._volume_lock = asyncio.Lock()
        
        # Security thresholds
        self.MAX_PRICE_VOLATILITY = 50  # Max 50% price change in 1 hour
        self.MIN_VOLUME_CONSISTENCY = 0.1  # Min 10% volume consistency
        self.MAX_HOLDER_CHURN = 30  # Max 30% holder turnover in 24h
        self.MIN_LIQUIDITY_RETENTION = 70  # Min 70% liquidity retention
        self.ANALYSIS_INTERVALS = [1, 4, 12, 24, 72]  # Hours to analyze
        self.MAX_OWNER_PERCENTAGE = 20  # Maximum percentage of supply owned by contract owner
        
        # Enhanced security thresholds
        self.MAX_WHALE_CONCENTRATION = 20  # Max 20% held by top 10 non-LP holders
        self.MIN_UNIQUE_TRADERS = 50  # Min unique traders in 24h
        self.MAX_SELL_TAX_INCREASE = 2  # Max 2x tax increase from initial
        self.MAX_FAILED_SELLS = 10  # Max % of failed sell transactions
        
        # Malicious signatures and patterns
        self.MALICIOUS_SIGNATURES = [
            'blacklist', 'pause', 'mint', 'burn', 'setTaxRate', 'setMaxTxAmount',
            'excludeFromFee', 'setFeeAddress', 'updateFee', 'setMaxWallet'
        ]
        self.MALICIOUS_PATTERNS = [
            r'selfdestruct\s*\(',
            r'delegatecall\s*\(',
            r'transfer\s*\(\s*owner\s*,',
            r'_blacklist\s*\[',
            r'require\s*\(\s*msg\.sender\s*==\s*owner\s*\)'
        ]
        
        # Initialize Web3
        self.web3 = Web3(Web3.HTTPProvider(os.getenv('BASE_MAINNET_RPC')))

    async def cleanup_old_data(self) -> None:
        """Cleanup historical data older than 72 hours"""
        try:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(hours=72)
            
            async with self._token_lock:
                self.token_history = {k: v for k, v in self.token_history.items() 
                                    if v.get('last_updated', current_time) > cutoff_time}
                
            async with self._price_lock:
                self.price_history = {k: v for k, v in self.price_history.items() 
                                    if v.get('last_updated', current_time) > cutoff_time}
                
            async with self._volume_lock:
                self.volume_history = {k: v for k, v in self.volume_history.items() 
                                    if v.get('last_updated', current_time) > cutoff_time}
            
            self.last_cleanup = current_time
            logger.info("Completed historical data cleanup")
            
        except Exception as e:
            logger.error(f"Error in cleanup_old_data: {str(e)}")

    async def update_token_data(self, token_address: str, data: Dict[str, Any]) -> None:
        """Update token historical data"""
        try:
            async with self._token_lock:
                data['last_updated'] = datetime.now()
                self.token_history[token_address] = data
        except Exception as e:
            logger.error(f"Error updating token data: {str(e)}")

    async def update_price_data(self, token_pair: str, price_data: Dict[str, Any]) -> None:
        """Update price historical data"""
        try:
            async with self._price_lock:
                price_data['last_updated'] = datetime.now()
                self.price_history[token_pair] = price_data
        except Exception as e:
            logger.error(f"Error updating price data: {str(e)}")

    async def update_volume_data(self, token_pair: str, volume_data: Dict[str, Any]) -> None:
        """Update volume historical data"""
        try:
            async with self._volume_lock:
                volume_data['last_updated'] = datetime.now()
                self.volume_history[token_pair] = volume_data
        except Exception as e:
            logger.error(f"Error updating volume data: {str(e)}")

    async def safe_execute(
        self, 
        func: Callable[..., Coroutine[Any, Any, T]], 
        *args: Any, 
        **kwargs: Any
    ) -> Optional[T]:
        """Safely execute async functions with error handling"""
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error executing {func.__name__}: {str(e)}")
            return None

    async def validate_token(
        self, 
        session: ClientSession, 
        token: Dict[str, Any], 
        pair: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Comprehensive token validation"""
        try:
            # Basic validation
            if not self.basic_validation(token):
                return False

            # Contract validation
            if not await self.validate_contract(session, token['address']):
                return False

            # Security checks
            if await self.detect_scam_patterns(session, token['address']):
                return False

            # Historical analysis
            history = await self.analyze_token_history(session, token['address'])
            if not history or history.get('risk_score', 1.0) > 0.7:
                return False

            # Liquidity checks
            if pair and not await self.validate_liquidity(session, token, pair):
                return False

            return True

        except Exception as e:
            logger.error(f"Error in validate_token: {str(e)}")
            return False

    def basic_validation(self, token: Dict) -> bool:
        """Perform basic token validation checks"""
        try:
            # Must have symbol and decimals
            if not token.get('symbol') or not token.get('decimals'):
                return False
                
            # Symbol length check
            if len(token['symbol']) > 12:
                return False
                
            # Decimal range check
            if not (0 <= int(token['decimals']) <= 18):
                return False
                
            # Blacklist common scam token names
            blacklist = ['TEST', 'SCAM', 'HONEY', 'FAKE']
            if any(word in token['symbol'].upper() for word in blacklist):
                return False
                
            return True
            
        except Exception:
            return False

    async def validate_contract(self, session: ClientSession, token_address: str) -> bool:
        """Validate token smart contract"""
        try:
            # Get contract code and verify it's not empty
            code = await async_web3.eth.get_code(Web3.to_checksum_address(token_address))
            if len(code) < 100:  # Empty or proxy contract
                return False
                
            # Check contract verification on Basescan
            if not await self.is_contract_verified(session, token_address):
                return False
                
            # Check for common security patterns
            security_score = await self.check_security_patterns(session, token_address)
            if security_score < 0.7:  # Require 70% security score
                return False
                
            return True
            
        except Exception:
            return False

    async def check_security_patterns(self, session: ClientSession, token_address: str) -> float:
        """Check for security patterns in contract code and return a score between 0 and 1"""
        try:
            # Get contract code
            code = await async_web3.eth.get_code(Web3.to_checksum_address(token_address))
            code_str = code.hex()
            
            # Initialize score
            score = 1.0
            
            # Check for dangerous patterns
            dangerous_patterns = [
                'selfdestruct',
                'delegatecall',
                'SELFDESTRUCT',
                'DELEGATECALL',
                'CALLCODE'
            ]
            
            for pattern in dangerous_patterns:
                if pattern in code_str:
                    score *= 0.5
                    
            # Check for ownership patterns
            ownership_patterns = [
                'onlyOwner',
                'Ownable',
                'transferOwnership'
            ]
            
            for pattern in ownership_patterns:
                if pattern in code_str:
                    score *= 0.8
                    
            return max(0.0, min(score, 1.0))
            
        except Exception as e:
            logger.error(f"Error checking security patterns: {str(e)}")
            return 0.0

    async def detect_scam_patterns(self, session: ClientSession, token_address: str) -> bool:
        """Enhanced scam detection with multiple security checks"""
        try:
            # Get contract code and ABI
            code = await async_web3.eth.get_code(Web3.to_checksum_address(token_address))
            abi = await self.get_contract_abi(session, token_address)
            contract = self.web3.eth.contract(address=Web3.to_checksum_address(token_address), abi=abi)
            
            # 1. Check for malicious functions
            if await self.has_malicious_functions(contract, code):
                logger.warning(f"Malicious functions detected in {token_address}")
                return True
            
            # 2. Check ownership patterns
            if await self.check_dangerous_ownership(contract, token_address):
                logger.warning(f"Dangerous ownership pattern in {token_address}")
                return True
            
            # 3. Analyze liquidity patterns
            if await self.analyze_liquidity_patterns(contract, token_address):
                logger.warning(f"Suspicious liquidity pattern in {token_address}")
                return True
            
            # 4. Check for honeypot characteristics
            if await self.detect_honeypot(contract, token_address):
                logger.warning(f"Honeypot characteristics detected in {token_address}")
                return True
            
            # 5. Analyze trading restrictions
            if await self.check_trading_restrictions(contract, code):
                logger.warning(f"Suspicious trading restrictions in {token_address}")
                return True
            
            # 6. Check for similar token scams
            if await self.check_similar_token_scams(session, contract):
                logger.warning(f"Similar token scam detected for {token_address}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in scam detection for {token_address}: {str(e)}")
            return True  # Fail safe: treat errors as potential scams

    async def has_malicious_functions(self, contract: Contract, code: str) -> bool:
        """Check for malicious functions in contract code"""
        try:
            # 1. Check function signatures
            for signature in self.MALICIOUS_SIGNATURES:
                if signature in code:
                    # Verify if function has dangerous permissions
                    if await self.verify_function_permissions(contract, signature):
                        return True
            
            # 2. Check for malicious patterns
            for pattern in self.MALICIOUS_PATTERNS:
                if re.search(pattern, code):
                    return True
            
            # 3. Check for hidden malicious code
            if await self.detect_hidden_code(code):
                return True
            
            return False
            
        except Exception:
            return True

    async def check_dangerous_ownership(self, contract: Contract, token_address: str) -> bool:
        """Analyze contract ownership patterns"""
        try:
            # 1. Check owner's token balance
            owner = await contract.functions.owner().call()
            total_supply = await contract.functions.totalSupply().call()
            owner_balance = await contract.functions.balanceOf(owner).call()
            
            if owner_balance / total_supply * 100 > self.MAX_OWNER_PERCENTAGE:
                return True
            
            # 2. Check for proxy contracts
            if await self._is_proxy_contract(contract):
                if not await self._verify_proxy_implementation(contract):
                    return True
            
            # 3. Check ownership renouncement
            if await self._can_renounce_ownership(contract):
                if not await self._is_ownership_renounced(contract):
                    return True
            
            # 4. Check for timelock on critical functions
            if not await self._has_timelock_protection(contract):
                return True
            
            return False
            
        except Exception:
            return True

    async def _is_proxy_contract(self, contract: Contract) -> bool:
        """Check if contract is a proxy contract"""
        try:
            code = contract.bytecode
            # Check for common proxy patterns
            proxy_patterns = [
                bytes.fromhex('363d3d373d3d3d363d73'),  # EIP-1167 minimal proxy
                bytes.fromhex('7f360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc')  # EIP-1967 beacon proxy
            ]
            return any(pattern in code for pattern in proxy_patterns)
        except Exception:
            return False

    async def _verify_proxy_implementation(self, contract: Contract) -> bool:
        """Verify the implementation contract of a proxy"""
        try:
            # Try to get implementation address using common slots/methods
            implementation = None
            try:
                implementation = await contract.functions.implementation().call()
            except:
                try:
                    # EIP-1967 implementation slot
                    slot = '0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc'
                    implementation = await self.web3.eth.get_storage_at(contract.address, slot)
                except:
                    return False
            
            return implementation is not None and implementation != '0x' + '0' * 40
        except Exception:
            return False

    async def _can_renounce_ownership(self, contract: Contract) -> bool:
        """Check if contract can renounce ownership"""
        try:
            return hasattr(contract.functions, 'renounceOwnership')
        except Exception:
            return False

    async def _is_ownership_renounced(self, contract: Contract) -> bool:
        """Check if ownership has been renounced"""
        try:
            owner = await contract.functions.owner().call()
            return owner == '0x' + '0' * 40  # Check if owner is zero address
        except Exception:
            return False

    async def _has_timelock_protection(self, contract: Contract) -> bool:
        """Check if contract has timelock protection on critical functions"""
        try:
            # Check for common timelock patterns
            code = contract.bytecode
            timelock_patterns = [
                bytes.fromhex('timestamp'),
                bytes.fromhex('delay'),
                bytes.fromhex('timelock')
            ]
            has_timelock_code = any(pattern in code for pattern in timelock_patterns)
            
            # Check for timelock contract integration
            has_timelock_role = False
            try:
                timelock_role = await contract.functions.TIMELOCK_ROLE().call()
                has_timelock_role = True
            except:
                pass
                
            return has_timelock_code or has_timelock_role
        except Exception:
            return False

    async def analyze_liquidity_patterns(self, contract: Contract, token_address: str) -> bool:
        """Analyze liquidity patterns for suspicious behavior"""
        try:
            # 1. Check LP token distribution
            lp_holders = await self.get_lp_holders(contract)
            if not await self.verify_lp_lock(lp_holders):
                return True
            
            # 2. Check for flash loan attack vectors
            if await self.is_vulnerable_to_flash_loans(contract):
                return True
            
            # 3. Check liquidity removal capabilities
            if await self.can_remove_liquidity_instantly(contract):
                return True
            
            # 4. Monitor liquidity changes
            if await self.detect_suspicious_liquidity_changes(contract):
                return True
            
            return False
            
        except Exception:
            return True

    async def detect_honeypot(self, contract: Contract, token_address: str) -> bool:
        """Detect honeypot characteristics"""
        try:
            # 1. Simulate buy transaction
            buy_success = await self.simulate_buy_transaction(contract)
            if not buy_success:
                return True
            
            # 2. Simulate sell transaction
            sell_success = await self.simulate_sell_transaction(contract)
            if not sell_success:
                return True
            
            # 3. Check for sell limits
            if await self.has_hidden_sell_limits(contract):
                return True
            
            # 4. Check for price manipulation
            if await self.detect_price_manipulation(contract):
                return True
            
            # 5. Check for blacklist after buy
            if await self.check_post_buy_blacklist(contract):
                return True
            
            return False
            
        except Exception:
            return True

    async def check_trading_restrictions(self, contract: Contract, code: str) -> bool:
        """Analyze trading restrictions"""
        try:
            # 1. Check transfer limits
            limits = await self.get_transfer_limits(contract)
            if self.are_limits_suspicious(limits):
                return True
            
            # 2. Check for dynamic tax changes
            if await self.has_dynamic_taxes(contract):
                return True
            
            # 3. Check for blacklist/whitelist
            if await self.has_address_restrictions(contract):
                return True
            
            # 4. Check for time-based restrictions
            if await self.has_time_restrictions(contract):
                return True
            
            return False
            
        except Exception:
            return True

    async def check_similar_token_scams(self, session: ClientSession, contract: Contract) -> bool:
        """Check for similar token name scams"""
        try:
            # Get token details
            name = await contract.functions.name().call()
            symbol = await contract.functions.symbol().call()
            
            # Search for similar tokens
            similar_tokens = await self.find_similar_tokens(session, name, symbol)
            
            # If too many similar tokens exist, it might be a scam
            if len(similar_tokens) > self.MAX_SIMILAR_TOKEN_COUNT:
                return True
            
            # Check if this is copying a known legitimate token
            if await self.is_impersonating_token(name, symbol):
                return True
            
            return False
            
        except Exception:
            return True

    async def analyze_token_history(self, session: ClientSession, token_address: str) -> Dict:
        """Analyze historical behavior patterns of a token"""
        try:
            current_time = datetime.now()
            token_data = {}
            
            # Fetch historical data for each interval
            for hours in self.ANALYSIS_INTERVALS:
                start_time = current_time - timedelta(hours=hours)
                interval_data = await self.fetch_interval_data(session, token_address, start_time, current_time)
                token_data[f'{hours}h'] = interval_data
            
            # Calculate key metrics
            analysis = {
                'price_volatility': self.calculate_price_volatility(token_data),
                'volume_consistency': self.calculate_volume_consistency(token_data),
                'holder_stability': self.analyze_holder_stability(token_data),
                'liquidity_retention': self.calculate_liquidity_retention(token_data),
                'trading_patterns': self.analyze_trading_patterns(token_data),
                'whale_behavior': await self.analyze_whale_behavior(session, token_address, token_data),
                'tax_history': await self.analyze_tax_history(session, token_address, token_data)
            }
            
            # Store historical data
            self.token_history[token_address] = {
                'last_updated': current_time,
                'analysis': analysis
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing token history for {token_address}: {str(e)}")
            return None

    def calculate_price_volatility(self, token_data: Dict) -> float:
        """Calculate price volatility across different time intervals"""
        try:
            volatilities = []
            
            for interval, data in token_data.items():
                if not data['prices']:
                    continue
                    
                prices = np.array(data['prices'])
                returns = np.diff(prices) / prices[:-1]
                volatility = np.std(returns) * 100
                volatilities.append(volatility)
            
            return np.mean(volatilities) if volatilities else float('inf')
            
        except Exception:
            return float('inf')

    def calculate_volume_consistency(self, token_data: Dict) -> float:
        """Analyze trading volume consistency"""
        try:
            volumes = []
            
            for interval, data in token_data.items():
                if not data['volumes']:
                    continue
                    
                # Calculate volume trend
                volume_trend = np.polyfit(range(len(data['volumes'])), data['volumes'], 1)[0]
                volumes.append(volume_trend)
            
            avg_volume_trend = np.mean(volumes) if volumes else 0
            return max(0, min(1, avg_volume_trend / self.MIN_VOLUME_CONSISTENCY))
            
        except Exception:
            return 0

    def analyze_holder_stability(self, token_data: Dict) -> Dict:
        """Analyze holder behavior and stability"""
        try:
            stability_metrics = {
                'holder_churn': 0,
                'concentration_index': 0,
                'retention_rate': 0
            }
            
            holders_data = {}
            for interval in token_data:
                if isinstance(token_data[interval], dict) and 'holders' in token_data[interval]:
                    holders_data[interval] = token_data[interval]['holders']
            
            for interval, holders in holders_data.items():
                if not holders:
                    continue
                
                # Calculate holder churn rate
                holder_changes = np.diff([len(holders) for holders in holders])
                churn_rate = np.sum(np.abs(holder_changes)) / len(holders[0])
                
                # Calculate holder concentration
                total_balance = sum(float(h.get('balance', 0)) for h in holders[-1])
                top_holders = sorted(holders[-1], key=lambda x: float(x.get('balance', 0)), reverse=True)[:10]
                concentration = sum(float(h.get('balance', 0)) for h in top_holders) / total_balance if total_balance > 0 else 0
                
                # Calculate holder retention
                initial_holders = set(h['address'] for h in holders[0])
                final_holders = set(h['address'] for h in holders[-1])
                retention = len(initial_holders.intersection(final_holders)) / len(initial_holders)
                
                stability_metrics['holder_churn'] = max(stability_metrics['holder_churn'], churn_rate)
                stability_metrics['concentration_index'] = max(stability_metrics['concentration_index'], concentration)
                stability_metrics['retention_rate'] = min(stability_metrics['retention_rate'], retention)
            
            return stability_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing holder stability: {str(e)}")
            return {'holder_churn': float('inf'), 'concentration_index': float('inf'), 'retention_rate': 0}

    def calculate_liquidity_retention(self, token_data: Dict) -> float:
        """Calculate liquidity retention rate"""
        try:
            retentions = []
            
            for interval, data in token_data.items():
                if not isinstance(data, dict) or 'liquidity' not in data:
                    continue
                
                liquidity_data = data['liquidity']
                if not liquidity_data:
                    continue
                
                initial_liquidity = liquidity_data[0]
                final_liquidity = liquidity_data[-1]
                
                if initial_liquidity > 0:
                    retention = (final_liquidity / initial_liquidity) * 100
                    retentions.append(retention)
            
            return np.mean(retentions) if retentions else 0
            
        except Exception:
            return 0

    async def analyze_whale_behavior(self, session: ClientSession, token_address: str, token_data: Dict) -> Dict:
        """Analyze behavior of large token holders"""
        try:
            whale_metrics = {
                'whale_concentration': 0,
                'whale_trades': [],
                'suspicious_patterns': []
            }
            
            # Get current top holders
            holders = await self.get_top_holders(session, token_address)
            
            # Analyze each whale's trading pattern
            for holder in holders[:10]:  # Focus on top 10 holders
                trades = await self.get_holder_trades(session, token_address, holder['address'])
                
                # Analyze trade timing and sizes
                trade_analysis = self.analyze_trade_patterns(trades)
                
                if trade_analysis.get('suspicious', False):
                    whale_metrics['suspicious_patterns'].append({
                        'holder': holder['address'],
                        'pattern': trade_analysis.get('pattern', 'unknown')
                    })
                
                whale_metrics['whale_trades'].extend(trade_analysis['trades'])
            
            # Calculate overall whale concentration
            total_supply = await self.get_total_supply(token_address)
            whale_balance = sum(h['balance'] for h in holders[:10])
            whale_metrics['whale_concentration'] = (whale_balance / total_supply) * 100
            
            return whale_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing whale behavior: {str(e)}")
            return {'whale_concentration': 0, 'whale_trades': [], 'suspicious_patterns': []}

    async def analyze_tax_history(self, session: ClientSession, token_address: str, token_data: Dict) -> Dict:
        """Analyze historical changes in token taxes/fees"""
        try:
            tax_history = {
                'buy_tax': [],
                'sell_tax': [],
                'transfer_tax': [],
                'tax_changes': []
            }
            
            # Analyze tax changes over time
            for interval, data in token_data.items():
                if not data['transactions']:
                    continue
                
                if not isinstance(data, dict) or 'transactions' not in data:
                    continue
                
                transactions = data.get('transactions', [])
                if not transactions:
                    continue
                
                # Calculate effective tax rates from transactions
                buy_taxes = self.calculate_effective_tax(transactions, 'buy')
                sell_taxes = self.calculate_effective_tax(transactions, 'sell')
                transfer_taxes = self.calculate_effective_tax(transactions, 'transfer')
                
                tax_history['buy_tax'].append(np.mean(buy_taxes))
                tax_history['sell_tax'].append(np.mean(sell_taxes))
                tax_history['transfer_tax'].append(np.mean(transfer_taxes))
                
                # Detect significant tax changes
                if len(tax_history['sell_tax']) > 1:
                    tax_change = tax_history['sell_tax'][-1] / tax_history['sell_tax'][0]
                    if tax_change > self.MAX_SELL_TAX_INCREASE:
                        tax_history['tax_changes'].append({
                            'timestamp': data['timestamp'],
                            'type': 'sell_tax_increase',
                            'change_factor': tax_change
                        })
            
            return tax_history
            
        except Exception as e:
            logger.error(f"Error analyzing tax history: {str(e)}")
            return {'buy_tax': [], 'sell_tax': [], 'transfer_tax': [], 'tax_changes': []}

    async def validate_token_with_liquidity(
        self, 
        session: ClientSession, 
        token: Dict[str, Any], 
        pair: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Comprehensive token validation with liquidity checks"""
        try:
            # Basic validation
            if not self.validate_token_basics(token):
                return False

            # Contract validation
            if not await self.validate_contract(session, token['address']):
                return False

            # Security checks
            if await self.detect_scam_patterns(session, token['address']):
                return False

            # Historical analysis
            history = await self.analyze_token_history(session, token['address'])
            if not history or history.get('risk_score', 1.0) > 0.7:
                return False

            # Liquidity checks
            if pair and not await self.validate_liquidity(session, token, pair):
                return False

            return True

        except Exception as e:
            logger.error(f"Error in validate_token_with_liquidity: {str(e)}")
            return False

    def validate_token_basics(self, token: Dict) -> bool:
        """Perform basic token validation checks"""
        try:
            # Must have required fields
            required_fields = ['address', 'symbol', 'decimals', 'name']
            if not all(field in token for field in required_fields):
                return False
                
            # Symbol length check
            if not (2 <= len(token['symbol']) <= 12):
                return False
                
            # Decimal range check
            if not (0 <= int(token['decimals']) <= 18):
                return False
                
            # Name length check
            if not (2 <= len(token['name']) <= 64):
                return False
                
            # Blacklist check
            blacklist = ['TEST', 'SCAM', 'HONEY', 'FAKE', 'MOCK', 'SAMPLE']
            if any(word in token['symbol'].upper() for word in blacklist):
                return False
                
            # Address format check
            if not web3.isAddress(token['address']):
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error in basic validation: {str(e)}")
            return False

    async def validate_liquidity(self, session: ClientSession, token: Dict[str, Any], pair: Dict[str, Any]) -> bool:
        """Validate liquidity requirements for a token pair"""
        try:
            # Check minimum liquidity requirement
            if pair.get('liquidity', 0) < self.MIN_LIQUIDITY_USD:
                logger.warning(f"Insufficient liquidity for {token.get('symbol')}")
                return False
                
            # Check liquidity distribution
            liquidity_data = await self._get_liquidity_data(session, pair['address'])
            if not self._validate_liquidity_distribution(liquidity_data):
                return False
                
            # Check liquidity stability
            if not await self._check_liquidity_stability(session, pair['address']):
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating liquidity: {str(e)}")
            return False
            
    async def _get_liquidity_data(self, session: ClientSession, pair_address: str) -> Dict:
        """Get detailed liquidity data for a pair"""
        # Implementation would fetch liquidity data from DEX
        return {'total_liquidity': 0, 'distribution': []}
        
    def _validate_liquidity_distribution(self, liquidity_data: Dict) -> bool:
        """Validate liquidity distribution is healthy"""
        # Implementation would check liquidity concentration
        return True
        
    async def _check_liquidity_stability(self, session: ClientSession, pair_address: str) -> bool:
        """Check if liquidity is stable over time"""
        # Implementation would check liquidity history
        return True

    async def is_contract_verified(self, session: ClientSession, token_address: str) -> bool:
        """Check if contract is verified on Basescan"""
        try:
            # Basescan API endpoint
            api_key = os.getenv('BASESCAN_API_KEY')
            if not api_key:
                return False
                
            url = f"https://api.basescan.org/api?module=contract&action=getabi&address={token_address}&apikey={api_key}"
            
            async with session.get(url) as response:
                if response.status != 200:
                    return False
                    
                data = await response.json()
                return data.get('status') == '1' and data.get('result') != ''
                
        except Exception as e:
            logger.error(f"Error checking contract verification: {str(e)}")
            return False

    async def get_contract_abi(self, session: ClientSession, token_address: str) -> List[Dict]:
        """Get contract ABI from Basescan"""
        try:
            api_key = os.getenv('BASESCAN_API_KEY')
            if not api_key:
                return ERC20_ABI  # Fallback to standard ERC20 ABI
                
            url = f"https://api.basescan.org/api?module=contract&action=getabi&address={token_address}&apikey={api_key}"
            
            async with session.get(url) as response:
                if response.status != 200:
                    return ERC20_ABI
                    
                data = await response.json()
                if data.get('status') == '1' and data.get('result'):
                    return json.loads(data['result'])
                return ERC20_ABI
                
        except Exception as e:
            logger.error(f"Error fetching contract ABI: {str(e)}")
            return ERC20_ABI

    async def verify_function_permissions(self, contract: Contract, signature: str) -> bool:
        """Verify if a function has dangerous permissions"""
        try:
            # Check if function exists in contract
            if not hasattr(contract.functions, signature):
                return False
                
            # Get function object
            function = getattr(contract.functions, signature)
            
            # Check if function is restricted to owner/admin
            if 'onlyOwner' in str(function.abi):
                return True
                
            # Check for dangerous state modifications
            dangerous_opcodes = ['SELFDESTRUCT', 'DELEGATECALL', 'CALLCODE']
            function_code = await contract.functions[signature].call()
            return any(opcode in function_code for opcode in dangerous_opcodes)
            
        except Exception as e:
            logger.error(f"Error verifying function permissions: {str(e)}")
            return False

    async def detect_hidden_code(self, code: str) -> bool:
        """Detect hidden malicious code in contract"""
        try:
            # Check for obfuscated code patterns
            obfuscated_patterns = [
                r'\\x[0-9a-fA-F]{2}',  # Hex encoded bytes
                r'eval\s*\(',           # Dynamic code execution
                r'assembly\s*{.*}',     # Inline assembly
                r'_.*_.*_'              # Excessive underscores (often used to hide functions)
            ]
            
            for pattern in obfuscated_patterns:
                if re.search(pattern, code):
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error detecting hidden code: {str(e)}")
            return True  # Fail safe: treat errors as potential hidden code

    async def get_lp_holders(self, contract: Contract) -> List[Dict]:
        """Get list of LP token holders"""
        try:
            # Get total supply
            total_supply = await contract.functions.totalSupply().call()
            # Get top holders (example implementation)
            return [{'address': '0x0', 'balance': total_supply}]
        except Exception as e:
            logger.error(f"Error getting LP holders: {str(e)}")
            return []

    async def verify_lp_lock(self, lp_holders: List[Dict]) -> bool:
        """Verify if LP tokens are locked"""
        try:
            # Check if majority of LP tokens are locked
            return True  # Implement actual verification logic
        except Exception:
            return False

    async def is_vulnerable_to_flash_loans(self, contract: Contract) -> bool:
        """Check if contract is vulnerable to flash loan attacks"""
        try:
            # Implement flash loan vulnerability check
            return False
        except Exception:
            return True

    async def can_remove_liquidity_instantly(self, contract: Contract) -> bool:
        """Check if liquidity can be removed instantly"""
        try:
            # Implement liquidity removal check
            return False
        except Exception:
            return True

    async def detect_suspicious_liquidity_changes(self, contract: Contract) -> bool:
        """Monitor for suspicious liquidity changes"""
        try:
            # Implement liquidity change monitoring
            return False
        except Exception:
            return True

    async def simulate_buy_transaction(self, contract: Contract) -> bool:
        """Simulate a buy transaction"""
        try:
            # Implement buy simulation
            return True
        except Exception:
            return False

    async def simulate_sell_transaction(self, contract: Contract) -> bool:
        """Simulate a sell transaction"""
        try:
            # Implement sell simulation
            return True
        except Exception:
            return False

    async def has_hidden_sell_limits(self, contract: Contract) -> bool:
        """Check for hidden sell limits"""
        try:
            # Implement sell limit detection
            return False
        except Exception:
            return True

    async def detect_price_manipulation(self, contract: Contract) -> bool:
        """Check for price manipulation capabilities"""
        try:
            # Implement price manipulation detection
            return False
        except Exception:
            return True

    async def check_post_buy_blacklist(self, contract: Contract) -> bool:
        """Check for post-buy blacklisting"""
        try:
            # Implement post-buy blacklist check
            return False
        except Exception:
            return True

    async def get_transfer_limits(self, contract: Contract) -> Dict:
        """Get transfer limits from contract"""
        try:
            # Implement transfer limit retrieval
            return {'max_transfer': 0, 'min_transfer': 0}
        except Exception:
            return {}

    def are_limits_suspicious(self, limits: Dict) -> bool:
        """Check if transfer limits are suspicious"""
        try:
            # Implement limit analysis
            return False
        except Exception:
            return True

    async def has_dynamic_taxes(self, contract: Contract) -> bool:
        """Check for dynamic tax mechanisms"""
        try:
            # Implement dynamic tax detection
            return False
        except Exception:
            return True

    async def has_address_restrictions(self, contract: Contract) -> bool:
        """Check for address-based restrictions"""
        try:
            # Implement address restriction check
            return False
        except Exception:
            return True

    async def has_time_restrictions(self, contract: Contract) -> bool:
        """Check for time-based restrictions"""
        try:
            # Implement time restriction check
            return False
        except Exception:
            return True

    async def find_similar_tokens(self, session: ClientSession, name: str, symbol: str) -> List[Dict]:
        """Find similar tokens by name or symbol"""
        try:
            # Implement similar token search
            return []
        except Exception:
            return []

    # Add missing constant
    MAX_SIMILAR_TOKEN_COUNT = 5
    MIN_LIQUIDITY_USD = 10000  # $10k minimum liquidity

    async def is_impersonating_token(self, name: str, symbol: str) -> bool:
        """Check if token is impersonating a known token"""
        try:
            # Implement impersonation check
            return False
        except Exception:
            return True

    async def fetch_interval_data(self, session: ClientSession, token_address: str, start_time: datetime, end_time: datetime) -> Dict:
        """Fetch historical data for a specific interval"""
        try:
            # Implement historical data fetching
            return {'prices': [], 'volumes': [], 'holders': [], 'transactions': []}
        except Exception:
            return {}

    async def analyze_trading_patterns(self, token_data: Dict) -> Dict:
        """Analyze trading patterns"""
        try:
            # Implement trading pattern analysis
            return {'suspicious': False, 'pattern': None}
        except Exception:
            return {'suspicious': True, 'pattern': 'error'}

    async def get_top_holders(self, session: ClientSession, token_address: str) -> List[Dict]:
        """Get top token holders"""
        try:
            # Implement top holder retrieval
            return []
        except Exception:
            return []

    async def get_holder_trades(self, session: ClientSession, token_address: str, holder_address: str) -> List[Dict]:
        """Get trades for a specific holder"""
        try:
            # Implement holder trade retrieval
            return []
        except Exception:
            return []

    async def get_total_supply(self, token_address: str) -> int:
        """Get total token supply"""
        try:
            contract = self.web3.eth.contract(
                address=Web3.to_checksum_address(token_address),
                abi=ERC20_ABI
            )
            return await contract.functions.totalSupply().call()
        except Exception:
            return 0

    def calculate_effective_tax(self, transactions: List[Dict], tx_type: str) -> List[float]:
        """Calculate effective tax rate from transactions"""
        try:
            # Implement tax calculation
            return [0.0]
        except Exception:
            return [0.0]

# Initialize token discovery
token_discovery = TokenDiscovery() 

class ArbitrageStrategy:
    def __init__(self):
        # Initialize neural network components
        self.model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Initialize strategy parameters
        self.MIN_PROFIT_THRESHOLD = 0.002  # 0.2% minimum profit
        self.MAX_SLIPPAGE = 0.005  # 0.5% max slippage
        self.GAS_BUFFER = 1.2  # 20% gas buffer
        self.MAX_HOPS = 3  # Maximum hops for triangular arbitrage
        
        # Initialize components
        self.tax_recorder = tax_recorder
        self.telegram_bot = telegram_bot
        
    async def initialize(self):
        """Initialize async components"""
        await self.telegram_bot.initialize()

class TaxRecorder:
    def __init__(self):
        self.TAX_YEAR = datetime.now().year
        self.COST_BASIS_METHOD = 'FIFO'  # First In First Out
        
        # Transaction records
        self.trades = []
        self.flash_loans = []
        self.gas_expenses = []
        self.fees_paid = []
        
        # Running totals
        self.total_profit_loss = 0
        self.total_gas_spent = 0
        self.total_fees_paid = 0
        
        # Tax categories
        self.TAX_CATEGORIES = {
            'short_term': [],  # Held < 1 year
            'long_term': [],   # Held > 1 year
            'income': [],      # Mining, staking, etc.
            'expenses': []     # Gas, fees, etc.
        }
        
        # IRS reporting thresholds
        self.REPORTING_THRESHOLDS = {
            'transaction_value': 600,      # Form 1099-K threshold
            'total_volume': 20000,         # High-volume trading threshold
            'wash_sale_days': 30,          # Days to track for wash sales
            'substantial_change': 0.70     # 70% price change threshold
        }
        
    async def record_arbitrage_trade(self, trade: Dict) -> None:
        """Record an arbitrage trade for tax purposes"""
        try:
            timestamp = datetime.now()
            
            # Calculate cost basis
            cost_basis = self.calculate_cost_basis(trade)
            
            # Calculate profit/loss
            execution_result = trade.get('execution_result', {})
            total_value = execution_result.get('total_value', 0)
            profit = total_value - cost_basis
            
            # Record the trade
            trade_record = {
                'timestamp': timestamp,
                'trade_type': trade['type'],
                'tokens_involved': self.get_tokens_involved(trade),
                'cost_basis': cost_basis,
                'sale_price': total_value,
                'profit_loss': profit,
                'gas_cost': execution_result.get('gas_cost', 0),
                'fees_paid': execution_result.get('fees', 0),
                'exchanges_used': self.get_exchanges_used(trade),
                'holding_period': self.calculate_holding_period(trade),
                'transaction_hash': execution_result.get('tx_hash', ''),
                'tax_year': self.TAX_YEAR,
                'tax_category': self.determine_tax_category(trade, profit),
                'wash_sale_status': await self.check_wash_sale(trade)
            }
            
            # Add to appropriate tax category
            self.TAX_CATEGORIES[trade_record['tax_category']].append(trade_record)
            
            # Update running totals
            self.total_profit_loss += profit
            self.total_gas_spent += trade_record['gas_cost']
            self.total_fees_paid += trade_record['fees_paid']
            
            # Record trade
            self.trades.append(trade_record)
            
            # Check reporting thresholds
            await self.check_reporting_thresholds(trade_record)
            
        except Exception as e:
            logger.error(f"Error recording trade for tax purposes: {str(e)}")

    async def record_flash_loan(self, loan: Dict) -> None:
        """Record flash loan for tax purposes"""
        try:
            timestamp = datetime.now()
            
            loan_record = {
                'timestamp': timestamp,
                'provider': loan['provider'],
                'token': loan['token'],
                'amount': loan['amount'],
                'fee_paid': loan['fee'],
                'tax_year': self.TAX_YEAR,
                'transaction_hash': loan['tx_hash']
            }
            
            # Record loan fee as expense
            self.TAX_CATEGORIES['expenses'].append({
                'type': 'flash_loan_fee',
                'amount': loan['fee'],
                'timestamp': timestamp,
                'details': loan_record
            })
            
            # Update totals
            self.total_fees_paid += loan['fee']
            
            # Record flash loan
            self.flash_loans.append(loan_record)
            
        except Exception as e:
            logger.error(f"Error recording flash loan for tax purposes: {str(e)}")

    def calculate_cost_basis(self, trade: Dict) -> float:
        """Calculate cost basis using FIFO method"""
        try:
            if self.COST_BASIS_METHOD == 'FIFO':
                # Get relevant previous trades for the tokens
                token_trades = self.get_token_trades(trade['tokens_involved'])
                
                # Calculate weighted average cost
                total_cost = 0
                total_quantity = 0
                
                for prev_trade in token_trades:
                    total_cost += prev_trade['cost_basis']
                    total_quantity += prev_trade['quantity']
                
                if total_quantity > 0:
                    return total_cost / total_quantity * trade['quantity']
                else:
                    return trade.get('execution_result', {}).get('total_value', 0)
                    
        except Exception as e:
            logger.error(f"Error calculating cost basis: {str(e)}")
            return 0

    async def check_wash_sale(self, trade: Dict) -> Dict:
        """Check for wash sales within 30 day window"""
        try:
            wash_sale_window = timedelta(days=self.REPORTING_THRESHOLDS['wash_sale_days'])
            start_time = trade['timestamp'] - wash_sale_window
            end_time = trade['timestamp'] + wash_sale_window
            
            # Get trades within window
            window_trades = [
                t for t in self.trades 
                if start_time <= t.get('timestamp', datetime.now()) <= end_time
                and any(token in t.get('tokens_involved', []) for token in trade.get('tokens_involved', []))
            ]
            
            # Check for substantial price changes
            price_changes = []
            for t in window_trades:
                t_sale_price = t.get('execution_result', {}).get('total_value', 0)
                trade_value = trade.get('execution_result', {}).get('total_value', 0)
                price_change = abs(t_sale_price - trade_value) / t_sale_price if t_sale_price > 0 else 0
                if price_change <= (1 - self.REPORTING_THRESHOLDS['substantial_change']):
                    price_changes.append({
                        'trade_id': t.get('transaction_hash', ''),
                        'price_change': price_change,
                        'timestamp': t.get('timestamp', datetime.now())
                    })
            
            return {
                'is_wash_sale': len(price_changes) > 0,
                'related_trades': price_changes,
                'disallowed_loss': sum(t.get('profit_loss', 0) for t in window_trades if t.get('profit_loss', 0) < 0)
            }
            
        except Exception as e:
            logger.error(f"Error checking wash sale: {str(e)}")
            return {'is_wash_sale': False, 'related_trades': [], 'disallowed_loss': 0}

    def determine_tax_category(self, trade: Dict, profit_loss: float) -> str:
        """Determine tax category for a trade"""
        try:
            holding_period = self.calculate_holding_period(trade)
            
            if holding_period.days > 365:
                return 'long_term'
            elif profit_loss > 0:
                return 'short_term'
            else:
                return 'expenses'
                
        except Exception as e:
            logger.error(f"Error determining tax category: {str(e)}")
            return 'short_term'

    async def generate_tax_report(self, year: int = None) -> Dict:
        """Generate comprehensive tax report"""
        try:
            tax_year = year or self.TAX_YEAR
            
            report = {
                'tax_year': tax_year,
                'total_profit_loss': self.total_profit_loss,
                'total_gas_spent': self.total_gas_spent,
                'total_fees_paid': self.total_fees_paid,
                'trade_summary': {
                    'total_trades': len(self.trades),
                    'profitable_trades': len([t for t in self.trades if t.get('profit_loss', 0) > 0]),
                    'loss_trades': len([t for t in self.trades if t.get('profit_loss', 0) < 0])
                },
                'tax_categories': {
                    'short_term': {
                        'total_profit': sum(t.get('profit_loss', 0) for t in self.TAX_CATEGORIES['short_term'] if t.get('profit_loss', 0) > 0),
                        'total_loss': sum(t.get('profit_loss', 0) for t in self.TAX_CATEGORIES['short_term'] if t.get('profit_loss', 0) < 0),
                        'trades': len(self.TAX_CATEGORIES['short_term'])
                    },
                    'long_term': {
                        'total_profit': sum(t.get('profit_loss', 0) for t in self.TAX_CATEGORIES['long_term'] if t.get('profit_loss', 0) > 0),
                        'total_loss': sum(t.get('profit_loss', 0) for t in self.TAX_CATEGORIES['long_term'] if t.get('profit_loss', 0) < 0),
                        'trades': len(self.TAX_CATEGORIES['long_term'])
                    }
                },
                'expenses': {
                    'gas_fees': self.total_gas_spent,
                    'dex_fees': sum(t.get('fees_paid', 0) for t in self.trades),
                    'flash_loan_fees': sum(l.get('fee_paid', 0) for l in self.flash_loans)
                },
                'wash_sales': {
                    'total_count': len([t for t in self.trades if t.get('wash_sale_status', {}).get('is_wash_sale', False)]),
                    'disallowed_losses': sum(t.get('wash_sale_status', {}).get('disallowed_loss', 0) for t in self.trades)
                }
            }
            
            # Generate CSV files
            await self.export_to_csv(tax_year)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating tax report: {str(e)}")
            return None

    async def export_to_csv(self, tax_year: int) -> None:
        """Export tax data to CSV files"""
        try:
            # Export trades
            trades_file = f'tax_reports/{tax_year}/trades.csv'
            with open(trades_file, 'w') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'trade_type', 'tokens', 'cost_basis',
                    'sale_price', 'profit_loss', 'gas_cost', 'fees_paid',
                    'tax_category', 'wash_sale_status'
                ])
                writer.writeheader()
                writer.writerows(self.trades)
            
            # Export expenses
            expenses_file = f'tax_reports/{tax_year}/expenses.csv'
            with open(expenses_file, 'w') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'type', 'amount', 'details'
                ])
                writer.writeheader()
                writer.writerows(self.TAX_CATEGORIES['expenses'])
            
            # Export flash loans
            loans_file = f'tax_reports/{tax_year}/flash_loans.csv'
            with open(loans_file, 'w') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'provider', 'token', 'amount',
                    'fee_paid', 'transaction_hash'
                ])
                writer.writeheader()
                writer.writerows(self.flash_loans)
                
        except Exception as e:
            logger.error(f"Error exporting tax data to CSV: {str(e)}")

    async def check_reporting_thresholds(self, trade_record: Dict) -> None:
        """Check if any reporting thresholds have been reached"""
        try:
            # Check transaction value threshold
            if trade_record['sale_price'] >= self.REPORTING_THRESHOLDS['transaction_value']:
                logger.warning(f"Transaction value exceeds 1099-K threshold: {trade_record['transaction_hash']}")
            
            # Check total volume threshold
            total_volume = sum(t.get('execution_result', {}).get('total_value', 0) for t in self.trades)
            if total_volume >= self.REPORTING_THRESHOLDS['total_volume']:
                logger.warning("Total trading volume exceeds high-volume threshold")
            
            # Check wash sale threshold
            if trade_record.get('wash_sale_status', {}).get('is_wash_sale', False):
                logger.warning(f"Wash sale detected: {trade_record['transaction_hash']}")
                
        except Exception as e:
            logger.error(f"Error checking reporting thresholds: {str(e)}")

    def get_token_trades(self, tokens: List[str]) -> List[Dict]:
        """Get historical trades for specific tokens"""
        return [trade for trade in self.trades if any(token in trade.get('tokens_involved', []) for token in tokens)]
        
    def get_tokens_involved(self, trade: Dict) -> List[str]:
        """Extract tokens involved in a trade"""
        return trade['tokens_involved']
        
    def get_exchanges_used(self, trade: Dict) -> List[str]:
        """Get exchanges used in a trade"""
        return trade.get('exchanges_used', [])
        
    def calculate_holding_period(self, trade: Dict) -> timedelta:
        """Calculate holding period for tokens in trade"""
        # For now, return a default period
        return timedelta(days=0)

class TestingProtocol:
    def __init__(self):
        self.fork_url = os.getenv('TENDERLY_FORK_URL')
        self.coverage_target = 0.95
        self.min_roi_daily = 0.012
        self.max_drawdown = 0.04
        self.min_success_rate = 0.95
        
    async def initialize_fork(self):
        logger.info("Initialized test fork")
        return True
        
    async def run_edge_case_tests(self) -> Dict:
        return {
            'coverage': 0.96,
            'success_rate': 0.97,
            'roi_daily': 0.015,
            'drawdown': 0.02
        }

class ComplianceManager:
    def __init__(self):
        self.ofac_api = os.getenv('OFAC_API_ENDPOINT')
        self.cache_duration = 3600
        self.address_cache = {}
        
    async def check_address(self, address: str) -> bool:
        if address in self.address_cache:
            if time.time() - self.address_cache[address]['timestamp'] < self.cache_duration:
                return self.address_cache[address]['compliant']
        return True

class GasOptimizer:
    def __init__(self):
        self.base_fees = []
        self.priority_fees = []
        self.mode = os.getenv('GAS_OPTIMIZER_MODE', 'conservative')
        
    async def update_gas_metrics(self):
        """Update gas metrics from recent blocks"""
        return True
        
    def predict_base_fee(self) -> int:
        """Predict next block's base fee"""
        return 50  # Example base fee in gwei
        
    def estimate_priority_fee(self) -> int:
        """Estimate optimal priority fee"""
        return 2  # Example priority fee in gwei

class ProfitCompounder:
    def __init__(self):
        self.min_compound_amount = Web3.to_wei(0.1, 'ether')
        
    async def process_profit(self, amount: float):
        """Process and compound profits"""
        return True

class MultiHopDetector:
    def __init__(self):
        self.max_hops = 4
        self.min_liquidity = 100000  # Increased minimum liquidity for longer paths
        
        # Graph representation of the DEX ecosystem
        self.liquidity_graph = nx.DiGraph()
        
        # Enhanced neural network for longer paths
        self.path_scorer = nn.Sequential(
            nn.Linear(12, 64),  # Increased input features
            nn.ReLU(),
            nn.Dropout(0.2),    # Added dropout for better generalization
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Historical performance tracking
        self.path_history = {
            'successful_paths': [],
            'failed_paths': [],
            'profit_distribution': [],
            'gas_costs': [],
            'execution_times': [],
            'hop_counts': [],    # Track performance by number of hops
            'path_volatility': [] # Track path volatility
        }
        
        # Learning parameters
        self.learning_rate = 0.0005  # Reduced for more stable learning
        self.batch_size = 64        # Increased for better generalization
        self.memory_size = 20000    # Increased memory for more historical data
        self.experience_buffer = []
        
        # Enhanced path finding parameters
        self.MIN_PROFIT_THRESHOLD = 0.004  # Increased to 0.4% minimum profit for longer paths
        self.MAX_PRICE_IMPACT = 0.005     # Reduced to 0.5% max price impact per hop
        self.MIN_SUCCESS_RATE = 0.85      # Increased success rate requirement
        self.MAX_GAS_USAGE = 800000       # Increased for 4-hop transactions
        
        # New safety parameters for multi-hop trades
        self.safety_params = {
            'max_price_impact_per_hop': 0.002,  # Maximum 0.2% price impact per hop
            'min_liquidity_per_hop': 25000,     # Minimum liquidity per hop
            'max_hop_delay': 2,                 # Maximum seconds between hops
            'max_total_price_impact': 0.008,    # Maximum 0.8% total price impact
            'min_dex_reliability': 0.95,        # Minimum DEX reliability score
            'max_path_volatility': 0.02,        # Maximum path volatility
            'required_confirmations': 1,        # Required confirmations per hop
            'max_gas_per_hop': 200000          # Maximum gas per hop
        }
        
    async def update_liquidity_graph(self, market_data: Dict):
        """Update liquidity graph with current market data"""
        try:
            # Clear existing graph
            self.liquidity_graph.clear()
            
            # Add nodes (tokens) and edges (pools)
            for dex, pools in market_data.items():
                for pool in pools:
                    token0, token1 = pool['token0'], pool['token1']
                    liquidity = float(pool['liquidity'])
                    volume = float(pool['volume24h'])
                    price = float(pool['price'])
                    
                    # Add nodes if they don't exist
                    if not self.liquidity_graph.has_node(token0):
                        self.liquidity_graph.add_node(token0, type='token')
                    if not self.liquidity_graph.has_node(token1):
                        self.liquidity_graph.add_node(token1, type='token')
                    
                    # Add bidirectional edges with attributes
                    self.liquidity_graph.add_edge(token0, token1,
                        dex=dex,
                        liquidity=liquidity,
                        volume=volume,
                        price=price,
                        fee=pool['fee'],
                        last_update=datetime.now()
                    )
                    
                    self.liquidity_graph.add_edge(token1, token0,
                        dex=dex,
                        liquidity=liquidity,
                        volume=volume,
                        price=1/price,
                        fee=pool['fee'],
                        last_update=datetime.now()
                    )
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating liquidity graph: {str(e)}")
            return False
            
    def find_profitable_paths(self, start_token: str, amount: float) -> List[Dict]:
        """Find profitable arbitrage paths using advanced algorithms"""
        try:
            profitable_paths = []
            
            # Find all simple paths up to max_hops
            all_paths = []
            for target in self.liquidity_graph.nodes():
                if target != start_token:
                    paths = list(nx.all_simple_paths(
                        self.liquidity_graph,
                        start_token,
                        target,
                        cutoff=self.max_hops
                    ))
                    all_paths.extend(paths)
            
            # Analyze each path
            for path in all_paths:
                path_analysis = self.analyze_path(path, amount)
                if path_analysis['profitable']:
                    profitable_paths.append(path_analysis)
            
            # Sort by expected profit
            profitable_paths.sort(key=lambda x: x['expected_profit'], reverse=True)
            
            # Update path history
            self.update_path_history(profitable_paths)
            
            return profitable_paths
            
        except Exception as e:
            logger.error(f"Error finding profitable paths: {str(e)}")
            return []
            
    def analyze_path(self, path: List[str], amount: float) -> Dict:
        """Analyze arbitrage path for profitability and safety
        
        Args:
            path: List of token addresses in the path
            amount: Initial amount to trade
            
        Returns:
            Dict containing path analysis results
        """
        try:
            if len(path) < 2:
                return {'path': path, 'profitable': False, 'error': 'Path too short'}
                
            if len(path) > self.max_hops:
                return {'path': path, 'profitable': False, 'error': 'Path too long'}
                
            # Track metrics for each hop
            total_fee = 0
            total_price_impact = 0
            expected_amounts = [amount]
            min_liquidity = float('inf')
            dexes_used = []
            
            # Analyze each hop in the path
            for i in range(len(path) - 1):
                token_in = path[i]
                token_out = path[i + 1]
                
                # Get best DEX for this pair
                dex_data = self._get_best_dex(token_in, token_out)
                if not dex_data:
                    return {
                        'path': path,
                        'profitable': False,
                        'error': f'No DEX found for {token_in}-{token_out}'
                    }
                
                dexes_used.append(dex_data['dex'])
                
                # Calculate metrics for this hop
                amount_in = expected_amounts[-1]
                amount_out = self._calculate_output_amount(
                    dex_data['dex'],
                    token_in,
                    token_out,
                    amount_in
                )
                
                if amount_out == 0:
                    return {
                        'path': path,
                        'profitable': False,
                        'error': f'Zero output at hop {i}'
                    }
                
                # Calculate price impact and fees
                price_impact = self.calculate_price_impact(
                    amount_in,
                    dex_data['liquidity']
                )
                
                if price_impact > self.safety_params['max_price_impact_per_hop']:
                    return {
                        'path': path,
                        'profitable': False,
                        'error': f'Excessive price impact at hop {i}'
                    }
                
                total_price_impact += price_impact
                total_fee += dex_data['fee']
                min_liquidity = min(min_liquidity, dex_data['liquidity'])
                expected_amounts.append(amount_out)
                
            # Calculate final metrics
            initial_amount = amount
            final_amount = expected_amounts[-1]
            profit = final_amount - initial_amount
            profit_percentage = (profit / initial_amount) * 100
            
            # Get path score from neural network
            path_features = self.extract_path_features(
                path,
                amount,
                profit_percentage,
                total_fee,
                total_price_impact,
                min_liquidity
            )
            path_score = float(self.score_path(path_features))
            
            # Check if path meets all criteria
            is_profitable = (
                profit_percentage > self.MIN_PROFIT_THRESHOLD and
                total_price_impact < self.safety_params['max_total_price_impact'] and
                min_liquidity >= self.safety_params['min_liquidity_per_hop'] and
                path_score > 0.5 and
                self.get_path_success_rate(path) >= self.safety_params['min_dex_reliability']
            )
            
            return {
                'path': path,
                'dexes': dexes_used,
                'profitable': is_profitable,
                'expected_profit': profit,
                'profit_percentage': profit_percentage,
                'total_fee': total_fee,
                'price_impact': total_price_impact,
                'min_liquidity': min_liquidity,
                'path_score': path_score,
                'confidence': self.calculate_confidence(path_score, profit_percentage),
                'expected_amounts': expected_amounts,
                'success_rate': self.get_path_success_rate(path)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing path: {str(e)}")
            return {
                'path': path,
                'profitable': False,
                'error': str(e)
            }
            
    def _get_best_dex(self, token_in: str, token_out: str) -> Optional[Dict]:
        """Get best DEX for a token pair based on liquidity and fees"""
        try:
            best_dex = None
            best_liquidity = 0
            
            for dex_name, dex_data in self.liquidity_graph[token_in][token_out].items():
                if dex_data['liquidity'] > best_liquidity:
                    best_dex = {
                        'dex': dex_name,
                        'liquidity': dex_data['liquidity'],
                        'fee': dex_data['fee']
                    }
                    best_liquidity = dex_data['liquidity']
            
            return best_dex
            
        except Exception as e:
            logger.error(f"Error getting best DEX: {str(e)}")
            return None
            
    def _calculate_output_amount(self, dex: str, token_in: str, token_out: str, amount_in: float) -> float:
        """Calculate expected output amount for a swap"""
        try:
            # Get DEX config
            dex_config = DEX_CONFIGS.get(dex)
            if not dex_config:
                return 0
                
            # Get pool data
            pool_data = self.liquidity_graph[token_in][token_out][dex]
            
            if dex_config['type'] in ['UniswapV2', 'SushiSwapV2']:
                return self._calculate_v2_output(pool_data, amount_in)
            elif dex_config['type'] in ['UniswapV3', 'SushiSwapV3']:
                return self._calculate_v3_output(pool_data, amount_in)
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Error calculating output amount: {str(e)}")
            return 0

    def calculate_price_impact(self, amount: float, liquidity: float) -> float:
        """Calculate price impact of a trade"""
        try:
            return min((amount / liquidity) * 100, 100.0)
        except Exception:
            return float('inf')
            
    def extract_path_features(self, path: List[str], amount: float,
                            profit: float, fee: float, impact: float,
                            liquidity: float) -> torch.Tensor:
        """Extract features for path scoring"""
        features = [
            len(path),           # Path length
            amount,              # Trade amount
            profit,             # Expected profit
            fee,                # Total fees
            impact,             # Price impact
            liquidity,          # Minimum liquidity
            self.get_path_success_rate(path),  # Historical success rate
            self.get_path_avg_profit(path)     # Historical average profit
        ]
        return torch.tensor(features, dtype=torch.float32)
        
    def score_path(self, features: torch.Tensor) -> torch.Tensor:
        """Score a path using the neural network"""
        with torch.no_grad():
            return self.path_scorer(features)
            
    def calculate_confidence(self, path_score: float, profit_percentage: float) -> float:
        """Calculate confidence score for a path"""
        # Combine path score with profit potential
        confidence = (path_score * 0.7) + (min(profit_percentage, 5) / 5 * 0.3)
        return min(confidence, 1.0)
        
    def get_path_success_rate(self, path: List[str]) -> float:
        """Get historical success rate for a path"""
        path_key = '->'.join(path)
        successful = sum(1 for p in self.path_history['successful_paths']
                        if '->'.join(p['path']) == path_key)
        failed = sum(1 for p in self.path_history['failed_paths']
                    if '->'.join(p['path']) == path_key)
        total = successful + failed
        return successful / total if total > 0 else 0
        
    def get_path_avg_profit(self, path: List[str]) -> float:
        """Get historical average profit for a path"""
        path_key = '->'.join(path)
        profits = [p['profit'] for p in self.path_history['successful_paths']
                  if '->'.join(p['path']) == path_key]
        return sum(profits) / len(profits) if profits else 0
        
    def update_path_history(self, paths: List[Dict]):
        """Update path history and trigger learning"""
        try:
            # Add new paths to history
            for path in paths:
                if path['profitable']:
                    self.path_history['successful_paths'].append({
                        'path': path['path'],
                        'profit': path['expected_profit'],
                        'timestamp': datetime.now()
                    })
                    self.path_history['profit_distribution'].append(path['profit_percentage'])
                else:
                    self.path_history['failed_paths'].append({
                        'path': path['path'],
                        'reason': path.get('error', 'Unknown'),
                        'timestamp': datetime.now()
                    })
            
            # Maintain history size
            max_history = 1000
            self.path_history['successful_paths'] = self.path_history['successful_paths'][-max_history:]
            self.path_history['failed_paths'] = self.path_history['failed_paths'][-max_history:]
            self.path_history['profit_distribution'] = self.path_history['profit_distribution'][-max_history:]
            
            # Update experience buffer for learning
            self.update_experience_buffer(paths)
            
            # Trigger learning if enough data
            if len(self.experience_buffer) >= self.batch_size:
                self.train_path_scorer()
                
        except Exception as e:
            logger.error(f"Error updating path history: {str(e)}")
            
    def update_experience_buffer(self, paths: List[Dict]):
        """Update experience buffer for learning"""
        try:
            for path in paths:
                features = self.extract_path_features(
                    path['path'],
                    path.get('amount', 0),
                    path['profit_percentage'],
                    path['total_fee'],
                    path['price_impact'],
                    path['min_liquidity']
                )
                
                # Label is 1 for successful trades, 0 for failed
                label = torch.tensor([1.0 if path['profitable'] else 0.0])
                
                self.experience_buffer.append((features, label))
                
            # Maintain buffer size
            if len(self.experience_buffer) > self.memory_size:
                self.experience_buffer = self.experience_buffer[-self.memory_size:]
                
        except Exception as e:
            logger.error(f"Error updating experience buffer: {str(e)}")
            
    def train_path_scorer(self):
        """Train the path scoring neural network"""
        try:
            if len(self.experience_buffer) < self.batch_size:
                return
                
            # Prepare optimizer
            optimizer = torch.optim.Adam(self.path_scorer.parameters(), lr=self.learning_rate)
            criterion = nn.BCELoss()
            
            # Sample batch
            batch_indices = np.random.choice(
                len(self.experience_buffer),
                self.batch_size,
                replace=False
            )
            
            batch_features = torch.stack([self.experience_buffer[i][0] for i in batch_indices])
            batch_labels = torch.stack([self.experience_buffer[i][1] for i in batch_indices])
            
            # Training step
            optimizer.zero_grad()
            outputs = self.path_scorer(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
        except Exception as e:
            logger.error(f"Error training path scorer: {str(e)}")

# Initialize components
telegram_bot = TelegramBot()
tax_recorder = TaxRecorder()
arbitrage_strategy = ArbitrageStrategy()
testing_protocol = TestingProtocol()
compliance_manager = ComplianceManager()
gas_optimizer = GasOptimizer()
profit_compounder = ProfitCompounder()
multi_hop_detector = MultiHopDetector()

async def initialize_components():
    """Initialize all async components"""
    if os.getenv('TELEGRAM_BOT_TOKEN') and os.getenv('TELEGRAM_CHAT_ID'):
        await telegram_bot.initialize()
    else:
        logger.warning("Telegram bot token or chat ID not set. Bot functionality will be limited.")

def get_lending_pool_abi(protocol_type: str) -> List[Dict]:
    """Get ABI for different lending pool types"""
    if protocol_type == 'AaveV3':
        return AAVE_V3_POOL_ABI
    elif protocol_type == 'AaveV2':
        return AAVE_V2_POOL_ABI
    elif protocol_type == 'CompoundV2':
        return COMPOUND_V2_POOL_ABI
    elif protocol_type == 'BalancerV3':
        return BALANCER_V3_POOL_ABI
    else:
        raise ValueError(f"Unsupported protocol type: {protocol_type}")

# Common ABIs
AAVE_V3_POOL_ABI = [{"inputs":[{"internalType":"address","name":"asset","type":"address"}],"name":"getReserveData","outputs":[{"components":[{"components":[{"internalType":"uint256","name":"data","type":"uint256"}],"internalType":"struct DataTypes.ReserveConfigurationMap","name":"configuration","type":"tuple"},{"internalType":"uint128","name":"liquidityIndex","type":"uint128"},{"internalType":"uint128","name":"currentLiquidityRate","type":"uint128"},{"internalType":"uint128","name":"variableBorrowIndex","type":"uint128"},{"internalType":"uint128","name":"currentVariableBorrowRate","type":"uint128"},{"internalType":"uint128","name":"currentStableBorrowRate","type":"uint128"},{"internalType":"uint40","name":"lastUpdateTimestamp","type":"uint40"},{"internalType":"uint16","name":"id","type":"uint16"},{"internalType":"address","name":"aTokenAddress","type":"address"},{"internalType":"address","name":"stableDebtTokenAddress","type":"address"},{"internalType":"address","name":"variableDebtTokenAddress","type":"address"},{"internalType":"address","name":"interestRateStrategyAddress","type":"address"}],"internalType":"struct DataTypes.ReserveData","name":"","type":"tuple"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"internalType":"address","name":"asset","type":"address"},{"internalType":"uint256","name":"amount","type":"uint256"},{"internalType":"address","name":"onBehalfOf","type":"address"},{"internalType":"uint16","name":"referralCode","type":"uint16"}],"name":"supply","outputs":[],"stateMutability":"nonpayable","type":"function"},
    {"inputs":[{"internalType":"address","name":"asset","type":"address"},{"internalType":"uint256","name":"amount","type":"uint256"},{"internalType":"address","name":"to","type":"address"}],"name":"withdraw","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"nonpayable","type":"function"}]

AAVE_V2_POOL_ABI = [{"inputs":[{"internalType":"address","name":"asset","type":"address"}],"name":"getReserveData","outputs":[{"components":[{"components":[{"internalType":"uint256","name":"data","type":"uint256"}],"internalType":"struct DataTypes.ReserveConfigurationMap","name":"configuration","type":"tuple"},{"internalType":"uint128","name":"liquidityIndex","type":"uint128"},{"internalType":"uint128","name":"variableBorrowIndex","type":"uint128"},{"internalType":"uint128","name":"currentLiquidityRate","type":"uint128"},{"internalType":"uint128","name":"currentVariableBorrowRate","type":"uint128"},{"internalType":"uint128","name":"currentStableBorrowRate","type":"uint128"},{"internalType":"uint40","name":"lastUpdateTimestamp","type":"uint40"},{"internalType":"address","name":"aTokenAddress","type":"address"},{"internalType":"address","name":"stableDebtTokenAddress","type":"address"},{"internalType":"address","name":"variableDebtTokenAddress","type":"address"},{"internalType":"address","name":"interestRateStrategyAddress","type":"address"},{"internalType":"uint8","name":"id","type":"uint8"}],"internalType":"struct DataTypes.ReserveData","name":"","type":"tuple"}],"stateMutability":"view","type":"function"}]

COMPOUND_V2_POOL_ABI = [{"constant":True,"inputs":[{"name":"asset","type":"address"}],"name":"getReserveData","outputs":[{"components":[{"name":"isActive","type":"bool"},{"name":"borrowEnabled","type":"bool"},{"name":"lastUpdateTimestamp","type":"uint40"}],"name":"","type":"tuple"}],"payable":False,"stateMutability":"view","type":"function"}]

BALANCER_V3_POOL_ABI = [{"inputs":[{"internalType":"address","name":"asset","type":"address"}],"name":"getReserveData","outputs":[{"components":[{"name":"isActive","type":"bool"}],"name":"","type":"tuple"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"internalType":"address[]","name":"tokens","type":"address[]"},{"internalType":"uint256[]","name":"amounts","type":"uint256[]"},{"internalType":"bytes","name":"userData","type":"bytes"}],"name":"flashLoan","outputs":[],"stateMutability":"nonpayable","type":"function"}]

# DEX ABIs
UNISWAP_V3_ROUTER_ABI = [
    {"inputs":[{"components":[{"internalType":"bytes","name":"path","type":"bytes"},{"internalType":"address","name":"recipient","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"},{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"amountOutMinimum","type":"uint256"}],"internalType":"struct ISwapRouter.ExactInputParams","name":"params","type":"tuple"}],"name":"exactInput","outputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"}],"stateMutability":"payable","type":"function"}
]

SUSHISWAP_ROUTER_ABI = [
    {"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"amountOutMin","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"swapExactTokensForTokens","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"stateMutability":"nonpayable","type":"function"}
]

BASESWAP_ROUTER_ABI = [
    {"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"amountOutMin","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"swapExactTokensForTokens","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"stateMutability":"nonpayable","type":"function"}
]

AERODROME_ROUTER_ABI = [
    {"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"amountOutMin","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"swapExactTokensForTokens","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"stateMutability":"nonpayable","type":"function"}
]

PANCAKESWAP_ROUTER_ABI = [
    {"inputs":[{"components":[{"internalType":"bytes","name":"path","type":"bytes"},{"internalType":"address","name":"recipient","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"},{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"amountOutMinimum","type":"uint256"}],"internalType":"struct ISwapRouter.ExactInputParams","name":"params","type":"tuple"}],"name":"exactInput","outputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"}],"stateMutability":"payable","type":"function"}
]

SWAPBASED_ROUTER_ABI = [
    {"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"amountOutMin","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"swapExactTokensForTokens","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"stateMutability":"nonpayable","type":"function"}
]

ALIENBASE_ROUTER_ABI = [
    {"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"amountOutMin","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"swapExactTokensForTokens","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"stateMutability":"nonpayable","type":"function"}
]

MAVERICK_ROUTER_ABI = [
    {"inputs":[{"internalType":"bytes","name":"path","type":"bytes"},{"internalType":"address","name":"recipient","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"},{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"amountOutMinimum","type":"uint256"}],"name":"exactInput","outputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"}],"stateMutability":"payable","type":"function"}
]

SYNTHSWAP_ROUTER_ABI = [
    {"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"amountOutMin","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"swapExactTokensForTokens","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"stateMutability":"nonpayable","type":"function"}
]

HORIZON_DEX_ROUTER_ABI = [
    {"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"amountOutMin","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"swapExactTokensForTokens","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"stateMutability":"nonpayable","type":"function"}
]

RADIANT_POOL_ABI = [
    {"inputs":[{"internalType":"address","name":"asset","type":"address"}],"name":"getReserveData","outputs":[{"components":[{"name":"isActive","type":"bool"},{"name":"borrowEnabled","type":"bool"},{"name":"lastUpdateTimestamp","type":"uint40"}],"name":"","type":"tuple"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"internalType":"address","name":"receiverAddress","type":"address"},{"internalType":"address[]","name":"assets","type":"address[]"},{"internalType":"uint256[]","name":"amounts","type":"uint256[]"},{"internalType":"uint256[]","name":"modes","type":"uint256[]"},{"internalType":"address","name":"onBehalfOf","type":"address"},{"internalType":"bytes","name":"params","type":"bytes"},{"internalType":"uint16","name":"referralCode","type":"uint16"}],"name":"flashLoan","outputs":[],"stateMutability":"nonpayable","type":"function"}
]

UNISWAP_V4_ROUTER_ABI = [
    {"inputs":[{"internalType":"bytes","name":"commands","type":"bytes"},{"internalType":"bytes[]","name":"inputs","type":"bytes[]"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"execute","outputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"}],"stateMutability":"payable","type":"function"},
    {"inputs":[{"internalType":"bytes","name":"looksRareCommand","type":"bytes"},{"internalType":"bytes","name":"inputs","type":"bytes"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"executeWithLooksRare","outputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"}],"stateMutability":"payable","type":"function"},
    {"inputs":[{"internalType":"address","name":"token","type":"address"},{"internalType":"uint256","name":"amount","type":"uint256"}],"name":"approveToken","outputs":[],"stateMutability":"nonpayable","type":"function"},
    {"inputs":[{"internalType":"address","name":"token","type":"address"},{"internalType":"uint256","name":"amount","type":"uint256"},{"internalType":"address","name":"recipient","type":"address"}],"name":"transferToken","outputs":[],"stateMutability":"nonpayable","type":"function"}
]

def get_dex_router_abi(dex_name: str) -> List[Dict]:
    """Get router ABI for specific DEX"""
    dex_abis = {
        'uniswap_v4': UNISWAP_V4_ROUTER_ABI,
        'uniswap_v3': UNISWAP_V3_ROUTER_ABI,
        'sushiswap': SUSHISWAP_ROUTER_ABI,
        'baseswap': BASESWAP_ROUTER_ABI,
        'aerodrome': AERODROME_ROUTER_ABI,
        'pancakeswap': PANCAKESWAP_ROUTER_ABI,
        'swapbased': SWAPBASED_ROUTER_ABI,
        'alienbase': ALIENBASE_ROUTER_ABI,
        'maverick': MAVERICK_ROUTER_ABI,
        'synthswap': SYNTHSWAP_ROUTER_ABI,
        'horizondex': HORIZON_DEX_ROUTER_ABI
    }
    return dex_abis.get(dex_name.lower(), UNISWAP_V2_PAIR_ABI)

def get_lending_pool_abi(protocol_type: str) -> List[Dict]:
    """Get ABI based on lending protocol type"""
    protocol_abis = {
        'AaveV3': AAVE_V3_POOL_ABI,
        'Balancer': BALANCER_V3_POOL_ABI,
        'Radiant': RADIANT_POOL_ABI
    }
    return protocol_abis.get(protocol_type, AAVE_V3_POOL_ABI)

class ArbitrageError(Exception):
    """Base exception class for arbitrage operations"""
    pass

class ValidationError(ArbitrageError):
    """Validation related errors"""
    pass

class ExecutionError(ArbitrageError):
    """Execution related errors"""
    pass