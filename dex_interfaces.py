import aiohttp
from typing import Dict, List
import json
import logging
from web3 import Web3
from superchain_utils import DEX_CONFIGS, LENDING_CONFIGS, TOKEN_ADDRESSES

logger = logging.getLogger(__name__)

# Uniswap V3 subgraph URL
UNISWAP_V3_SUBGRAPH = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"

# Uniswap V2 subgraph URL
UNISWAP_V2_SUBGRAPH = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v2"

# Base Chain Graph URLs
GRAPH_ENDPOINTS = {
    'aerodrome': 'https://api.thegraph.com/subgraphs/name/aerodrome-finance/aerodrome-v2-base',
    'baseswap': 'https://api.studio.thegraph.com/query/50526/baseswap-v2/v0.0.1',
    'alienbase': 'https://api.thegraph.com/subgraphs/name/alienbase/exchange'
}

async def fetch_uniswap_v3_data(session: aiohttp.ClientSession, pool_address: str) -> Dict:
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
    """ % pool_address.lower()

    try:
        async with session.post(UNISWAP_V3_SUBGRAPH, json={'query': query}) as response:
            data = await response.json()
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
            
    except Exception as e:
        logger.error(f"Error fetching Uniswap V3 data: {str(e)}")
        return None

async def fetch_uniswap_v2_data(session: aiohttp.ClientSession, pair_address: str) -> Dict:
    """Fetch market data from Uniswap V2"""
    query = """
    {
      pair(id: "%s") {
        token0Price
        token1Price
        reserveUSD
        volumeUSD
        reserve0
        reserve1
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
    """ % pair_address.lower()

    try:
        async with session.post(UNISWAP_V2_SUBGRAPH, json={'query': query}) as response:
            data = await response.json()
            pair = data['data']['pair']
            
            return {
                'price_impact': calculate_price_impact_v2(pair),
                'liquidity': float(pair['reserveUSD']),
                'volatility': calculate_volatility_v2(pair),
                'volume_24h': float(pair['volumeUSD']),
                'reserve0': float(pair['reserve0']),
                'reserve1': float(pair['reserve1'])
            }
            
    except Exception as e:
        logger.error(f"Error fetching Uniswap V2 data: {str(e)}")
        return None

def calculate_price_impact_v3(pool: Dict) -> float:
    """Calculate price impact for Uniswap V3 pool"""
    liquidity = float(pool['liquidity'])
    volume = float(pool['volumeUSD'])
    
    if volume == 0:
        return 1.0
        
    # Simplified price impact calculation
    return min(1.0, 1 / (liquidity / volume))

def calculate_price_impact_v2(pair: Dict) -> float:
    """Calculate price impact for Uniswap V2 pair"""
    reserve_usd = float(pair['reserveUSD'])
    volume = float(pair['volumeUSD'])
    
    if volume == 0:
        return 1.0
        
    # Simplified price impact calculation
    return min(1.0, 1 / (reserve_usd / volume))

def calculate_volatility_v3(pool: Dict) -> float:
    """Calculate volatility for Uniswap V3 pool"""
    # Implement volatility calculation using tick data
    tick = int(pool['tick'])
    sqrt_price = float(pool['sqrtPrice'])
    
    # Simplified volatility calculation
    return abs(tick) / 100000

def calculate_volatility_v2(pair: Dict) -> float:
    """Calculate volatility for Uniswap V2 pair"""
    # Implement volatility calculation using reserves
    reserve0 = float(pair['reserve0'])
    reserve1 = float(pair['reserve1'])
    
    # Simplified volatility calculation
    return abs(1 - (reserve0 / reserve1)) / 100 

async def fetch_dex_data(session: aiohttp.ClientSession, dex_name: str, dex_info: Dict) -> Dict:
    """Fetch market data from a specific DEX on Base"""
    if dex_name not in GRAPH_ENDPOINTS:
        logger.error(f"No subgraph endpoint for {dex_name}")
        return None

    # Query for DEX-specific data
    query = """
    {
      pairs(first: 100, orderBy: reserveUSD, orderDirection: desc) {
        id
        token0 {
          id
          symbol
          decimals
        }
        token1 {
          id
          symbol
          decimals
        }
        reserve0
        reserve1
        reserveUSD
        volumeUSD
        token0Price
        token1Price
      }
    }
    """

    try:
        async with session.post(GRAPH_ENDPOINTS[dex_name], json={'query': query}) as response:
            data = await response.json()
            pairs = data['data']['pairs']
            
            # Process pairs and return relevant data
            processed_pairs = {}
            for pair in pairs:
                key = f"{pair['token0']['symbol']}/{pair['token1']['symbol']}"
                processed_pairs[key] = {
                    'address': pair['id'],
                    'reserve0': float(pair['reserve0']),
                    'reserve1': float(pair['reserve1']),
                    'reserveUSD': float(pair['reserveUSD']),
                    'volumeUSD': float(pair['volumeUSD']),
                    'price0': float(pair['token0Price']),
                    'price1': float(pair['token1Price']),
                    'liquidity_score': calculate_liquidity_score(pair),
                    'volume_score': calculate_volume_score(pair)
                }
            
            return processed_pairs
            
    except Exception as e:
        logger.error(f"Error fetching {dex_name} data: {str(e)}")
        return None

async def fetch_lending_protocol_data(session: aiohttp.ClientSession, protocol_name: str) -> Dict:
    """Fetch data from lending protocols on Base"""
    protocol = LENDING_CONFIGS[protocol_name]
    
    try:
        if protocol['type'] == 'AaveV3':
            return await fetch_aave_v3_data(session, protocol['address'])
        elif protocol['type'] == 'CompoundV2':
            return await fetch_moonwell_data(session, protocol['address'])
        else:
            logger.error(f"Unsupported protocol type: {protocol['type']}")
            return None
            
    except Exception as e:
        logger.error(f"Error fetching lending protocol data: {str(e)}")
        return None

async def fetch_aave_v3_data(session: aiohttp.ClientSession, address: str) -> Dict:
    """Fetch Aave V3 protocol data on Base"""
    query = """
    {
      reserves(first: 100) {
        id
        symbol
        decimals
        liquidityRate
        variableBorrowRate
        totalATokenSupply
        availableLiquidity
        totalDeposits
        totalCurrentVariableDebt
      }
    }
    """
    
    try:
        async with session.post('https://api.thegraph.com/subgraphs/name/aave/protocol-v3-base', 
                              json={'query': query}) as response:
            data = await response.json()
            return process_lending_data(data['data']['reserves'], 'AaveV3')
    except Exception as e:
        logger.error(f"Error fetching Aave V3 data: {str(e)}")
        return None

async def fetch_moonwell_data(session: aiohttp.ClientSession, address: str) -> Dict:
    """Fetch Moonwell (Compound V2) protocol data on Base"""
    query = """
    {
      markets(first: 100) {
        id
        symbol
        supplyRate
        borrowRate
        totalSupply
        totalBorrows
        exchangeRate
        underlyingAddress
      }
    }
    """
    
    try:
        async with session.post('https://api.thegraph.com/subgraphs/name/moonwell-finance/moonwell-base', 
                              json={'query': query}) as response:
            data = await response.json()
            return process_lending_data(data['data']['markets'], 'CompoundV2')
    except Exception as e:
        logger.error(f"Error fetching Moonwell data: {str(e)}")
        return None

def calculate_liquidity_score(pair: Dict) -> float:
    """Calculate liquidity score for a trading pair"""
    reserve_usd = float(pair['reserveUSD'])
    # Score from 0 to 1 based on liquidity
    return min(1.0, reserve_usd / 1_000_000)  # Normalize to 1M USD

def calculate_volume_score(pair: Dict) -> float:
    """Calculate volume score for a trading pair"""
    volume_usd = float(pair['volumeUSD'])
    # Score from 0 to 1 based on 24h volume
    return min(1.0, volume_usd / 100_000)  # Normalize to 100k USD

def process_lending_data(reserves: List[Dict], protocol_type: str) -> Dict:
    """Process lending protocol data"""
    processed_data = {}
    
    for reserve in reserves:
        token_data = {
            'symbol': reserve['symbol'],
            'decimals': int(reserve['decimals']) if 'decimals' in reserve else 18,
            'liquidity': float(reserve.get('availableLiquidity', 0)),
            'total_supply': float(reserve.get('totalATokenSupply', reserve.get('totalSupply', 0))),
            'borrow_rate': float(reserve.get('variableBorrowRate', reserve.get('borrowRate', 0))),
            'supply_rate': float(reserve.get('liquidityRate', reserve.get('supplyRate', 0))),
            'utilization': calculate_utilization(reserve, protocol_type)
        }
        
        processed_data[reserve['symbol']] = token_data
        
    return processed_data

def calculate_utilization(reserve: Dict, protocol_type: str) -> float:
    """Calculate utilization rate for a lending market"""
    if protocol_type == 'AaveV3':
        total_debt = float(reserve['totalCurrentVariableDebt'])
        total_deposits = float(reserve['totalDeposits'])
    else:  # CompoundV2
        total_debt = float(reserve['totalBorrows'])
        total_supply = float(reserve['totalSupply'])
        exchange_rate = float(reserve['exchangeRate'])
        total_deposits = total_supply * exchange_rate
        
    if total_deposits == 0:
        return 0
        
    return min(1.0, total_debt / total_deposits) 