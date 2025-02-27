"""
Yield Farming Analysis Module

This module provides comprehensive analysis of yield farming opportunities
across multiple DeFi protocols, with intelligent optimization strategies
and risk assessment.
"""

from typing import Dict, List, Tuple, Optional, Any, Union, TypedDict
import logging
import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from decimal import Decimal
from web3 import Web3
from ..utils.cache import AsyncCache
from ..utils.metrics import MetricsManager
from ..integrations.defillama import DefiLlamaClient
from ..core.token_price import PriceProvider

logger = logging.getLogger(__name__)

@dataclass
class PoolData:
    """Yield farming pool details"""
    id: str
    name: str
    protocol_id: str
    protocol_name: str
    tokens: List[str]
    tvl: float
    apy: float
    rewards: List[str] = field(default_factory=list)
    rewards_tokens: List[Dict[str, Any]] = field(default_factory=list)
    fee_apy: float = 0.0
    rewards_apy: float = 0.0
    il_risk: float = 0.0
    last_updated: float = field(default_factory=time.time)
    pool_url: Optional[str] = None
    leverage_options: Optional[Dict[str, Any]] = None

@dataclass
class FarmingOpportunity:
    """Complete yield farming opportunity analysis"""
    pool: PoolData
    token_address: str
    token_share: float
    entry_cost: float
    exit_cost: float
    apy: float
    adjusted_apy: float
    projected_earnings: float
    adjusted_earnings: float
    il_risk: float
    risk_score: float
    time_horizon: int
    position_size: float
    leverage: float = 1.0
    compound_frequency: str = "daily"
    optimal_harvest_frequency: int = 7  # days
    strategy: Dict[str, Any] = field(default_factory=dict)
    
class YieldFarmingAnalyzer:
    """
    Advanced yield farming opportunity analyzer with:
    - Multi-protocol support
    - APY calculation with fee/reward breakdown
    - Impermanent loss simulation
    - Gas-optimized strategy generation
    - Compound frequency optimization
    - Risk assessment and scoring
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        cache: AsyncCache,
        metrics: MetricsManager,
        web3: Optional[Web3] = None
    ):
        self.config = config
        self.cache = cache
        self.metrics = metrics
        self.web3 = web3 or Web3()
        
        # Initialize clients
        self.defillama = DefiLlamaClient(config.get('defillama_url', 'https://yields.llama.fi/'))
        self.price_provider = PriceProvider(cache)
        
        # Initialize circuit breakers
        self.circuit_breakers = {
            'defillama': {'failures': 0, 'threshold': 5, 'open': False, 'last_attempt': 0},
            'protocol_api': {'failures': 0, 'threshold': 5, 'open': False, 'last_attempt': 0}
        }
        
        # Protocol-specific configs
        self.protocol_configs = self._load_protocol_configs()
        
        # Risk assessment parameters
        self.risk_params = {
            'tvl_weight': 0.3,
            'age_weight': 0.15,
            'audit_weight': 0.25,
            'complexity_weight': 0.1,
            'il_weight': 0.2,
            'min_tvl_threshold': 500000,  # $500k
            'max_apy_threshold': 1000.0,  # 1000% APY is suspicious
        }
        
    def _load_protocol_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load protocol-specific configurations"""
        # This would ideally load from a configuration file
        # For now, we'll hardcode some common protocols
        return {
            'aave': {
                'type': 'lending',
                'entry_gas': 180000,
                'exit_gas': 140000,
                'harvest_gas': 120000,
                'risk_score': 0.2,  # Lower is better
                'audited': True,
                'complexity': 'medium',
            },
            'compound': {
                'type': 'lending',
                'entry_gas': 160000,
                'exit_gas': 130000,
                'harvest_gas': 110000,
                'risk_score': 0.25,
                'audited': True,
                'complexity': 'medium',
            },
            'curve': {
                'type': 'dex',
                'entry_gas': 240000,
                'exit_gas': 190000,
                'harvest_gas': 150000,
                'risk_score': 0.3,
                'audited': True,
                'complexity': 'high',
            },
            'convex': {
                'type': 'yield_aggregator',
                'entry_gas': 280000,
                'exit_gas': 220000,
                'harvest_gas': 180000,
                'risk_score': 0.35,
                'audited': True,
                'complexity': 'high',
            },
            'yearn': {
                'type': 'yield_aggregator',
                'entry_gas': 220000,
                'exit_gas': 180000,
                'harvest_gas': 160000,
                'risk_score': 0.3,
                'audited': True,
                'complexity': 'high',
            },
            'sushiswap': {
                'type': 'dex',
                'entry_gas': 200000,
                'exit_gas': 160000,
                'harvest_gas': 140000,
                'risk_score': 0.4,
                'audited': True,
                'complexity': 'medium',
            },
            'uniswap': {
                'type': 'dex',
                'entry_gas': 180000,
                'exit_gas': 150000,
                'harvest_gas': 0,  # No harvesting in Uniswap
                'risk_score': 0.25,
                'audited': True,
                'complexity': 'medium',
            },
        }
        
    async def _call_with_circuit_breaker(self, service_name: str, call_fn, *args, **kwargs):
        """Call external API with circuit breaker pattern"""
        # Check if circuit is open (too many failures)
        breaker = self.circuit_breakers.get(service_name)
        if not breaker:
            logger.warning(f"No circuit breaker configured for {service_name}")
            return await call_fn(*args, **kwargs)
            
        if breaker['open']:
            # Circuit is open, check if we should retry
            if time.time() - breaker['last_attempt'] < 60:  # Wait at least 60s
                logger.warning(f"Circuit breaker open for {service_name}, request blocked")
                self.metrics.record_api_error(service_name, "circuit_open")
                return None

            # Try to reset circuit
            logger.info(f"Attempting to reset circuit breaker for {service_name}")
            breaker['open'] = False
            breaker['failures'] = 0
        
        try:
            # Make API call
            result = await call_fn(*args, **kwargs)
            
            # Reset failures on success
            breaker['failures'] = 0
            return result
            
        except Exception as e:
            # Increment failures
            breaker['failures'] += 1
            breaker['last_attempt'] = time.time()
            
            # Open circuit if threshold reached
            if breaker['failures'] >= breaker['threshold']:
                breaker['open'] = True
                logger.warning(f"Circuit breaker opened for {service_name} due to {str(e)}")
                self.metrics.record_api_error(service_name, "circuit_tripped")
                
            # Re-raise for caller to handle
            raise
    
    async def get_yield_opportunities(
        self,
        token_address: str,
        amount: float,
        time_horizon: int = 30,  # days
        chain_id: int = 1  # Ethereum mainnet by default
    ) -> List[FarmingOpportunity]:
        """
        Find and analyze yield farming opportunities for a token
        
        Args:
            token_address: Address of the token to farm
            amount: Amount of tokens to invest
            time_horizon: Investment time horizon in days
            chain_id: Chain ID to search for opportunities
            
        Returns:
            List of yield farming opportunities sorted by adjusted APY
        """
        # Record metrics
        start_time = time.time()
        self.metrics.record_api_call('yield_farming', 'get_opportunities')
        
        try:
            # Get token price for calculations
            token_price = await self.price_provider.get_token_price(token_address, chain_id)
            if not token_price:
                logger.warning(f"Could not get price for token {token_address}")
                token_price = 1.0  # Default to 1.0 for calculations
                
            # Convert amount to USD for consistent comparisons
            amount_usd = amount * token_price
            
            # Get all yield farming pools from DefiLlama
            yield_data = await self._call_with_circuit_breaker(
                'defillama',
                self.defillama.get_yields,
                chain_id
            )
            
            if not yield_data:
                logger.error("Failed to fetch yield data from DefiLlama")
                return []
                
            # Find pools that accept this token
            compatible_pools = await self._find_compatible_pools(
                token_address,
                yield_data.get('data', []),
                chain_id
            )
            
            if not compatible_pools:
                logger.info(f"No compatible yield farming pools found for {token_address}")
                return []
                
            # Analyze each pool for detailed metrics
            opportunities = []
            
            for pool_data in compatible_pools:
                try:
                    opportunity = await self._analyze_pool_opportunity(
                        pool_data,
                        token_address,
                        amount,
                        amount_usd,
                        time_horizon,
                        chain_id
                    )
                    
                    if opportunity:
                        opportunities.append(opportunity)
                        
                except Exception as e:
                    logger.error(f"Error analyzing pool opportunity: {str(e)}")
                    
            # Sort by adjusted APY
            opportunities.sort(key=lambda x: x.adjusted_apy, reverse=True)
            
            # Record metrics
            duration = time.time() - start_time
            self.metrics.record_duration('yield_farming', duration)
            logger.info(f"Found {len(opportunities)} yield farming opportunities in {duration:.2f}s")
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error getting yield opportunities: {str(e)}")
            self.metrics.record_api_error('yield_farming', str(e))
            return []
            
    async def _find_compatible_pools(
        self,
        token_address: str,
        pools: List[Dict[str, Any]],
        chain_id: int
    ) -> List[PoolData]:
        """Find pools compatible with the given token"""
        compatible_pools = []
        checksum_address = Web3.to_checksum_address(token_address)
        
        for pool in pools:
            try:
                # Skip pools from other chains
                if pool.get('chain') != chain_id:
                    continue
                    
                # Check token compatibility
                pool_tokens = pool.get('tokens', [])
                
                # Handle different formats - some APIs use addresses, some use symbols
                token_found = False
                
                # Check by address
                if pool_tokens and any(t.lower() == token_address.lower() for t in pool_tokens):
                    token_found = True
                    
                # Check by underlying assets if not found
                if not token_found and 'underlyingTokens' in pool:
                    underlying = pool.get('underlyingTokens', [])
                    if any(t.lower() == token_address.lower() for t in underlying):
                        token_found = True
                        
                if token_found:
                    # Convert to our internal format
                    pool_data = PoolData(
                        id=pool.get('pool', ''),
                        name=pool.get('name', 'Unknown Pool'),
                        protocol_id=pool.get('project', ''),
                        protocol_name=pool.get('projectName', ''),
                        tokens=pool.get('tokens', []),
                        tvl=pool.get('tvl', 0),
                        apy=pool.get('apy', 0),
                        rewards=pool.get('rewardTokens', []),
                        fee_apy=pool.get('apyBase', 0),
                        rewards_apy=pool.get('apyReward', 0),
                        last_updated=time.time()
                    )
                    
                    compatible_pools.append(pool_data)
                    
            except Exception as e:
                logger.debug(f"Error checking pool compatibility: {str(e)}")
                
        return compatible_pools
        
    async def _analyze_pool_opportunity(
        self,
        pool: PoolData,
        token_address: str,
        amount: float,
        amount_usd: float,
        time_horizon: int,
        chain_id: int
    ) -> Optional[FarmingOpportunity]:
        """Analyze a pool for yield farming opportunity"""
        try:
            # Get protocol information
            protocol_info = self.protocol_configs.get(
                pool.protocol_id.lower(),
                {
                    'type': 'unknown',
                    'entry_gas': 200000,
                    'exit_gas': 150000,
                    'harvest_gas': 120000,
                    'risk_score': 0.5,
                    'audited': False,
                    'complexity': 'unknown'
                }
            )
            
            # Get gas price
            gas_price = await self._get_gas_price(chain_id)
            
            # Calculate costs
            entry_cost = await self._calculate_gas_cost(
                protocol_info['entry_gas'],
                gas_price,
                chain_id
            )
            
            exit_cost = await self._calculate_gas_cost(
                protocol_info['exit_gas'],
                gas_price,
                chain_id
            )
            
            # Calculate token's share in the pool
            token_share = await self._calculate_token_share(pool, token_address)
            
            # Calculate impermanent loss risk
            il_risk = await self._calculate_il_risk(pool, token_address, time_horizon)
            
            # Calculate risk score
            risk_score = await self._calculate_risk_score(pool, protocol_info, il_risk)
            
            # Calculate optimal compound frequency
            compound_frequency, optimal_harvest_interval = await self._calculate_optimal_compound_frequency(
                pool.apy,
                protocol_info['harvest_gas'],
                gas_price,
                chain_id,
                amount_usd
            )
            
            # Calculate projected earnings
            projected_earnings = await self._calculate_projected_earnings(
                amount,
                pool.apy,
                time_horizon,
                compound_frequency
            )
            
            # Adjust earnings for costs and risks
            adjusted_earnings = await self._calculate_adjusted_earnings(
                projected_earnings,
                entry_cost,
                exit_cost,
                il_risk,
                risk_score,
                time_horizon
            )
            
            # Calculate adjusted APY
            adjusted_apy = (adjusted_earnings / amount) * (365 / time_horizon)
            
            # Create strategy
            strategy = await self._generate_farming_strategy(
                pool,
                token_address,
                protocol_info,
                compound_frequency,
                optimal_harvest_interval
            )
            
            # Create opportunity object
            opportunity = FarmingOpportunity(
                pool=pool,
                token_address=token_address,
                token_share=token_share,
                entry_cost=entry_cost,
                exit_cost=exit_cost,
                apy=pool.apy,
                adjusted_apy=adjusted_apy,
                projected_earnings=projected_earnings,
                adjusted_earnings=adjusted_earnings,
                il_risk=il_risk,
                risk_score=risk_score,
                time_horizon=time_horizon,
                position_size=amount,
                compound_frequency=compound_frequency,
                optimal_harvest_frequency=optimal_harvest_interval,
                strategy=strategy
            )
            
            return opportunity
            
        except Exception as e:
            logger.error(f"Error analyzing pool opportunity: {str(e)}")
            return None
            
    async def _get_gas_price(self, chain_id: int) -> int:
        """Get current gas price for the chain"""
        cache_key = f"gas_price:{chain_id}"
        
        # Try cache
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
            
        # Default gas prices by chain
        default_gas = {
            1: 50 * 10**9,    # Ethereum: 50 gwei
            10: 0.001 * 10**9,  # Optimism: 0.001 gwei
            42161: 0.1 * 10**9, # Arbitrum: 0.1 gwei
            137: 50 * 10**9,    # Polygon: 50 gwei
            56: 5 * 10**9       # BSC: 5 gwei
        }
        
        try:
            # This would connect to RPC nodes to get real gas prices
            # For now, use defaults
            gas_price = default_gas.get(chain_id, 50 * 10**9)
            
            # Cache for a short time
            await self.cache.set(cache_key, gas_price, ttl=60)
            
            return gas_price
            
        except Exception as e:
            logger.error(f"Error getting gas price: {str(e)}")
            return default_gas.get(chain_id, 50 * 10**9)
            
    async def _calculate_gas_cost(self, gas_units: int, gas_price: int, chain_id: int) -> float:
        """Calculate gas cost in USD"""
        # Get native token price
        native_token = {
            1: "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # ETH (WETH)
            10: "0x4200000000000000000000000000000000000006",  # ETH on Optimism
            42161: "0x82af49447d8a07e3bd95bd0d56f35241523fbab1",  # ETH on Arbitrum
            137: "0x0d500b1d8e8ef31e21c99d1db9a6444d3adf1270",  # MATIC (WMATIC)
            56: "0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c"   # BNB (WBNB)
        }
        
        token_address = native_token.get(chain_id, native_token[1])
        token_price = await self.price_provider.get_token_price(token_address, chain_id)
        
        if not token_price:
            # Default prices if lookup fails
            token_price = {
                1: 2000.0,  # ETH
                10: 2000.0,  # ETH on Optimism
                42161: 2000.0,  # ETH on Arbitrum
                137: 0.7,   # MATIC
                56: 300.0   # BNB
            }.get(chain_id, 2000.0)
            
        # Calculate cost
        gas_cost = (gas_units * gas_price) / 10**18  # Convert to native token
        usd_cost = gas_cost * token_price
        
        return usd_cost
        
    async def _calculate_token_share(self, pool: PoolData, token_address: str) -> float:
        """Calculate token's share in the pool"""
        if len(pool.tokens) == 1:
            return 1.0
            
        # If weights are available, use them
        # Otherwise assume equal weighting
        return 1.0 / len(pool.tokens)
        
    async def _calculate_il_risk(self, pool: PoolData, token_address: str, time_horizon: int) -> float:
        """Calculate impermanent loss risk"""
        # Single-asset pools have no IL
        if len(pool.tokens) == 1:
            return 0.0
            
        # Stablecoin pools have minimal IL
        if 'stable' in pool.name.lower():
            return 0.02
            
        # For other pools, base on volatility
        # This is a simplified model - could use historical data for better estimates
        risk_by_pool_type = {
            "stable": 0.05,
            "eth": 0.15,
            "btc": 0.15,
            "bluechip": 0.2,
            "midcap": 0.3,
            "volatile": 0.4
        }
        
        # Try to determine pool type from name
        pool_type = "volatile"  # Default
        for key in risk_by_pool_type:
            if key in pool.name.lower():
                pool_type = key
                break
                
        base_risk = risk_by_pool_type[pool_type]
        
        # Adjust for time horizon - longer horizon means more IL risk
        time_factor = min(time_horizon / 30, 3.0)  # Cap at 3x for very long horizons
        
        return base_risk * time_factor
        
    async def _calculate_risk_score(self, pool: PoolData, protocol_info: Dict[str, Any], il_risk: float) -> float:
        """Calculate overall risk score (0-1, lower is better)"""
        # Base components
        tvl_risk = max(0, min(1, self.risk_params['min_tvl_threshold'] / max(pool.tvl, 1)))
        protocol_risk = protocol_info.get('risk_score', 0.5)
        
        # Complexity risk
        complexity_map = {
            "low": 0.1,
            "medium": 0.3,
            "high": 0.5,
            "unknown": 0.7
        }
        complexity_risk = complexity_map.get(protocol_info.get('complexity', 'unknown'), 0.5)
        
        # Audit risk
        audit_risk = 0.2 if protocol_info.get('audited', False) else 0.8
        
        # Suspiciously high APY risk
        apy_risk = min(1.0, pool.apy / self.risk_params['max_apy_threshold'])
        
        # Weighted risk score
        risk_score = (
            tvl_risk * self.risk_params['tvl_weight'] +
            protocol_risk * self.risk_params['age_weight'] +
            audit_risk * self.risk_params['audit_weight'] +
            complexity_risk * self.risk_params['complexity_weight'] +
            il_risk * self.risk_params['il_weight'] +
            apy_risk * 0.1  # Additional small factor for suspiciously high APYs
        )
        
        return min(1.0, max(0.1, risk_score))
        
    async def _calculate_optimal_compound_frequency(
        self,
        apy: float,
        harvest_gas: int,
        gas_price: int,
        chain_id: int,
        amount_usd: float
    ) -> Tuple[str, int]:
        """
        Calculate optimal compounding frequency based on gas costs vs. returns
        
        Returns a tuple of (compound_frequency, harvest_interval_days)
        """
        if harvest_gas == 0:
            # Protocol doesn't support harvesting (e.g., Uniswap)
            return "none", 0
            
        # Calculate gas cost per harvest
        harvest_cost = await self._calculate_gas_cost(harvest_gas, gas_price, chain_id)
        
        # Convert APY to daily rate
        daily_rate = apy / 365
        
        # Calculate earnings per day
        daily_earnings = amount_usd * daily_rate
        
        if daily_earnings <= 0:
            return "none", 0
            
        # Calculate optimal harvest interval (days)
        # Harvest when earnings > 2x gas cost (to account for volatility and make it worthwhile)
        optimal_days = max(1, int(harvest_cost * 2 / daily_earnings))
        
        # Map days to human-readable frequency
        if optimal_days == 1:
            return "daily", 1
        elif optimal_days <= 3:
            return "every 2-3 days", optimal_days
        elif optimal_days <= 7:
            return "weekly", 7
        elif optimal_days <= 14:
            return "biweekly", 14
        elif optimal_days <= 30:
            return "monthly", 30
        else:
            return "quarterly", 90
            
    async def _calculate_projected_earnings(
        self,
        amount: float,
        apy: float,
        time_horizon: int,
        compound_frequency: str
    ) -> float:
        """Calculate projected earnings with compounding"""
        # Convert APY to periodic rate based on compound frequency
        if compound_frequency == "daily":
            periods = time_horizon
            periodic_rate = apy / 365
        elif compound_frequency == "weekly" or compound_frequency == "every 2-3 days":
            periods = time_horizon / 7
            periodic_rate = apy / 52
        elif compound_frequency == "biweekly":
            periods = time_horizon / 14
            periodic_rate = apy / 26
        elif compound_frequency == "monthly":
            periods = time_horizon / 30
            periodic_rate = apy / 12
        elif compound_frequency == "quarterly":
            periods = time_horizon / 90
            periodic_rate = apy / 4
        else:
            # No compounding
            return amount * (apy * time_horizon / 365)
            
        # Compound interest formula
        future_value = amount * (1 + periodic_rate) ** periods
        earnings = future_value - amount
        
        return earnings
        
    async def _calculate_adjusted_earnings(
        self,
        projected_earnings: float,
        entry_cost: float,
        exit_cost: float,
        il_risk: float,
        risk_score: float,
        time_horizon: int
    ) -> float:
        """Calculate earnings adjusted for costs and risks"""
        # Subtract costs
        net_earnings = projected_earnings - entry_cost - exit_cost
        
        # Adjust for impermanent loss
        il_adjusted = net_earnings * (1 - il_risk)
        
        # Adjust for other risks (more impact for longer time horizons)
        risk_factor = risk_score * min(time_horizon / 180, 1.0)  # Scale with time up to 6 months
        risk_adjusted = il_adjusted * (1 - risk_factor)
        
        return max(0, risk_adjusted)
        
    async def _generate_farming_strategy(
        self,
        pool: PoolData,
        token_address: str,
        protocol_info: Dict[str, Any],
        compound_frequency: str,
        optimal_harvest_interval: int
    ) -> Dict[str, Any]:
        """Generate yield farming strategy and instructions"""
        # Generic strategy template
        strategy = {
            "name": f"{pool.protocol_name} {pool.name} Strategy",
            "description": f"Yield farming strategy for {pool.name} on {pool.protocol_name}",
            "steps": [
                {
                    "step": 1,
                    "action": f"Approve {pool.protocol_name} to use your tokens",
                    "technical": f"Call approve() on the token contract at {token_address}",
                    "gas_estimate": protocol_info.get('entry_gas', 200000) // 4
                },
                {
                    "step": 2,
                    "action": f"Deposit tokens into {pool.name}",
                    "technical": f"Call deposit() or add_liquidity() on the pool contract",
                    "gas_estimate": protocol_info.get('entry_gas', 200000) * 3 // 4
                }
            ],
            "maintenance": {
                "compound_frequency": compound_frequency,
                "harvest_interval": f"Every {optimal_harvest_interval} days",
                "gas_efficiency": f"Most efficient to harvest when gas price is below {50} gwei"
            },
            "exit_strategy": {
                "step": 1,
                "action": f"Withdraw tokens from {pool.name}",
                "technical": "Call withdraw() or remove_liquidity() on the pool contract",
                "gas_estimate": protocol_info.get('exit_gas', 150000)
            },
            "risks": [
                "Smart contract risk",
                "Protocol risk",
                "Market volatility risk"
            ]
        }
        
        # Add impermanent loss risk if applicable
        if len(pool.tokens) > 1:
            strategy["risks"].append("Impermanent loss risk")
            
        # Add protocol-specific steps and info
        if protocol_info['type'] == 'lending':
            # Lending protocols (Aave, Compound)
            strategy["steps"][1]["action"] = f"Supply tokens to {pool.protocol_name}"
            strategy["steps"][1]["technical"] = "Call supply() or mint() on the protocol contract"
            
        elif protocol_info['type'] == 'dex':
            # DEX protocols (Uniswap, Sushiswap, Curve)
            strategy["steps"].insert(1, {
                "step": 2,
                "action": "Acquire paired tokens if needed",
                "technical": "Swap for paired tokens if providing LP",
                "gas_estimate": 150000
            })
            strategy["steps"][2]["step"] = 3
            strategy["steps"][2]["action"] = "Add liquidity to the pool"
            
        elif protocol_info['type'] == 'yield_aggregator':
            # Yield aggregators (Yearn, Convex)
            strategy["steps"][1]["action"] = f"Deposit tokens into {pool.protocol_name} vault"
            strategy["technical"] = "Call deposit() on the vault contract"
            
        return strategy
