import torch
import statistics
from typing import Dict, List, Tuple, Optional, TypedDict, Union, Any
import logging
from datetime import datetime
import time
import json
import math
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent directory to Python path to resolve core imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.types import (
    TokenPair, MarketDataType, OpportunityType,
    FlashLoanOpportunityType, MarketValidationResult
)

logger = logging.getLogger(__name__)

# Type definitions
class NetworkStatusType(TypedDict):
    is_healthy: bool
    block_time: float
    gas_price: int
    pending_transactions: int
    network_load: float
    reason: Optional[str]

class TimeSeriesFeatures:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.price_history = []
        self.volume_history = []
        
    def update(self, price: float, volume: float) -> None:
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        if len(self.price_history) > self.window_size:
            self.price_history = self.price_history[-self.window_size:]
            self.volume_history = self.volume_history[-self.window_size:]
            
    def get_features(self) -> torch.Tensor:
        if len(self.price_history) < 2:
            return torch.zeros(5)
            
        returns = torch.tensor([p2/p1 - 1 for p1, p2 in zip(self.price_history[:-1], self.price_history[1:])])
        volatility = float(torch.std(returns))
        volume_ma = float(torch.tensor(self.volume_history[-24:]).mean()) if len(self.volume_history) >= 24 else 0
        volume_trend = volume_ma / float(torch.tensor(self.volume_history).mean()) if self.volume_history else 0
        
        return torch.tensor([
            volatility,
            volume_trend,
            self._calculate_momentum(),
            self._calculate_relative_strength(),
            self._calculate_liquidity_depth()
        ])
        
    def _calculate_momentum(self) -> float:
        if len(self.price_history) < 2:
            return 0.0
        return self.price_history[-1] / self.price_history[0] - 1
        
    def _calculate_relative_strength(self) -> float:
        if len(self.price_history) < 14:
            return 0.0
        gains = []
        losses = []
        for i in range(1, len(self.price_history)):
            change = self.price_history[i] - self.price_history[i-1]
            if change >= 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(-change)
        avg_gain = float(torch.tensor(gains[-14:]).mean())
        avg_loss = float(torch.tensor(losses[-14:]).mean())
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))
        
    def _calculate_liquidity_depth(self) -> float:
        if len(self.volume_history) < 24:
            return 0.0
        volume_tensor = torch.tensor(self.volume_history[-24:])
        mean_volume = float(volume_tensor.mean())
        std_volume = float(volume_tensor.std())
        return mean_volume / std_volume if std_volume > 0 else 0.0

class CrossChainAnalyzer:
    def __init__(self):
        self.eth_correlation = []
        self.l2_metrics = {}
        
    async def analyze_opportunities(self, token_pair: Tuple[str, str]) -> Dict:
        return {
            'eth_correlation': self._calculate_eth_correlation(),
            'l2_gas_efficiency': self._calculate_l2_efficiency(),
            'bridge_liquidity': await self._get_bridge_liquidity(),
            'cross_chain_volume': await self._get_cross_chain_volume()
        }
        
    def _calculate_eth_correlation(self) -> float:
        if len(self.eth_correlation) < 2:
            return 0
        return float(torch.tensor(self.eth_correlation).corrcoef()[0, 1])
        
    def _calculate_l2_efficiency(self) -> float:
        gas_savings = 0.85  # Base typically saves 85% on gas compared to L1
        return gas_savings * self._get_l2_multiplier()
        
    async def _get_bridge_liquidity(self) -> float:
        return 0.0  # Implement bridge liquidity analysis
        
    async def _get_cross_chain_volume(self) -> float:
        return 0.0  # Implement cross-chain volume analysis
        
    def _get_l2_multiplier(self) -> float:
        return 1.2  # Base has good efficiency due to OP Stack

class MEVProtection:
    def __init__(self):
        self.sandwich_threshold = 0.02  # 2% price impact threshold
        self.frontrun_threshold = 0.01  # 1% slippage threshold
        
    def calculate_mev_risk(self, trade_params: Dict) -> Dict:
        return {
            'sandwich_risk': self._estimate_sandwich_risk(trade_params),
            'frontrunning_risk': self._estimate_frontrunning_risk(trade_params),
            'backrunning_risk': self._estimate_backrunning_risk(trade_params),
            'optimal_block_position': self._calculate_optimal_block_position(trade_params)
        }
        
    def _estimate_sandwich_risk(self, trade_params: Dict) -> float:
        return min(trade_params.get('price_impact', 0) / self.sandwich_threshold, 1.0)
        
    def _estimate_frontrunning_risk(self, trade_params: Dict) -> float:
        return min(trade_params.get('slippage', 0) / self.frontrun_threshold, 1.0)
        
    def _estimate_backrunning_risk(self, trade_params: Dict) -> float:
        volume = trade_params.get('volume_24h', 0)
        return 1.0 / (1.0 + volume/1e6) if volume > 0 else 1.0
        
    def _calculate_optimal_block_position(self, trade_params: Dict) -> int:
        gas_price = trade_params.get('gas_price', 0)
        return 1 if gas_price > 100 else 0  # First position if gas price is high

class GasOptimizer:
    def __init__(self):
        self.base_fee_history = []
        self.priority_fee_history = []
        
    def optimize_execution(self, trade_params: Dict) -> Dict:
        return {
            'optimal_gas_price': self._calculate_optimal_gas_price(),
            'base_fee_prediction': self._predict_base_fee_next_blocks(),
            'priority_fee_strategy': self._calculate_priority_fee(),
            'block_space_analysis': self._analyze_block_space()
        }
        
    def _calculate_optimal_gas_price(self) -> int:
        return max(int(self.base_fee_history[-1] * 1.2) if self.base_fee_history else 0, 1)
        
    def _predict_base_fee_next_blocks(self) -> List[int]:
        return self.base_fee_history[-5:] if self.base_fee_history else []
        
    def _calculate_priority_fee(self) -> int:
        return int(statistics.mean(self.priority_fee_history[-10:]) if self.priority_fee_history else 1)
        
    def _analyze_block_space(self) -> Dict:
        return {'utilization': 0.8, 'congestion': 'medium'}

class TokenEconomicsAnalyzer:
    def __init__(self):
        self.supply_history = {}
        self.holder_data = {}
        
    def analyze_token_metrics(self, token: str) -> Dict:
        return {
            'supply_dynamics': self._analyze_supply_changes(token),
            'holder_concentration': self._analyze_holder_distribution(token),
            'vesting_schedules': self._track_vesting_events(token),
            'protocol_revenue': self._analyze_protocol_revenue(token)
        }
        
    def _analyze_supply_changes(self, token: str) -> Dict:
        return {'inflation_rate': 0.0, 'burn_rate': 0.0, 'circulating_ratio': 0.8}
        
    def _analyze_holder_distribution(self, token: str) -> Dict:
        return {'top_holders_share': 0.0, 'gini_coefficient': 0.5}
        
    def _track_vesting_events(self, token: str) -> List:
        return []
        
    def _analyze_protocol_revenue(self, token: str) -> Dict:
        return {'revenue_30d': 0.0, 'revenue_growth': 0.0}

class DefiLlamaIntegration:
    def __init__(self):
        self.cache_duration = 300  # 5 minutes cache
        self.tvl_cache = {}
        self.volume_cache = {}
        
    async def get_protocol_data(self, protocol_slug: str) -> Dict:
        try:
            current_time = datetime.now().timestamp()
            if protocol_slug in self.tvl_cache and current_time - self.tvl_cache[protocol_slug]['timestamp'] < self.cache_duration:
                return self.tvl_cache[protocol_slug]['data']
            
            # Implement actual DeFiLlama API call here
            return {
                'tvl': 0,
                'volume_24h': 0,
                'fees_24h': 0,
                'mcap_tvl': 0,
                'historical_tvl': [],
                'timestamp': current_time
            }
        except Exception as e:
            logger.error(f"Error fetching DeFiLlama data: {str(e)}")
            return {}

class MarketAnalyzer:
    """Handles market analysis and opportunity detection"""
    
    VOLATILITY_THRESHOLD = 0.15  # 15% volatility threshold for warnings
    
    def __init__(self):
        # Original components
        self.time_series = TimeSeriesFeatures()
        self.cross_chain = CrossChainAnalyzer()
        self.mev_protection = MEVProtection()
        self.gas_optimizer = GasOptimizer()
        self.token_economics = TokenEconomicsAnalyzer()
        
        # Original tracking
        self.volatility_history = []
        self.price_history = {}
        self.volume_history = {}
        self.slippage_history = {}
        
        # Original configuration
        self.history_window = 1000
        self.volatility_window = 24
        self.cleanup_interval = 3600
        self.last_cleanup = time.time()
        
        # Original DeFi integrations
        self.defillama = DefiLlamaIntegration()
        self.supported_dexes = self._load_dex_configs()
        self.protocol_slugs = self._load_protocol_slugs()
        
    def monitor_volatility(self, market_data: MarketDataType) -> None:
        """Monitor market volatility and log warnings if exceeds thresholds"""
        try:
            current_volatility = market_data.get('volatility', 0.0)
            self.volatility_history.append((time.time(), current_volatility))
            
            # Keep last hour of volatility data
            cutoff_time = time.time() - 3600
            self.volatility_history = [
                (t, v) for t, v in self.volatility_history 
                if t > cutoff_time
            ]
            
            # Calculate rolling volatility
            if len(self.volatility_history) > 1:
                rolling_vol = statistics.stdev([v for _, v in self.volatility_history])
                if rolling_vol > self.VOLATILITY_THRESHOLD:
                    logger.warning(
                        f"High market volatility detected: {rolling_vol:.4f} "
                        f"(threshold: {self.VOLATILITY_THRESHOLD})"
                    )
                    
        except Exception as e:
            logger.error(f"Error monitoring volatility: {str(e)}")
            
    def get_volatility_adjustment(self) -> float:
        """Calculate confidence adjustment factor based on volatility"""
        try:
            if not self.volatility_history:
                return 1.0
                
            recent_volatility = statistics.mean([
                v for _, v in self.volatility_history[-10:]  # Last 10 readings
            ])
            
            # Exponential decay for high volatility
            return math.exp(-2 * recent_volatility)
            
        except Exception as e:
            logger.error(f"Error calculating volatility adjustment: {str(e)}")
            return 1.0
            
    async def calculate_market_depth(self, token_pair: TokenPair) -> Dict:
        """Calculate market depth using DeFiLlama data"""
        total_liquidity = 0
        total_volume = 0
        
        for dex_name, dex_info in self.supported_dexes.items():
            if dex_name in self.protocol_slugs:
                protocol_data = await self.defillama.get_protocol_data(
                    self.protocol_slugs[dex_name]
                )
                if protocol_data:
                    total_liquidity += protocol_data['tvl']
                    total_volume += protocol_data['volume_24h']
                    
        return {
            'total_liquidity': total_liquidity,
            'total_volume': total_volume,
            'liquidity_depth': total_liquidity / max(total_volume, 1),
            'market_impact': self.estimate_market_impact(total_liquidity)
        }
        
    async def analyze_protocol_health(self) -> Dict:
        """Analyze protocol health metrics from DeFiLlama"""
        protocol_health = {}
        for dex_name, slug in self.protocol_slugs.items():
            protocol_data = await self.defillama.get_protocol_data(slug)
            if protocol_data:
                protocol_health[dex_name] = {
                    'tvl_trend': self.calculate_tvl_trend(
                        protocol_data['historical_tvl']
                    ),
                    'volume_quality': protocol_data['volume_24h'] / 
                                    max(protocol_data['tvl'], 1),
                    'fee_efficiency': protocol_data['fees_24h'] / 
                                    max(protocol_data['volume_24h'], 1)
                }
        return protocol_health
        
    def calculate_tvl_trend(self, historical_tvl: List) -> float:
        """Calculate TVL trend from historical data"""
        if not historical_tvl or len(historical_tvl) < 2:
            return 0
        
        recent_tvl = historical_tvl[-7:]  # Last 7 days
        if len(recent_tvl) < 2:
            return 0
            
        return (recent_tvl[-1] - recent_tvl[0]) / recent_tvl[0]
        
    def estimate_market_impact(self, liquidity: float) -> float:
        """Estimate market impact based on liquidity"""
        return 1 / (1 + liquidity/1e6)  # Simplified impact model 

    async def fetch_market_data(self) -> MarketDataType:
        """Fetch current market data from various sources"""
        try:
            # Fetch DEX data
            dex_data = await self._fetch_dex_data()
            
            # Fetch token data
            token_data = await self._fetch_token_data()
            
            # Fetch network data
            network_data = await self._fetch_network_data()
            
            # Combine all data
            market_data = {
                **dex_data,
                **token_data,
                **network_data,
                'timestamp': datetime.now().isoformat()
            }
            
            # Monitor volatility
            self.monitor_volatility(market_data)
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            return {}

    async def get_current_price(self, token_pair: TokenPair) -> float:
        """Get current price for token pair"""
        try:
            # Try getting price from DEX
            dex_price = await self._get_dex_price(token_pair)
            if dex_price > 0:
                return dex_price
            
            # Fallback to aggregator price
            agg_price = await self._get_aggregator_price(token_pair)
            if agg_price > 0:
                return agg_price
            
            # Final fallback to cached price
            return self._get_cached_price(token_pair)
            
        except Exception as e:
            logger.error(f"Error getting current price: {str(e)}")
            return 0.0

    async def get_current_liquidity(self, token_pair: TokenPair) -> float:
        """Get current liquidity for token pair"""
        try:
            total_liquidity = 0
            
            # Get liquidity from all DEXs
            for dex in self.supported_dexes:
                liquidity = await self._get_dex_liquidity(dex, token_pair)
                total_liquidity += liquidity
            
            return total_liquidity
            
        except Exception as e:
            logger.error(f"Error getting current liquidity: {str(e)}")
            return 0.0

    async def check_network_status(self) -> NetworkStatusType:
        """Check current network health status"""
        try:
            # Get block time
            block_time = await self._get_average_block_time()
            
            # Get network load
            network_load = await self._get_network_load()
            
            # Get pending transactions
            pending_tx = await self._get_pending_transactions()
            
            # Get gas price
            gas_price = await self._get_current_gas_price()
            
            # Determine network health
            is_healthy = (
                block_time < 15 and  # Less than 15 seconds block time
                network_load < 0.8 and  # Less than 80% network load
                pending_tx < 50000  # Less than 50k pending transactions
            )
            
            return NetworkStatusType(
                is_healthy=is_healthy,
                block_time=block_time,
                gas_price=gas_price,
                pending_transactions=pending_tx,
                network_load=network_load,
                reason=None if is_healthy else self._get_unhealthy_reason(
                    block_time, network_load, pending_tx
                )
            )
            
        except Exception as e:
            logger.error(f"Error checking network status: {str(e)}")
            return NetworkStatusType(
                is_healthy=False,
                block_time=0,
                gas_price=0,
                pending_transactions=0,
                network_load=0,
                reason=f"Error checking network status: {str(e)}"
            )

    # Helper methods
    async def _fetch_dex_data(self) -> Dict:
        """Fetch data from DEXs"""
        return {}  # Empty dict instead of pass

    async def _fetch_token_data(self) -> Dict:
        """Fetch token specific data"""
        return {}  # Empty dict instead of pass

    async def _fetch_network_data(self) -> Dict:
        """Fetch network metrics"""
        return {}  # Empty dict instead of pass

    async def _get_dex_price(self, token_pair: TokenPair) -> float:
        """Get price from DEX"""
        return 0.0  # Zero instead of pass

    async def _get_aggregator_price(self, token_pair: TokenPair) -> float:
        """Get price from aggregator"""
        return 0.0  # Zero instead of pass

    def _get_cached_price(self, token_pair: TokenPair) -> float:
        """Get cached price"""
        return 0.0  # Zero instead of pass

    async def _get_dex_liquidity(self, dex: str, token_pair: TokenPair) -> float:
        """Get liquidity from specific DEX"""
        return 0.0  # Zero instead of pass

    async def _get_average_block_time(self) -> float:
        """Calculate average block time"""
        return 0.0  # Zero instead of pass

    async def _get_network_load(self) -> float:
        """Get current network load"""
        return 0.0  # Zero instead of pass

    async def _get_pending_transactions(self) -> int:
        """Get number of pending transactions"""
        return 0  # Zero instead of pass

    async def _get_current_gas_price(self) -> int:
        """Get current gas price"""
        return 0  # Zero instead of pass

    def _get_unhealthy_reason(
        self,
        block_time: float,
        network_load: float,
        pending_tx: int
    ) -> str:
        """Get reason for unhealthy network status"""
        reasons = []
        if block_time >= 15:
            reasons.append(f"High block time: {block_time:.1f}s")
        if network_load >= 0.8:
            reasons.append(f"High network load: {network_load*100:.1f}%")
        if pending_tx >= 50000:
            reasons.append(f"Many pending transactions: {pending_tx}")
        return ", ".join(reasons)

    async def analyze_market(
        self,
        token_pair: TokenPair,
        amount: float
    ) -> Dict[str, Any]:
        """Comprehensive market analysis with all original capabilities"""
        try:
            # Time series analysis
            time_series_features = self.time_series.get_features()
            
            # Cross-chain analysis
            cross_chain_metrics = await self.cross_chain.analyze_opportunities(token_pair)
            
            # MEV risk analysis
            mev_risk = self.mev_protection.calculate_mev_risk({
                'amount': amount,
                'token_pair': token_pair
            })
            
            # Gas optimization
            gas_strategy = self.gas_optimizer.optimize_execution({
                'amount': amount,
                'token_pair': token_pair
            })
            
            # Token economics
            token_metrics = self.token_economics.analyze_token_metrics(token_pair[0])
            
            # Market depth
            market_depth = await self.calculate_market_depth(token_pair)
            
            # Protocol health
            protocol_health = await self.analyze_protocol_health()
            
            # Combine all analyses
            return {
                'time_series': time_series_features,
                'cross_chain': cross_chain_metrics,
                'mev_risk': mev_risk,
                'gas_strategy': gas_strategy,
                'token_economics': token_metrics,
                'market_depth': market_depth,
                'protocol_health': protocol_health,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in market analysis: {str(e)}")
            return {}
            
    def update_market_state(
        self,
        token_pair: TokenPair,
        price: float,
        volume: float,
        timestamp: float
    ) -> None:
        """Update all market state tracking"""
        try:
            # Update time series
            self.time_series.update(price, volume)
            
            # Update histories
            if token_pair not in self.price_history:
                self.price_history[token_pair] = []
            self.price_history[token_pair].append(price)
            
            if token_pair not in self.volume_history:
                self.volume_history[token_pair] = []
            self.volume_history[token_pair].append(volume)
            
            # Cleanup if needed
            current_time = time.time()
            if current_time - self.last_cleanup > self.cleanup_interval:
                self._cleanup_histories()
                
        except Exception as e:
            logger.error(f"Error updating market state: {str(e)}")
            
    def _cleanup_histories(self) -> None:
        """Cleanup all historical data"""
        try:
            # Cleanup price history
            for pair in self.price_history:
                self.price_history[pair] = self.price_history[pair][-self.history_window:]
                
            # Cleanup volume history
            for pair in self.volume_history:
                self.volume_history[pair] = self.volume_history[pair][-self.history_window:]
                
            # Cleanup slippage history
            for pair in self.slippage_history:
                self.slippage_history[pair] = self.slippage_history[pair][-self.history_window:]
                
            self.last_cleanup = time.time()
            
        except Exception as e:
            logger.error(f"Error cleaning up histories: {str(e)}")
            
    def _load_dex_configs(self) -> Dict[str, Any]:
        """Load DEX configurations"""
        try:
            with open('config/dex_configs.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading DEX configs: {str(e)}")
            return {}
            
    def _load_protocol_slugs(self) -> Dict[str, str]:
        """Load protocol slugs for DeFiLlama"""
        try:
            with open('config/protocol_slugs.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading protocol slugs: {str(e)}")
            return {} 