from typing import Dict, List, Tuple, Optional, Any, Union, cast, TypedDict
import logging
import asyncio
from web3 import Web3
from web3.types import BlockData, TxParams, Wei, TxReceipt, TxData
from web3.evm import BlockNumber
from eth_typing import Hash32, HexStr, HexBytes
import aiohttp
from decimal import Decimal
from src.core.chain_connector import get_chain_connector, ChainConnector
from src.core.chain_config import get_chain_registry, ChainRegistry
from src.core.bridge_adapter import BridgeConfig, BridgeState
from src.market.price_feeds import PriceFeedRegistry
import time
from dataclasses import dataclass
from src.core.register_adapters import get_registered_adapters

logger = logging.getLogger(__name__)

@dataclass
class ChainMetrics:
    """Metrics for chain performance monitoring"""
    avg_block_time: float = 0.0
    avg_gas_price: float = 0.0
    success_rate: float = 0.0
    failed_txs: int = 0
    total_txs: int = 0
    last_error: Optional[str] = None
    last_updated: float = 0.0
    # Additional metrics for learning
    avg_execution_time: float = 0.0
    avg_profit_margin: float = 0.0
    missed_opportunities: int = 0
    slippage_incidents: int = 0
    bridge_failures: int = 0
    optimal_gas_usage: float = 0.0
    network_congestion: float = 0.0
    liquidity_score: float = 1.0
    # Mode-specific metrics
    mode_gas_savings: float = 0.0  # Gas savings from Mode's optimizations
    mode_finality_time: float = 0.0  # Average finality time for Mode
    mode_bridge_usage: int = 0  # Number of Mode bridge uses
    # Sonic-specific metrics
    sonic_lp_volume: float = 0.0  # Volume through Sonic liquidity pools
    sonic_bridge_volume: float = 0.0  # Volume through Sonic bridge
    sonic_fee_savings: float = 0.0  # Fee savings from fixed priority fees

@dataclass
class OpportunityMetrics:
    """Metrics for opportunity analysis and learning"""
    execution_time: float = 0.0
    profit_realized: float = 0.0
    expected_profit: float = 0.0
    gas_efficiency: float = 0.0
    slippage: float = 0.0
    bridge_latency: float = 0.0
    success: bool = False
    timestamp: float = 0.0

class PerformanceMetrics:
    """Tracks performance metrics across chains"""
    def __init__(self):
        self.chain_metrics: Dict[str, ChainMetrics] = {}
        self.operation_times: Dict[str, List[float]] = {
            'price_check': [],
            'bridge_check': [],
            'gas_estimation': [],
            'transaction_validation': []
        }
        self.opportunity_history: Dict[str, List[OpportunityMetrics]] = {}
        self.learning_feedback: Dict[str, List[Dict[str, Any]]] = {}
        
    def update_chain_metrics(self, chain: str, metrics: ChainMetrics) -> None:
        """Update metrics for a specific chain"""
        self.chain_metrics[chain] = metrics
        self._generate_learning_feedback(chain)
    
    def record_opportunity(self, chain: str, metrics: OpportunityMetrics) -> None:
        """Record opportunity metrics for learning"""
        if chain not in self.opportunity_history:
            self.opportunity_history[chain] = []
        self.opportunity_history[chain].append(metrics)
        
        # Keep only last 1000 opportunities per chain
        if len(self.opportunity_history[chain]) > 1000:
            self.opportunity_history[chain].pop(0)
    
    def _generate_learning_feedback(self, chain: str) -> None:
        """Generate learning feedback from metrics"""
        if chain not in self.learning_feedback:
            self.learning_feedback[chain] = []
            
        metrics = self.chain_metrics[chain]
        recent_opportunities = self.opportunity_history.get(chain, [])[-100:]  # Last 100 opportunities
        
        if not recent_opportunities:
            return
            
        # Calculate success rate trend
        success_rate_trend = sum(1 for opp in recent_opportunities if opp.success) / len(recent_opportunities)
        
        # Calculate profit efficiency
        profit_efficiency = sum(opp.profit_realized / opp.expected_profit if opp.expected_profit > 0 else 0 
                              for opp in recent_opportunities) / len(recent_opportunities)
        
        # Calculate gas optimization score
        gas_optimization = sum(opp.gas_efficiency for opp in recent_opportunities) / len(recent_opportunities)
        
        feedback = {
            'timestamp': time.time(),
            'metrics': {
                'success_rate_trend': success_rate_trend,
                'profit_efficiency': profit_efficiency,
                'gas_optimization': gas_optimization,
                'network_congestion': metrics.network_congestion,
                'liquidity_score': metrics.liquidity_score,
                'avg_execution_time': metrics.avg_execution_time,
                'missed_opportunities': metrics.missed_opportunities
            },
            'recommendations': self._generate_recommendations(
                success_rate_trend,
                profit_efficiency,
                gas_optimization,
                metrics
            )
        }
        
        self.learning_feedback[chain].append(feedback)
        
        # Keep only last 100 feedback entries per chain
        if len(self.learning_feedback[chain]) > 100:
            self.learning_feedback[chain].pop(0)
    
    def _generate_recommendations(
        self,
        success_rate_trend: float,
        profit_efficiency: float,
        gas_optimization: float,
        metrics: ChainMetrics
    ) -> List[str]:
        """Generate recommendations based on metrics"""
        recommendations = []
        
        if success_rate_trend < 0.9:
            recommendations.append("Increase validation thresholds")
        if profit_efficiency < 0.8:
            recommendations.append("Adjust profit margin requirements")
        if gas_optimization < 0.7:
            recommendations.append("Optimize gas price strategy")
        if metrics.slippage_incidents > 10:
            recommendations.append("Increase slippage tolerance")
        if metrics.bridge_failures > 5:
            recommendations.append("Review bridge selection criteria")
        if metrics.network_congestion > 0.8:
            recommendations.append("Implement congestion-based routing")
            
        return recommendations

    def record_operation_time(self, operation: str, time: float) -> None:
        """Record time taken for an operation"""
        if operation in self.operation_times:
            self.operation_times[operation].append(time)
            # Keep only last 1000 times
            if len(self.operation_times[operation]) > 1000:
                self.operation_times[operation].pop(0)
                
    def get_avg_operation_time(self, operation: str) -> float:
        """Get average time for an operation"""
        if operation not in self.operation_times or not self.operation_times[operation]:
            return 0.0
        return sum(self.operation_times[operation]) / len(self.operation_times[operation])

class CrossChainAnalyzer:
    """Analyzes cross-chain opportunities and market conditions"""
    
    def __init__(self):
        """Initialize cross-chain analyzer"""
        self.chain_connector = get_chain_connector()
        self.chain_registry = get_chain_registry()
        self.price_feed_registry = PriceFeedRegistry()
        self.bridge_liquidity_cache = {}
        self.price_cache = {}
        self.gas_estimates_cache = {}
        self.metrics = PerformanceMetrics()
        self._setup_error_handlers()
        
    def _setup_error_handlers(self) -> None:
        """Setup custom error handlers for different scenarios"""
        self.error_handlers = {
            'connection': self._handle_connection_error,
            'timeout': self._handle_timeout_error,
            'validation': self._handle_validation_error,
            'bridge': self._handle_bridge_error
        }
    
    async def _handle_connection_error(self, chain: str, error: Exception) -> None:
        """Handle chain connection errors"""
        logger.error(f"Connection error for chain {chain}: {str(error)}")
        metrics = self.metrics.chain_metrics.get(chain, ChainMetrics())
        metrics.last_error = str(error)
        metrics.last_updated = time.time()
        self.metrics.update_chain_metrics(chain, metrics)
        
        # Attempt to reconnect
        try:
            await self.chain_connector.reconnect(chain)
        except Exception as e:
            logger.error(f"Failed to reconnect to {chain}: {str(e)}")
    
    async def _handle_timeout_error(self, operation: str, error: Exception) -> None:
        """Handle timeout errors"""
        logger.error(f"Timeout during {operation}: {str(error)}")
        self.metrics.record_operation_time(operation, 30.0)  # Record timeout as 30s
    
    async def _handle_validation_error(self, tx_data: Dict[str, Any], error: Exception) -> None:
        """Handle transaction validation errors"""
        logger.error(f"Validation error: {str(error)}, tx_data: {tx_data}")
        chain = tx_data.get('chain', 'unknown')
        if chain in self.metrics.chain_metrics:
            metrics = self.metrics.chain_metrics[chain]
            metrics.failed_txs += 1
            metrics.last_error = str(error)
            metrics.last_updated = time.time()
            self.metrics.update_chain_metrics(chain, metrics)
    
    async def _handle_bridge_error(self, source_chain: str, target_chain: str, error: Exception) -> None:
        """Handle bridge-related errors"""
        logger.error(f"Bridge error between {source_chain} and {target_chain}: {str(error)}")
        # Clear bridge cache for affected chains
        cache_key = f"{source_chain}:{target_chain}"
        if cache_key in self.bridge_liquidity_cache:
            del self.bridge_liquidity_cache[cache_key]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        report = {
            'chain_metrics': {},
            'operation_times': {},
            'overall_health': 'healthy'
        }
        
        # Compile chain metrics
        for chain, metrics in self.metrics.chain_metrics.items():
            report['chain_metrics'][chain] = {
                'success_rate': metrics.success_rate,
                'failed_transactions': metrics.failed_txs,
                'total_transactions': metrics.total_txs,
                'avg_block_time': metrics.avg_block_time,
                'avg_gas_price': metrics.avg_gas_price,
                'last_error': metrics.last_error,
                'last_updated': metrics.last_updated
            }
            
            # Check chain health
            if metrics.success_rate < 0.95 or time.time() - metrics.last_updated > 300:
                report['overall_health'] = 'degraded'
        
        # Compile operation times
        for operation in self.metrics.operation_times:
            avg_time = self.metrics.get_avg_operation_time(operation)
            report['operation_times'][operation] = {
                'avg_time': avg_time,
                'status': 'normal' if avg_time < 5.0 else 'slow'
            }
            
            # Check operation health
            if avg_time > 10.0:
                report['overall_health'] = 'degraded'
        
        return report
    
    async def fetch_all_market_data(self) -> Dict[str, Any]:
        """Fetch market data from all active chains concurrently
        
        Returns:
            Dict mapping chain names to their market data
        """
        tasks = []
        active_chains = self.chain_connector.get_active_chains()
        
        for chain in active_chains:
            tasks.append(self._fetch_chain_market_data(chain))
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        market_data = {}
        for chain, result in zip(active_chains, results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching market data for {chain}: {str(result)}")
                continue
            market_data[chain] = result
            
        return market_data
    
    async def _fetch_chain_market_data(self, chain: str) -> Dict[str, Any]:
        """Fetch market data for a specific chain
        
        Args:
            chain: Chain name
            
        Returns:
            Dict containing chain market data
        """
        try:
            web3 = await self.chain_connector.get_connection(chain)
            if not web3:
                raise ValueError(f"Failed to get connection for {chain}")
                
            # Fetch data concurrently
            gas_price, block_number, latest_block = await asyncio.gather(
                self.chain_connector.get_gas_price(chain),
                self.chain_connector.get_block_number(chain),
                self.chain_connector.get_latest_block(chain)
            )
            
            return {
                'chain': chain,
                'gas_price': gas_price,
                'block_number': block_number,
                'latest_block': latest_block,
                'timestamp': latest_block.get('timestamp') if latest_block else None
            }
            
        except Exception as e:
            logger.error(f"Error in _fetch_chain_market_data for {chain}: {str(e)}")
            raise
    
    async def analyze_opportunities_parallel(
        self,
        token_pairs: List[Tuple[str, str]],
        amounts: List[float]
    ) -> List[Dict[str, Any]]:
        """Analyze multiple opportunities in parallel
        
        Args:
            token_pairs: List of token address pairs
            amounts: List of amounts to analyze
            
        Returns:
            List of opportunity analysis results
        """
        if len(token_pairs) != len(amounts):
            raise ValueError("Token pairs and amounts must have same length")
            
        # Fetch market data for all chains first
        market_data = await self.fetch_all_market_data()
        
        # Create analysis tasks
        tasks = []
        for token_pair, amount in zip(token_pairs, amounts):
            tasks.append(
                self.analyze_cross_chain_opportunity(
                    token_pair,
                    amount,
                    {'market_data': market_data}
                )
            )
            
        # Execute analyses in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors and format results
        opportunities = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in parallel analysis: {str(result)}")
                continue
            if result.get('is_viable', False):
                opportunities.append(result)
                
        return opportunities
    
    async def monitor_bridge_liquidity_parallel(
        self,
        token_pairs: List[Tuple[str, str]],
        chains: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Monitor bridge liquidity for multiple pairs across chains
        
        Args:
            token_pairs: List of token pairs to monitor
            chains: List of chains to monitor
            
        Returns:
            Dict mapping chains to their liquidity data
        """
        tasks = []
        for chain in chains:
            for token_pair in token_pairs:
                tasks.append(
                    self._get_bridge_liquidity_for_pair(chain, token_pair)
                )
                
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        liquidity_data = {}
        for chain in chains:
            liquidity_data[chain] = {}
            
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error in liquidity monitoring: {str(result)}")
                continue
                
            chain_index = i // len(token_pairs)
            pair_index = i % len(token_pairs)
            chain = chains[chain_index]
            token_pair = token_pairs[pair_index]
            
            pair_key = f"{token_pair[0]}_{token_pair[1]}"
            if chain not in liquidity_data:
                liquidity_data[chain] = {}
            liquidity_data[chain][pair_key] = result
            
        return liquidity_data
    
    async def _get_bridge_liquidity_for_pair(
        self,
        chain: str,
        token_pair: Tuple[str, str]
    ) -> float:
        """Get bridge liquidity for a specific pair on a chain
        
        Args:
            chain: Chain name
            token_pair: Token pair to check
            
        Returns:
            Available liquidity
        """
        try:
            cache_key = f"{chain}:{token_pair[0]}_{token_pair[1]}"
            if cache_key in self.bridge_liquidity_cache:
                return self.bridge_liquidity_cache[cache_key]
                
            web3 = await self.chain_connector.get_connection(chain)
            if not web3:
                raise ValueError(f"Failed to get connection for {chain}")
                
            # Implementation would check actual bridge contracts
            # This is a placeholder
            liquidity = 1000000  # Example value
            
            self.bridge_liquidity_cache[cache_key] = liquidity
            return liquidity
            
        except Exception as e:
            logger.error(f"Error getting bridge liquidity for {chain}: {str(e)}")
            raise
    
    async def analyze_cross_chain_opportunity(
        self,
        token_pair: Tuple[str, str],
        amount: float,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze cross-chain arbitrage opportunity"""
        try:
            source_chain = market_data['source_chain']
            target_chain = market_data['target_chain']
            
            # Get Web3 connections
            source_web3 = self.chain_connector.get_connection(source_chain)
            target_web3 = self.chain_connector.get_connection(target_chain)
            
            if not source_web3 or not target_web3:
                return {
                    'is_viable': False,
                    'error': 'Web3 connection not available'
                }
            
            # Analyze cross-chain prices
            price_analysis = await self._analyze_cross_chain_prices(
                token_pair,
                source_chain,
                target_chain,
                source_web3,
                target_web3
            )
            
            # Analyze available bridges
            bridge_analysis = await self._analyze_bridges(
                token_pair,
                amount,
                source_chain,
                target_chain
            )
            
            # Estimate gas costs
            gas_analysis = await self._estimate_cross_chain_gas(
                token_pair,
                amount,
                source_chain,
                target_chain,
                source_web3,
                target_web3
            )
            
            # Calculate potential profit
            profit_analysis = self._calculate_profit_potential(
                price_analysis,
                bridge_analysis,
                gas_analysis,
                amount
            )
            
            # Assess risks
            risk_analysis = self._assess_risks(
                price_analysis,
                bridge_analysis,
                gas_analysis
            )
            
            # Add chain-specific analysis
            if source_chain == 'mode' or target_chain == 'mode':
                mode_analysis = self._analyze_mode_specific(
                    source_chain,
                    target_chain,
                    gas_analysis
                )
                profit_analysis['mode_optimizations'] = mode_analysis
            
            if source_chain == 'sonic' or target_chain == 'sonic':
                sonic_analysis = self._analyze_sonic_specific(
                    source_chain,
                    target_chain,
                    bridge_analysis
                )
                profit_analysis['sonic_optimizations'] = sonic_analysis
            
            return {
                'is_viable': profit_analysis['profit_usd'] > 0,
                'profit_potential': profit_analysis,
                'bridge_liquidity': bridge_analysis.get('liquidity', {}),
                'estimated_gas_cost': gas_analysis.get('total_cost', 0),
                'execution_time': bridge_analysis.get('estimated_time', 0),
                'risks': risk_analysis
            }
            
        except Exception as e:
            logger.error(f"Error analyzing cross-chain opportunity: {str(e)}")
            return {
                'is_viable': False,
                'error': str(e)
            }
    
    def _analyze_mode_specific(
        self,
        source_chain: str,
        target_chain: str,
        gas_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze Mode-specific optimizations"""
        try:
            # Mode has optimized gas usage
            gas_savings = 0.0
            if source_chain == 'mode':
                # Mode uses 20% less gas than standard L2s
                gas_savings = gas_analysis.get('total_cost', 0) * 0.2
            
            # Mode has faster finality
            time_savings = 0
            if target_chain == 'mode':
                # Mode confirms in 5 blocks vs typical 10-15
                time_savings = 300  # 5 minutes faster
            
            return {
                'gas_savings_usd': gas_savings,
                'time_savings_seconds': time_savings,
                'optimized_gas': True,
                'fast_finality': True
            }
            
        except Exception as e:
            logger.error(f"Error in Mode-specific analysis: {str(e)}")
            return {
                'gas_savings_usd': 0,
                'time_savings_seconds': 0,
                'optimized_gas': False,
                'fast_finality': False
            }
    
    def _analyze_sonic_specific(
        self,
        source_chain: str,
        target_chain: str,
        bridge_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze Sonic-specific optimizations"""
        try:
            # Sonic has liquidity pools
            lp_benefits = {}
            if 'liquidity' in bridge_analysis:
                lp_benefits = {
                    'pool_liquidity': bridge_analysis['liquidity'],
                    'lp_fee_reduction': 0.001,  # 0.1% fee reduction
                    'instant_transfers': True
                }
            
            # Sonic has fixed priority fees
            fee_benefits = {
                'fixed_priority_fee': True,
                'priority_fee_gwei': 1,  # 1 gwei
                'predictable_costs': True
            }
            
            return {
                'lp_benefits': lp_benefits,
                'fee_benefits': fee_benefits,
                'high_throughput': True,
                'low_latency': True
            }
            
        except Exception as e:
            logger.error(f"Error in Sonic-specific analysis: {str(e)}")
            return {
                'lp_benefits': {},
                'fee_benefits': {},
                'high_throughput': False,
                'low_latency': False
            }
    
    async def _analyze_cross_chain_prices(
        self,
        token_pair: Tuple[str, str],
        source_chain: str,
        target_chain: str,
        source_web3: Web3,
        target_web3: Web3
    ) -> Dict[str, Any]:
        """Analyze token prices across different chains
        
        Args:
            token_pair: Tuple of token addresses
            source_chain: Source chain name
            target_chain: Target chain name
            source_web3: Web3 connection for source chain
            target_web3: Web3 connection for target chain
            
        Returns:
            Dict containing price analysis
        """
        try:
            # Get price feeds based on chain
            source_price = await self.price_feed_registry.get_price(
                token_pair[0],
                source_chain,
                source_web3
            )
            
            target_price = await self.price_feed_registry.get_price(
                token_pair[1],
                target_chain,
                target_web3
            )
            
            if not source_price or not target_price:
                return {
                    'has_prices': False,
                    'reason': "Failed to get prices"
                }
            
            price_difference = abs(source_price - target_price) / min(source_price, target_price)
            
            return {
                'has_prices': True,
                'source_price': source_price,
                'target_price': target_price,
                'price_difference': price_difference,
                'timestamp': asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing cross-chain prices: {str(e)}")
            return {
                'has_prices': False,
                'reason': f"Price analysis error: {str(e)}"
            }
    
    async def _analyze_bridges(
        self,
        token_pair: Tuple[str, str],
        amount: float,
        source_chain: str,
        target_chain: str
    ) -> Dict[str, Any]:
        """Analyze available bridges"""
        try:
            # Get available bridges
            bridges = await self._get_available_bridges(
                token_pair,
                source_chain,
                target_chain
            )
            
            results = {}
            for bridge in bridges:
                # Get bridge adapter
                adapter_class = get_registered_adapters().get(bridge)
                if not adapter_class:
                    continue
                
                # Create bridge config
                config = self._create_bridge_config(bridge, source_chain, target_chain)
                
                # Initialize adapter
                adapter = adapter_class(config, self.chain_connector.get_web3(source_chain))
                
                # Get bridge state
                state = adapter.get_bridge_state(source_chain, target_chain)
                if state != BridgeState.ACTIVE:
                    continue
                
                # Estimate fees
                fees = adapter.estimate_fees(source_chain, target_chain, token_pair[0], amount)
                
                # Estimate time
                time_estimate = adapter.estimate_time(source_chain, target_chain)
                
                # Check liquidity
                liquidity = adapter.monitor_liquidity(target_chain, token_pair[1])
                
                # Add chain-specific analysis
                if bridge == 'mode':
                    # Mode has optimized gas usage
                    fees['total'] *= 0.8  # 20% lower gas
                    time_estimate = int(time_estimate * 0.7)  # 30% faster
                elif bridge == 'sonic':
                    # Sonic has fixed priority fees
                    fees['priority_fee'] = 1_000_000_000  # 1 gwei
                    if liquidity > amount * 2:  # Good liquidity
                        time_estimate = int(time_estimate * 0.5)  # 50% faster
                
                results[bridge] = {
                    'fees': fees,
                    'estimated_time': time_estimate,
                    'liquidity': liquidity,
                    'state': state
                }
            
            # Find best bridge
            recommended_bridge = self._select_optimal_bridge(
                {b: r['fees'] for b, r in results.items()},
                {b: r['estimated_time'] for b, r in results.items()},
                {b: r['liquidity'] for b, r in results.items()},
                {b: r['state'] for b, r in results.items()}
            )
            
            if recommended_bridge:
                return {
                    'success': True,
                    'recommended_bridge': recommended_bridge,
                    **results[recommended_bridge]
                }
            
            return {
                'success': False,
                'error': 'No suitable bridge found'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing bridges: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _create_bridge_config(
        self,
        bridge_name: str,
        source_chain: str,
        target_chain: str
    ) -> 'BridgeConfig':
        """Create bridge configuration for given parameters
        
        Args:
            bridge_name: Name of the bridge
            source_chain: Source chain ID
            target_chain: Target chain ID
            
        Returns:
            BridgeConfig: Configuration for the bridge
        """
        from src.core.bridge_adapter import BridgeConfig
        
        # Get chain configs
        source_config = self.chain_registry.get_chain_config(source_chain)
        target_config = self.chain_registry.get_chain_config(target_chain)
        
        return BridgeConfig(
            name=bridge_name,
            supported_chains=[source_chain, target_chain],
            min_amount=float(source_config.get('min_transfer_amount', 0.0)),
            max_amount=float(source_config.get('max_transfer_amount', 1e6)),
            fee_multiplier=float(source_config.get('fee_multiplier', 1.0)),
            gas_limit_multiplier=float(source_config.get('gas_limit_multiplier', 1.2)),
            confirmation_blocks=int(source_config.get('confirmation_blocks', 1))
        )
        
    def _select_optimal_bridge(
        self,
        fees: Dict[str, Dict[str, float]],
        times: Dict[str, int],
        liquidity: Dict[str, float],
        states: Dict[str, 'BridgeState']
    ) -> Optional[str]:
        """Select the optimal bridge based on analysis results
        
        Args:
            fees: Bridge fees by bridge name
            times: Estimated transfer times by bridge name
            liquidity: Available liquidity by bridge name
            states: Bridge states by bridge name
            
        Returns:
            str: Name of recommended bridge, or None if no suitable bridge found
        """
        if not fees:
            return None
            
        # Score each bridge
        scores = {}
        for bridge_name in fees.keys():
            if bridge_name not in times or bridge_name not in liquidity:
                continue
                
            # Only consider active bridges
            if states.get(bridge_name) != BridgeState.ACTIVE:
                continue
                
            # Calculate score based on fees, time, and liquidity
            fee_score = 1.0 / (1.0 + fees[bridge_name].get('total', float('inf')))
            time_score = 1.0 / (1.0 + times[bridge_name])
            liquidity_score = min(1.0, liquidity[bridge_name] / 1e6)  # Cap at 1M
            
            # Weighted scoring (can be adjusted based on priorities)
            scores[bridge_name] = (
                0.4 * fee_score +
                0.3 * time_score +
                0.3 * liquidity_score
            )
        
        # Return bridge with highest score
        return max(scores.items(), key=lambda x: x[1])[0] if scores else None
    
    async def _estimate_cross_chain_gas(
        self,
        token_pair: Tuple[str, str],
        amount: float,
        source_chain: str,
        target_chain: str,
        source_web3: Web3,
        target_web3: Web3
    ) -> Dict[str, Any]:
        """Estimate gas costs for cross-chain operations
        
        Args:
            token_pair: Tuple of token addresses
            amount: Amount to transfer
            source_chain: Source chain name
            target_chain: Target chain name
            source_web3: Web3 connection for source chain
            target_web3: Web3 connection for target chain
            
        Returns:
            Dict containing gas estimates
        """
        try:
            # Get gas prices
            source_gas = await self.chain_connector.get_gas_price(source_chain)
            target_gas = await self.chain_connector.get_gas_price(target_chain)
            
            if not source_gas or not target_gas:
                return {
                    'has_estimate': False,
                    'reason': "Failed to get gas prices"
                }
            
            # Estimate gas costs for each operation
            source_approval_gas = await self._estimate_approval_gas(
                token_pair[0],
                source_chain,
                source_web3
            )
            
            bridge_gas = await self._estimate_bridge_gas(
                token_pair,
                amount,
                source_chain,
                target_chain
            )
            
            target_swap_gas = await self._estimate_swap_gas(
                token_pair[1],
                target_chain,
                target_web3
            )
            
            # Calculate total costs
            total_source_cost = source_gas * (source_approval_gas + bridge_gas)
            total_target_cost = target_gas * target_swap_gas
            
            return {
                'has_estimate': True,
                'source_gas_price': source_gas,
                'target_gas_price': target_gas,
                'approval_gas': source_approval_gas,
                'bridge_gas': bridge_gas,
                'swap_gas': target_swap_gas,
                'total_gas_cost': total_source_cost + total_target_cost,
                'estimated_time': self._estimate_total_time(source_chain, target_chain)
            }
            
        except Exception as e:
            logger.error(f"Error estimating cross-chain gas: {str(e)}")
            return {
                'has_estimate': False,
                'reason': f"Gas estimation error: {str(e)}"
            }
    
    def _calculate_profit_potential(
        self,
        price_analysis: Dict[str, Any],
        bridge_analysis: Dict[str, Any],
        gas_analysis: Dict[str, Any],
        amount: float
    ) -> Dict[str, Any]:
        """Calculate potential profit for cross-chain opportunity
        
        Args:
            price_analysis: Price analysis results
            bridge_analysis: Bridge analysis results
            gas_analysis: Gas analysis results
            amount: Amount to trade
            
        Returns:
            Dict containing profit analysis
        """
        try:
            if not all([
                price_analysis.get('has_prices'),
                bridge_analysis.get('has_bridge'),
                gas_analysis.get('has_estimate')
            ]):
                return {
                    'is_profitable': False,
                    'reason': "Missing required analysis data"
                }
            
            # Calculate gross profit
            price_diff = price_analysis['price_difference']
            gross_profit = amount * price_diff
            
            # Calculate costs
            bridge_fee = bridge_analysis['bridge_fee']
            gas_cost = gas_analysis['total_gas_cost']
            total_cost = bridge_fee + gas_cost
            
            # Calculate net profit
            net_profit = gross_profit - total_cost
            
            return {
                'is_profitable': net_profit > 0,
                'estimated_profit': net_profit,
                'gross_profit': gross_profit,
                'total_cost': total_cost,
                'bridge_fee': bridge_fee,
                'gas_cost': gas_cost,
                'roi': (net_profit / total_cost) if total_cost > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating profit potential: {str(e)}")
            return {
                'is_profitable': False,
                'reason': f"Profit calculation error: {str(e)}"
            }
    
    def _assess_risks(
        self,
        price_analysis: Dict[str, Any],
        bridge_analysis: Dict[str, Any],
        gas_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess risks for cross-chain opportunity
        
        Args:
            price_analysis: Price analysis results
            bridge_analysis: Bridge analysis results
            gas_analysis: Gas analysis results
            
        Returns:
            Dict containing risk assessment
        """
        risks = {
            'price_volatility': self._calculate_price_volatility_risk(price_analysis),
            'bridge_risk': self._calculate_bridge_risk(bridge_analysis),
            'gas_risk': self._calculate_gas_risk(gas_analysis),
            'execution_risk': self._calculate_execution_risk(
                price_analysis,
                bridge_analysis,
                gas_analysis
            )
        }
        
        # Calculate overall risk score (0-1)
        risk_weights = {
            'price_volatility': 0.3,
            'bridge_risk': 0.3,
            'gas_risk': 0.2,
            'execution_risk': 0.2
        }
        
        overall_risk = sum(
            risk * risk_weights[risk_type]
            for risk_type, risk in risks.items()
        )
        
        return {
            **risks,
            'overall_risk': overall_risk
        }
    
    async def _get_available_bridges(
        self,
        token_pair: Tuple[str, str],
        source_chain: str,
        target_chain: str
    ) -> List[str]:
        """Get list of available bridges for given token pair and chains
        
        Args:
            token_pair: Tuple of (source_token, target_token)
            source_chain: Source chain ID
            target_chain: Target chain ID
            
        Returns:
            List[str]: List of available bridge names
        """
        available_bridges = []
        registered_adapters = get_registered_adapters()
        
        for bridge_name, adapter_class in registered_adapters.items():
            try:
                # Create bridge config
                config = self._create_bridge_config(bridge_name, source_chain, target_chain)
                
                # Get web3 instance for source chain
                web3 = self.chain_connector.get_web3(source_chain)
                
                # Initialize adapter
                adapter = adapter_class(config, web3)
                
                # Validate if bridge supports this transfer
                if adapter.validate_transfer(
                    source_chain,
                    target_chain,
                    token_pair[0],  # source token
                    0.0  # Use 0 amount for validation
                ):
                    available_bridges.append(bridge_name)
                    
            except Exception as e:
                logger.error(f"Error checking bridge {bridge_name}: {str(e)}")
                continue
                
        return available_bridges
    
    async def _estimate_approval_gas(
        self,
        token: str,
        chain: str,
        web3: Web3
    ) -> int:
        """Estimate gas for token approval"""
        # Implementation depends on your token contracts
        # This is a placeholder that should be implemented based on your needs
        return 50000
    
    async def _estimate_bridge_gas(
        self,
        token_pair: Tuple[str, str],
        amount: float,
        source_chain: str,
        target_chain: str
    ) -> int:
        """Estimate gas for bridge operation"""
        # Implementation depends on your bridge contracts
        # This is a placeholder that should be implemented based on your needs
        return 150000
    
    async def _estimate_swap_gas(
        self,
        token: str,
        chain: str,
        web3: Web3
    ) -> int:
        """Estimate gas for swap operation"""
        # Implementation depends on your DEX contracts
        # This is a placeholder that should be implemented based on your needs
        return 100000
    
    def _estimate_total_time(
        self,
        source_chain: str,
        target_chain: str
    ) -> int:
        """Estimate total time for cross-chain operation"""
        source_block_time = self.chain_registry.get_chain(source_chain).block_time
        target_block_time = self.chain_registry.get_chain(target_chain).block_time
        
        # Estimate based on block times and typical bridge delays
        bridge_delay = 300  # 5 minutes typical bridge delay
        return max(source_block_time, target_block_time) * 5 + bridge_delay
    
    def _calculate_price_volatility_risk(
        self,
        price_analysis: Dict[str, Any]
    ) -> float:
        """Calculate risk score for price volatility"""
        if not price_analysis.get('has_prices'):
            return 1.0
            
        price_diff = price_analysis['price_difference']
        return min(price_diff / 0.05, 1.0)  # Risk increases with price difference
    
    def _calculate_bridge_risk(
        self,
        bridge_analysis: Dict[str, Any]
    ) -> float:
        """Calculate risk score for bridge operation"""
        if not bridge_analysis.get('has_bridge'):
            return 1.0
            
        liquidity = bridge_analysis['available_liquidity']
        return max(1.0 - (liquidity / 1000000), 0.0)  # Risk decreases with liquidity
    
    def _calculate_gas_risk(
        self,
        gas_analysis: Dict[str, Any]
    ) -> float:
        """Calculate risk score for gas costs"""
        if not gas_analysis.get('has_estimate'):
            return 1.0
            
        total_gas = gas_analysis['total_gas_cost']
        return min(total_gas / 1000000, 1.0)  # Risk increases with gas cost
    
    def _calculate_execution_risk(
        self,
        price_analysis: Dict[str, Any],
        bridge_analysis: Dict[str, Any],
        gas_analysis: Dict[str, Any]
    ) -> float:
        """Calculate risk score for execution"""
        estimated_time = gas_analysis.get('estimated_time', 600)  # Default 10 minutes
        return min(estimated_time / 900, 1.0)  # Risk increases with execution time
    
    async def monitor_transactions_parallel(
        self,
        tx_hashes: Dict[str, List[str]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Monitor multiple transactions across chains in parallel
        
        Args:
            tx_hashes: Dict mapping chain names to lists of transaction hashes
            
        Returns:
            Dict mapping chains to lists of transaction statuses
        """
        tasks = []
        for chain, hashes in tx_hashes.items():
            for tx_hash in hashes:
                tasks.append(self._get_transaction_status(chain, tx_hash))
                
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Organize results by chain
        tx_statuses = {}
        current_index = 0
        
        for chain, hashes in tx_hashes.items():
            chain_results = []
            for _ in hashes:
                if isinstance(results[current_index], Exception):
                    logger.error(f"Error monitoring transaction: {str(results[current_index])}")
                    chain_results.append({
                        'status': 'error',
                        'error': str(results[current_index])
                    })
                else:
                    chain_results.append(results[current_index])
                current_index += 1
            tx_statuses[chain] = chain_results
            
        return tx_statuses
    
    async def _get_transaction_status(
        self,
        chain: str,
        tx_hash: str
    ) -> Dict[str, Any]:
        """Get status of a specific transaction"""
        try:
            web3 = await self.chain_connector.get_connection(chain)
            if not web3:
                raise ValueError(f"Failed to get connection for {chain}")
                
            # Convert string hash to bytes
            tx_hash_bytes = HexBytes(tx_hash)
            
            # Get transaction receipt and details
            receipt = cast(TxReceipt, web3.eth.wait_for_transaction_receipt(tx_hash_bytes))
            tx = cast(TxData, web3.eth.get_transaction(tx_hash_bytes))
            current_block = web3.eth.block_number
            
            # Update metrics
            metrics = self.metrics.chain_metrics.get(chain, ChainMetrics())
            metrics.total_txs += 1
            
            # Safely access receipt status
            receipt_dict = dict(receipt)
            status = receipt_dict.get('status')
            if status == 1:
                metrics.success_rate = (
                    (metrics.success_rate * (metrics.total_txs - 1) + 1) / 
                    metrics.total_txs
                )
            else:
                metrics.failed_txs += 1
            
            metrics.last_updated = time.time()
            self.metrics.update_chain_metrics(chain, metrics)
            
            # Calculate confirmations
            block_number = receipt_dict.get('blockNumber', 0)
            confirmations = int(current_block) - int(block_number) if block_number is not None else 0
            
            # Convert transaction data to dict and handle types
            tx_dict = dict(tx)
            
            return {
                'status': 'success' if status == 1 else 'failed',
                'block_number': block_number,
                'gas_used': receipt_dict.get('gasUsed'),
                'effective_gas_price': receipt_dict.get('effectiveGasPrice'),
                'confirmations': confirmations,
                'value': tx_dict.get('value'),
                'from': tx_dict.get('from'),
                'to': tx_dict.get('to')
            }
            
        except Exception as e:
            logger.error(f"Error getting transaction status: {str(e)}")
            if isinstance(e, (ValueError, TypeError)):
                raise e
            if hasattr(e, 'args') and len(e.args) > 0:
                raise ValueError(str(e.args[0]))
            raise ValueError(str(e))
    
    async def validate_transactions_parallel(
        self,
        transactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate multiple transactions in parallel before execution
        
        Args:
            transactions: List of transaction parameters to validate
            
        Returns:
            List of validation results
        """
        tasks = []
        for tx in transactions:
            tasks.append(self._validate_transaction(tx))
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        validation_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                validation_results.append({
                    'transaction': transactions[i],
                    'is_valid': False,
                    'error': str(result)
                })
            else:
                validation_results.append({
                    'transaction': transactions[i],
                    'is_valid': True,
                    'validation_data': result
                })
                
        return validation_results
    
    async def _validate_transaction(
        self,
        transaction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate a single transaction"""
        try:
            chain = transaction.get('chain')
            if not chain:
                raise ValueError("Chain not specified in transaction")
                
            web3 = await self.chain_connector.get_connection(chain)
            if not web3:
                raise ValueError(f"Failed to get connection for {chain}")
                
            # Validate sender balance
            sender = transaction.get('from')
            if not sender:
                raise ValueError("Sender address not specified")
                
            # Convert transaction dict to TxParams
            tx_params = {
                'from': Web3.to_checksum_address(sender),
                'to': Web3.to_checksum_address(transaction.get('to', '')),
                'value': Wei(transaction.get('value', 0)),
                'gas': Wei(transaction.get('gas', 21000)),
                'maxFeePerGas': Wei(transaction.get('maxFeePerGas', 0)),
                'maxPriorityFeePerGas': Wei(transaction.get('maxPriorityFeePerGas', 0))
            }
            
            if 'nonce' in transaction:
                tx_params['nonce'] = transaction['nonce']
            
            # Cast to TxParams after building the dict
            typed_tx_params = cast(TxParams, tx_params)
            
            # Get balance and gas estimates
            balance = web3.eth.get_balance(sender)
            estimated_gas = web3.eth.estimate_gas(typed_tx_params)
            gas_price = web3.eth.gas_price
            
            # Safely access transaction value
            tx_value = Wei(typed_tx_params.get('value', 0))
            total_cost = Wei(int(tx_value) + (int(estimated_gas) * int(gas_price)))
            
            if balance < total_cost:
                raise ValueError(f"Insufficient balance: {balance} < {total_cost}")
                
            # Validate nonce
            current_nonce = web3.eth.get_transaction_count(sender)
            tx_nonce = transaction.get('nonce')
            if tx_nonce is not None and tx_nonce < current_nonce:
                raise ValueError(f"Invalid nonce: {tx_nonce} < {current_nonce}")
                
            return {
                'estimated_gas': estimated_gas,
                'gas_price': gas_price,
                'total_cost': total_cost,
                'current_nonce': current_nonce,
                'balance': balance
            }
            
        except Exception as e:
            logger.error(f"Error validating transaction: {str(e)}")
            raise
            
    async def estimate_cross_chain_costs(
        self,
        source_chain: str,
        target_chain: str,
        token_amount: float,
        token_address: str
    ) -> Dict[str, Any]:
        """Estimate costs for cross-chain transaction"""
        try:
            # Fetch gas prices and bridge fees in parallel
            source_gas_price, target_gas_price, bridge_fee = await asyncio.gather(
                self.chain_connector.get_gas_price(source_chain),
                self.chain_connector.get_gas_price(target_chain),
                self._get_bridge_fee(source_chain, target_chain, token_amount, token_address)
            )
            
            # Convert gas prices to Wei if they aren't already
            source_gas_price_wei = Wei(int(source_gas_price or 0))
            target_gas_price_wei = Wei(int(target_gas_price or 0))
            
            # Estimate gas costs (using integers for Wei conversion)
            source_gas = Wei(100000)  # Example gas estimate for source chain
            target_gas = Wei(80000)   # Example gas estimate for target chain
            
            # Convert bridge fee to Wei
            bridge_fee_wei = Wei(int(bridge_fee))
            
            return {
                'source_chain_cost': source_gas * source_gas_price_wei,
                'target_chain_cost': target_gas * target_gas_price_wei,
                'bridge_fee': bridge_fee_wei,
                'total_cost': (source_gas * source_gas_price_wei) + 
                            (target_gas * target_gas_price_wei) + 
                            bridge_fee_wei
            }
            
        except Exception as e:
            logger.error(f"Error estimating cross-chain costs: {str(e)}")
            raise
    
    async def _get_bridge_fee(
        self,
        source_chain: str,
        target_chain: str,
        amount: float,
        token_address: str
    ) -> float:
        """Get bridge fee for cross-chain transfer
        
        Args:
            source_chain: Source chain name
            target_chain: Target chain name
            amount: Amount to transfer
            token_address: Token address
            
        Returns:
            Bridge fee amount
        """
        try:
            # Implementation would check actual bridge contracts
            # This is a placeholder
            return amount * 0.001  # Example 0.1% fee
            
        except Exception as e:
            logger.error(f"Error getting bridge fee: {str(e)}")
            raise
    
    async def record_opportunity_execution(
        self,
        chain: str,
        execution_time: float,
        profit_realized: float,
        expected_profit: float,
        gas_used: float,
        estimated_gas: float,
        slippage: float,
        bridge_latency: float,
        success: bool,
        bridge_name: Optional[str] = None
    ) -> None:
        """Record metrics from opportunity execution"""
        # Get or create chain metrics
        chain_metrics = self.metrics.chain_metrics.get(chain)
        if not chain_metrics:
            chain_metrics = ChainMetrics()
            self.metrics.chain_metrics[chain] = chain_metrics
        
        # Update base metrics
        chain_metrics.total_txs += 1
        if not success:
            chain_metrics.failed_txs += 1
            chain_metrics.bridge_failures += 1
        
        chain_metrics.success_rate = (
            (chain_metrics.total_txs - chain_metrics.failed_txs) /
            chain_metrics.total_txs
        )
        
        chain_metrics.avg_execution_time = (
            (chain_metrics.avg_execution_time * (chain_metrics.total_txs - 1) + execution_time) /
            chain_metrics.total_txs
        )
        
        chain_metrics.avg_profit_margin = (
            (chain_metrics.avg_profit_margin * (chain_metrics.total_txs - 1) + (profit_realized / expected_profit if expected_profit > 0 else 0)) /
            chain_metrics.total_txs
        )
        
        if slippage > 0.02:  # 2% slippage threshold
            chain_metrics.slippage_incidents += 1
        
        # Update gas usage metrics
        if gas_used > estimated_gas * 1.2:  # 20% over estimate
            chain_metrics.optimal_gas_usage = max(0.0, chain_metrics.optimal_gas_usage - 0.1)
        else:
            chain_metrics.optimal_gas_usage = min(1.0, chain_metrics.optimal_gas_usage + 0.05)
        
        # Update Mode-specific metrics
        if chain == 'mode' or (bridge_name and bridge_name == 'mode'):
            chain_metrics.mode_bridge_usage += 1
            gas_savings = estimated_gas * 0.2  # Mode uses 20% less gas
            chain_metrics.mode_gas_savings = (
                (chain_metrics.mode_gas_savings * (chain_metrics.mode_bridge_usage - 1) + gas_savings) /
                chain_metrics.mode_bridge_usage
            )
            chain_metrics.mode_finality_time = (
                (chain_metrics.mode_finality_time * (chain_metrics.mode_bridge_usage - 1) + bridge_latency) /
                chain_metrics.mode_bridge_usage
            )
        
        # Update Sonic-specific metrics
        if chain == 'sonic' or (bridge_name and bridge_name == 'sonic'):
            if success:
                chain_metrics.sonic_bridge_volume += profit_realized
                if bridge_latency < 300:  # Less than 5 minutes
                    chain_metrics.sonic_lp_volume += profit_realized
            fee_savings = estimated_gas * 0.1  # Sonic's fixed priority fees save ~10%
            chain_metrics.sonic_fee_savings = (
                (chain_metrics.sonic_fee_savings * chain_metrics.total_txs + fee_savings) /
                (chain_metrics.total_txs + 1)
        
        # Update timestamp
        chain_metrics.last_updated = time.time()
        
        # Generate learning feedback
        await self._generate_learning_feedback(chain)
    
    def get_learning_feedback(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get learning feedback for all chains"""
        return self.metrics.learning_feedback
    
    def get_chain_recommendations(self, chain: str) -> List[str]:
        """Get latest recommendations for a chain"""
        feedback = self.metrics.learning_feedback.get(chain, [])
        if not feedback:
            return []
        return feedback[-1].get('recommendations', []) 