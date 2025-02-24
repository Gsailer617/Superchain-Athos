import logging
from typing import Dict, List, Tuple, Any, Optional, Union, Sequence
import numpy as np
from scipy import stats
import time
from datetime import datetime, timedelta
from decimal import Decimal
from ..market.types import Position

logger = logging.getLogger(__name__)

class RiskAnalysis:
    """Centralized risk analysis and calculations"""
    
    @staticmethod
    def _convert_to_float_array(data: Sequence[Union[int, float]]) -> np.ndarray:
        """Convert sequence to float numpy array"""
        return np.array(data, dtype=np.float64)

    @staticmethod
    def calculate_position_health_risk(position: Position) -> float:
        """Calculate position health risk"""
        try:
            if position.health_factor == float('inf'):
                return 0.0
                
            # Higher risk as health factor approaches 1
            return max(0.0, min(1.0, 1.0 / position.health_factor))
            
        except Exception as e:
            logger.error(f"Error calculating health risk: {str(e)}")
            return 1.0

    @staticmethod
    def calculate_liquidation_risk(position: Position) -> float:
        """Calculate position liquidation risk"""
        try:
            if position.borrowed == 0:
                return 0.0
                
            current_ratio = float(position.supplied / position.borrowed)
            liquidation_threshold = getattr(position, 'liquidation_threshold', 0.8)
            threshold_ratio = 1.0 / float(liquidation_threshold)
            
            risk = max(0.0, min(1.0, 1.0 - (current_ratio / threshold_ratio)))
            return float(risk)
            
        except Exception as e:
            logger.error(f"Error calculating liquidation risk: {str(e)}")
            return 1.0

    @staticmethod
    def calculate_market_risk(market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive market risk metrics"""
        try:
            # Get volatility metrics
            vol_metrics = RiskAnalysis.calculate_volatility_metrics(market_data)
            volatility = vol_metrics['volatility']
            
            # Calculate liquidity risk
            liquidity_risk = RiskAnalysis.analyze_liquidity_depth(
                market_data.get('amount', 0),
                market_data
            )
            
            # Calculate network risk
            network_risk = RiskAnalysis.calculate_network_congestion(market_data)
            
            # Calculate overall market risk
            overall_risk = (
                volatility * 0.4 +
                liquidity_risk * 0.4 +
                network_risk * 0.2
            )
            
            return {
                'volatility': float(volatility),
                'liquidity_risk': float(liquidity_risk),
                'network_risk': float(network_risk),
                'overall_risk': float(overall_risk)
            }
            
        except Exception as e:
            logger.error(f"Error calculating market risk: {str(e)}")
            return {
                'volatility': 1.0,
                'liquidity_risk': 1.0,
                'network_risk': 1.0,
                'overall_risk': 1.0
            }

    @staticmethod
    def estimate_price_impact(
        amount: float,
        market_data: Dict[str, Any]
    ) -> float:
        """Estimate price impact of a trade
        
        Args:
            amount: Trade amount
            market_data: Market data dictionary
            
        Returns:
            Estimated price impact as a fraction
        """
        try:
            liquidity = float(market_data.get('liquidity', 0))
            if liquidity <= 0:
                return 1.0
                
            # Calculate impact using square root formula
            impact = amount / (2 * liquidity)
            return min(1.0, impact ** 0.5)
            
        except Exception as e:
            logger.error(f"Error estimating price impact: {str(e)}")
            return 1.0
            
    @staticmethod
    def calculate_slippage_risk(market_data: Dict[str, Any]) -> float:
        """Calculate slippage risk based on market conditions
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Slippage risk score between 0 and 1
        """
        try:
            # Get relevant metrics
            volatility = float(market_data.get('volatility_24h', 0))
            volume = float(market_data.get('volume_24h', 0))
            liquidity = float(market_data.get('liquidity', 0))
            
            if volume <= 0 or liquidity <= 0:
                return 1.0
                
            # Calculate components
            vol_factor = min(volatility / 100, 1.0)
            liq_factor = min(1e6 / liquidity, 1.0) if liquidity > 0 else 1.0
            vol_factor = min(1e6 / volume, 1.0) if volume > 0 else 1.0
            
            # Weighted combination
            risk = (
                0.4 * vol_factor +
                0.4 * liq_factor +
                0.2 * vol_factor
            )
            
            return float(risk)
            
        except Exception as e:
            logger.error(f"Error calculating slippage risk: {str(e)}")
            return 1.0
            
    @staticmethod
    def calculate_volatility_metrics(market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive volatility metrics"""
        try:
            prices = market_data.get('price_history', [])
            if not prices:
                return {
                    'volatility': 1.0,
                    'trend': 0.0,
                    'momentum': 0.0
                }
                
            prices_array = RiskAnalysis._convert_to_float_array(prices)
            returns = np.diff(np.log(prices_array))
            
            # Calculate metrics
            volatility = float(np.std(returns) * np.sqrt(365 * 24))
            
            # Calculate trend using float arrays
            x = np.arange(len(prices_array), dtype=np.float64)
            slope, _, r_value, _, _ = stats.linregress(x, prices_array)
            trend = float(slope) * float(len(prices_array)) / float(prices_array[-1])
            
            # Calculate momentum
            momentum = float(
                (prices_array[-1] / prices_array[0] - 1) if len(prices_array) > 1 else 0
            )
            
            return {
                'volatility': min(float(volatility), 1.0),
                'trend': float(trend),
                'momentum': float(momentum)
            }
            
        except Exception as e:
            logger.error(f"Error calculating volatility metrics: {str(e)}")
            return {
                'volatility': 1.0,
                'trend': 0.0,
                'momentum': 0.0
            }
            
    @staticmethod
    def analyze_liquidity_depth(
        amount: float,
        market_data: Dict[str, Any]
    ) -> float:
        """Analyze liquidity depth for a given amount
        
        Args:
            amount: Trade amount
            market_data: Market data dictionary
            
        Returns:
            Liquidity risk score between 0 and 1
        """
        try:
            # Get liquidity data
            liquidity = float(market_data.get('liquidity', 0))
            volume_24h = float(market_data.get('volume_24h', 0))
            
            if liquidity <= 0 or volume_24h <= 0:
                return 1.0
                
            # Calculate metrics
            depth_ratio = amount / liquidity
            turnover_ratio = volume_24h / liquidity
            
            # Combine metrics
            risk = (
                0.6 * min(depth_ratio, 1.0) +
                0.4 * (1.0 / (1.0 + turnover_ratio))
            )
            
            return float(risk)
            
        except Exception as e:
            logger.error(f"Error analyzing liquidity depth: {str(e)}")
            return 1.0
            
    @staticmethod
    def calculate_network_congestion(market_data: Dict[str, Any]) -> float:
        """Calculate network congestion risk
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Congestion risk score between 0 and 1
        """
        try:
            # Get network metrics
            gas_price = float(market_data.get('gas_price', 0))
            pending_tx = float(market_data.get('pending_tx_count', 0))
            block_time = float(market_data.get('block_time', 0))
            
            # Calculate components
            gas_factor = min(gas_price / 200, 1.0)  # Normalized to 200 gwei
            pending_factor = min(pending_tx / 10000, 1.0)  # Normalized to 10k tx
            block_factor = min(block_time / 15, 1.0)  # Normalized to 15 seconds
            
            # Weighted combination
            congestion = (
                0.4 * gas_factor +
                0.4 * pending_factor +
                0.2 * block_factor
            )
            
            return float(congestion)
            
        except Exception as e:
            logger.error(f"Error calculating network congestion: {str(e)}")
            return 1.0
            
    @staticmethod
    def analyze_block_time_stability(market_data: Dict[str, Any]) -> float:
        """Analyze block time stability"""
        try:
            block_times = market_data.get('block_time_history', [])
            if not block_times:
                return 1.0
                
            block_times_array = RiskAnalysis._convert_to_float_array(block_times)
            
            # Calculate statistics
            mean_time = float(np.mean(block_times_array))
            std_time = float(np.std(block_times_array))
            cv = float(std_time / mean_time) if mean_time > 0 else 1.0
            
            # Calculate trend using float arrays
            x = np.arange(len(block_times_array), dtype=np.float64)
            slope, _, r_value, _, _ = stats.linregress(x, block_times_array)
            trend = float(slope) * float(len(block_times_array)) / float(mean_time)
            
            # Combine metrics with explicit float casting
            risk = float(
                0.5 * min(float(cv), 1.0) +
                0.5 * min(abs(float(trend)), 1.0)
            )
            
            return risk
            
        except Exception as e:
            logger.error(f"Error analyzing block time stability: {str(e)}")
            return 1.0
            
    @staticmethod
    def calculate_execution_complexity(amount: float) -> float:
        """Calculate execution complexity risk
        
        Args:
            amount: Trade amount
            
        Returns:
            Complexity risk score between 0 and 1
        """
        try:
            # Define thresholds
            low_threshold = 1000  # $1k
            high_threshold = 100000  # $100k
            
            if amount <= low_threshold:
                return 0.1
            elif amount >= high_threshold:
                return 1.0
            else:
                # Linear interpolation
                return 0.1 + 0.9 * (amount - low_threshold) / (high_threshold - low_threshold)
                
        except Exception as e:
            logger.error(f"Error calculating execution complexity: {str(e)}")
            return 1.0
            
    @staticmethod
    def estimate_timing_risk(
        token_pair: Tuple[str, str],
        market_data: Optional[Dict[str, Any]] = None
    ) -> float:
        """Estimate timing risk for execution
        
        Args:
            token_pair: Token addresses
            market_data: Optional market data
            
        Returns:
            Timing risk score between 0 and 1
        """
        try:
            if not market_data:
                return 0.5
                
            # Get relevant metrics
            volatility = float(market_data.get('volatility_24h', 0))
            volume_profile = market_data.get('volume_profile', {})
            current_hour = datetime.now().hour
            
            # Calculate volatility component
            vol_risk = min(volatility / 100, 1.0)
            
            # Calculate time-of-day component
            hour_vol = volume_profile.get(str(current_hour), 0)
            avg_vol = np.mean(list(volume_profile.values())) if volume_profile else 0
            time_risk = 1.0 - (hour_vol / avg_vol) if avg_vol > 0 else 0.5
            
            # Combine risks
            risk = 0.7 * vol_risk + 0.3 * time_risk
            
            return float(risk)
            
        except Exception as e:
            logger.error(f"Error estimating timing risk: {str(e)}")
            return 0.5
            
    @staticmethod
    def calculate_protocol_risk(metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate protocol-specific risk metrics"""
        try:
            # Calculate TVL risk
            tvl = float(metrics.get('tvl', 0))
            tvl_risk = 1.0 / (1.0 + tvl / 1e6)  # Normalized to $1M TVL
            
            # Calculate utilization risk
            total_borrowed = float(metrics.get('total_borrowed', 0))
            total_supplied = float(metrics.get('total_supplied', 1))  # Avoid div by 0
            utilization_risk = min(total_borrowed / total_supplied, 1.0)
            
            # Calculate protocol health risk
            health_score = float(metrics.get('health', 0))
            health_risk = 1.0 - min(health_score / 100, 1.0)
            
            # Calculate smart contract risk
            audit_score = float(metrics.get('audit_score', 0))
            bug_bounty = float(metrics.get('bug_bounty', 0))
            age_days = float(metrics.get('age', 0))
            
            contract_risk = (
                (1.0 - min(audit_score / 100, 1.0)) * 0.4 +
                (1.0 / (1.0 + bug_bounty / 1e6)) * 0.3 +  # Normalized to $1M bounty
                (1.0 / (1.0 + age_days / 365)) * 0.3  # Normalized to 1 year
            )
            
            # Calculate overall protocol risk
            overall_risk = (
                tvl_risk * 0.25 +
                utilization_risk * 0.25 +
                health_risk * 0.25 +
                contract_risk * 0.25
            )
            
            return {
                'tvl_risk': float(tvl_risk),
                'utilization_risk': float(utilization_risk),
                'health_risk': float(health_risk),
                'contract_risk': float(contract_risk),
                'overall_risk': float(overall_risk)
            }
            
        except Exception as e:
            logger.error(f"Error calculating protocol risk: {str(e)}")
            return {
                'tvl_risk': 1.0,
                'utilization_risk': 1.0,
                'health_risk': 1.0,
                'contract_risk': 1.0,
                'overall_risk': 1.0
            }

class StressTestSimulator:
    """Simulates stress conditions for testing"""
    
    @staticmethod
    async def test_high_volatility(
        token_pair: Tuple[str, str],
        amount: float,
        num_trials: int = 100
    ) -> Dict[str, Any]:
        """Test strategy under high volatility conditions
        
        Args:
            token_pair: Token addresses
            amount: Trade amount
            num_trials: Number of simulation trials
            
        Returns:
            Dict containing test results
        """
        try:
            # Simulation parameters
            volatility_multiplier = 3.0
            success_threshold = 0.02  # 2% slippage tolerance
            
            successes = 0
            slippages = []
            drawdowns = []
            recovery_times = []
            gas_costs = []
            errors = 0
            
            for _ in range(num_trials):
                try:
                    # Simulate volatile price movement
                    price_change = np.random.normal(0, volatility_multiplier * 0.02)
                    simulated_slippage = abs(price_change)
                    
                    # Check if trade would succeed
                    if simulated_slippage <= success_threshold:
                        successes += 1
                        slippages.append(simulated_slippage)
                        
                        # Simulate drawdown and recovery
                        drawdown = abs(price_change) * amount
                        drawdowns.append(drawdown)
                        
                        recovery_time = np.random.exponential(300)  # Mean 5 minutes
                        recovery_times.append(recovery_time)
                        
                        # Simulate gas costs
                        gas_cost = np.random.uniform(0.8, 1.2)  # ±20% variation
                        gas_costs.append(gas_cost)
                    else:
                        errors += 1
                        
                except Exception:
                    errors += 1
                    
            # Calculate results
            return {
                'success_rate': successes / num_trials,
                'avg_slippage': np.mean(slippages) if slippages else 1.0,
                'max_drawdown': max(drawdowns) if drawdowns else amount,
                'recovery_time': np.mean(recovery_times) if recovery_times else 3600,
                'gas_efficiency': np.mean(gas_costs) if gas_costs else 0.0,
                'error_rate': errors / num_trials
            }
            
        except Exception as e:
            logger.error(f"Error in volatility stress test: {str(e)}")
            return {
                'success_rate': 0.0,
                'avg_slippage': 1.0,
                'max_drawdown': amount,
                'recovery_time': 3600,
                'gas_efficiency': 0.0,
                'error_rate': 1.0
            }
            
    @staticmethod
    async def test_network_congestion(
        token_pair: Tuple[str, str],
        num_trials: int = 100
    ) -> Dict[str, Any]:
        """Test strategy under network congestion
        
        Args:
            token_pair: Token addresses
            num_trials: Number of simulation trials
            
        Returns:
            Dict containing test results
        """
        try:
            # Simulation parameters
            base_gas_price = 50  # gwei
            congestion_multiplier = 5
            timeout_threshold = 600  # 10 minutes
            
            successes = 0
            recovery_times = []
            gas_costs = []
            errors = 0
            
            for _ in range(num_trials):
                try:
                    # Simulate congested network
                    gas_price = base_gas_price * (1 + np.random.exponential(congestion_multiplier))
                    block_time = np.random.exponential(15)  # Mean 15 seconds
                    
                    # Check if transaction would succeed
                    if block_time <= timeout_threshold:
                        successes += 1
                        
                        # Simulate recovery
                        recovery_time = block_time * np.random.uniform(1, 5)
                        recovery_times.append(recovery_time)
                        
                        # Calculate gas efficiency
                        gas_efficiency = base_gas_price / gas_price
                        gas_costs.append(gas_efficiency)
                    else:
                        errors += 1
                        
                except Exception:
                    errors += 1
                    
            # Calculate results
            return {
                'success_rate': successes / num_trials,
                'avg_slippage': 0.0,  # Not applicable
                'max_drawdown': 0.0,  # Not applicable
                'recovery_time': np.mean(recovery_times) if recovery_times else timeout_threshold,
                'gas_efficiency': np.mean(gas_costs) if gas_costs else 0.0,
                'error_rate': errors / num_trials
            }
            
        except Exception as e:
            logger.error(f"Error in congestion stress test: {str(e)}")
            return {
                'success_rate': 0.0,
                'avg_slippage': 0.0,
                'max_drawdown': 0.0,
                'recovery_time': timeout_threshold,
                'gas_efficiency': 0.0,
                'error_rate': 1.0
            }
            
    @staticmethod
    async def test_low_liquidity(
        token_pair: Tuple[str, str],
        amount: float,
        num_trials: int = 100
    ) -> Dict[str, Any]:
        """Test strategy under low liquidity conditions
        
        Args:
            token_pair: Token addresses
            amount: Trade amount
            num_trials: Number of simulation trials
            
        Returns:
            Dict containing test results
        """
        try:
            # Simulation parameters
            liquidity_multiplier = 0.2  # Reduce liquidity to 20%
            max_acceptable_impact = 0.05  # 5% max price impact
            
            successes = 0
            slippages = []
            drawdowns = []
            recovery_times = []
            gas_costs = []
            errors = 0
            
            for _ in range(num_trials):
                try:
                    # Simulate reduced liquidity
                    available_liquidity = amount * 2 * liquidity_multiplier * np.random.uniform(0.5, 1.5)
                    price_impact = amount / (2 * available_liquidity) if available_liquidity > 0 else 1.0
                    
                    # Check if trade would succeed
                    if price_impact <= max_acceptable_impact:
                        successes += 1
                        slippages.append(price_impact)
                        
                        # Simulate drawdown
                        drawdown = price_impact * amount
                        drawdowns.append(drawdown)
                        
                        # Simulate recovery
                        recovery_time = np.random.exponential(900)  # Mean 15 minutes
                        recovery_times.append(recovery_time)
                        
                        # Simulate gas costs
                        gas_cost = np.random.uniform(0.6, 1.4)  # ±40% variation
                        gas_costs.append(gas_cost)
                    else:
                        errors += 1
                        
                except Exception:
                    errors += 1
                    
            # Calculate results
            return {
                'success_rate': successes / num_trials,
                'avg_slippage': np.mean(slippages) if slippages else 1.0,
                'max_drawdown': max(drawdowns) if drawdowns else amount,
                'recovery_time': np.mean(recovery_times) if recovery_times else 3600,
                'gas_efficiency': np.mean(gas_costs) if gas_costs else 0.0,
                'error_rate': errors / num_trials
            }
            
        except Exception as e:
            logger.error(f"Error in liquidity stress test: {str(e)}")
            return {
                'success_rate': 0.0,
                'avg_slippage': 1.0,
                'max_drawdown': amount,
                'recovery_time': 3600,
                'gas_efficiency': 0.0,
                'error_rate': 1.0
            } 