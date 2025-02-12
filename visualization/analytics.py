"""
Analytics and Data Processing Module

This module provides comprehensive data analysis and processing capabilities
for the visualization dashboard, including trend detection, statistical analysis,
and real-time metric calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import structlog
from dataclasses import dataclass
from scipy import stats
import asyncio
from prometheus_client import Histogram, Counter

logger = structlog.get_logger(__name__)

# Analytics metrics
METRICS = {
    'analysis_time': Histogram(
        'analytics_processing_seconds',
        'Time spent on data analysis',
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
    ),
    'outlier_count': Counter(
        'analytics_outliers_total',
        'Total number of detected outliers'
    )
}

@dataclass
class AnalyticsConfig:
    """Configuration for analytics calculations"""
    trend_window: int = 20  # Window size for trend calculations
    outlier_threshold: float = 2.5  # Z-score threshold for outliers
    min_data_points: int = 5  # Minimum points needed for calculations
    volatility_window: int = 24  # Hours for volatility calculation
    momentum_window: int = 12  # Hours for momentum calculation
    risk_free_rate: float = 0.0  # Risk-free rate for Sharpe ratio
    confidence_level: float = 0.95  # Confidence level for statistical tests
    update_interval: int = 60  # Seconds between analytics updates
    profit_window: int = 24  # Hours
    gas_efficiency_threshold: float = 0.8
    min_success_rate: float = 0.7
    enable_advanced_metrics: bool = True

class DataAnalyzer:
    """
    Comprehensive data analysis system for arbitrage metrics.
    
    Features:
    - Real-time metric calculations
    - Trend detection and analysis
    - Statistical analysis and outlier detection
    - Performance metrics and risk assessment
    - Automated alerts and notifications
    """
    
    def __init__(self, config: Optional[AnalyticsConfig] = None):
        """Initialize analyzer with configuration"""
        self.config = config or AnalyticsConfig()
        self.last_update = datetime.now()
        self.cached_metrics: Dict[str, Any] = {}
        self._setup_metrics()
    
    def _setup_metrics(self) -> None:
        """Initialize metric tracking"""
        self.metrics = {
            'profit_metrics': {},
            'gas_metrics': {},
            'opportunity_metrics': {},
            'risk_metrics': {},
            'trend_metrics': {},
            'alerts': []
        }
    
    async def analyze_data(
        self,
        data: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive data analysis.
        
        Args:
            data: Dictionary containing various data series
            
        Returns:
            Dictionary containing all computed metrics and analyses
        """
        try:
            with METRICS['analysis_time'].time():
                # Check if update is needed
                if (datetime.now() - self.last_update).seconds < self.config.update_interval:
                    return self.cached_metrics
                
                # Calculate core metrics
                profit_metrics = await self._analyze_profits(data.get('profits', []))
                gas_metrics = await self._analyze_gas(data.get('gas_prices', []))
                opportunity_metrics = await self._analyze_opportunities(data.get('opportunities', []))
                
                # Calculate advanced metrics
                risk_metrics = await self._calculate_risk_metrics(data)
                trend_metrics = await self._analyze_trends(data)
                
                # Detect anomalies and generate alerts
                alerts = await self._detect_anomalies(data)
                
                # Update cached metrics
                self.cached_metrics = {
                    'profit_metrics': profit_metrics,
                    'gas_metrics': gas_metrics,
                    'opportunity_metrics': opportunity_metrics,
                    'risk_metrics': risk_metrics,
                    'trend_metrics': trend_metrics,
                    'alerts': alerts,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.last_update = datetime.now()
                return self.cached_metrics
                
        except Exception as e:
            logger.error("Error in data analysis", error=str(e))
            return self.cached_metrics
    
    async def _analyze_profits(
        self,
        profit_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Analyze profit metrics"""
        try:
            if len(profit_data) < self.config.min_data_points:
                return {}
            
            profits = [p['profit'] for p in profit_data]
            
            return {
                'total_profit': sum(profits),
                'average_profit': np.mean(profits),
                'profit_std': np.std(profits),
                'max_profit': max(profits),
                'min_profit': min(profits),
                'profit_sharpe': self._calculate_sharpe_ratio(profits),
                'profit_volatility': self._calculate_volatility(profits),
                'profit_momentum': self._calculate_momentum(profits),
                'win_rate': sum(1 for p in profits if p > 0) / len(profits)
            }
            
        except Exception as e:
            logger.error("Error analyzing profits", error=str(e))
            return {}
    
    async def _analyze_gas(
        self,
        gas_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Analyze gas metrics"""
        try:
            if len(gas_data) < self.config.min_data_points:
                return {}
            
            base_fees = [g['base_fee'] for g in gas_data]
            priority_fees = [g['priority_fee'] for g in gas_data]
            total_fees = [b + p for b, p in zip(base_fees, priority_fees)]
            
            return {
                'average_base_fee': np.mean(base_fees),
                'average_priority_fee': np.mean(priority_fees),
                'gas_volatility': self._calculate_volatility(total_fees),
                'gas_trend': self._calculate_trend(total_fees),
                'gas_efficiency': self._calculate_gas_efficiency(gas_data),
                'optimal_gas_threshold': self._calculate_optimal_gas(total_fees)
            }
            
        except Exception as e:
            logger.error("Error analyzing gas", error=str(e))
            return {}
    
    async def _analyze_opportunities(
        self,
        opportunity_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Analyze opportunity metrics"""
        try:
            if len(opportunity_data) < self.config.min_data_points:
                return {}
            
            expected_profits = [op['expected_profit'] for op in opportunity_data]
            confidences = [op['confidence'] for op in opportunity_data]
            executed = [op['executed'] for op in opportunity_data]
            
            return {
                'opportunity_count': len(opportunity_data),
                'average_expected_profit': np.mean(expected_profits),
                'average_confidence': np.mean(confidences),
                'execution_rate': sum(executed) / len(executed),
                'profit_realization': self._calculate_profit_realization(
                    expected_profits,
                    executed
                ),
                'opportunity_frequency': self._calculate_opportunity_frequency(
                    opportunity_data
                )
            }
            
        except Exception as e:
            logger.error("Error analyzing opportunities", error=str(e))
            return {}
    
    async def _calculate_risk_metrics(
        self,
        data: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        try:
            profits = [p['profit'] for p in data.get('profits', [])]
            if len(profits) < self.config.min_data_points:
                return {}
            
            return {
                'sharpe_ratio': self._calculate_sharpe_ratio(profits),
                'sortino_ratio': self._calculate_sortino_ratio(profits),
                'max_drawdown': self._calculate_max_drawdown(profits),
                'value_at_risk': self._calculate_var(profits),
                'expected_shortfall': self._calculate_expected_shortfall(profits),
                'risk_adjusted_return': self._calculate_risk_adjusted_return(profits)
            }
            
        except Exception as e:
            logger.error("Error calculating risk metrics", error=str(e))
            return {}
    
    async def _analyze_trends(
        self,
        data: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Analyze trends in various metrics"""
        try:
            trends = {}
            
            # Analyze profit trends
            if 'profits' in data:
                profits = [p['profit'] for p in data['profits']]
                trends['profit_trend'] = self._calculate_trend(profits)
                trends['profit_momentum'] = self._calculate_momentum(profits)
            
            # Analyze gas trends
            if 'gas_prices' in data:
                gas_prices = [g['base_fee'] + g['priority_fee'] for g in data['gas_prices']]
                trends['gas_trend'] = self._calculate_trend(gas_prices)
            
            # Analyze opportunity trends
            if 'opportunities' in data:
                expected_profits = [op['expected_profit'] for op in data['opportunities']]
                trends['opportunity_trend'] = self._calculate_trend(expected_profits)
            
            return trends
            
        except Exception as e:
            logger.error("Error analyzing trends", error=str(e))
            return {}
    
    async def _detect_anomalies(
        self,
        data: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Detect anomalies and generate alerts"""
        alerts = []
        
        try:
            # Check profit anomalies
            if 'profits' in data:
                profit_anomalies = self._detect_metric_anomalies(
                    [p['profit'] for p in data['profits']],
                    'profit'
                )
                alerts.extend(profit_anomalies)
            
            # Check gas anomalies
            if 'gas_prices' in data:
                gas_anomalies = self._detect_metric_anomalies(
                    [g['base_fee'] + g['priority_fee'] for g in data['gas_prices']],
                    'gas'
                )
                alerts.extend(gas_anomalies)
            
            # Update metrics
            METRICS['outlier_count'].inc(len(alerts))
            
            return alerts
            
        except Exception as e:
            logger.error("Error detecting anomalies", error=str(e))
            return []
    
    def _calculate_sharpe_ratio(
        self,
        returns: List[float],
        annualize: bool = True
    ) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(returns) < self.config.min_data_points:
                return 0.0
            
            excess_returns = np.array(returns) - self.config.risk_free_rate
            sharpe = np.mean(excess_returns) / np.std(excess_returns)
            
            return sharpe * np.sqrt(365) if annualize else sharpe
            
        except Exception:
            return 0.0
    
    def _calculate_sortino_ratio(
        self,
        returns: List[float],
        annualize: bool = True
    ) -> float:
        """Calculate Sortino ratio"""
        try:
            if len(returns) < self.config.min_data_points:
                return 0.0
            
            excess_returns = np.array(returns) - self.config.risk_free_rate
            downside_returns = excess_returns[excess_returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else np.inf
            
            sortino = np.mean(excess_returns) / downside_std
            return sortino * np.sqrt(365) if annualize else sortino
            
        except Exception:
            return 0.0
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        try:
            if len(returns) < self.config.min_data_points:
                return 0.0
            
            cumulative = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (running_max - cumulative) / running_max
            return np.max(drawdowns)
            
        except Exception:
            return 0.0
    
    def _calculate_var(
        self,
        returns: List[float],
        confidence_level: Optional[float] = None
    ) -> float:
        """Calculate Value at Risk"""
        try:
            if len(returns) < self.config.min_data_points:
                return 0.0
            
            confidence = confidence_level or self.config.confidence_level
            return -np.percentile(returns, 100 * (1 - confidence))
            
        except Exception:
            return 0.0
    
    def _calculate_expected_shortfall(
        self,
        returns: List[float],
        confidence_level: Optional[float] = None
    ) -> float:
        """Calculate Expected Shortfall (CVaR)"""
        try:
            if len(returns) < self.config.min_data_points:
                return 0.0
            
            confidence = confidence_level or self.config.confidence_level
            var = self._calculate_var(returns, confidence)
            return -np.mean([r for r in returns if r < -var])
            
        except Exception:
            return 0.0
    
    def _calculate_risk_adjusted_return(self, returns: List[float]) -> float:
        """Calculate risk-adjusted return metric"""
        try:
            if len(returns) < self.config.min_data_points:
                return 0.0
            
            return np.mean(returns) / (np.std(returns) + 1e-10)
            
        except Exception:
            return 0.0
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, float]:
        """Calculate trend metrics"""
        try:
            if len(values) < self.config.min_data_points:
                return {}
            
            x = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            
            return {
                'slope': slope,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'std_error': std_err
            }
            
        except Exception:
            return {}
    
    def _calculate_momentum(self, values: List[float]) -> float:
        """Calculate momentum indicator"""
        try:
            if len(values) < self.config.momentum_window:
                return 0.0
            
            recent_values = values[-self.config.momentum_window:]
            weights = np.linspace(0, 1, len(recent_values))
            return np.average(recent_values, weights=weights)
            
        except Exception:
            return 0.0
    
    def _calculate_volatility(
        self,
        values: List[float],
        annualize: bool = True
    ) -> float:
        """Calculate volatility"""
        try:
            if len(values) < self.config.volatility_window:
                return 0.0
            
            returns = np.diff(values) / values[:-1]
            vol = np.std(returns)
            return vol * np.sqrt(365) if annualize else vol
            
        except Exception:
            return 0.0
    
    def _calculate_gas_efficiency(self, gas_data: List[Dict[str, Any]]) -> float:
        """Calculate gas usage efficiency"""
        try:
            if not gas_data or 'profits' not in gas_data[0]:
                return 0.0
            
            total_profit = sum(g['profits'] for g in gas_data)
            total_gas = sum(g['base_fee'] + g['priority_fee'] for g in gas_data)
            return total_profit / (total_gas + 1e-10)
            
        except Exception:
            return 0.0
    
    def _calculate_optimal_gas(self, gas_prices: List[float]) -> float:
        """Calculate optimal gas price threshold"""
        try:
            if len(gas_prices) < self.config.min_data_points:
                return 0.0
            
            # Use 75th percentile as optimal threshold
            return np.percentile(gas_prices, 75)
            
        except Exception:
            return 0.0
    
    def _calculate_profit_realization(
        self,
        expected_profits: List[float],
        executed: List[bool]
    ) -> float:
        """Calculate profit realization rate"""
        try:
            if not expected_profits or not executed:
                return 0.0
            
            executed_profits = [p for p, e in zip(expected_profits, executed) if e]
            if not executed_profits:
                return 0.0
            
            return np.mean(executed_profits) / np.mean(expected_profits)
            
        except Exception:
            return 0.0
    
    def _calculate_opportunity_frequency(
        self,
        opportunities: List[Dict[str, Any]]
    ) -> float:
        """Calculate opportunity frequency (opportunities per hour)"""
        try:
            if len(opportunities) < 2:
                return 0.0
            
            timestamps = [datetime.fromisoformat(op['timestamp']) for op in opportunities]
            time_diffs = np.diff(timestamps)
            avg_time_diff = np.mean([td.total_seconds() for td in time_diffs])
            return 3600 / (avg_time_diff + 1e-10)  # Convert to opportunities per hour
            
        except Exception:
            return 0.0
    
    def _detect_metric_anomalies(
        self,
        values: List[float],
        metric_name: str
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in metric values"""
        anomalies = []
        
        try:
            if len(values) < self.config.min_data_points:
                return anomalies
            
            mean = np.mean(values)
            std = np.std(values)
            z_scores = np.abs((values - mean) / (std + 1e-10))
            
            for i, z_score in enumerate(z_scores):
                if z_score > self.config.outlier_threshold:
                    anomalies.append({
                        'metric': metric_name,
                        'value': values[i],
                        'z_score': z_score,
                        'timestamp': datetime.now().isoformat(),
                        'severity': 'high' if z_score > 2 * self.config.outlier_threshold else 'medium'
                    })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies for {metric_name}", error=str(e))
            return anomalies
    
    def calculate_metrics(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate various performance metrics"""
        try:
            metrics = {
                'profit_metrics': self._calculate_profit_metrics(data),
                'gas_metrics': self._calculate_gas_metrics(data),
                'opportunity_metrics': self._calculate_opportunity_metrics(data)
            }
            return metrics
        except Exception as e:
            logger.error("Error calculating metrics", error=str(e))
            return {}
    
    def detect_trends(self, data: pd.Series) -> Dict[str, Any]:
        """Detect trends in the data"""
        try:
            # Calculate moving averages
            ma_short = data.rolling(window=self.config.trend_window).mean()
            ma_long = data.rolling(window=self.config.trend_window * 2).mean()
            
            # Determine trend direction
            current_trend = 'neutral'
            if ma_short.iloc[-1] > ma_long.iloc[-1]:
                current_trend = 'upward'
            elif ma_short.iloc[-1] < ma_long.iloc[-1]:
                current_trend = 'downward'
            
            return {
                'trend_direction': current_trend,
                'strength': abs(ma_short.iloc[-1] - ma_long.iloc[-1]) / ma_long.iloc[-1],
                'ma_short': ma_short.iloc[-1],
                'ma_long': ma_long.iloc[-1]
            }
        except Exception as e:
            logger.error("Error detecting trends", error=str(e))
            return {'trend_direction': 'unknown'}
    
    def detect_outliers(self, data: pd.Series) -> np.ndarray:
        """Detect outliers using statistical methods"""
        try:
            if len(data) < self.config.min_data_points:
                return np.zeros(len(data), dtype=bool)
            
            # Calculate z-scores
            z_scores = np.abs((data - data.mean()) / data.std())
            
            # Mark outliers based on z-score threshold
            outliers = z_scores > self.config.outlier_threshold
            
            return outliers.values
        except Exception as e:
            logger.error("Error detecting outliers", error=str(e))
            return np.zeros(len(data), dtype=bool)
    
    def _calculate_profit_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate profit-related metrics"""
        try:
            metrics = {
                'total_profit': float(data['profit'].sum()),
                'average_profit': float(data['profit'].mean()),
                'profit_std': float(data['profit'].std()),
                'win_rate': float((data['profit'] > 0).mean())
            }
            return metrics
        except Exception as e:
            logger.error("Error calculating profit metrics", error=str(e))
            return {}
    
    def _calculate_gas_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate gas-related metrics"""
        try:
            metrics = {
                'average_gas': float(data['gas_price'].mean()),
                'gas_std': float(data['gas_price'].std()),
                'gas_efficiency': float((data['profit'] / data['gas_price']).mean())
            }
            return metrics
        except Exception as e:
            logger.error("Error calculating gas metrics", error=str(e))
            return {}
    
    def _calculate_opportunity_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate opportunity-related metrics"""
        try:
            metrics = {
                'opportunity_count': float(len(data)),
                'success_rate': float(data['success_rate'].mean()),
                'opportunity_frequency': float(len(data) / (data['timestamp'].max() - data['timestamp'].min()).total_seconds())
            }
            return metrics
        except Exception as e:
            logger.error("Error calculating opportunity metrics", error=str(e))
            return {}

def calculate_metrics(
    performance_data: pd.DataFrame,
    config: Optional[AnalyticsConfig] = None
) -> Dict[str, float]:
    """Calculate performance metrics from data
    
    Args:
        performance_data: DataFrame with columns [timestamp, profit, gas_used, success_rate]
        config: Optional analytics configuration
        
    Returns:
        Dictionary of calculated metrics
    """
    if config is None:
        config = AnalyticsConfig()
        
    # Calculate basic metrics
    metrics = {
        'avg_profit': float(np.float64(performance_data['profit'].mean())),
        'total_profit': float(np.float64(performance_data['profit'].sum())),
        'success_rate': float(np.float64(performance_data['success_rate'].mean())),
        'gas_efficiency': float(np.float64(
            performance_data['profit'].sum() / 
            performance_data['gas_used'].sum()
            if performance_data['gas_used'].sum() > 0 else 0
        ))
    }
    
    # Add advanced metrics if enabled
    if config.enable_advanced_metrics:
        # Calculate volatility
        metrics['profit_volatility'] = float(np.float64(performance_data['profit'].std()))
        
        # Calculate trend using numpy's polyfit
        trend_coef = np.polyfit(
            range(len(performance_data)), 
            performance_data['profit'].values,
            deg=1
        )
        metrics['profit_trend'] = float(np.float64(trend_coef[0]))
        
        # Calculate efficiency score
        metrics['efficiency_score'] = float(np.float64(
            metrics['success_rate'] * 
            metrics['gas_efficiency'] * 
            (1 - metrics['profit_volatility'])
        ))
    
    return metrics 