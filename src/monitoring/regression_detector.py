import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import structlog
from prometheus_client import Histogram, Counter, Gauge
from scipy import stats

logger = structlog.get_logger(__name__)

class RegressionDetector:
    def __init__(self, lookback_days: int = 7, z_score_threshold: float = 2.0):
        self.lookback_days = lookback_days
        self.z_score_threshold = z_score_threshold
        
        # Regression metrics
        self.regression_counter = Counter(
            'performance_regressions_total',
            'Total number of detected performance regressions',
            ['metric_name', 'severity']
        )
        
        self.regression_score = Gauge(
            'regression_score',
            'Z-score indicating deviation from historical performance',
            ['metric_name']
        )
        
        self.baseline_values = Gauge(
            'performance_baseline',
            'Baseline values for performance metrics',
            ['metric_name', 'statistic']
        )

    async def analyze_metric(
        self,
        metric_name: str,
        current_value: float,
        historical_values: List[float]
    ) -> Tuple[bool, float, Dict[str, float]]:
        """
        Analyze a metric for performance regression using statistical methods.
        
        Returns:
        - is_regression: Whether a regression is detected
        - z_score: How many standard deviations from the mean
        - statistics: Dictionary of relevant statistical measures
        """
        if len(historical_values) < 2:
            logger.warning("Insufficient historical data for regression analysis",
                         metric_name=metric_name)
            return False, 0.0, {}

        # Calculate baseline statistics
        baseline_mean = np.mean(historical_values)
        baseline_std = np.std(historical_values)
        
        # Calculate z-score
        z_score = (current_value - baseline_mean) / baseline_std if baseline_std > 0 else 0
        
        # Update baseline metrics
        self.baseline_values.labels(metric_name=metric_name, statistic='mean').set(baseline_mean)
        self.baseline_values.labels(metric_name=metric_name, statistic='std').set(baseline_std)
        
        # Record regression score
        self.regression_score.labels(metric_name=metric_name).set(z_score)
        
        # Detect regression based on z-score
        is_regression = abs(z_score) > self.z_score_threshold
        
        if is_regression:
            severity = 'critical' if abs(z_score) > self.z_score_threshold * 1.5 else 'warning'
            self.regression_counter.labels(
                metric_name=metric_name,
                severity=severity
            ).inc()
            
            logger.warning("Performance regression detected",
                         metric_name=metric_name,
                         z_score=z_score,
                         severity=severity)

        # Calculate additional statistics
        stats_dict = {
            'mean': baseline_mean,
            'std': baseline_std,
            'z_score': z_score,
            'percentile': stats.percentileofscore(historical_values, current_value)
        }
        
        return is_regression, z_score, stats_dict

    async def analyze_latency_distribution(
        self,
        current_latencies: List[float],
        historical_latencies: List[List[float]]
    ) -> Dict[str, float]:
        """
        Analyze latency distributions using Kolmogorov-Smirnov test.
        """
        if not historical_latencies or not current_latencies:
            return {}
            
        # Combine historical latencies
        baseline_latencies = np.concatenate(historical_latencies)
        
        # Perform KS test
        ks_statistic, p_value = stats.ks_2samp(current_latencies, baseline_latencies)
        
        # Calculate percentiles
        current_percentiles = np.percentile(current_latencies, [50, 90, 95, 99])
        baseline_percentiles = np.percentile(baseline_latencies, [50, 90, 95, 99])
        
        return {
            'ks_statistic': ks_statistic,
            'p_value': p_value,
            'regression_detected': p_value < 0.05,
            'percentile_changes': {
                'p50': current_percentiles[0] - baseline_percentiles[0],
                'p90': current_percentiles[1] - baseline_percentiles[1],
                'p95': current_percentiles[2] - baseline_percentiles[2],
                'p99': current_percentiles[3] - baseline_percentiles[3]
            }
        }

    async def analyze_error_patterns(
        self,
        current_errors: Dict[str, int],
        historical_errors: List[Dict[str, int]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze error patterns for anomalies.
        """
        results = {}
        
        for error_type in current_errors:
            # Get historical values for this error type
            historical_values = [h.get(error_type, 0) for h in historical_errors]
            
            # Calculate baseline statistics
            baseline_mean = np.mean(historical_values)
            baseline_std = np.std(historical_values)
            current_value = current_errors[error_type]
            
            # Calculate z-score
            z_score = (current_value - baseline_mean) / baseline_std if baseline_std > 0 else 0
            
            results[error_type] = {
                'z_score': z_score,
                'is_anomaly': abs(z_score) > self.z_score_threshold,
                'current_rate': current_value,
                'baseline_rate': baseline_mean
            }
            
        return results

    async def analyze_strategy_performance(
        self,
        strategy: str,
        current_metrics: Dict[str, float],
        historical_metrics: List[Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze strategy performance metrics for regressions.
        """
        results = {}
        
        for metric_name, current_value in current_metrics.items():
            historical_values = [h.get(metric_name, 0) for h in historical_metrics]
            
            is_regression, z_score, stats_dict = await self.analyze_metric(
                f"strategy_{strategy}_{metric_name}",
                current_value,
                historical_values
            )
            
            results[metric_name] = {
                'is_regression': is_regression,
                'z_score': z_score,
                **stats_dict
            }
            
        return results 