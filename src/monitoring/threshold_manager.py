import numpy as np
from typing import Dict, List, Optional, Tuple
import structlog
from prometheus_client import Gauge
from datetime import datetime, timedelta

logger = structlog.get_logger(__name__)

class ThresholdManager:
    def __init__(self):
        # Dynamic threshold metrics
        self.current_threshold = Gauge(
            'metric_dynamic_threshold',
            'Current dynamic threshold for metrics',
            ['metric_name', 'threshold_type']
        )
        
        self.threshold_adjustment = Gauge(
            'threshold_adjustment_percent',
            'Percentage adjustment of threshold from baseline',
            ['metric_name']
        )

    async def calculate_dynamic_threshold(
        self,
        metric_name: str,
        values: List[float],
        window_size: int = 24,
        sensitivity: float = 2.0
    ) -> Dict[str, float]:
        """
        Calculate dynamic thresholds using statistical methods.
        
        Args:
            metric_name: Name of the metric
            values: Historical values
            window_size: Size of the rolling window
            sensitivity: Number of standard deviations for threshold
        """
        if len(values) < window_size:
            logger.warning("Insufficient data for threshold calculation",
                         metric_name=metric_name)
            return {}
            
        try:
            # Calculate rolling statistics
            values_array = np.array(values)
            rolling_mean = np.mean(values_array[-window_size:])
            rolling_std = np.std(values_array[-window_size:])
            
            # Calculate thresholds
            upper_threshold = rolling_mean + (sensitivity * rolling_std)
            lower_threshold = rolling_mean - (sensitivity * rolling_std)
            
            # Record current thresholds
            self.current_threshold.labels(
                metric_name=metric_name,
                threshold_type='upper'
            ).set(upper_threshold)
            
            self.current_threshold.labels(
                metric_name=metric_name,
                threshold_type='lower'
            ).set(lower_threshold)
            
            # Calculate and record adjustment percentage
            baseline = np.mean(values_array[:window_size])  # Initial baseline
            adjustment = ((upper_threshold - baseline) / baseline) * 100
            self.threshold_adjustment.labels(metric_name=metric_name).set(adjustment)
            
            return {
                'upper_threshold': upper_threshold,
                'lower_threshold': lower_threshold,
                'rolling_mean': rolling_mean,
                'rolling_std': rolling_std
            }
            
        except Exception as e:
            logger.error("Error calculating dynamic thresholds",
                        metric_name=metric_name,
                        error=str(e))
            return {}

    async def detect_threshold_violations(
        self,
        metric_name: str,
        current_value: float,
        thresholds: Dict[str, float]
    ) -> Tuple[bool, str]:
        """
        Detect if current value violates thresholds.
        
        Returns:
            Tuple of (is_violation, violation_type)
        """
        if not thresholds:
            return False, ""
            
        try:
            if current_value > thresholds['upper_threshold']:
                return True, "upper_threshold_violation"
            elif current_value < thresholds['lower_threshold']:
                return True, "lower_threshold_violation"
            return False, ""
            
        except Exception as e:
            logger.error("Error detecting threshold violations",
                        metric_name=metric_name,
                        error=str(e))
            return False, ""

    async def adjust_sensitivity(
        self,
        metric_name: str,
        false_positives: int,
        false_negatives: int,
        current_sensitivity: float
    ) -> float:
        """
        Adjust threshold sensitivity based on alert accuracy.
        
        Returns:
            New sensitivity value
        """
        try:
            # Calculate error ratio
            total_errors = false_positives + false_negatives
            if total_errors == 0:
                return current_sensitivity
                
            fp_ratio = false_positives / total_errors
            
            # Adjust sensitivity
            if fp_ratio > 0.6:  # Too many false positives
                new_sensitivity = current_sensitivity * 1.1  # Increase threshold
            elif fp_ratio < 0.4:  # Too many false negatives
                new_sensitivity = current_sensitivity * 0.9  # Decrease threshold
            else:
                return current_sensitivity
                
            logger.info("Adjusted threshold sensitivity",
                       metric_name=metric_name,
                       old_sensitivity=current_sensitivity,
                       new_sensitivity=new_sensitivity)
                       
            return new_sensitivity
            
        except Exception as e:
            logger.error("Error adjusting sensitivity",
                        metric_name=metric_name,
                        error=str(e))
            return current_sensitivity 