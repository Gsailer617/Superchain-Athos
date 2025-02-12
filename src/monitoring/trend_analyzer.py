import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import structlog
from prometheus_client import Histogram, Counter, Gauge
from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

logger = structlog.get_logger(__name__)

class TrendAnalyzer:
    def __init__(self):
        # Trend metrics
        self.trend_strength = Gauge(
            'metric_trend_strength',
            'Strength of trend in metric (positive or negative)',
            ['metric_name']
        )
        
        self.seasonality_strength = Gauge(
            'metric_seasonality_strength',
            'Strength of seasonal patterns in metric',
            ['metric_name']
        )
        
        self.forecast_value = Gauge(
            'metric_forecast_value',
            'Forecasted value for metric',
            ['metric_name', 'horizon']
        )
        
        self.forecast_error = Gauge(
            'metric_forecast_error',
            'Mean absolute percentage error of forecasts',
            ['metric_name']
        )

    async def analyze_trend(
        self,
        metric_name: str,
        values: List[float],
        timestamps: List[datetime]
    ) -> Dict[str, float]:
        """
        Analyze trend components using time series decomposition.
        """
        if len(values) < 2:
            return {}
            
        try:
            # Convert to numpy array
            values_array = np.array(values)
            
            # Perform seasonal decomposition
            result = seasonal_decompose(
                values_array,
                period=min(len(values) // 2, 24),  # Assume daily seasonality if enough data
                extrapolate_trend='freq'
            )
            
            # Calculate trend strength
            trend_strength = 1 - np.var(result.resid) / np.var(values_array - result.seasonal)
            self.trend_strength.labels(metric_name=metric_name).set(trend_strength)
            
            # Calculate seasonality strength
            seasonality_strength = 1 - np.var(result.resid) / np.var(values_array - result.trend)
            self.seasonality_strength.labels(metric_name=metric_name).set(seasonality_strength)
            
            return {
                'trend_strength': trend_strength,
                'seasonality_strength': seasonality_strength,
                'trend_direction': np.mean(np.diff(result.trend)),
                'last_trend_value': result.trend[-1],
                'last_seasonal_value': result.seasonal[-1]
            }
            
        except Exception as e:
            logger.error("Error analyzing trend",
                        metric_name=metric_name,
                        error=str(e))
            return {}

    async def forecast_metric(
        self,
        metric_name: str,
        values: List[float],
        horizon: int = 24
    ) -> Dict[str, List[float]]:
        """
        Generate forecasts using Holt-Winters method.
        """
        if len(values) < horizon:
            return {}
            
        try:
            # Fit model
            model = ExponentialSmoothing(
                values,
                seasonal_periods=min(len(values) // 2, 24),
                trend='add',
                seasonal='add'
            ).fit()
            
            # Generate forecast
            forecast = model.forecast(horizon)
            
            # Calculate forecast error
            mape = np.mean(np.abs(model.resid / values)) * 100
            self.forecast_error.labels(metric_name=metric_name).set(mape)
            
            # Record forecasts
            for h, value in enumerate(forecast):
                self.forecast_value.labels(
                    metric_name=metric_name,
                    horizon=str(h+1)
                ).set(value)
            
            return {
                'forecast': forecast.tolist(),
                'mape': mape,
                'lower_bound': (forecast - 2 * model.resid.std()).tolist(),
                'upper_bound': (forecast + 2 * model.resid.std()).tolist()
            }
            
        except Exception as e:
            logger.error("Error generating forecast",
                        metric_name=metric_name,
                        error=str(e))
            return {}

    async def detect_anomalies(
        self,
        values: List[float],
        window_size: int = 24
    ) -> List[Tuple[int, float]]:
        """
        Detect anomalies using rolling statistics.
        """
        if len(values) < window_size:
            return []
            
        try:
            values_array = np.array(values)
            anomalies = []
            
            # Calculate rolling mean and std
            for i in range(window_size, len(values)):
                window = values_array[i-window_size:i]
                mean = np.mean(window)
                std = np.std(window)
                
                # Check if current value is anomalous
                z_score = (values_array[i] - mean) / std if std > 0 else 0
                if abs(z_score) > 3:  # 3 sigma rule
                    anomalies.append((i, z_score))
            
            return anomalies
            
        except Exception as e:
            logger.error("Error detecting anomalies", error=str(e))
            return []

    async def analyze_correlations(
        self,
        metrics: Dict[str, List[float]]
    ) -> Dict[Tuple[str, str], float]:
        """
        Analyze correlations between different metrics.
        """
        correlations = {}
        
        try:
            metric_names = list(metrics.keys())
            
            for i in range(len(metric_names)):
                for j in range(i + 1, len(metric_names)):
                    name1, name2 = metric_names[i], metric_names[j]
                    values1, values2 = metrics[name1], metrics[name2]
                    
                    if len(values1) == len(values2):
                        correlation = stats.pearsonr(values1, values2)[0]
                        correlations[(name1, name2)] = correlation
            
            return correlations
            
        except Exception as e:
            logger.error("Error analyzing correlations", error=str(e))
            return {} 