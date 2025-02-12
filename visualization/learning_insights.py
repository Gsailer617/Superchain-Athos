from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.history.trade_history import TradeHistoryManager
import structlog

logger = structlog.get_logger(__name__)

class LearningInsightsVisualizer:
    """Visualizes insights from historical trading data"""
    
    def __init__(self, history_manager: TradeHistoryManager):
        self.history_manager = history_manager
    
    def create_learning_curve(
        self,
        lookback_period: str = '30d',
        window_size: str = '1d'
    ) -> go.Figure:
        """Create learning curve showing improvement over time
        
        Args:
            lookback_period: Period to analyze
            window_size: Window size for rolling calculations
            
        Returns:
            Plotly figure with learning curves
        """
        try:
            # Get historical data
            df = self.history_manager.get_history(
                start_time=datetime.now() - pd.Timedelta(lookback_period)
            )
            
            if df.empty:
                return go.Figure()
            
            # Calculate rolling metrics
            df['rolling_success'] = df['success'].rolling(window_size).mean()
            df['rolling_profit'] = df['profit'].rolling(window_size).mean()
            df['rolling_execution'] = df['execution_time'].rolling(window_size).mean()
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=(
                    'Success Rate Learning Curve',
                    'Profit Learning Curve',
                    'Execution Time Learning Curve'
                ),
                shared_xaxes=True
            )
            
            # Success rate curve
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['rolling_success'],
                    mode='lines',
                    name='Success Rate',
                    line=dict(color='green')
                ),
                row=1, col=1
            )
            
            # Profit curve
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['rolling_profit'],
                    mode='lines',
                    name='Average Profit',
                    line=dict(color='blue')
                ),
                row=2, col=1
            )
            
            # Execution time curve
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['rolling_execution'],
                    mode='lines',
                    name='Execution Time',
                    line=dict(color='orange')
                ),
                row=3, col=1
            )
            
            fig.update_layout(
                height=900,
                title_text='Trading Strategy Learning Curves',
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error("Error creating learning curve", error=str(e))
            return go.Figure()
    
    def create_strategy_evolution(
        self,
        lookback_period: str = '30d',
        top_n: int = 5
    ) -> go.Figure:
        """Create visualization of strategy evolution
        
        Args:
            lookback_period: Period to analyze
            top_n: Number of top strategies to show
            
        Returns:
            Plotly figure with strategy evolution
        """
        try:
            df = self.history_manager.get_history(
                start_time=datetime.now() - pd.Timedelta(lookback_period)
            )
            
            if df.empty:
                return go.Figure()
            
            # Get top strategies by profit
            top_strategies = (
                df.groupby('strategy')['profit']
                .sum()
                .sort_values(ascending=False)
                .head(top_n)
                .index
            )
            
            # Filter for top strategies
            df = df[df['strategy'].isin(top_strategies)]
            
            # Create figure
            fig = go.Figure()
            
            for strategy in top_strategies:
                strategy_df = df[df['strategy'] == strategy]
                
                # Calculate cumulative metrics
                cumulative_profit = strategy_df['profit'].cumsum()
                
                fig.add_trace(
                    go.Scatter(
                        x=strategy_df['timestamp'],
                        y=cumulative_profit,
                        mode='lines',
                        name=strategy,
                        hovertemplate=(
                            'Strategy: %{fullData.name}<br>' +
                            'Time: %{x}<br>' +
                            'Cumulative Profit: $%{y:.2f}<br>' +
                            '<extra></extra>'
                        )
                    )
                )
            
            fig.update_layout(
                title='Strategy Evolution Over Time',
                xaxis_title='Time',
                yaxis_title='Cumulative Profit ($)',
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error("Error creating strategy evolution", error=str(e))
            return go.Figure()
    
    def create_feature_importance(
        self,
        lookback_period: str = '7d'
    ) -> go.Figure:
        """Create feature importance visualization
        
        Args:
            lookback_period: Period to analyze
            
        Returns:
            Plotly figure with feature importance
        """
        try:
            # Get learning features
            features = self.history_manager.get_learning_features(lookback_period)
            
            if features.empty:
                return go.Figure()
            
            # Calculate feature correlations with profit
            correlations = features.corr()['profit'].sort_values(ascending=True)
            
            # Create bar chart
            fig = go.Figure(
                go.Bar(
                    x=correlations.values,
                    y=correlations.index,
                    orientation='h'
                )
            )
            
            fig.update_layout(
                title='Feature Importance (Correlation with Profit)',
                xaxis_title='Correlation Coefficient',
                yaxis_title='Feature',
                height=max(400, len(correlations) * 20)  # Dynamic height
            )
            
            return fig
            
        except Exception as e:
            logger.error("Error creating feature importance", error=str(e))
            return go.Figure()
    
    def create_performance_prediction(
        self,
        lookback_period: str = '7d',
        forecast_hours: int = 24
    ) -> go.Figure:
        """Create performance prediction visualization
        
        Args:
            lookback_period: Period to analyze
            forecast_hours: Hours to forecast
            
        Returns:
            Plotly figure with performance prediction
        """
        try:
            df = self.history_manager.get_history(
                start_time=datetime.now() - pd.Timedelta(lookback_period)
            )
            
            if df.empty:
                return go.Figure()
            
            # Calculate hourly profits
            hourly = df.set_index('timestamp').resample('1H').agg({
                'profit': 'sum',
                'success': 'mean',
                'gas_price': 'mean'
            })
            
            # Simple exponential smoothing for prediction
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            
            model = ExponentialSmoothing(
                hourly['profit'],
                seasonal_periods=24,
                trend='add',
                seasonal='add'
            ).fit()
            
            # Generate forecast
            forecast = model.forecast(forecast_hours)
            
            # Create figure
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(
                go.Scatter(
                    x=hourly.index,
                    y=hourly['profit'],
                    mode='lines',
                    name='Historical Profit',
                    line=dict(color='blue')
                )
            )
            
            # Forecast
            fig.add_trace(
                go.Scatter(
                    x=pd.date_range(
                        start=hourly.index[-1],
                        periods=forecast_hours + 1,
                        freq='H'
                    ),
                    y=forecast,
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red', dash='dash')
                )
            )
            
            fig.update_layout(
                title='Profit Forecast',
                xaxis_title='Time',
                yaxis_title='Hourly Profit ($)',
                showlegend=True,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error("Error creating performance prediction", error=str(e))
            return go.Figure() 