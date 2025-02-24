"""
Visualization module for learning insights and performance metrics
"""

from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import structlog

from ..specialized.trade_monitor import TradeMonitor

logger = structlog.get_logger(__name__)

class LearningInsightsVisualizer:
    """Visualize learning insights from trading data"""
    
    def __init__(self, trade_monitor: TradeMonitor):
        """Initialize visualizer
        
        Args:
            trade_monitor: Trade monitoring instance
        """
        self.trade_monitor = trade_monitor
        
    def create_performance_dashboard(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """Create performance dashboard
        
        Args:
            days: Number of days of data to include
            
        Returns:
            Dashboard configuration
        """
        try:
            # Get trade history
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            trades_df = self.trade_monitor.get_trade_history(
                start_time=start_time,
                end_time=end_time
            )
            
            if trades_df.empty:
                return self._empty_dashboard()
                
            # Create subplots
            fig = make_subplots(
                rows=3,
                cols=2,
                subplot_titles=(
                    'Profit Over Time',
                    'Success Rate',
                    'Gas Costs',
                    'Execution Times',
                    'Token Performance',
                    'DEX Performance'
                )
            )
            
            # Add profit timeline
            fig.add_trace(
                go.Scatter(
                    x=trades_df['timestamp'],
                    y=trades_df['profit'].cumsum(),
                    name='Cumulative Profit'
                ),
                row=1,
                col=1
            )
            
            # Add success rate
            success_rate = trades_df['success'].rolling(window=100).mean() * 100
            fig.add_trace(
                go.Scatter(
                    x=trades_df['timestamp'],
                    y=success_rate,
                    name='Success Rate (%)'
                ),
                row=1,
                col=2
            )
            
            # Add gas costs
            fig.add_trace(
                go.Scatter(
                    x=trades_df['timestamp'],
                    y=trades_df['gas_cost'],
                    name='Gas Cost'
                ),
                row=2,
                col=1
            )
            
            # Add execution times
            fig.add_trace(
                go.Scatter(
                    x=trades_df['timestamp'],
                    y=trades_df['execution_time'],
                    name='Execution Time'
                ),
                row=2,
                col=2
            )
            
            # Add token performance
            token_stats = self._get_token_performance()
            fig.add_trace(
                go.Bar(
                    x=list(token_stats.keys()),
                    y=[s['profit'] for s in token_stats.values()],
                    name='Token Profit'
                ),
                row=3,
                col=1
            )
            
            # Add DEX performance
            dex_stats = self._get_dex_performance()
            fig.add_trace(
                go.Bar(
                    x=list(dex_stats.keys()),
                    y=[s['profit'] for s in dex_stats.values()],
                    name='DEX Profit'
                ),
                row=3,
                col=2
            )
            
            # Update layout
            fig.update_layout(
                height=1000,
                width=1200,
                showlegend=True,
                title_text='Trading Performance Dashboard'
            )
            
            return {
                'figure': fig,
                'metrics': self._get_summary_metrics(trades_df)
            }
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            return self._empty_dashboard()
            
    def create_learning_curves(
        self,
        days: int = 30,
        window_size: int = 100
    ) -> Dict[str, Any]:
        """Create learning curve visualizations
        
        Args:
            days: Number of days of data
            window_size: Rolling window size
            
        Returns:
            Learning curves configuration
        """
        try:
            # Get trade history
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            trades_df = self.trade_monitor.get_trade_history(
                start_time=start_time,
                end_time=end_time
            )
            
            if trades_df.empty:
                return self._empty_dashboard()
                
            # Create subplots
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    'Profit Learning Curve',
                    'Success Rate Learning Curve',
                    'Gas Optimization Curve',
                    'Execution Time Learning Curve'
                )
            )
            
            # Add profit learning curve
            rolling_profit = trades_df['profit'].rolling(window=window_size).mean()
            fig.add_trace(
                go.Scatter(
                    x=trades_df['timestamp'],
                    y=rolling_profit,
                    name='Average Profit'
                ),
                row=1,
                col=1
            )
            
            # Add success rate curve
            rolling_success = trades_df['success'].rolling(window=window_size).mean() * 100
            fig.add_trace(
                go.Scatter(
                    x=trades_df['timestamp'],
                    y=rolling_success,
                    name='Success Rate (%)'
                ),
                row=1,
                col=2
            )
            
            # Add gas optimization curve
            rolling_gas = trades_df['gas_cost'].rolling(window=window_size).mean()
            fig.add_trace(
                go.Scatter(
                    x=trades_df['timestamp'],
                    y=rolling_gas,
                    name='Average Gas Cost'
                ),
                row=2,
                col=1
            )
            
            # Add execution time curve
            rolling_time = trades_df['execution_time'].rolling(window=window_size).mean()
            fig.add_trace(
                go.Scatter(
                    x=trades_df['timestamp'],
                    y=rolling_time,
                    name='Average Execution Time'
                ),
                row=2,
                col=2
            )
            
            # Update layout
            fig.update_layout(
                height=800,
                width=1200,
                showlegend=True,
                title_text='Trading Learning Curves'
            )
            
            return {
                'figure': fig,
                'metrics': self._get_learning_metrics(trades_df, window_size)
            }
            
        except Exception as e:
            logger.error(f"Error creating learning curves: {str(e)}")
            return self._empty_dashboard()
            
    def _get_token_performance(self) -> Dict[str, Dict[str, float]]:
        """Get token performance statistics"""
        return self.trade_monitor.get_token_analytics()
        
    def _get_dex_performance(self) -> Dict[str, Dict[str, float]]:
        """Get DEX performance statistics"""
        return self.trade_monitor.get_dex_analytics()
        
    def _get_summary_metrics(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary metrics"""
        try:
            return {
                'total_trades': len(trades_df),
                'success_rate': (trades_df['success'].mean() * 100),
                'total_profit': trades_df['profit'].sum(),
                'total_gas_cost': trades_df['gas_cost'].sum(),
                'avg_execution_time': trades_df['execution_time'].mean(),
                'best_profit': trades_df['profit'].max(),
                'worst_profit': trades_df['profit'].min()
            }
        except Exception:
            return {}
            
    def _get_learning_metrics(
        self,
        trades_df: pd.DataFrame,
        window_size: int
    ) -> Dict[str, Any]:
        """Calculate learning metrics"""
        try:
            # Calculate improvements
            profit_improvement = (
                trades_df['profit'].rolling(window=window_size).mean().iloc[-1] /
                trades_df['profit'].rolling(window=window_size).mean().iloc[window_size]
            ) - 1
            
            success_improvement = (
                trades_df['success'].rolling(window=window_size).mean().iloc[-1] /
                trades_df['success'].rolling(window=window_size).mean().iloc[window_size]
            ) - 1
            
            gas_improvement = (
                trades_df['gas_cost'].rolling(window=window_size).mean().iloc[window_size] /
                trades_df['gas_cost'].rolling(window=window_size).mean().iloc[-1]
            ) - 1
            
            time_improvement = (
                trades_df['execution_time'].rolling(window=window_size).mean().iloc[window_size] /
                trades_df['execution_time'].rolling(window=window_size).mean().iloc[-1]
            ) - 1
            
            return {
                'profit_improvement': profit_improvement * 100,
                'success_improvement': success_improvement * 100,
                'gas_optimization': gas_improvement * 100,
                'time_optimization': time_improvement * 100
            }
            
        except Exception:
            return {}
            
    def _empty_dashboard(self) -> Dict[str, Any]:
        """Create empty dashboard when no data available"""
        fig = go.Figure()
        fig.update_layout(
            title_text='No Trading Data Available',
            annotations=[{
                'text': 'No trading data available for the selected time period',
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': {'size': 20}
            }]
        )
        
        return {
            'figure': fig,
            'metrics': {}
        }

    def visualize_performance(
        self,
        metrics: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> None:
        """Visualize performance metrics
        
        Args:
            metrics: Performance metrics dictionary
            output_path: Optional path to save visualization
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Profit Distribution',
                    'Success Rate by Strategy',
                    'Gas Efficiency',
                    'Execution Time Distribution',
                    'Hourly Performance',
                    'Strategy Performance'
                )
            )
            
            # Profit distribution
            if 'strategy_performance' in metrics:
                profits = [
                    data['avg_profit']
                    for data in metrics['strategy_performance'].values()
                ]
                fig.add_trace(
                    go.Histogram(x=profits, name='Profit Distribution'),
                    row=1, col=1
                )
            
            # Success rate by strategy
            if 'strategy_performance' in metrics:
                strategies = list(metrics['strategy_performance'].keys())
                success_rates = [
                    data['success_rate']
                    for data in metrics['strategy_performance'].values()
                ]
                fig.add_trace(
                    go.Bar(
                        x=strategies,
                        y=success_rates,
                        name='Success Rate'
                    ),
                    row=1, col=2
                )
            
            # Gas efficiency
            if 'dex_performance' in metrics:
                dexes = list(metrics['dex_performance'].keys())
                gas_prices = [
                    data['avg_gas_price']
                    for data in metrics['dex_performance'].values()
                ]
                fig.add_trace(
                    go.Bar(
                        x=dexes,
                        y=gas_prices,
                        name='Gas Price'
                    ),
                    row=2, col=1
                )
            
            # Execution time distribution
            if 'strategy_performance' in metrics:
                exec_times = [
                    data['avg_execution_time']
                    for data in metrics['strategy_performance'].values()
                ]
                fig.add_trace(
                    go.Box(y=exec_times, name='Execution Time'),
                    row=2, col=2
                )
            
            # Hourly performance
            if 'hourly_performance' in metrics:
                hours = list(metrics['hourly_performance'].keys())
                profits = [
                    data['avg_profit']
                    for data in metrics['hourly_performance'].values()
                ]
                fig.add_trace(
                    go.Scatter(
                        x=hours,
                        y=profits,
                        mode='lines+markers',
                        name='Hourly Profit'
                    ),
                    row=3, col=1
                )
            
            # Strategy performance comparison
            if 'strategy_performance' in metrics:
                strategies = list(metrics['strategy_performance'].keys())
                total_profits = [
                    data['total_profit']
                    for data in metrics['strategy_performance'].values()
                ]
                trade_counts = [
                    data['trade_count']
                    for data in metrics['strategy_performance'].values()
                ]
                
                fig.add_trace(
                    go.Bar(
                        x=strategies,
                        y=total_profits,
                        name='Total Profit',
                        yaxis='y1'
                    ),
                    row=3, col=2
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=strategies,
                        y=trade_counts,
                        mode='lines+markers',
                        name='Trade Count',
                        yaxis='y2'
                    ),
                    row=3, col=2
                )
            
            # Update layout
            fig.update_layout(
                height=1200,
                width=1600,
                title_text='Trading Performance Insights',
                showlegend=True
            )
            
            # Save if path provided
            if output_path:
                fig.write_html(output_path)
            
            # Show figure
            fig.show()
            
        except Exception as e:
            logger.error("Error creating performance visualization", error=str(e))
    
    def visualize_learning_metrics(
        self,
        timeframe: str = '7d',
        output_path: Optional[str] = None
    ) -> None:
        """Visualize ML model metrics and learning progress
        
        Args:
            timeframe: Time window for analysis
            output_path: Optional path to save visualization
        """
        try:
            # Get model metrics
            metrics = self.trade_monitor._model_metrics
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Profit Prediction Error',
                    'Risk Assessment Metrics',
                    'Feature Importance',
                    'Anomaly Detection'
                )
            )
            
            # Profit prediction metrics
            if 'profit_prediction' in metrics:
                mse = metrics['profit_prediction']['mse']
                mae = metrics['profit_prediction']['mae']
                r2 = metrics['profit_prediction']['r2']
                
                fig.add_trace(
                    go.Scatter(
                        y=mse,
                        mode='lines',
                        name='MSE'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        y=mae,
                        mode='lines',
                        name='MAE'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        y=r2,
                        mode='lines',
                        name='R²'
                    ),
                    row=1, col=1
                )
            
            # Risk assessment metrics
            if 'risk_assessment' in metrics:
                accuracy = metrics['risk_assessment']['accuracy']
                precision = metrics['risk_assessment']['precision']
                recall = metrics['risk_assessment']['recall']
                
                fig.add_trace(
                    go.Scatter(
                        y=accuracy,
                        mode='lines',
                        name='Accuracy'
                    ),
                    row=1, col=2
                )
                
                fig.add_trace(
                    go.Scatter(
                        y=precision,
                        mode='lines',
                        name='Precision'
                    ),
                    row=1, col=2
                )
                
                fig.add_trace(
                    go.Scatter(
                        y=recall,
                        mode='lines',
                        name='Recall'
                    ),
                    row=1, col=2
                )
            
            # Feature importance
            prediction = self.trade_monitor.predict_profit(
                strategy='*',
                token_pair='*',
                market_conditions={}
            )
            
            if prediction and prediction.features_importance:
                features = list(prediction.features_importance.keys())
                importance = list(prediction.features_importance.values())
                
                fig.add_trace(
                    go.Bar(
                        x=features,
                        y=importance,
                        name='Feature Importance'
                    ),
                    row=2, col=1
                )
            
            # Anomaly detection
            if 'anomaly_detection' in metrics:
                anomaly_ratio = metrics['anomaly_detection']['anomaly_ratio']
                
                fig.add_trace(
                    go.Scatter(
                        y=anomaly_ratio,
                        mode='lines',
                        name='Anomaly Ratio'
                    ),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                height=1000,
                width=1600,
                title_text='Learning Metrics and Model Performance',
                showlegend=True
            )
            
            # Save if path provided
            if output_path:
                fig.write_html(output_path)
            
            # Show figure
            fig.show()
            
        except Exception as e:
            logger.error("Error creating learning metrics visualization", error=str(e))
    
    def create_performance_report(
        self,
        timeframe: str = '24h',
        output_path: Optional[str] = None
    ) -> None:
        """Create comprehensive performance report
        
        Args:
            timeframe: Time window for analysis
            output_path: Optional path to save report
        """
        try:
            # Get performance metrics
            metrics = self.trade_monitor.analyze_performance(timeframe=timeframe)
            
            # Create visualizations
            self.visualize_performance(metrics)
            self.visualize_learning_metrics(timeframe)
            
            if output_path:
                # Create report summary
                summary = {
                    'timestamp': datetime.now().isoformat(),
                    'timeframe': timeframe,
                    'metrics': metrics,
                    'model_metrics': self.trade_monitor._model_metrics
                }
                
                # Save report
                import json
                with open(output_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                logger.info(
                    "Performance report created",
                    path=output_path
                )
            
        except Exception as e:
            logger.error("Error creating performance report", error=str(e)) 