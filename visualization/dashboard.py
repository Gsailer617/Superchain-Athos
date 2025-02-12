"""Dashboard module for visualization"""

import dash
from dash import html, dcc
import plotly.graph_objects as go
from typing import Optional, Dict, Any
import pandas as pd
from .charts import create_performance_chart
from .analytics import calculate_metrics, AnalyticsConfig
from .learning_insights import LearningInsightsVisualizer
from src.history.trade_history import TradeHistoryManager
from src.monitoring.monitor_manager import MonitorManager
import dash_bootstrap_components as dbc

# Create the Dash application instance
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

def create_app(
    performance_data: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None,
    history_manager: Optional[TradeHistoryManager] = None,
    monitor_manager: Optional[MonitorManager] = None
) -> dash.Dash:
    """Create the dashboard application
    
    Args:
        performance_data: Performance data DataFrame
        config: Configuration dictionary
        history_manager: Trade history manager instance
        monitor_manager: Monitor manager instance for enhanced learning
        
    Returns:
        Dash application instance
    """
    # Initialize learning visualizer if history manager is provided
    learning_viz = None
    if history_manager:
        learning_viz = LearningInsightsVisualizer(history_manager)
    
    # Create intervals for updates
    intervals = html.Div([
        dcc.Interval(id='fast-interval', interval=10*1000),  # 10 seconds
        dcc.Interval(id='medium-interval', interval=30*1000),  # 30 seconds
        dcc.Interval(id='slow-interval', interval=300*1000)  # 5 minutes
    ])
    
    # Create main layout
    app.layout = dbc.Container([
        html.H1('Arbitrage Monitoring Dashboard', className='mb-4'),
        
        intervals,
        
        # Performance Metrics Section
        dbc.Row([
            dbc.Col([
                html.H2('Performance Metrics'),
                html.Div(id='metrics-container')
            ])
        ], className='mb-4'),
        
        # Charts Section
        dbc.Row([
            dbc.Col([
                html.H2('Performance Charts'),
                dcc.Graph(id='performance-chart')
            ])
        ], className='mb-4'),
        
        # Learning Insights Section
        dbc.Row([
            dbc.Col([
                html.H2('Learning Insights'),
                dbc.Tabs([
                    dbc.Tab([
                        dcc.Graph(id='learning-curve')
                    ], label='Learning Curve'),
                    dbc.Tab([
                        dcc.Graph(id='strategy-evolution')
                    ], label='Strategy Evolution'),
                    dbc.Tab([
                        dcc.Graph(id='feature-importance')
                    ], label='Feature Importance'),
                    dbc.Tab([
                        dcc.Graph(id='performance-prediction')
                    ], label='Performance Prediction'),
                    dbc.Tab([
                        html.Div(id='anomaly-detection')
                    ], label='Anomaly Detection'),
                    dbc.Tab([
                        html.Div(id='optimization-suggestions')
                    ], label='Optimization Suggestions')
                ])
            ])
        ], className='mb-4')
    ], fluid=True)
    
    # Add callback to update chart
    @app.callback(
        dash.Output('performance-chart', 'figure'),
        dash.Input('medium-interval', 'n_intervals')
    )
    def update_chart(n: int) -> go.Figure:
        if performance_data is not None:
            return create_performance_chart(performance_data)
        return go.Figure()
    
    # Add callback to update metrics
    @app.callback(
        dash.Output('metrics-container', 'children'),
        dash.Input('fast-interval', 'n_intervals')
    )
    def update_metrics(n: int) -> list:
        if performance_data is not None:
            metrics = calculate_metrics(performance_data)
            return [
                html.Div([
                    html.H3(key.replace('_', ' ').title()),
                    html.P(f'{value:.4f}')
                ]) for key, value in metrics.items()
            ]
        return []
    
    # Add callbacks for learning insights if available
    if learning_viz:
        @app.callback(
            dash.Output('learning-curve', 'figure'),
            dash.Input('slow-interval', 'n_intervals')
        )
        def update_learning_curve(n: int) -> go.Figure:
            return learning_viz.create_learning_curve()
        
        @app.callback(
            dash.Output('strategy-evolution', 'figure'),
            dash.Input('slow-interval', 'n_intervals')
        )
        def update_strategy_evolution(n: int) -> go.Figure:
            return learning_viz.create_strategy_evolution()
        
        @app.callback(
            dash.Output('feature-importance', 'figure'),
            dash.Input('slow-interval', 'n_intervals')
        )
        def update_feature_importance(n: int) -> go.Figure:
            return learning_viz.create_feature_importance()
        
        @app.callback(
            dash.Output('performance-prediction', 'figure'),
            dash.Input('slow-interval', 'n_intervals')
        )
        def update_performance_prediction(n: int) -> go.Figure:
            return learning_viz.create_performance_prediction()
    
    # Add callbacks for enhanced learning features if monitor manager is available
    if monitor_manager:
        @app.callback(
            dash.Output('anomaly-detection', 'children'),
            dash.Input('medium-interval', 'n_intervals')
        )
        async def update_anomaly_detection(n: int) -> list:
            insights = await monitor_manager.get_learning_insights()
            anomaly_scores = insights.get('anomaly_scores', [])
            
            if not anomaly_scores:
                return html.P("No anomaly detection data available")
            
            # Count anomalies
            anomaly_count = sum(1 for score in anomaly_scores if score == -1)
            
            return [
                html.H4(f"Detected Anomalies: {anomaly_count}"),
                html.P(f"Total Points Analyzed: {len(anomaly_scores)}"),
                html.P(f"Anomaly Rate: {(anomaly_count/len(anomaly_scores))*100:.2f}%")
            ]
        
        @app.callback(
            dash.Output('optimization-suggestions', 'children'),
            dash.Input('slow-interval', 'n_intervals')
        )
        async def update_optimization_suggestions(n: int) -> list:
            insights = await monitor_manager.get_learning_insights()
            suggestions = insights.get('optimization_suggestions', [])
            
            if not suggestions:
                return html.P("No optimization suggestions available")
            
            return [
                html.H4("Optimization Suggestions"),
                html.Ul([
                    html.Li(suggestion) for suggestion in suggestions
                ])
            ]
    
    return app

def init_dashboard(
    performance_data: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None,
    history_manager: Optional[TradeHistoryManager] = None,
    monitor_manager: Optional[MonitorManager] = None,
    debug: bool = False
) -> dash.Dash:
    """Initialize and start the dashboard
    
    Args:
        performance_data: Optional initial performance data
        config: Optional dashboard configuration
        history_manager: Optional trade history manager
        monitor_manager: Optional monitor manager for enhanced learning
        debug: Enable debug mode
        
    Returns:
        Running Dash application
    """
    global app
    app = create_app(performance_data, config, history_manager, monitor_manager)
    if debug:
        app.enable_dev_tools()
    return app

# Initialize the app with default settings
init_dashboard() 