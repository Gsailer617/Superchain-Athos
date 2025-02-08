"""
Enhanced Visualization Dashboard Module

This module provides a real-time, interactive dashboard for monitoring arbitrage operations.
Features include auto-refreshing metrics, interactive charts, and responsive design.
"""

from dash import Dash, dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, TypedDict
import logging
from dataclasses import dataclass
from functools import lru_cache
import time
from prometheus_client import Counter, Histogram
import structlog
import asyncio
from visualization.chart_utils import InteractiveChartGenerator, ChartTheme
from visualization.analytics import DataAnalyzer, AnalyticsConfig
from visualization.performance import PerformanceMonitor, PerformanceConfig

logger = structlog.get_logger(__name__)

# Metrics for performance monitoring
METRICS = {
    'render_time': Histogram(
        'visualization_render_seconds',
        'Time spent rendering visualizations',
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
    ),
    'update_count': Counter(
        'visualization_updates_total',
        'Total number of visualization updates'
    ),
    'error_count': Counter(
        'visualization_errors_total',
        'Total number of visualization errors',
        ['type']
    )
}

class Alert(TypedDict):
    message: str
    color: str
    timestamp: datetime

class Profit(TypedDict):
    timestamp: datetime
    profit: float
    cumulative_profit: float

class GasPrice(TypedDict):
    timestamp: datetime
    base_fee: int
    priority_fee: int

class Opportunity(TypedDict):
    timestamp: datetime
    expected_profit: float
    confidence: float
    executed: bool

class NetworkStatus(TypedDict):
    timestamp: datetime
    healthy: bool
    latency: float

class DataStore(TypedDict):
    profits: List[Profit]
    gas_prices: List[GasPrice]
    opportunities: List[Opportunity]
    network_status: List[NetworkStatus]

@dataclass
class ChartConfig:
    """Configuration for chart appearance and behavior"""
    theme: str = 'light'
    height: int = 400
    margin: Dict[str, int] = {'l': 40, 'r': 40, 't': 40, 'b': 40}
    animation_duration: int = 1000
    
    def __post_init__(self):
        pass  # No need for post init since we have default value

class ArbitrageVisualizer:
    """
    Enhanced visualization dashboard for arbitrage monitoring.
    
    Features:
    - Real-time metric updates
    - Interactive charts with Plotly
    - Responsive design with Bootstrap
    - Dark/Light theme switching
    - Performance monitoring and caching
    - Live alerts and notifications
    """
    
    def __init__(
        self,
        port: int = 8050,
        debug: bool = False,
        config: Optional[Dict] = None
    ):
        """Initialize the visualization dashboard"""
        self.port = port
        self.debug = debug
        self.config = config or {}
        
        # Initialize Dash app with Bootstrap
        self.app = Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )
        
        # Chart configuration
        self.chart_config = ChartConfig(
            theme=self.config.get('theme', 'light'),
            height=self.config.get('chart_height', 400),
            animation_duration=self.config.get('animation_duration', 1000)
        )
        
        # Initialize components
        self.chart_generator = InteractiveChartGenerator(
            theme=ChartTheme.get_light() if self.chart_config.theme == 'light' else ChartTheme.get_dark()
        )
        self.data_analyzer = DataAnalyzer(AnalyticsConfig())
        self.performance_monitor = PerformanceMonitor(PerformanceConfig())
        
        # Initialize data storage
        self.data: DataStore = {
            'profits': [],
            'gas_prices': [],
            'opportunities': [],
            'network_status': []
        }
        
        # Alert storage
        self.active_alerts: List[Alert] = []
        
        # Setup layout and callbacks
        self._setup_layout()
        self._setup_callbacks()
        
    def _setup_layout(self):
        """Setup the dashboard layout with Bootstrap components"""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("Arbitrage Dashboard", className="text-primary mb-4"),
                    dbc.Switch(
                        id='theme-switch',
                        label="Dark Mode",
                        value=self.chart_config.theme == 'dark'
                    )
                ])
            ], className="mt-4"),
            
            # Auto-refresh intervals
            dcc.Interval(
                id='fast-interval',
                interval=1*1000,  # 1 second for critical updates
                n_intervals=0
            ),
            dcc.Interval(
                id='medium-interval',
                interval=5*1000,  # 5 seconds for charts
                n_intervals=0
            ),
            dcc.Interval(
                id='slow-interval',
                interval=30*1000,  # 30 seconds for analytics
                n_intervals=0
            ),
            
            # Alert area
            dbc.Row([
                dbc.Col([
                    html.Div(id='alert-container')
                ])
            ], className="mt-2"),
            
            # Main content
            dbc.Row([
                # Left column - Key Metrics
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Key Metrics"),
                        dbc.CardBody([
                            html.Div(id='key-metrics-container')
                        ])
                    ])
                ], width=4),
                
                # Right column - Charts
                dbc.Col([
                    dbc.Tabs([
                        dbc.Tab([
                            dcc.Loading(
                                dcc.Graph(id='profit-chart'),
                                type="circle"
                            )
                        ], label="Profit Analysis"),
                        dbc.Tab([
                            dcc.Loading(
                                dcc.Graph(id='gas-chart'),
                                type="circle"
                            )
                        ], label="Gas Analysis")
                    ])
                ], width=8)
            ], className="mt-4"),
            
            # Bottom row - Detailed Stats
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Opportunity Analysis"),
                        dbc.CardBody([
                            dcc.Loading(
                                dcc.Graph(id='opportunity-chart'),
                                type="circle"
                            )
                        ])
                    ])
                ])
            ], className="mt-4"),
            
            # Performance Metrics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("System Performance"),
                        dbc.CardBody([
                            html.Div(id='performance-metrics-container')
                        ])
                    ])
                ])
            ], className="mt-4")
            
        ], fluid=True)
        
    def _setup_callbacks(self):
        """Setup all dashboard callbacks"""
        
        @self.app.callback(
            Output('key-metrics-container', 'children'),
            Input('fast-interval', 'n_intervals')
        )
        def update_key_metrics(n):
            """Update key metrics display"""
            try:
                with METRICS['render_time'].time():
                    metrics = self._get_current_metrics()
                    
                    return dbc.Row([
                        dbc.Col([
                            html.H4(f"${metrics['total_profit']:.2f}"),
                            html.P("Total Profit")
                        ]),
                        dbc.Col([
                            html.H4(f"{metrics['success_rate']:.1%}"),
                            html.P("Success Rate")
                        ]),
                        dbc.Col([
                            html.H4(f"{metrics['gas_price']} gwei"),
                            html.P("Current Gas")
                        ])
                    ])
            except Exception as e:
                logger.error("Error updating metrics", error=str(e))
                METRICS['error_count'].labels(type='metrics_update').inc()
                self._add_alert("Error updating metrics", "danger")
                raise PreventUpdate
        
        @self.app.callback(
            [Output('profit-chart', 'figure'),
             Output('gas-chart', 'figure'),
             Output('opportunity-chart', 'figure')],
            Input('medium-interval', 'n_intervals'),
            State('theme-switch', 'value')
        )
        def update_charts(n, dark_mode):
            """Update all charts"""
            try:
                with METRICS['render_time'].time():
                    # Update chart theme based on dark mode
                    self.chart_generator = InteractiveChartGenerator(
                        theme=ChartTheme.get_dark() if dark_mode else ChartTheme.get_light()
                    )
                    
                    # Generate charts
                    profit_fig, gas_fig, opportunity_fig = self.update_charts(dark_mode)
                    
                    return profit_fig, gas_fig, opportunity_fig
                    
            except Exception as e:
                logger.error("Error updating charts", error=str(e))
                METRICS['error_count'].labels(type='chart_update').inc()
                self._add_alert("Error updating charts", "danger")
                raise PreventUpdate
        
        @self.app.callback(
            Output('performance-metrics-container', 'children'),
            Input('slow-interval', 'n_intervals')
        )
        async def update_performance_metrics(n):
            """Update performance metrics display"""
            try:
                metrics = await self.performance_monitor.run_health_check()
                suggestions = self.performance_monitor._generate_optimization_suggestions(metrics)
                
                return [
                    dbc.Row([
                        dbc.Col([
                            html.H6("CPU Usage"),
                            html.P(f"{metrics['cpu_usage']*100:.1f}%")
                        ]),
                        dbc.Col([
                            html.H6("Memory Usage"),
                            html.P(f"{metrics['memory_percent']:.1f}%")
                        ]),
                        dbc.Col([
                            html.H6("Render Time"),
                            html.P(f"{metrics.get('render_time_avg', 0):.3f}s")
                        ])
                    ]),
                    html.Hr(),
                    html.H6("Optimization Suggestions"),
                    html.Ul([html.Li(s) for s in suggestions])
                ]
                
            except Exception as e:
                logger.error("Error updating performance metrics", error=str(e))
                METRICS['error_count'].labels(type='performance_update').inc()
                self._add_alert("Error updating performance metrics", "warning")
                raise PreventUpdate
        
        @self.app.callback(
            Output('alert-container', 'children'),
            Input('fast-interval', 'n_intervals')
        )
        def update_alerts(n):
            """Update alert display"""
            return [
                dbc.Alert(
                    alert['message'],
                    color=alert['color'],
                    dismissable=True,
                    is_open=True,
                    duration=5000
                )
                for alert in self.active_alerts[-5:]  # Show last 5 alerts
            ]
    
    def _add_alert(self, message: str, color: str) -> None:
        """Add a new alert to the active alerts list"""
        self.active_alerts.append({
            'message': message,
            'color': color,
            'timestamp': datetime.now()
        })
        # Keep only last 10 alerts
        self.active_alerts = self.active_alerts[-10:]
    
    @lru_cache(maxsize=100)
    def _get_current_metrics(self) -> Dict[str, float]:
        """Get current system metrics with caching"""
        try:
            return {
                'total_profit': sum(p['profit'] for p in self.data['profits']),
                'success_rate': self._calculate_success_rate(),
                'gas_price': self._get_current_gas_price()
            }
        except Exception as e:
            logger.error("Error getting metrics", error=str(e))
            return {'total_profit': 0.0, 'success_rate': 0.0, 'gas_price': 0}
    
    def _calculate_success_rate(self) -> float:
        """Calculate the success rate of arbitrage opportunities"""
        try:
            opportunities = self.data['opportunities']
            if not opportunities:
                return 0.0
            
            successful = sum(1 for op in opportunities if op['executed'])
            return successful / len(opportunities)
        except Exception as e:
            logger.error("Error calculating success rate", error=str(e))
            return 0.0
    
    def _get_current_gas_price(self) -> int:
        """Get the current gas price"""
        try:
            gas_prices = self.data['gas_prices']
            if not gas_prices:
                return 0
            return gas_prices[-1]['base_fee'] + gas_prices[-1]['priority_fee']
        except Exception as e:
            logger.error("Error getting gas price", error=str(e))
            return 0
    
    def update_data(self, new_data: Dict[str, Any]) -> None:
        """Update dashboard data with new metrics"""
        try:
            timestamp = datetime.now()
            
            # Update profits
            if 'profit' in new_data:
                self.data['profits'].append({
                    'timestamp': timestamp,
                    'profit': new_data['profit'],
                    'cumulative_profit': sum(p['profit'] for p in self.data['profits']) + new_data['profit']
                })
            
            # Update gas prices
            if 'gas_price' in new_data:
                self.data['gas_prices'].append({
                    'timestamp': timestamp,
                    'base_fee': new_data['gas_price']['base_fee'],
                    'priority_fee': new_data['gas_price']['priority_fee']
                })
            
            # Update opportunities
            if 'opportunity' in new_data:
                self.data['opportunities'].append({
                    'timestamp': timestamp,
                    'expected_profit': new_data['opportunity']['expected_profit'],
                    'confidence': new_data['opportunity']['confidence'],
                    'executed': new_data['opportunity'].get('executed', False)
                })
            
            # Update network status
            if 'network_status' in new_data:
                self.data['network_status'].append({
                    'timestamp': timestamp,
                    'healthy': new_data['network_status']['healthy'],
                    'latency': new_data['network_status']['latency']
                })
            
            # Trim old data
            self._trim_old_data()
            
            METRICS['update_count'].inc()
            
        except Exception as e:
            logger.error("Error updating data", error=str(e))
            METRICS['error_count'].labels(type='data_update').inc()
            self._add_alert("Error updating dashboard data", "danger")
    
    def _trim_old_data(self, max_age: timedelta = timedelta(hours=24)):
        """Trim data older than max_age"""
        try:
            cutoff = datetime.now() - max_age
            
            # Trim each data type separately
            self.data['profits'] = [
                item for item in self.data['profits']
                if item['timestamp'] > cutoff
            ]
            self.data['gas_prices'] = [
                item for item in self.data['gas_prices']
                if item['timestamp'] > cutoff
            ]
            self.data['opportunities'] = [
                item for item in self.data['opportunities']
                if item['timestamp'] > cutoff
            ]
            self.data['network_status'] = [
                item for item in self.data['network_status']
                if item['timestamp'] > cutoff
            ]
        except Exception as e:
            logger.error("Error trimming old data", error=str(e))
        
    def _prepare_chart_data(self) -> Dict[str, pd.DataFrame]:
        """Prepare data for charts in a consistent format"""
        try:
            chart_data = {
                'profits': pd.DataFrame(self.data['profits']),
                'gas': pd.DataFrame(self.data['gas_prices']),
                'opportunities': pd.DataFrame(self.data['opportunities']),
                'network': pd.DataFrame(self.data['network_status'])
            }
            return chart_data
        except Exception as e:
            logger.error("Error preparing chart data", error=str(e))
            return {}

    def create_opportunity_chart(self) -> go.Figure:
        """Create opportunity analysis chart"""
        try:
            chart_data = self._prepare_chart_data()
            if 'opportunities' not in chart_data:
                return go.Figure()
            
            df = chart_data['opportunities']
            return self.chart_generator.create_opportunity_chart(
                opportunities=df['expected_profit'].astype(float).tolist(),
                timestamps=df['timestamp'].tolist(),
                height=self.chart_config.height,
                margin=self.chart_config.margin
            )
        except Exception as e:
            logger.error("Error creating opportunity chart", error=str(e))
            return go.Figure()

    def update_charts(self, dark_mode: bool) -> tuple[go.Figure, go.Figure, go.Figure]:
        """Update all charts with consistent data"""
        try:
            with METRICS['render_time'].time():
                # Update chart theme based on dark mode
                self.chart_generator = InteractiveChartGenerator(
                    theme=ChartTheme.get_dark() if dark_mode else ChartTheme.get_light()
                )
                
                # Prepare data once for all charts
                chart_data = self._prepare_chart_data()
                if not chart_data:
                    raise ValueError("No chart data available")
                
                # Generate charts using the prepared data
                profit_fig = self.chart_generator.create_profit_chart(
                    data=chart_data['profits'].to_dict('records'),
                    height=self.chart_config.height,
                    margin=self.chart_config.margin
                )
                
                gas_fig = self.chart_generator.create_gas_analysis_chart(
                    data=chart_data['gas'].to_dict('records'),
                    height=self.chart_config.height,
                    margin=self.chart_config.margin
                )
                
                opportunity_fig = self.create_opportunity_chart()
                
                return profit_fig, gas_fig, opportunity_fig
                
        except Exception as e:
            logger.error("Error updating charts", error=str(e))
            METRICS['error_count'].labels(type='chart_update').inc()
            self._add_alert("Error updating charts", "danger")
            raise PreventUpdate
    
    def run(self):
        """Run the dashboard server"""
        try:
            # Start performance monitoring
            asyncio.run(self.performance_monitor.start_monitoring())
            
            # Run the Dash server
            self.app.run_server(
                port=self.port,
                debug=self.debug
            )
        except Exception as e:
            logger.error("Error running dashboard", error=str(e))
            raise
        finally:
            # Stop performance monitoring
            asyncio.run(self.performance_monitor.stop_monitoring())