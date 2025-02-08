"""
Interactive Chart Utilities Module

This module provides enhanced interactive chart generation using Plotly
with support for real-time updates, customization, and caching.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import structlog
from functools import lru_cache
import json

logger = structlog.get_logger(__name__)

@dataclass
class ChartTheme:
    """Configuration for chart theming"""
    template: str
    color_scheme: List[str]
    font_family: str = "Arial"
    title_font_size: int = 24
    axis_font_size: int = 14
    
    @classmethod
    def get_light(cls) -> 'ChartTheme':
        """Get light theme configuration"""
        return cls(
            template='plotly_white',
            color_scheme=[
                '#1f77b4',  # Blue
                '#2ca02c',  # Green
                '#ff7f0e',  # Orange
                '#d62728',  # Red
                '#9467bd',  # Purple
                '#8c564b',  # Brown
                '#e377c2',  # Pink
                '#7f7f7f',  # Gray
                '#bcbd22',  # Yellow-green
                '#17becf'   # Cyan
            ]
        )
    
    @classmethod
    def get_dark(cls) -> 'ChartTheme':
        """Get dark theme configuration"""
        return cls(
            template='plotly_dark',
            color_scheme=[
                '#17becf',  # Cyan
                '#2ecc71',  # Green
                '#f1c40f',  # Yellow
                '#e74c3c',  # Red
                '#9b59b6',  # Purple
                '#3498db',  # Blue
                '#e67e22',  # Orange
                '#95a5a6',  # Gray
                '#1abc9c',  # Turquoise
                '#34495e'   # Navy
            ]
        )

class InteractiveChartGenerator:
    """
    Generates interactive Plotly charts for the dashboard.
    
    Features:
    - Real-time updates
    - Interactive tooltips and legends
    - Customizable themes and layouts
    - Performance optimized with caching
    - Advanced chart types and overlays
    """
    
    def __init__(self, theme: Optional[ChartTheme] = None):
        """Initialize chart generator with theme"""
        self.theme = theme or ChartTheme.get_light()
        self._setup_cache()
    
    def _setup_cache(self) -> None:
        """Setup chart caching"""
        self.cache = {}
        self.cache_ttl = 60  # 60 seconds cache TTL
    
    @lru_cache(maxsize=100)
    def _get_cached_figure(self, cache_key: str) -> Optional[go.Figure]:
        """Get cached figure if available and not expired"""
        if cache_key in self.cache:
            timestamp, figure = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.cache_ttl:
                return figure
            del self.cache[cache_key]
        return None
    
    def _cache_figure(self, cache_key: str, figure: go.Figure) -> None:
        """Cache figure with timestamp"""
        self.cache[cache_key] = (datetime.now(), figure)
    
    def create_profit_chart(
        self,
        data: List[Dict[str, Any]],
        height: int = 400,
        margin: Optional[Dict[str, int]] = None,
        show_trends: bool = True,
        show_annotations: bool = True
    ) -> go.Figure:
        """
        Create interactive profit analysis chart.
        
        Features:
        - Cumulative and individual profit views
        - Trend lines and moving averages
        - Annotated key events
        - Custom hover information
        """
        try:
            cache_key = f"profit_{json.dumps(data)}"
            cached_fig = self._get_cached_figure(cache_key)
            if cached_fig:
                return cached_fig
            
            df = pd.DataFrame(data)
            
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Cumulative Profit', 'Individual Trades'),
                specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
            )
            
            # Cumulative profit line
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['cumulative_profit'],
                    mode='lines',
                    name='Cumulative Profit',
                    line=dict(
                        color=self.theme.color_scheme[0],
                        width=2
                    ),
                    hovertemplate=(
                        'Time: %{x}<br>' +
                        'Profit: $%{y:.2f}<br>' +
                        '<extra></extra>'
                    )
                ),
                row=1, col=1
            )
            
            if show_trends:
                # Add moving average
                window = 20
                df['ma'] = df['cumulative_profit'].rolling(window=window).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['ma'],
                        mode='lines',
                        name=f'{window}-Period MA',
                        line=dict(
                            color=self.theme.color_scheme[1],
                            width=1,
                            dash='dash'
                        )
                    ),
                    row=1, col=1
                )
            
            # Individual trades scatter
            colors = [
                self.theme.color_scheme[0] if p > 0 
                else self.theme.color_scheme[3]
                for p in df['profit']
            ]
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['profit'],
                    mode='markers',
                    name='Individual Trades',
                    marker=dict(
                        color=colors,
                        size=8,
                        symbol='circle'
                    ),
                    hovertemplate=(
                        'Time: %{x}<br>' +
                        'Profit: $%{y:.2f}<br>' +
                        '<extra></extra>'
                    )
                ),
                row=2, col=1
            )
            
            if show_annotations:
                # Add annotations for significant events
                significant_profits = df[abs(df['profit']) > df['profit'].std() * 2]
                for _, row in significant_profits.iterrows():
                    fig.add_annotation(
                        x=row['timestamp'],
                        y=row['profit'],
                        text="Significant Trade",
                        showarrow=True,
                        arrowhead=1,
                        row=2, col=1
                    )
            
            # Update layout
            fig.update_layout(
                template=self.theme.template,
                height=height,
                margin=margin or dict(l=40, r=40, t=40, b=40),
                showlegend=True,
                hovermode='x unified'
            )
            
            self._cache_figure(cache_key, fig)
            return fig
            
        except Exception as e:
            logger.error("Error creating profit chart", error=str(e))
            return go.Figure()
    
    def create_liquidity_chart(
        self,
        data: List[Dict[str, Any]],
        height: int = 400,
        margin: Optional[Dict[str, int]] = None,
        show_depth: bool = True
    ) -> go.Figure:
        """
        Create liquidity analysis chart.
        
        Features:
        - Pool liquidity over time
        - Market depth visualization
        - Volume profile
        - Slippage indicators
        """
        try:
            df = pd.DataFrame(data)
            
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Liquidity Depth', 'Volume Profile')
            )
            
            # Liquidity depth
            if show_depth:
                buy_depth = df['buy_depth'] if 'buy_depth' in df else df['liquidity']
                sell_depth = df['sell_depth'] if 'sell_depth' in df else df['liquidity']
                
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=buy_depth,
                        fill='tonexty',
                        name='Buy Depth',
                        line=dict(color=self.theme.color_scheme[1])
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=-sell_depth,
                        fill='tonexty',
                        name='Sell Depth',
                        line=dict(color=self.theme.color_scheme[3])
                    ),
                    row=1, col=1
                )
            
            # Volume profile
            fig.add_trace(
                go.Bar(
                    x=df['timestamp'],
                    y=df['volume'],
                    name='Volume',
                    marker_color=self.theme.color_scheme[2]
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                template=self.theme.template,
                height=height,
                margin=margin or dict(l=40, r=40, t=40, b=40),
                showlegend=True,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error("Error creating liquidity chart", error=str(e))
            return go.Figure()
    
    def create_risk_heatmap(
        self,
        data: List[Dict[str, Any]],
        height: int = 400,
        margin: Optional[Dict[str, int]] = None
    ) -> go.Figure:
        """
        Create risk analysis heatmap.
        
        Features:
        - Risk exposure by time and metric
        - Correlation visualization
        - Interactive tooltips
        """
        try:
            df = pd.DataFrame(data)
            
            # Create correlation matrix
            risk_metrics = ['volatility', 'var', 'sharpe', 'sortino']
            corr_matrix = df[risk_metrics].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdYlBu',
                zmin=-1,
                zmax=1,
                hoverongaps=False
            ))
            
            # Update layout
            fig.update_layout(
                template=self.theme.template,
                height=height,
                margin=margin or dict(l=40, r=40, t=40, b=40),
                title='Risk Metric Correlations'
            )
            
            return fig
            
        except Exception as e:
            logger.error("Error creating risk heatmap", error=str(e))
            return go.Figure()
    
    def create_gas_optimization_chart(
        self,
        data: List[Dict[str, Any]],
        height: int = 400,
        margin: Optional[Dict[str, int]] = None,
        show_predictions: bool = True
    ) -> go.Figure:
        """
        Create gas optimization analysis chart.
        
        Features:
        - Gas price trends
        - Optimal gas predictions
        - Success rate overlay
        - Cost efficiency metrics
        """
        try:
            df = pd.DataFrame(data)
            
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Gas Prices', 'Success Rate')
            )
            
            # Gas prices
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['gas_price'],
                    mode='lines',
                    name='Gas Price',
                    line=dict(color=self.theme.color_scheme[0])
                ),
                row=1, col=1
            )
            
            if show_predictions:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['optimal_gas'],
                        mode='lines',
                        name='Optimal Gas',
                        line=dict(
                            color=self.theme.color_scheme[1],
                            dash='dash'
                        )
                    ),
                    row=1, col=1
                )
            
            # Success rate
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['success_rate'],
                    mode='lines',
                    name='Success Rate',
                    line=dict(color=self.theme.color_scheme[2])
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                template=self.theme.template,
                height=height,
                margin=margin or dict(l=40, r=40, t=40, b=40),
                showlegend=True,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error("Error creating gas optimization chart", error=str(e))
            return go.Figure()
    
    def create_opportunity_analysis(
        self,
        data: List[Dict[str, Any]],
        height: int = 400,
        margin: Optional[Dict[str, int]] = None,
        show_confidence: bool = True
    ) -> go.Figure:
        """
        Create opportunity analysis chart.
        
        Features:
        - Expected vs realized profit
        - Confidence scoring
        - Success rate metrics
        - Time-based patterns
        """
        try:
            df = pd.DataFrame(data)
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Profit Distribution', 'Success by Hour'),
                specs=[[{"type": "violin"}, {"type": "bar"}]]
            )
            
            # Profit distribution
            fig.add_trace(
                go.Violin(
                    y=df['profit'],
                    name='Profit Distribution',
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=self.theme.color_scheme[0],
                    line_color=self.theme.color_scheme[1]
                ),
                row=1, col=1
            )
            
            # Success by hour
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            success_by_hour = df.groupby('hour')['success'].mean()
            
            fig.add_trace(
                go.Bar(
                    x=success_by_hour.index,
                    y=success_by_hour.values,
                    name='Success Rate by Hour',
                    marker_color=self.theme.color_scheme[2]
                ),
                row=1, col=2
            )
            
            if show_confidence:
                # Add confidence bands
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['confidence'],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=df['confidence'],
                            colorscale='Viridis',
                            showscale=True
                        ),
                        name='Confidence'
                    ),
                    row=1, col=1
                )
            
            # Update layout
            fig.update_layout(
                template=self.theme.template,
                height=height,
                margin=margin or dict(l=40, r=40, t=40, b=40),
                showlegend=True,
                hovermode='closest'
            )
            
            return fig
            
        except Exception as e:
            logger.error("Error creating opportunity analysis", error=str(e))
            return go.Figure()
    
    def create_network_health_dashboard(
        self,
        data: List[Dict[str, Any]],
        height: int = 400,
        margin: Optional[Dict[str, int]] = None
    ) -> go.Figure:
        """
        Create network health monitoring dashboard.
        
        Features:
        - Block time monitoring
        - Gas price trends
        - Network congestion
        - Health indicators
        """
        try:
            df = pd.DataFrame(data)
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Block Time',
                    'Gas Price',
                    'Network Load',
                    'Health Score'
                )
            )
            
            # Block time
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['block_time'],
                    mode='lines',
                    name='Block Time',
                    line=dict(color=self.theme.color_scheme[0])
                ),
                row=1, col=1
            )
            
            # Gas price
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['gas_price'],
                    mode='lines',
                    name='Gas Price',
                    line=dict(color=self.theme.color_scheme[1])
                ),
                row=1, col=2
            )
            
            # Network load
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['network_load'],
                    mode='lines',
                    name='Network Load',
                    line=dict(color=self.theme.color_scheme[2])
                ),
                row=2, col=1
            )
            
            # Health score gauge
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=df['health_score'].iloc[-1],
                    gauge={
                        'axis': {'range': [0, 100]},
                        'steps': [
                            {'range': [0, 30], 'color': self.theme.color_scheme[3]},
                            {'range': [30, 70], 'color': self.theme.color_scheme[2]},
                            {'range': [70, 100], 'color': self.theme.color_scheme[1]}
                        ]
                    }
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                template=self.theme.template,
                height=height,
                margin=margin or dict(l=40, r=40, t=40, b=40),
                showlegend=True,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error("Error creating network health dashboard", error=str(e))
            return go.Figure()
    
    def create_gas_analysis_chart(
        self,
        data: List[Dict[str, Any]],
        height: int = 400,
        margin: Optional[Dict[str, int]] = None
    ) -> go.Figure:
        """Create gas analysis chart"""
        try:
            df = pd.DataFrame(data)
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['gas_price'],
                    mode='lines',
                    name='Gas Price',
                    line=dict(color=self.theme.color_scheme[0])
                )
            )
            
            fig.update_layout(
                template=self.theme.template,
                height=height,
                margin=margin or dict(l=40, r=40, t=40, b=40),
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error("Error creating gas chart", error=str(e))
            return go.Figure()
    
    def create_opportunity_chart(
        self,
        opportunities: List[float],
        timestamps: List[datetime],
        height: int = 400,
        margin: Optional[Dict[str, int]] = None
    ) -> go.Figure:
        """Create opportunity analysis chart"""
        try:
            df = pd.DataFrame({
                'timestamp': timestamps,
                'opportunities': opportunities
            })
            
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['opportunities'],
                    mode='lines+markers',
                    name='Opportunities',
                    line=dict(color=self.theme.color_scheme[0])
                )
            )
            
            fig.update_layout(
                template=self.theme.template,
                height=height,
                margin=margin or dict(l=40, r=40, t=40, b=40),
                showlegend=True,
                title='Opportunity Analysis',
                yaxis_title='Number of Opportunities',
                xaxis_title='Time'
            )
            
            return fig
            
        except Exception as e:
            logger.error("Error creating opportunity chart", error=str(e))
            return go.Figure() 