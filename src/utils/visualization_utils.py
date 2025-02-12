import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import io
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import structlog
import matplotlib.dates as mdates

logger = structlog.get_logger(__name__)

class VisualizationUtils:
    """
    Enhanced visualization utilities for dashboard and notifications.
    
    Features:
    - Static chart generation for reports/exports
    - Interactive Plotly charts for dashboard
    - Consistent styling and theming
    - Advanced analytics visualizations
    - Performance optimized rendering
    """
    
    # Color schemes
    COLORS = {
        'profit': '#2ecc71',
        'loss': '#e74c3c',
        'neutral': '#3498db',
        'warning': '#f1c40f',
        'gas': '#9b59b6'
    }
    
    @staticmethod
    def setup_style(theme: str = 'dark') -> None:
        """Setup consistent plotting style"""
        if theme == 'dark':
            plt.style.use('dark_background')
            sns.set_theme(style="darkgrid", palette="deep")
        else:
            plt.style.use('default')
            sns.set_theme(style="whitegrid", palette="deep")
    
    @staticmethod
    def create_interactive_profit_chart(
        data: List[Dict],
        height: int = 400,
        dark_mode: bool = True
    ) -> go.Figure:
        """Create interactive profit chart with Plotly"""
        try:
            df = pd.DataFrame(data)
            
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Cumulative Profit', 'Individual Trades')
            )
            
            # Cumulative profit line
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['cumulative_profit'],
                    mode='lines',
                    name='Cumulative Profit',
                    line=dict(
                        color=VisualizationUtils.COLORS['profit'],
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
            
            # Individual trade scatter
            colors = [
                VisualizationUtils.COLORS['profit'] if p > 0 
                else VisualizationUtils.COLORS['loss']
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
            
            # Update layout
            fig.update_layout(
                template='plotly_dark' if dark_mode else 'plotly_white',
                height=height,
                showlegend=True,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error("Error creating interactive profit chart", error=str(e))
            return go.Figure()
    
    @staticmethod
    def create_interactive_gas_analysis(
        data: List[Dict],
        height: int = 400,
        dark_mode: bool = True
    ) -> go.Figure:
        """Create interactive gas analysis chart with Plotly"""
        try:
            df = pd.DataFrame(data)
            
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Gas Prices', 'Gas Usage Efficiency')
            )
            
            # Gas prices
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['base_fee'],
                    mode='lines',
                    name='Base Fee',
                    line=dict(color=VisualizationUtils.COLORS['gas'])
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['priority_fee'],
                    mode='lines',
                    name='Priority Fee',
                    line=dict(color=VisualizationUtils.COLORS['warning'])
                ),
                row=1, col=1
            )
            
            # Gas efficiency (if profit data available)
            if 'profit' in df.columns:
                efficiency = df['profit'] / (df['base_fee'] + df['priority_fee'])
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=efficiency,
                        mode='lines',
                        name='Gas Efficiency',
                        line=dict(color=VisualizationUtils.COLORS['neutral'])
                    ),
                    row=2, col=1
                )
            
            # Update layout
            fig.update_layout(
                template='plotly_dark' if dark_mode else 'plotly_white',
                height=height,
                showlegend=True,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error("Error creating interactive gas analysis", error=str(e))
            return go.Figure()
    
    @staticmethod
    def create_opportunity_heatmap(
        data: List[Dict],
        height: int = 400,
        dark_mode: bool = True
    ) -> go.Figure:
        """Create opportunity analysis heatmap with Plotly"""
        try:
            df = pd.DataFrame(data)
            
            # Create time bins and profit bins
            df['hour'] = df['timestamp'].dt.hour
            df['profit_bin'] = pd.qcut(df['expected_profit'], q=10, labels=False)
            
            # Create heatmap data
            heatmap_data = pd.crosstab(df['hour'], df['profit_bin'])
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='Viridis',
                hoverongaps=False
            ))
            
            # Update layout
            fig.update_layout(
                template='plotly_dark' if dark_mode else 'plotly_white',
                height=height,
                title='Opportunity Distribution by Hour',
                xaxis_title='Profit Level (0=Low, 9=High)',
                yaxis_title='Hour of Day'
            )
            
            return fig
            
        except Exception as e:
            logger.error("Error creating opportunity heatmap", error=str(e))
            return go.Figure()
    
    @staticmethod
    def create_network_status_gauge(
        data: Dict[str, float],
        height: int = 200,
        dark_mode: bool = True
    ) -> go.Figure:
        """Create network status gauge chart with Plotly"""
        try:
            fig = go.Figure()
            
            # Add network health gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=data.get('network_health', 0) * 100,
                title={'text': "Network Health"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': VisualizationUtils.COLORS['neutral']},
                    'steps': [
                        {'range': [0, 30], 'color': VisualizationUtils.COLORS['loss']},
                        {'range': [30, 70], 'color': VisualizationUtils.COLORS['warning']},
                        {'range': [70, 100], 'color': VisualizationUtils.COLORS['profit']}
                    ]
                }
            ))
            
            # Update layout
            fig.update_layout(
                template='plotly_dark' if dark_mode else 'plotly_white',
                height=height
            )
            
            return fig
            
        except Exception as e:
            logger.error("Error creating network status gauge", error=str(e))
            return go.Figure()
    
    @staticmethod
    def create_token_analytics_chart(
        token_analytics: Dict,
        top_n: int = 10,
        height: int = 400,
        dark_mode: bool = True
    ) -> go.Figure:
        """Create interactive token analytics visualization"""
        try:
            # Convert to DataFrame and sort by profit
            df = pd.DataFrame.from_dict(token_analytics, orient='index')
            df = df.sort_values('profit', ascending=False).head(top_n)
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Token Profits', 'Success Rates'),
                vertical_spacing=0.2
            )
            
            # Profit by token
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['profit'],
                    name='Profit',
                    marker_color=VisualizationUtils.COLORS['profit']
                ),
                row=1, col=1
            )
            
            # Success rate by token
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['success_rate'],
                    name='Success Rate',
                    marker_color=VisualizationUtils.COLORS['neutral']
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                template='plotly_dark' if dark_mode else 'plotly_white',
                height=height,
                showlegend=True,
                hovermode='x unified'
            )
            
            # Update axes
            fig.update_xaxes(tickangle=45)
            
            return fig
            
        except Exception as e:
            logger.error("Error creating token analytics chart", error=str(e))
            return go.Figure()
    
    @staticmethod
    def export_chart_as_image(
        fig: Union[go.Figure, Figure],
        format: str = 'png'
    ) -> io.BytesIO:
        """Export chart as image buffer"""
        try:
            buf = io.BytesIO()
            
            if isinstance(fig, go.Figure):
                fig.write_image(buf, format=format)
            else:
                fig.savefig(buf, format=format, bbox_inches='tight')
                plt.close(fig)
            
            buf.seek(0)
            return buf
            
        except Exception as e:
            logger.error("Error exporting chart", error=str(e))
            return io.BytesIO()
    
    @staticmethod
    def create_correlation_matrix(
        data: pd.DataFrame,
        height: int = 400,
        dark_mode: bool = True
    ) -> go.Figure:
        """Create correlation matrix heatmap"""
        try:
            # Calculate correlation matrix
            corr_matrix = data.corr()
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmin=-1,
                zmax=1
            ))
            
            # Update layout
            fig.update_layout(
                template='plotly_dark' if dark_mode else 'plotly_white',
                height=height,
                title='Metric Correlations'
            )
            
            return fig
            
        except Exception as e:
            logger.error("Error creating correlation matrix", error=str(e))
            return go.Figure()
    
    @staticmethod
    def create_performance_indicators(
        metrics: Dict[str, float],
        height: int = 200,
        dark_mode: bool = True
    ) -> go.Figure:
        """Create performance indicator gauges"""
        try:
            fig = make_subplots(
                rows=1,
                cols=3,
                specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]]
            )
            
            # Success Rate
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=metrics.get('success_rate', 0) * 100,
                    title={'text': "Success Rate (%)"},
                    gauge={'axis': {'range': [0, 100]}},
                    domain={'row': 0, 'column': 0}
                ),
                row=1, col=1
            )
            
            # ROI
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=metrics.get('roi', 0) * 100,
                    title={'text': "ROI (%)"},
                    gauge={'axis': {'range': [-50, 50]}},
                    domain={'row': 0, 'column': 1}
                ),
                row=1, col=2
            )
            
            # Gas Efficiency
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=metrics.get('gas_efficiency', 0) * 100,
                    title={'text': "Gas Efficiency (%)"},
                    gauge={'axis': {'range': [0, 100]}},
                    domain={'row': 0, 'column': 2}
                ),
                row=1, col=3
            )
            
            # Update layout
            fig.update_layout(
                template='plotly_dark' if dark_mode else 'plotly_white',
                height=height
            )
            
            return fig
            
        except Exception as e:
            logger.error("Error creating performance indicators", error=str(e))
            return go.Figure()

    @staticmethod
    def create_profit_chart(
        timestamps: List[datetime],
        profits: List[float],
        title: str = "Profit Over Time"
    ) -> io.BytesIO:
        """Create profit over time chart"""
        plt.figure(figsize=(10, 6))
        timestamps_num = mdates.date2num(timestamps)  # Convert to numeric format
        plt.plot(timestamps_num, profits, color='green', linewidth=2)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Profit (ETH)')
        plt.grid(True, alpha=0.3)
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf
        
    @staticmethod
    def create_performance_summary(
        performance: Dict,
        timeframe: str = "24h"
    ) -> io.BytesIO:
        """Create performance summary visualization"""
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart of trade outcomes
        outcomes = [
            performance['successful_trades'],
            performance['failed_trades']
        ]
        labels = ['Successful', 'Failed']
        colors = ['green', 'red']
        
        ax1.pie(outcomes, labels=labels, colors=colors, autopct='%1.1f%%')
        ax1.set_title(f'Trade Outcomes ({timeframe})')
        
        # Bar chart of profits and gas costs
        metrics = ['Total Profit', 'Gas Spent', 'Net Profit']
        values = [
            performance['total_profit'],
            performance['total_gas_spent'],
            performance['total_profit'] - performance['total_gas_spent']
        ]
        
        ax2.bar(metrics, values, color=['green', 'orange', 'blue'])
        ax2.set_title(f'Profit Metrics ({timeframe})')
        ax2.set_ylabel('ETH')
        plt.xticks(rotation=45)
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf
        
    @staticmethod
    def create_token_analytics_static_chart(
        token_analytics: Dict,
        top_n: int = 10
    ) -> io.BytesIO:
        """Create token analytics visualization"""
        # Convert to DataFrame and sort by profit
        df = pd.DataFrame.from_dict(token_analytics, orient='index')
        df = df.sort_values('profit', ascending=False).head(top_n)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Profit by token
        sns.barplot(data=df, x=df.index, y='profit', ax=ax1, color='green')
        ax1.set_title('Top Token Profits')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        
        # Success rate by token
        sns.barplot(data=df, x=df.index, y='success_rate', ax=ax2, color='blue')
        ax2.set_title('Token Success Rates')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf
        
    @staticmethod
    def create_dex_analytics_chart(
        dex_analytics: Dict,
        metrics: List[str] = ['profit', 'volume', 'success_rate']
    ) -> io.BytesIO:
        """Create DEX analytics visualization"""
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(dex_analytics, orient='index')
        
        # Create figure with subplots
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 5*len(metrics)))
        
        for i, metric in enumerate(metrics):
            sns.barplot(data=df, x=df.index, y=metric, ax=axes[i])
            axes[i].set_title(f'DEX {metric.replace("_", " ").title()}')
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf 