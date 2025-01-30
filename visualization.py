import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from collections import deque
import threading
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
import requests

logger = logging.getLogger(__name__)

class ArbitrageVisualizer:
    """Real-time visualization dashboard for arbitrage monitoring"""
    
    def __init__(self, max_points: int = 1000):
        self.max_points = max_points
        self.app = dash.Dash(__name__)
        
        # Real-time data storage
        self.data = {
            'timestamps': deque(maxlen=max_points),
            'profits': deque(maxlen=max_points),
            'confidence_scores': deque(maxlen=max_points),
            'risk_scores': deque(maxlen=max_points),
            'gas_prices': deque(maxlen=max_points),
            'volumes': deque(maxlen=max_points),
            'slippage': deque(maxlen=max_points),
            'price_impact': deque(maxlen=max_points),
            'execution_times': deque(maxlen=max_points)
        }
        
        # Performance tracking
        self.performance = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_profit': 0.0,
            'total_gas_spent': 0.0,
            'best_trade': None,
            'worst_trade': None
        }
        
        # Token pair analytics
        self.token_analytics = {}
        
        # DEX analytics
        self.dex_analytics = {}
        
        # Visualization settings
        plt.style.use('dark_background')
        sns.set_theme(style="darkgrid")
        
        self._setup_layout()
        self._setup_callbacks()
        
    def _setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = html.Div([
            html.H1('Base Chain Arbitrage Monitor', 
                   style={'textAlign': 'center', 'color': '#ffffff'}),
            
            # System Status
            html.Div([
                html.Div([
                    html.H3('System Status'),
                    html.H4(id='system-status', children='🟢 Active')
                ], className='status-box'),
                html.Div([
                    html.H3('24h Performance'),
                    html.Div(id='performance-summary')
                ], className='status-box'),
                html.Div([
                    html.H3('Current Network'),
                    html.H4('Base Chain'),
                    html.P(id='network-stats')
                ], className='status-box')
            ], className='status-row'),
            
            # Main Trading View
            html.Div([
                # Left panel - Real-time metrics
                html.Div([
                    dcc.Graph(id='profit-chart', className='chart'),
                    dcc.Graph(id='volume-chart', className='chart')
                ], className='panel'),
                
                # Right panel - Risk metrics
                html.Div([
                    dcc.Graph(id='risk-analysis', className='chart'),
                    dcc.Graph(id='gas-analysis', className='chart')
                ], className='panel')
            ], className='main-view'),
            
            # Bottom panel - Token & DEX Analytics
            html.Div([
                html.Div([
                    html.H3('Token Pair Performance'),
                    dcc.Graph(id='token-performance')
                ], className='bottom-panel'),
                html.Div([
                    html.H3('DEX Analytics'),
                    dcc.Graph(id='dex-analytics')
                ], className='bottom-panel')
            ], className='analytics-view'),
            
            # Active Opportunities
            html.Div([
                html.H3('Active Arbitrage Opportunities'),
                html.Div(id='opportunities-table')
            ], className='opportunities-view'),
            
            dcc.Interval(
                id='update-interval',
                interval=5000,  # Update every 5 seconds instead of 1 second
                n_intervals=0
            )
        ])
        
    def _setup_callbacks(self):
        """Setup dashboard callbacks"""
        @self.app.callback(
            [Output('profit-chart', 'figure'),
             Output('volume-chart', 'figure'),
             Output('risk-analysis', 'figure'),
             Output('gas-analysis', 'figure'),
             Output('token-performance', 'figure'),
             Output('dex-analytics', 'figure'),
             Output('performance-summary', 'children'),
             Output('network-stats', 'children'),
             Output('opportunities-table', 'children')],
            [Input('update-interval', 'n_intervals')]
        )
        def update_dashboard(_):
            return self._generate_dashboard_data()
            
    def _generate_dashboard_data(self):
        """Generate real-time dashboard data"""
        # Profit chart
        profit_fig = go.Figure()
        profit_fig.add_trace(go.Scatter(
            x=list(self.data['timestamps']),
            y=list(self.data['profits']),
            name='Profit (ETH)',
            line=dict(color='#00ff00')
        ))
        profit_fig.update_layout(
            title='Real-time Profit Analysis',
            template='plotly_dark',
            height=300
        )
        
        # Volume chart
        volume_fig = go.Figure()
        volume_fig.add_trace(go.Bar(
            x=list(self.data['timestamps']),
            y=list(self.data['volumes']),
            name='Trading Volume',
            marker_color='#4287f5'
        ))
        volume_fig.update_layout(
            title='Trading Volume',
            template='plotly_dark',
            height=300
        )
        
        # Risk analysis
        risk_fig = go.Figure()
        risk_fig.add_trace(go.Scatter(
            x=list(self.data['timestamps']),
            y=list(self.data['risk_scores']),
            name='Risk Score',
            line=dict(color='#ff0000')
        ))
        risk_fig.add_trace(go.Scatter(
            x=list(self.data['timestamps']),
            y=list(self.data['confidence_scores']),
            name='Confidence',
            line=dict(color='#00ff00')
        ))
        risk_fig.update_layout(
            title='Risk Analysis',
            template='plotly_dark',
            height=300
        )
        
        # Gas analysis
        gas_fig = go.Figure()
        gas_fig.add_trace(go.Scatter(
            x=list(self.data['timestamps']),
            y=list(self.data['gas_prices']),
            name='Gas Price (GWEI)',
            line=dict(color='#ffa500')
        ))
        gas_fig.update_layout(
            title='Gas Price Analysis',
            template='plotly_dark',
            height=300
        )
        
        # Token performance
        token_fig = go.Figure()
        if self.token_analytics:
            token_fig.add_trace(go.Bar(
                x=list(self.token_analytics.keys()),
                y=[data['profit'] for data in self.token_analytics.values()],
                name='Token Profit',
                marker_color='#00ff00'
            ))
        token_fig.update_layout(
            title='Token Pair Performance',
            template='plotly_dark',
            height=300
        )
        
        # DEX analytics
        dex_fig = go.Figure()
        if self.dex_analytics:
            dex_fig.add_trace(go.Bar(
                x=list(self.dex_analytics.keys()),
                y=[data['volume'] for data in self.dex_analytics.values()],
                name='DEX Volume',
                marker_color='#4287f5'
            ))
        dex_fig.update_layout(
            title='DEX Analytics',
            template='plotly_dark',
            height=300
        )
        
        # Performance summary
        performance_summary = html.Div([
            html.P(f"Total Trades: {self.performance['total_trades']}"),
            html.P(f"Success Rate: {self._calculate_success_rate():.1f}%"),
            html.P(f"Total Profit: {self.performance['total_profit']:.4f} ETH"),
            html.P(f"Gas Spent: {self.performance['total_gas_spent']:.4f} ETH")
        ])
        
        # Network stats
        network_stats = html.Div([
            html.P(f"Current Gas: {self.data['gas_prices'][-1] if self.data['gas_prices'] else 0} GWEI"),
            html.P(f"24h Volume: ${sum(self.data['volumes']):.2f}")
        ])
        
        # Active opportunities table
        opportunities_table = self._generate_opportunities_table()
        
        return (profit_fig, volume_fig, risk_fig, gas_fig, token_fig, dex_fig,
                performance_summary, network_stats, opportunities_table)
    
    def _calculate_success_rate(self) -> float:
        """Calculate current success rate"""
        if self.performance['total_trades'] == 0:
            return 0.0
        return (self.performance['successful_trades'] / self.performance['total_trades']) * 100
    
    def _generate_opportunities_table(self):
        """Generate active opportunities table"""
        # Get current opportunities
        current_opps = []
        for i, (profit, conf, risk) in enumerate(zip(
            self.data['profits'], 
            self.data['confidence_scores'], 
            self.data['risk_scores']
        )):
            if profit > 0 and conf > 0.8 and risk < 0.3:
                current_opps.append({
                    'id': f'opp_{i}',
                    'profit': profit,
                    'confidence': conf,
                    'risk': risk
                })
        
        # Create table
        return html.Table(
            [html.Tr([html.Th(col) for col in ['ID', 'Profit', 'Confidence', 'Risk']])] +
            [html.Tr([
                html.Td(opp['id']),
                html.Td(f"{opp['profit']:.4f} ETH"),
                html.Td(f"{opp['confidence']*100:.1f}%"),
                html.Td(f"{opp['risk']*100:.1f}%")
            ]) for opp in current_opps],
            className='opportunities-table'
        )
    
    def update_data(self, data: Dict):
        """Update real-time data"""
        timestamp = datetime.now()
        
        # Update time series data
        self.data['timestamps'].append(timestamp)
        self.data['profits'].append(data['profit_prediction'])
        self.data['confidence_scores'].append(data['confidence'])
        self.data['risk_scores'].append(data['risk_score'])
        self.data['gas_prices'].append(data['gas_price'])
        self.data['volumes'].append(data['volume_24h'])
        self.data['slippage'].append(data.get('slippage', 0))
        self.data['price_impact'].append(data.get('price_impact', 0))
        
        # Update token analytics
        token_pair = f"{data['tokens_involved'][0]}/{data['tokens_involved'][1]}"
        if token_pair not in self.token_analytics:
            self.token_analytics[token_pair] = {
                'trades': 0,
                'profit': 0,
                'volume': 0
            }
        self.token_analytics[token_pair]['trades'] += 1
        self.token_analytics[token_pair]['profit'] += data['profit_prediction']
        self.token_analytics[token_pair]['volume'] += data['volume_24h']
        
        # Update DEX analytics if available
        if 'dex' in data:
            dex_name = data['dex']
            if dex_name not in self.dex_analytics:
                self.dex_analytics[dex_name] = {
                    'trades': 0,
                    'volume': 0,
                    'profit': 0
                }
            self.dex_analytics[dex_name]['trades'] += 1
            self.dex_analytics[dex_name]['volume'] += data['volume_24h']
            self.dex_analytics[dex_name]['profit'] += data['profit_prediction']
    
    def update_trade_execution(self, trade: Dict):
        """Update trade execution data"""
        self.performance['total_trades'] += 1
        
        profit = trade.get('profit', 0)
        gas_used = trade.get('gas_used', 0)
        net_profit = profit - gas_used
        
        if trade.get('status') == 'completed':
            self.performance['successful_trades'] += 1
            self.performance['total_profit'] += profit
            self.performance['total_gas_spent'] += gas_used
            
            # Update best trade
            if not self.performance['best_trade'] or net_profit > self.performance['best_trade']['profit']:
                self.performance['best_trade'] = {
                    'profit': net_profit,
                    'tokens': trade['tokens_involved'],
                    'timestamp': datetime.now()
                }
        else:
            self.performance['failed_trades'] += 1
            
            # Update worst trade
            if not self.performance['worst_trade'] or net_profit < self.performance['worst_trade']['profit']:
                self.performance['worst_trade'] = {
                    'profit': net_profit,
                    'tokens': trade['tokens_involved'],
                    'timestamp': datetime.now()
                }
        
        # Add execution time to analytics
        if 'execution_time' in trade:
            self.data['execution_times'].append(trade['execution_time'])
    
    def run_in_thread(self):
        """Run the dashboard in a separate thread"""
        def run():
            self.app.run_server(debug=False, host='0.0.0.0', port=8050, use_reloader=False)
            
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        logger.info("Arbitrage visualization dashboard started on http://localhost:8050")
        
        # Wait a moment to ensure the server starts
        time.sleep(2)
        
        # Verify the server is running
        try:
            response = requests.get('http://localhost:8050')
            if response.status_code == 200:
                logger.info("Dashboard server is running successfully")
            else:
                logger.warning(f"Dashboard server returned status code: {response.status_code}")
        except Exception as e:
            logger.error(f"Error verifying dashboard server: {str(e)}")
    
    def stop(self):
        """Stop the visualization dashboard"""
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1)
        logger.info("Arbitrage visualization dashboard stopped")

# Create assets directory and add custom CSS
os.makedirs('assets', exist_ok=True)

with open('assets/custom.css', 'w') as f:
    f.write("""
body {
    background-color: #1a1a1a;
    color: #ffffff;
    font-family: Arial, sans-serif;
}

.status-row {
    display: flex;
    justify-content: space-between;
    margin: 20px;
}

.status-box {
    background-color: #2a2a2a;
    border-radius: 10px;
    padding: 15px;
    width: 30%;
    text-align: center;
}

.main-view {
    display: flex;
    justify-content: space-between;
    margin: 20px;
}

.panel {
    width: 48%;
    background-color: #2a2a2a;
    border-radius: 10px;
    padding: 15px;
}

.chart {
    margin-bottom: 20px;
}

.analytics-view {
    display: flex;
    justify-content: space-between;
    margin: 20px;
}

.bottom-panel {
    width: 48%;
    background-color: #2a2a2a;
    border-radius: 10px;
    padding: 15px;
}

.opportunities-view {
    margin: 20px;
    background-color: #2a2a2a;
    border-radius: 10px;
    padding: 15px;
}

.opportunities-table {
    width: 100%;
    border-collapse: collapse;
}

.opportunities-table th, .opportunities-table td {
    padding: 10px;
    text-align: left;
    border-bottom: 1px solid #3a3a3a;
}

.opportunities-table th {
    background-color: #3a3a3a;
}
""") 