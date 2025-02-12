"""Charts module for visualization"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, Any

def create_performance_chart(data: pd.DataFrame) -> go.Figure:
    """Create performance visualization chart"""
    fig = go.Figure()
    
    # Add profit line
    fig.add_trace(go.Scatter(
        x=data['timestamp'],
        y=data['profit'],
        name='Profit',
        line=dict(color='green', width=2)
    ))
    
    # Add gas usage line
    fig.add_trace(go.Scatter(
        x=data['timestamp'],
        y=data['gas_used'],
        name='Gas Used',
        yaxis='y2',
        line=dict(color='orange', width=2)
    ))
    
    # Add success rate line
    fig.add_trace(go.Scatter(
        x=data['timestamp'],
        y=data['success_rate'],
        name='Success Rate',
        yaxis='y3',
        line=dict(color='blue', width=2)
    ))
    
    # Update layout
    fig.update_layout(
        title='Performance Metrics Over Time',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Profit (ETH)', side='left'),
        yaxis2=dict(title='Gas Used', side='right', overlaying='y'),
        yaxis3=dict(title='Success Rate', side='right', overlaying='y'),
        hovermode='x unified',
        showlegend=True
    )
    
    return fig 