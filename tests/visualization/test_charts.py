"""
Test suite for chart generation and utilities
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from visualization.chart_utils import ChartTheme, InteractiveChartGenerator
from tests.utils.test_utils import create_mock_metrics

@pytest.fixture
def chart_theme():
    """Fixture for chart theme"""
    return ChartTheme(
        background_color='white',
        plot_background_color='white',
        grid_color='lightgray',
        text_color='black',
        font_family='Arial'
    )

@pytest.fixture
def chart_generator(chart_theme):
    """Fixture for chart generator"""
    return InteractiveChartGenerator(theme=chart_theme)

def test_line_chart_generation(chart_generator):
    """Test line chart generation"""
    data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=10),
        'value': np.random.randn(10)
    })
    
    fig = chart_generator.create_line_chart(
        data=data,
        x_column='timestamp',
        y_column='value',
        title='Test Line Chart'
    )
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == 'scatter'
    assert fig.data[0].mode == 'lines'

def test_bar_chart_generation(chart_generator):
    """Test bar chart generation"""
    data = pd.DataFrame({
        'category': ['A', 'B', 'C', 'D'],
        'value': np.random.randint(1, 100, 4)
    })
    
    fig = chart_generator.create_bar_chart(
        data=data,
        x_column='category',
        y_column='value',
        title='Test Bar Chart'
    )
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == 'bar'

def test_scatter_chart_generation(chart_generator):
    """Test scatter chart generation"""
    data = pd.DataFrame({
        'x': np.random.randn(20),
        'y': np.random.randn(20)
    })
    
    fig = chart_generator.create_scatter_chart(
        data=data,
        x_column='x',
        y_column='y',
        title='Test Scatter Chart'
    )
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == 'scatter'
    assert fig.data[0].mode == 'markers'

def test_theme_application(chart_generator, chart_theme):
    """Test theme application to charts"""
    fig = chart_generator.create_empty_figure()
    
    assert fig.layout.plot_bgcolor == chart_theme.plot_background_color
    assert fig.layout.paper_bgcolor == chart_theme.background_color
    assert fig.layout.font.family == chart_theme.font_family

def test_chart_update(chart_generator):
    """Test chart update functionality"""
    # Create initial chart
    initial_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=5),
        'value': np.random.randn(5)
    })
    
    fig = chart_generator.create_line_chart(
        data=initial_data,
        x_column='timestamp',
        y_column='value'
    )
    
    # Update with new data
    new_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-06', periods=5),
        'value': np.random.randn(5)
    })
    
    updated_fig = chart_generator.update_line_chart(
        fig=fig,
        data=new_data,
        x_column='timestamp',
        y_column='value'
    )
    
    assert len(updated_fig.data[0].x) == 10

def test_chart_layout_customization(chart_generator):
    """Test chart layout customization"""
    fig = chart_generator.create_empty_figure(
        title='Custom Chart',
        height=500,
        width=800,
        margin={'l': 50, 'r': 50, 't': 50, 'b': 50}
    )
    
    assert fig.layout.title.text == 'Custom Chart'
    assert fig.layout.height == 500
    assert fig.layout.width == 800
    assert fig.layout.margin.l == 50

def test_multiple_traces(chart_generator):
    """Test multiple trace handling"""
    data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=10),
        'value1': np.random.randn(10),
        'value2': np.random.randn(10)
    })
    
    fig = chart_generator.create_multi_line_chart(
        data=data,
        x_column='timestamp',
        y_columns=['value1', 'value2'],
        names=['Series 1', 'Series 2']
    )
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert all(trace.type == 'scatter' for trace in fig.data)

def test_chart_animation(chart_generator):
    """Test chart animation settings"""
    data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=10),
        'value': np.random.randn(10)
    })
    
    fig = chart_generator.create_animated_line_chart(
        data=data,
        x_column='timestamp',
        y_column='value',
        animation_duration=1000
    )
    
    assert isinstance(fig, go.Figure)
    assert fig.layout.transition.duration == 1000 