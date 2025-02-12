"""
Test suite for core visualization components
"""

import os
import sys
import pytest
import asyncio
from visualization.performance import PerformanceMonitor, PerformanceConfig
from visualization.chart_utils import ChartTheme, InteractiveChartGenerator
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

@pytest.fixture
def performance_monitor():
    """Fixture for performance monitoring"""
    config = PerformanceConfig(
        cpu_threshold=90.0,
        memory_threshold=90.0,
        render_time_threshold=5.0,
        monitoring_interval=0.1,
        enable_suggestions=True,
        enable_metrics=True
    )
    monitor = PerformanceMonitor(config)
    return monitor

@pytest.fixture
def chart_generator():
    """Fixture for chart generation"""
    return InteractiveChartGenerator(theme=ChartTheme.get_light())

@pytest.mark.asyncio
async def test_chart_generation(chart_generator):
    """Test basic chart generation"""
    # Create sample data
    data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=10),
        'value': np.random.randn(10)
    })
    
    # Generate line chart
    fig = chart_generator.create_line_chart(
        data=data,
        x_column='timestamp',
        y_column='value',
        title='Test Chart'
    )
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    assert fig.data[0].type == 'scatter'

@pytest.mark.asyncio
async def test_theme_application(chart_generator):
    """Test theme application to charts"""
    light_theme = ChartTheme.get_light()
    dark_theme = ChartTheme.get_dark()
    
    # Create chart with light theme
    fig_light = chart_generator.create_empty_figure()
    assert fig_light.layout.plot_bgcolor == light_theme.plot_background_color
    
    # Switch to dark theme
    chart_generator.theme = dark_theme
    fig_dark = chart_generator.create_empty_figure()
    assert fig_dark.layout.plot_bgcolor == dark_theme.plot_background_color

@pytest.mark.asyncio
async def test_chart_updates(chart_generator):
    """Test chart update functionality"""
    # Create initial chart
    data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=5),
        'value': np.random.randn(5)
    })
    
    fig = chart_generator.create_line_chart(
        data=data,
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
    
    assert len(updated_fig.data[0].x) == 10  # Combined length of old and new data 