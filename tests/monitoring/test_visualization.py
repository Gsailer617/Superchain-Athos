"""Tests for monitoring visualization components"""

import pytest
import dash
from dash.testing.application_runners import import_app
from dash.testing.composite import DashComposite
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.monitoring.monitor_manager import MonitorManager
from src.visualization.dashboard import create_app
from tests.utils.test_utils import (
    create_mock_metrics,
    create_mock_trade_history,
    create_mock_learning_insights
)

@pytest.fixture
def dash_duo():
    """Fixture for Dash testing"""
    with DashComposite() as dc:
        yield dc

@pytest.mark.visualization
def test_dashboard_initialization(dash_duo, monitor_manager):
    """Test dashboard initialization"""
    # Create app
    app = create_app(monitor_manager=monitor_manager)
    
    # Start dashboard
    dash_duo.start_server(app)
    
    # Check main components exist
    assert dash_duo.find_element("#profit-loss-chart")
    assert dash_duo.find_element("#gas-usage-chart")
    assert dash_duo.find_element("#execution-time-chart")
    assert dash_duo.find_element("#success-rate-chart")

@pytest.mark.visualization
def test_performance_charts(dash_duo, monitor_manager):
    """Test performance metrics visualization"""
    # Record some test data
    for i in range(10):
        monitor_manager.record_trade(
            strategy="test_strategy",
            token_pair="ETH-USDC",
            dex="uniswap",
            profit=1.0 * i,
            gas_price=50.0,
            execution_time=0.1 * i,
            success=True
        )
    
    # Create and start app
    app = create_app(monitor_manager=monitor_manager)
    dash_duo.start_server(app)
    
    # Wait for charts to update
    dash_duo.wait_for_element("#profit-loss-chart .js-plotly-plot")
    
    # Verify chart data
    profit_chart = dash_duo.find_element("#profit-loss-chart")
    assert profit_chart is not None
    assert "Profit/Loss" in profit_chart.text

@pytest.mark.visualization
def test_system_metrics_charts(dash_duo, monitor_manager):
    """Test system metrics visualization"""
    app = create_app(monitor_manager=monitor_manager)
    dash_duo.start_server(app)
    
    # Wait for system metrics charts
    dash_duo.wait_for_element("#cpu-usage-chart .js-plotly-plot")
    dash_duo.wait_for_element("#memory-usage-chart .js-plotly-plot")
    
    # Verify charts
    cpu_chart = dash_duo.find_element("#cpu-usage-chart")
    memory_chart = dash_duo.find_element("#memory-usage-chart")
    assert "CPU Usage" in cpu_chart.text
    assert "Memory Usage" in memory_chart.text

@pytest.mark.visualization
def test_learning_insights_charts(dash_duo, monitor_manager):
    """Test learning insights visualization"""
    # Generate test insights
    insights = create_mock_learning_insights()
    
    # Create app
    app = create_app(monitor_manager=monitor_manager)
    dash_duo.start_server(app)
    
    # Wait for learning charts
    dash_duo.wait_for_element("#learning-curve .js-plotly-plot")
    dash_duo.wait_for_element("#strategy-evolution .js-plotly-plot")
    dash_duo.wait_for_element("#feature-importance .js-plotly-plot")
    
    # Verify charts
    assert dash_duo.find_element("#learning-curve")
    assert dash_duo.find_element("#strategy-evolution")
    assert dash_duo.find_element("#feature-importance")

@pytest.mark.visualization
def test_anomaly_detection_visualization(dash_duo, monitor_manager):
    """Test anomaly detection visualization"""
    # Generate test data with anomalies
    history = create_mock_trade_history(num_trades=100)
    # Inject anomalies
    anomaly_indices = np.random.choice(len(history), size=5, replace=False)
    for idx in anomaly_indices:
        history.iloc[idx, history.columns.get_loc('profit')] = -1000.0
    
    # Record trades
    for _, trade in history.iterrows():
        monitor_manager.record_trade(
            strategy=trade['strategy'],
            token_pair=trade['token_pair'],
            dex=trade['dex'],
            profit=float(trade['profit']),
            gas_price=float(trade['gas_price']),
            execution_time=float(trade['execution_time']),
            success=bool(trade['success'])
        )
    
    # Create app
    app = create_app(monitor_manager=monitor_manager)
    dash_duo.start_server(app)
    
    # Wait for anomaly chart
    dash_duo.wait_for_element("#anomaly-detection-chart .js-plotly-plot")
    
    # Verify chart and alerts
    assert dash_duo.find_element("#anomaly-detection-chart")
    alerts = dash_duo.find_element("#anomaly-alerts")
    assert alerts is not None

@pytest.mark.visualization
def test_optimization_suggestions_display(dash_duo, monitor_manager):
    """Test optimization suggestions display"""
    # Record poor performance trades to trigger suggestions
    for i in range(10):
        monitor_manager.record_trade(
            strategy="test_strategy",
            token_pair="ETH-USDC",
            dex="uniswap",
            profit=-1.0,
            gas_price=100.0,
            execution_time=2.0,
            success=False
        )
    
    # Create app
    app = create_app(monitor_manager=monitor_manager)
    dash_duo.start_server(app)
    
    # Wait for suggestions
    dash_duo.wait_for_element("#optimization-suggestions")
    
    # Verify suggestions are displayed
    suggestions = dash_duo.find_element("#optimization-suggestions")
    assert suggestions is not None
    assert len(suggestions.text) > 0

@pytest.mark.visualization
def test_real_time_updates(dash_duo, monitor_manager):
    """Test real-time chart updates"""
    # Create app
    app = create_app(monitor_manager=monitor_manager)
    dash_duo.start_server(app)
    
    # Get initial chart state
    dash_duo.wait_for_element("#profit-loss-chart .js-plotly-plot")
    initial_chart = dash_duo.find_element("#profit-loss-chart")
    initial_text = initial_chart.text
    
    # Record new trade
    monitor_manager.record_trade(
        strategy="test_strategy",
        token_pair="ETH-USDC",
        dex="uniswap",
        profit=100.0,
        gas_price=50.0,
        execution_time=0.1,
        success=True
    )
    
    # Wait for update and verify
    dash_duo.wait_for_element("#profit-loss-chart .js-plotly-plot")
    updated_chart = dash_duo.find_element("#profit-loss-chart")
    assert updated_chart.text != initial_text

@pytest.mark.visualization
def test_chart_interactions(dash_duo, monitor_manager):
    """Test chart interactions"""
    # Create app with test data
    history = create_mock_trade_history(num_trades=100)
    for _, trade in history.iterrows():
        monitor_manager.record_trade(
            strategy=trade['strategy'],
            token_pair=trade['token_pair'],
            dex=trade['dex'],
            profit=float(trade['profit']),
            gas_price=float(trade['gas_price']),
            execution_time=float(trade['execution_time']),
            success=bool(trade['success'])
        )
    
    app = create_app(monitor_manager=monitor_manager)
    dash_duo.start_server(app)
    
    # Wait for charts
    dash_duo.wait_for_element("#profit-loss-chart .js-plotly-plot")
    
    # Test zoom interaction
    chart = dash_duo.find_element("#profit-loss-chart")
    dash_duo.click_at_coord_fractions(chart, 0.2, 0.2)
    dash_duo.click_at_coord_fractions(chart, 0.8, 0.8)
    
    # Verify chart responded to interaction
    assert dash_duo.find_element("#profit-loss-chart .js-plotly-plot") 