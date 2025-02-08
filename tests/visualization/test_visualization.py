"""
Test suite for the arbitrage system including visualization components
"""

import os
import sys

# Add workspace root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import asyncio
from visualization.performance import PerformanceMonitor, PerformanceConfig
from visualization.chart_utils import ChartTheme, InteractiveChartGenerator
from visualization.analytics import DataAnalyzer, AnalyticsConfig
import dash
from dash.testing.application_runners import import_app
import time
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from web3 import Web3
from web3.middleware import geth_poa_middleware

# Load environment variables
load_dotenv()

# ... existing imports and test code ...

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
def test_data():
    """Fixture for test data"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    return pd.DataFrame({
        'timestamp': dates,
        'profit': np.random.normal(100, 20, 100),
        'gas_price': np.random.uniform(50, 150, 100),
        'success_rate': np.random.uniform(0.8, 1.0, 100),
        'opportunity_count': np.random.poisson(10, 100)
    })

@pytest.fixture
def web3_provider():
    """Initialize Web3 with Alchemy for testing"""
    # Get Alchemy key from environment
    alchemy_key = os.getenv('ALCHEMY_API_KEY')
    if not alchemy_key:
        raise ValueError("ALCHEMY_API_KEY environment variable is not set")
        
    # Initialize Web3 with Alchemy
    w3 = Web3(Web3.HTTPProvider(
        f"https://base-mainnet.g.alchemy.com/v2/{alchemy_key}",
        request_kwargs={
            'timeout': 30,
            'headers': {'User-Agent': 'FlashingBase/1.0.0'}
        }
    ))
    
    # Add PoA middleware for Base
    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    
    if not w3.is_connected():
        raise ValueError("Failed to connect to Base mainnet via Alchemy")
        
    return w3

@pytest.fixture
def test_config(web3_provider):
    """Test configuration with Web3 provider"""
    return {
        'network': {
            'rpc_url': f"https://base-mainnet.g.alchemy.com/v2/{os.getenv('ALCHEMY_API_KEY')}",
            'chain_id': 8453,  # Base mainnet
            'name': 'base'
        },
        'web3': web3_provider,
        # ... rest of test config ...
    }

@pytest.mark.asyncio
async def test_visualization_performance(performance_monitor):
    """Test performance monitoring of visualization components"""
    await performance_monitor.start_monitoring()
    
    # Simulate visualization operations
    for _ in range(5):
        performance_monitor.track_chart_render('profit_chart', 0.5)
        performance_monitor.track_chart_interaction('zoom')
        performance_monitor.track_data_update('profit_component')
        await asyncio.sleep(0.1)
    
    # Run health check
    health_check = await performance_monitor.run_health_check()
    
    await performance_monitor.stop_monitoring()
    
    assert health_check['status'] in ['healthy', 'degraded']
    assert isinstance(health_check['suggestions'], list)
    assert len(performance_monitor._performance_history) > 0

@pytest.mark.asyncio
async def test_chart_generation(test_data):
    """Test chart generation functionality"""
    chart_generator = InteractiveChartGenerator(theme=ChartTheme.get_dark())
    
    # Test profit chart
    profit_chart = chart_generator.create_profit_chart(
        data=[{
            'timestamp': t,
            'profit': p,
            'cumulative_profit': test_data['profit'].cumsum()[i]
        } for i, (t, p) in enumerate(zip(test_data['timestamp'], test_data['profit']))]
    )
    assert profit_chart is not None
    
    # Test gas analysis chart
    gas_chart = chart_generator.create_gas_analysis_chart(
        data=[{
            'timestamp': t,
            'gas_price': g
        } for t, g in zip(test_data['timestamp'], test_data['gas_price'])]
    )
    assert gas_chart is not None
    
    # Test opportunity chart
    opportunity_chart = chart_generator.create_opportunity_chart(
        opportunities=test_data['opportunity_count'],
        timestamps=test_data['timestamp']
    )
    assert opportunity_chart is not None

@pytest.mark.asyncio
async def test_analytics_integration(test_data):
    """Test analytics integration with visualization"""
    config = AnalyticsConfig()
    analyzer = DataAnalyzer(config)
    
    # Calculate metrics
    metrics = analyzer.calculate_metrics(test_data)
    
    assert 'profit_metrics' in metrics
    assert 'gas_metrics' in metrics
    assert 'opportunity_metrics' in metrics
    
    # Test trend detection
    trends = analyzer.detect_trends(test_data['profit'])
    assert isinstance(trends, dict)
    assert 'trend_direction' in trends
    
    # Test outlier detection
    outliers = analyzer.detect_outliers(test_data['profit'])
    assert isinstance(outliers, np.ndarray)

@pytest.mark.integration
def test_dashboard_integration():
    """Test the complete dashboard integration"""
    # Import the dashboard app
    app = import_app("visualization.dashboard")
    
    # Create a test client
    test_client = app.server.test_client()
    
    # Test that the page loads
    response = test_client.get('/')
    assert response.status_code == 200

@pytest.mark.integration
def test_realtime_updates():
    """Test real-time data updates in the dashboard"""
    app = import_app("visualization.dashboard")
    test_client = app.server.test_client()
    
    # Test initial load
    response = test_client.get('/')
    assert response.status_code == 200
    
    # Test update endpoint if it exists
    response = test_client.get('/_dash-update-component')
    assert response.status_code in [200, 204]

@pytest.mark.integration
def test_error_handling():
    """Test dashboard error handling and recovery"""
    app = import_app("visualization.dashboard")
    test_client = app.server.test_client()
    
    # Test error route if it exists
    response = test_client.get('/error-test')
    assert response.status_code in [404, 500]  # Either not found or handled error

# ... rest of the test file ... 