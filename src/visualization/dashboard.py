"""
Visualization/Dashboard Module

This module provides visualization and dashboard capabilities:
- Real-time metrics visualization
- Performance dashboards
- System health monitoring
- Alert visualization
- Trend analysis
- Custom reporting
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import structlog
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
import json
from ..utils.metrics import metrics_manager, MetricType
from ..monitoring.performance_monitor import (
    performance_monitor, ResourceType, AlertSeverity
)

logger = structlog.get_logger(__name__)

@dataclass
class ChartConfig:
    """Configuration for a chart"""
    title: str
    metric_name: str
    chart_type: str
    timeframe: str = "1h"
    refresh_interval: int = 10
    height: int = 400
    width: int = 800

class Dashboard:
    """Real-time dashboard system"""
    
    def __init__(self, 
                 title: str = "System Dashboard",
                 refresh_interval: int = 10):
        self.title = title
        self.refresh_interval = refresh_interval
        self.app = FastAPI(title=title)
        self._setup_routes()
        self._active_websockets: List[WebSocket] = []
        self._chart_configs: Dict[str, ChartConfig] = {}

    def _setup_routes(self):
        """Setup FastAPI routes"""
        # Serve static files
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
        
        # Main dashboard
        @self.app.get("/", response_class=HTMLResponse)
        async def get_dashboard():
            return self._generate_dashboard_html()
        
        # Metrics endpoint
        @self.app.get("/metrics")
        async def get_metrics():
            return Response(
                generate_latest(metrics_manager.registry),
                media_type=CONTENT_TYPE_LATEST
            )
        
        # WebSocket for real-time updates
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self._active_websockets.append(websocket)
            try:
                while True:
                    data = await websocket.receive_text()
                    # Handle any client messages if needed
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}")
            finally:
                self._active_websockets.remove(websocket)
        
        # Chart data endpoints
        @self.app.get("/chart/{chart_id}")
        async def get_chart_data(chart_id: str):
            if chart_id not in self._chart_configs:
                raise HTTPException(status_code=404, detail="Chart not found")
            return await self._get_chart_data(chart_id)

    def _generate_dashboard_html(self) -> str:
        """Generate the main dashboard HTML"""
        charts_html = ""
        for chart_id, config in self._chart_configs.items():
            charts_html += f"""
            <div class="chart-container">
                <div id="{chart_id}" style="height: {config.height}px; width: {config.width}px;"></div>
            </div>
            """
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.title}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                }}
                .chart-container {{
                    margin: 20px;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
            </style>
        </head>
        <body>
            <h1>{self.title}</h1>
            <div id="alerts"></div>
            {charts_html}
            <script>
                const ws = new WebSocket(`ws://${{window.location.host}}/ws`);
                ws.onmessage = function(event) {{
                    const data = JSON.parse(event.data);
                    if (data.type === 'chart_update') {{
                        Plotly.update(data.chart_id, data.data, data.layout);
                    }} else if (data.type === 'alert') {{
                        updateAlerts(data.alerts);
                    }}
                }};
                
                function updateAlerts(alerts) {{
                    const alertsDiv = document.getElementById('alerts');
                    alertsDiv.innerHTML = alerts.map(alert => `
                        <div class="alert alert-${{alert.severity}}">
                            ${{alert.message}}
                        </div>
                    `).join('');
                }}
                
                // Initial chart loading
                {self._generate_chart_js()}
            </script>
        </body>
        </html>
        """

    def _generate_chart_js(self) -> str:
        """Generate JavaScript for initial chart loading"""
        js = ""
        for chart_id, config in self._chart_configs.items():
            js += f"""
            fetch('/chart/{chart_id}')
                .then(response => response.json())
                .then(data => {{
                    Plotly.newPlot('{chart_id}', data.data, data.layout);
                }});
            """
        return js

    async def _get_chart_data(self, chart_id: str) -> Dict[str, Any]:
        """Get data for a specific chart"""
        config = self._chart_configs[chart_id]
        
        # Get metric data from Prometheus
        metric = metrics_manager.get_metric(config.metric_name)
        if not metric:
            raise HTTPException(
                status_code=404,
                detail=f"Metric {config.metric_name} not found"
            )
        
        # Convert to pandas DataFrame for easier plotting
        samples = []
        for sample in metric.collect()[0].samples:
            samples.append({
                'timestamp': datetime.now(),
                'value': sample.value,
                **sample.labels
            })
        
        df = pd.DataFrame(samples)
        
        # Create appropriate plot based on chart type
        if config.chart_type == 'line':
            fig = px.line(
                df,
                x='timestamp',
                y='value',
                title=config.title
            )
        elif config.chart_type == 'bar':
            fig = px.bar(
                df,
                x='timestamp',
                y='value',
                title=config.title
            )
        elif config.chart_type == 'gauge':
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=df['value'].iloc[-1] if len(df) > 0 else 0,
                title={'text': config.title},
                gauge={'axis': {'range': [0, 100]}}
            ))
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported chart type: {config.chart_type}"
            )
        
        return {
            'data': fig.data,
            'layout': fig.layout
        }

    def add_chart(self, chart_id: str, config: ChartConfig):
        """Add a new chart to the dashboard"""
        self._chart_configs[chart_id] = config

    async def _broadcast_updates(self):
        """Broadcast updates to all connected clients"""
        while True:
            try:
                # Update chart data
                updates = []
                for chart_id, config in self._chart_configs.items():
                    chart_data = await self._get_chart_data(chart_id)
                    updates.append({
                        'type': 'chart_update',
                        'chart_id': chart_id,
                        'data': chart_data['data'],
                        'layout': chart_data['layout']
                    })
                
                # Get active alerts
                alerts = []
                for resource_type in ResourceType:
                    value = performance_monitor._resource_usage.labels(
                        resource_type=resource_type.value
                    )._value.get()
                    
                    if value > 90:
                        alerts.append({
                            'severity': AlertSeverity.CRITICAL.value,
                            'message': f"{resource_type.value} usage at {value:.1f}%"
                        })
                    elif value > 75:
                        alerts.append({
                            'severity': AlertSeverity.WARNING.value,
                            'message': f"{resource_type.value} usage at {value:.1f}%"
                        })
                
                # Send updates to all connected clients
                for websocket in self._active_websockets:
                    for update in updates:
                        await websocket.send_text(json.dumps(update))
                    await websocket.send_text(json.dumps({
                        'type': 'alert',
                        'alerts': alerts
                    }))
            
            except Exception as e:
                logger.error(f"Error broadcasting updates: {str(e)}")
            
            await asyncio.sleep(self.refresh_interval)

    async def start(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the dashboard server"""
        import uvicorn
        
        # Start update broadcast loop
        asyncio.create_task(self._broadcast_updates())
        
        # Start FastAPI server
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()

# Global dashboard instance
dashboard = Dashboard()

class ArbitrageVisualizer:
    """Visualization dashboard for arbitrage operations"""
    
    def __init__(
        self,
        port: int = 8050,
        debug: bool = False,
        config: Optional[Dict] = None
    ):
        self.port = port
        self.debug = debug
        self.config = config or {}
        self.metrics_manager = MetricsManager()
        self.monitoring = MonitoringManager(config or {})
        
        # Initialize components
        self.chart_generator = InteractiveChartGenerator(
            theme=ChartTheme.get_light() if self.chart_config.theme == 'light' else ChartTheme.get_dark()
        )
        
        # Setup monitoring callbacks
        self._setup_monitoring_callbacks()
    
    def _setup_monitoring_callbacks(self):
        """Setup callbacks for monitoring data"""
        @self.app.callback(
            Output('ml-metrics-container', 'children'),
            Input('fast-interval', 'n_intervals')
        )
        async def update_ml_metrics(n):
            metrics = await self.monitoring.get_ml_metrics()
            return self._create_ml_metrics_cards(metrics)
        
        @self.app.callback(
            Output('llm-insights-container', 'children'),
            Input('medium-interval', 'n_intervals')
        )
        async def update_llm_insights(n):
            insights = await self.monitoring.get_llm_insights()
            return self._create_llm_insights_cards(insights)
        
        @self.app.callback(
            Output('cross-chain-container', 'children'),
            Input('medium-interval', 'n_intervals')
        )
        async def update_cross_chain_metrics(n):
            metrics = await self.monitoring.get_cross_chain_metrics()
            return self._create_cross_chain_cards(metrics)
    
    def _create_ml_metrics_cards(self, metrics: Dict[str, Any]) -> List[Component]:
        """Create ML metrics visualization components"""
        cards = []
        
        # Model performance card
        cards.append(
            dbc.Card([
                dbc.CardHeader("Model Performance"),
                dbc.CardBody([
                    dcc.Graph(
                        figure=self.chart_generator.create_performance_chart(
                            metrics['model_performance']
                        )
                    )
                ])
            ])
        )
        
        # Training metrics card
        cards.append(
            dbc.Card([
                dbc.CardHeader("Training Metrics"),
                dbc.CardBody([
                    dcc.Graph(
                        figure=self.chart_generator.create_training_chart(
                            metrics['training_metrics']
                        )
                    )
                ])
            ])
        )
        
        return cards
    
    def _create_llm_insights_cards(self, insights: Dict[str, Any]) -> List[Component]:
        """Create LLM insights visualization components"""
        cards = []
        
        # Sentiment analysis card
        cards.append(
            dbc.Card([
                dbc.CardHeader("Market Sentiment"),
                dbc.CardBody([
                    dcc.Graph(
                        figure=self.chart_generator.create_sentiment_chart(
                            insights['sentiment_data']
                        )
                    )
                ])
            ])
        )
        
        # Strategy recommendations card
        cards.append(
            dbc.Card([
                dbc.CardHeader("Strategy Insights"),
                dbc.CardBody([
                    dcc.Graph(
                        figure=self.chart_generator.create_strategy_chart(
                            insights['strategy_data']
                        )
                    )
                ])
            ])
        )
        
        return cards
    
    def _create_cross_chain_cards(self, metrics: Dict[str, Any]) -> List[Component]:
        """Create cross-chain metrics visualization components"""
        cards = []
        
        # Bridge analytics card
        cards.append(
            dbc.Card([
                dbc.CardHeader("Bridge Analytics"),
                dbc.CardBody([
                    dcc.Graph(
                        figure=self.chart_generator.create_bridge_chart(
                            metrics['bridge_data']
                        )
                    )
                ])
            ])
        )
        
        # Chain performance card
        cards.append(
            dbc.Card([
                dbc.CardHeader("Chain Performance"),
                dbc.CardBody([
                    dcc.Graph(
                        figure=self.chart_generator.create_chain_performance_chart(
                            metrics['chain_performance']
                        )
                    )
                ])
            ])
        )
        
        return cards 