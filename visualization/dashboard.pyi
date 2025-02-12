"""Type stubs for dashboard module"""

from typing import Optional, Dict, Any, List
import pandas as pd
import dash
from src.history.trade_history import TradeHistoryManager
from src.monitoring.monitor_manager import MonitorManager
import plotly.graph_objects as go

def create_app(
    performance_data: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None,
    history_manager: Optional[TradeHistoryManager] = None,
    monitor_manager: Optional[MonitorManager] = None
) -> dash.Dash: ...

def update_chart(n: int) -> go.Figure: ...

def update_metrics(n: int) -> List[Any]: ...

def update_learning_curve(n: int) -> go.Figure: ...

def update_strategy_evolution(n: int) -> go.Figure: ...

def update_feature_importance(n: int) -> go.Figure: ...

def update_performance_prediction(n: int) -> go.Figure: ...

async def update_anomaly_detection(n: int) -> List[Any]: ...

async def update_optimization_suggestions(n: int) -> List[Any]: ...

def init_dashboard(
    performance_data: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None,
    history_manager: Optional[TradeHistoryManager] = None,
    monitor_manager: Optional[MonitorManager] = None,
    debug: bool = False
) -> dash.Dash: ... 