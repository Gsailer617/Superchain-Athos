"""Type stubs for chart utilities"""

from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
import pandas as pd
from dataclasses import dataclass

@dataclass
class ChartTheme:
    background_color: str
    plot_background_color: str
    font_family: str
    primary_color: str
    secondary_color: str
    accent_color: str
    success_color: str
    warning_color: str
    error_color: str
    grid_color: str

class InteractiveChartGenerator:
    def __init__(self, theme: Optional[ChartTheme] = None) -> None: ...
    
    def create_time_series(
        self,
        data: pd.DataFrame,
        x_column: str,
        y_columns: List[str],
        title: str,
        y_axis_titles: Optional[List[str]] = None
    ) -> go.Figure: ...
    
    def create_scatter_plot(
        self,
        data: pd.DataFrame,
        x_column: str,
        y_column: str,
        color_column: Optional[str] = None,
        size_column: Optional[str] = None,
        title: str = ''
    ) -> go.Figure: ...
    
    def create_bar_chart(
        self,
        data: pd.DataFrame,
        x_column: str,
        y_column: str,
        color_column: Optional[str] = None,
        title: str = ''
    ) -> go.Figure: ...
    
    def create_pie_chart(
        self,
        data: pd.DataFrame,
        names_column: str,
        values_column: str,
        title: str = ''
    ) -> go.Figure: ...
    
    def create_heatmap(
        self,
        data: pd.DataFrame,
        x_column: str,
        y_column: str,
        values_column: str,
        title: str = ''
    ) -> go.Figure: ...
    
    def _get_color(self, index: int) -> str: ... 