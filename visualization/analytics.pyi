"""Type stubs for analytics module"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd

@dataclass
class AnalyticsConfig:
    profit_window: int
    gas_efficiency_threshold: float
    min_success_rate: float
    enable_advanced_metrics: bool

def calculate_metrics(
    performance_data: pd.DataFrame,
    config: Optional[AnalyticsConfig] = None
) -> Dict[str, float]: ... 