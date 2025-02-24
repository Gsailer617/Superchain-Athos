"""Monitoring package providing comprehensive system and performance monitoring"""

from .monitor_manager import MonitorManager, MonitoringConfig
from .specialized.trade_monitor import TradeMonitor, TradeMetrics
from .specialized.system_monitor import SystemMonitor, ResourceType, ResourceThreshold
from .visualization.learning_insights import LearningInsightsVisualizer

__all__ = [
    # Main components
    'MonitorManager',
    'MonitoringConfig',
    
    # Specialized monitoring
    'TradeMonitor',
    'TradeMetrics',
    'SystemMonitor',
    'ResourceType',
    'ResourceThreshold',
    
    # Visualization
    'LearningInsightsVisualizer'
]

__version__ = '2.0.0' 