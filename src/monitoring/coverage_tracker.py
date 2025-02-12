import json
from datetime import datetime
from typing import Dict, List, Optional
import structlog
from prometheus_client import Gauge, Summary
import os

logger = structlog.get_logger(__name__)

class CoverageTracker:
    def __init__(self, history_file: str = "coverage_history.json"):
        self.history_file = history_file
        
        # Coverage metrics
        self.coverage_percentage = Gauge(
            'test_coverage_percent',
            'Test coverage percentage by component',
            ['component']
        )
        
        self.coverage_change = Gauge(
            'test_coverage_change',
            'Change in test coverage since last run',
            ['component']
        )
        
        self.uncovered_lines = Gauge(
            'uncovered_lines_total',
            'Number of uncovered lines by component',
            ['component']
        )
        
        self.test_execution_time = Summary(
            'test_execution_duration_seconds',
            'Time taken to execute test suite',
            ['component']
        )

    async def record_coverage(
        self,
        component: str,
        coverage_data: Dict[str, float],
        execution_time: float
    ):
        """Record coverage metrics for a component"""
        try:
            # Load historical data
            history = self._load_history()
            
            # Record current metrics
            current_coverage = coverage_data['percent_covered']
            self.coverage_percentage.labels(component=component).set(current_coverage)
            
            # Calculate and record change
            if component in history:
                last_coverage = history[component][-1]['coverage']
                change = current_coverage - last_coverage
                self.coverage_change.labels(component=component).set(change)
            
            # Record uncovered lines
            self.uncovered_lines.labels(component=component).set(
                coverage_data.get('uncovered_lines', 0)
            )
            
            # Record execution time
            self.test_execution_time.labels(component=component).observe(execution_time)
            
            # Update history
            if component not in history:
                history[component] = []
            
            history[component].append({
                'timestamp': datetime.now().isoformat(),
                'coverage': current_coverage,
                'execution_time': execution_time
            })
            
            # Trim history to keep last 30 days
            history[component] = history[component][-30:]
            
            # Save updated history
            self._save_history(history)
            
        except Exception as e:
            logger.error("Error recording coverage metrics",
                        component=component,
                        error=str(e))

    def get_coverage_trend(
        self,
        component: str,
        days: int = 7
    ) -> List[Dict]:
        """Get coverage trend for a component"""
        history = self._load_history()
        if component not in history:
            return []
            
        return history[component][-days:]

    def _load_history(self) -> Dict:
        """Load coverage history from file"""
        if not os.path.exists(self.history_file):
            return {}
            
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error("Error loading coverage history",
                        error=str(e))
            return {}

    def _save_history(self, history: Dict):
        """Save coverage history to file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(history, f)
        except Exception as e:
            logger.error("Error saving coverage history",
                        error=str(e)) 