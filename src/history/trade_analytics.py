"""
Trade analytics module for analyzing trade history and gas optimization performance.
Provides advanced analytics, visualization, and integration with gas and execution modules.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os
import structlog
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .trade_history import EnhancedTradeHistoryManager
from .enhanced_trade_metrics import EnhancedTradeMetrics, GasMetrics, ExecutionMetrics, TokenMetrics

logger = structlog.get_logger(__name__)

class TradeAnalytics:
    """Advanced analytics for trade history with gas and execution integration"""
    
    def __init__(
        self,
        trade_history_manager: Optional[EnhancedTradeHistoryManager] = None,
        storage_path: str = "data/trade_history",
        reports_path: str = "data/reports",
        thread_pool_size: int = 4
    ):
        """Initialize trade analytics
        
        Args:
            trade_history_manager: Existing trade history manager (creates new one if None)
            storage_path: Path to trade history storage (if creating new manager)
            reports_path: Path to store generated reports
            thread_pool_size: Size of thread pool for parallel operations
        """
        # Use provided manager or create new one
        self.trade_history = trade_history_manager or EnhancedTradeHistoryManager(
            storage_path=storage_path
        )
        
        # Reports path
        self.reports_path = Path(reports_path)
        self.reports_path.mkdir(parents=True, exist_ok=True)
        
        # Thread pool for parallel operations
        self.thread_pool = ThreadPoolExecutor(max_workers=thread_pool_size)
    
    def generate_performance_report(
        self,
        timeframe: str = '24h',
        include_gas_metrics: bool = True,
        include_charts: bool = True,
        save_report: bool = True,
        report_format: str = 'json'
    ) -> Dict[str, Any]:
        """Generate comprehensive performance report
        
        Args:
            timeframe: Timeframe for analysis (e.g., '24h', '7d', '30d')
            include_gas_metrics: Whether to include detailed gas metrics
            include_charts: Whether to generate and include chart data
            save_report: Whether to save report to disk
            report_format: Format to save report ('json' or 'csv')
            
        Returns:
            Dictionary with performance report
        """
        # Get performance metrics
        performance = self.trade_history.analyze_performance(
            timeframe=timeframe,
            include_gas_metrics=include_gas_metrics,
            include_charts=include_charts
        )
        
        # Get gas performance metrics
        gas_performance = self.trade_history.analyze_gas_performance(
            timeframe=timeframe
        )
        
        # Combine into report
        report = {
            'generated_at': datetime.now(),
            'timeframe': timeframe,
            'performance': performance,
            'gas_performance': gas_performance,
            'summary': self._generate_summary(performance, gas_performance)
        }
        
        # Save report if requested
        if save_report:
            self._save_report(report, report_format)
        
        return report
    
    def _generate_summary(
        self,
        performance: Dict[str, Any],
        gas_performance: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate summary of performance metrics
        
        Args:
            performance: Performance metrics
            gas_performance: Gas performance metrics
            
        Returns:
            Dictionary with summary metrics
        """
        summary = {}
        
        # Extract key metrics
        if performance:
            summary.update({
                'total_trades': performance.get('total_trades', 0),
                'successful_trades': performance.get('successful_trades', 0),
                'success_rate': performance.get('success_rate', 0),
                'total_profit': performance.get('total_profit', 0),
                'average_profit': performance.get('average_profit', 0),
            })
        
        # Extract gas metrics
        if gas_performance:
            summary.update({
                'total_gas_cost_usd': gas_performance.get('total_gas_cost_usd', 0),
                'average_gas_cost_usd': gas_performance.get('average_gas_cost_usd', 0),
                'total_estimated_savings_usd': gas_performance.get('total_estimated_savings_usd', 0),
            })
            
            # Calculate net profit (after gas costs)
            if 'total_profit' in summary and 'total_gas_cost_usd' in summary:
                summary['net_profit'] = summary['total_profit'] - summary['total_gas_cost_usd']
        
        return summary
    
    def _save_report(self, report: Dict[str, Any], format: str = 'json') -> str:
        """Save report to disk
        
        Args:
            report: Report data
            format: Format to save report ('json' or 'csv')
            
        Returns:
            Path to saved report
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'json':
            # Convert datetime objects to strings
            report_copy = self._prepare_report_for_json(report)
            
            # Save as JSON
            filepath = self.reports_path / f"report_{timestamp}.json"
            with open(filepath, 'w') as f:
                json.dump(report_copy, f, indent=2)
                
        elif format == 'csv':
            # Flatten report into DataFrame
            df = self._flatten_report_to_dataframe(report)
            
            # Save as CSV
            filepath = self.reports_path / f"report_{timestamp}.csv"
            df.to_csv(filepath, index=False)
        
        logger.info(
            "Saved performance report",
            filepath=str(filepath),
            format=format
        )
        
        return str(filepath)
    
    def _prepare_report_for_json(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare report for JSON serialization by converting datetime objects to strings
        
        Args:
            report: Report data
            
        Returns:
            Report with serializable values
        """
        def convert_value(value):
            if isinstance(value, datetime):
                return value.isoformat()
            elif isinstance(value, (np.int64, np.float64)):
                return float(value)
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [convert_value(item) for item in value]
            return value
        
        return {k: convert_value(v) for k, v in report.items()}
    
    def _flatten_report_to_dataframe(self, report: Dict[str, Any]) -> pd.DataFrame:
        """Flatten report into DataFrame for CSV export
        
        Args:
            report: Report data
            
        Returns:
            Flattened DataFrame
        """
        # Extract summary metrics
        flat_data = {}
        
        if 'summary' in report:
            for key, value in report['summary'].items():
                flat_data[key] = [value]
        
        # Extract performance metrics
        if 'performance' in report:
            for key, value in report['performance'].items():
                if not isinstance(value, dict):
                    flat_data[f"performance_{key}"] = [value]
        
        # Extract gas performance metrics
        if 'gas_performance' in report:
            for key, value in report['gas_performance'].items():
                if not isinstance(value, dict):
                    flat_data[f"gas_{key}"] = [value]
        
        return pd.DataFrame(flat_data)
    
    def visualize_performance(
        self,
        timeframe: str = '7d',
        metrics: List[str] = ['profit', 'gas_cost_usd', 'success_rate'],
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> Dict[str, Any]:
        """Visualize trade performance metrics
        
        Args:
            timeframe: Timeframe for visualization (e.g., '24h', '7d', '30d')
            metrics: List of metrics to visualize
            save_path: Path to save visualization (None to not save)
            show_plot: Whether to display the plot
            
        Returns:
            Dictionary with visualization data
        """
        # Calculate start time based on timeframe
        start_time = datetime.now() - pd.Timedelta(timeframe)
        
        # Get historical data
        df = self.trade_history.get_history(start_time=start_time)
        
        if df.empty:
            return {}
        
        # Set up plot
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)), sharex=True)
        if len(metrics) == 1:
            axes = [axes]
        
        # Set timestamp as index
        df_time = df.copy()
        df_time['timestamp'] = pd.to_datetime(df_time['timestamp'])
        df_time = df_time.set_index('timestamp')
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            if metric in df_time.columns:
                # Resample data
                if metric in ['success_rate']:
                    # For rate metrics, calculate mean
                    if metric == 'success_rate':
                        resampled = df_time['success'].resample('1H').mean() * 100
                    else:
                        resampled = df_time[metric].resample('1H').mean()
                else:
                    # For cumulative metrics, calculate sum
                    resampled = df_time[metric].resample('1H').sum()
                
                # Plot
                resampled.plot(ax=axes[i], marker='o', linestyle='-')
                axes[i].set_title(f"{metric.replace('_', ' ').title()} over time")
                axes[i].grid(True)
                
                # Add rolling average
                if len(resampled) > 5:
                    rolling_avg = resampled.rolling(window=5).mean()
                    rolling_avg.plot(ax=axes[i], color='red', linestyle='--', 
                                    label='5-period moving average')
                    axes[i].legend()
        
        # Format plot
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
            logger.info("Saved performance visualization", filepath=save_path)
        
        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # Return visualization data
        viz_data = {}
        for metric in metrics:
            if metric in df_time.columns:
                if metric == 'success_rate':
                    resampled = df_time['success'].resample('1H').mean() * 100
                else:
                    resampled = df_time[metric].resample('1H').sum()
                
                viz_data[metric] = {
                    'timestamps': resampled.index.tolist(),
                    'values': resampled.tolist()
                }
        
        return viz_data
    
    def analyze_gas_optimization_impact(
        self,
        timeframe: str = '30d'
    ) -> Dict[str, Any]:
        """Analyze impact of gas optimization on profitability
        
        Args:
            timeframe: Timeframe for analysis (e.g., '24h', '7d', '30d')
            
        Returns:
            Dictionary with gas optimization impact analysis
        """
        # Calculate start time based on timeframe
        start_time = datetime.now() - pd.Timedelta(timeframe)
        
        # Get historical data
        df = self.trade_history.get_history(start_time=start_time)
        
        if df.empty or 'optimization_mode' not in df.columns:
            return {}
        
        # Group by optimization mode
        mode_analysis = df.groupby('optimization_mode').agg({
            'profit': ['mean', 'sum', 'count'],
            'gas_cost_usd': ['mean', 'sum'],
            'execution_time': 'mean',
            'confirmation_time': 'mean' if 'confirmation_time' in df.columns else None,
            'success': 'mean',
        }).reset_index()
        
        # Calculate net profit (profit - gas cost)
        if 'gas_cost_usd' in df.columns:
            df['net_profit'] = df['profit'] - df['gas_cost_usd']
            
            mode_net_profit = df.groupby('optimization_mode').agg({
                'net_profit': ['mean', 'sum']
            }).reset_index()
            
            # Merge with mode analysis
            mode_analysis = pd.merge(
                mode_analysis, 
                mode_net_profit, 
                on='optimization_mode'
            )
        
        # Calculate profit per gas cost ratio
        if 'gas_cost_usd' in df.columns and df['gas_cost_usd'].sum() > 0:
            df['profit_per_gas'] = df['profit'] / df['gas_cost_usd']
            
            mode_efficiency = df.groupby('optimization_mode').agg({
                'profit_per_gas': ['mean', 'median']
            }).reset_index()
            
            # Merge with mode analysis
            mode_analysis = pd.merge(
                mode_analysis, 
                mode_efficiency, 
                on='optimization_mode'
            )
        
        # Calculate overall impact
        impact = {
            'optimization_mode_analysis': mode_analysis.to_dict(),
            'timeframe': timeframe,
            'start_time': start_time,
            'end_time': datetime.now(),
        }
        
        # Calculate estimated savings
        if 'optimization_savings' in df.columns and 'gas_cost_usd' in df.columns:
            total_gas_cost = df['gas_cost_usd'].sum()
            avg_savings_pct = df['optimization_savings'].mean()
            estimated_savings = total_gas_cost * avg_savings_pct / 100
            
            impact['estimated_savings_usd'] = estimated_savings
            impact['average_savings_percentage'] = avg_savings_pct
        
        return impact
    
    def compare_strategies(
        self,
        timeframe: str = '30d',
        metrics: List[str] = ['profit', 'success_rate', 'gas_cost_usd']
    ) -> Dict[str, Any]:
        """Compare performance of different strategies
        
        Args:
            timeframe: Timeframe for comparison (e.g., '24h', '7d', '30d')
            metrics: List of metrics to compare
            
        Returns:
            Dictionary with strategy comparison
        """
        # Calculate start time based on timeframe
        start_time = datetime.now() - pd.Timedelta(timeframe)
        
        # Get historical data
        df = self.trade_history.get_history(start_time=start_time)
        
        if df.empty or 'strategy' not in df.columns:
            return {}
        
        # Prepare metrics for aggregation
        agg_dict = {}
        for metric in metrics:
            if metric == 'success_rate' and 'success' in df.columns:
                agg_dict['success'] = 'mean'
            elif metric in df.columns:
                agg_dict[metric] = ['mean', 'sum', 'count']
        
        # Group by strategy
        strategy_analysis = df.groupby('strategy').agg(agg_dict).reset_index()
        
        # Calculate net profit if possible
        if 'profit' in df.columns and 'gas_cost_usd' in df.columns:
            df['net_profit'] = df['profit'] - df['gas_cost_usd']
            
            strategy_net_profit = df.groupby('strategy').agg({
                'net_profit': ['mean', 'sum']
            }).reset_index()
            
            # Merge with strategy analysis
            strategy_analysis = pd.merge(
                strategy_analysis, 
                strategy_net_profit, 
                on='strategy'
            )
        
        # Format success rate as percentage
        if 'success' in strategy_analysis.columns:
            strategy_analysis['success_rate'] = strategy_analysis['success'] * 100
        
        return {
            'strategy_comparison': strategy_analysis.to_dict(),
            'timeframe': timeframe,
            'start_time': start_time,
            'end_time': datetime.now(),
            'total_strategies': strategy_analysis['strategy'].nunique()
        }
    
    async def generate_async_report(
        self,
        timeframe: str = '7d',
        include_gas_metrics: bool = True,
        include_charts: bool = True,
        save_report: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive performance report asynchronously
        
        Args:
            timeframe: Timeframe for analysis (e.g., '24h', '7d', '30d')
            include_gas_metrics: Whether to include detailed gas metrics
            include_charts: Whether to generate and include chart data
            save_report: Whether to save report to disk
            
        Returns:
            Dictionary with performance report
        """
        # Use ThreadPoolExecutor to avoid blocking
        loop = asyncio.get_event_loop()
        report = await loop.run_in_executor(
            self.thread_pool,
            lambda: self.generate_performance_report(
                timeframe=timeframe,
                include_gas_metrics=include_gas_metrics,
                include_charts=include_charts,
                save_report=save_report
            )
        )
        
        return report
    
    def close(self) -> None:
        """Close analytics and clean up resources"""
        try:
            # Close trade history manager
            self.trade_history.close()
            
            # Shutdown thread pool
            self.thread_pool.shutdown()
            
            logger.info("Trade analytics closed")
            
        except Exception as e:
            logger.error("Error closing trade analytics", error=str(e))


class TradeGasExecutionIntegrator:
    """Integrates trade history with gas optimization and execution modules"""
    
    def __init__(
        self,
        trade_history_manager: Optional[EnhancedTradeHistoryManager] = None,
        storage_path: str = "data/trade_history"
    ):
        """Initialize integrator
        
        Args:
            trade_history_manager: Existing trade history manager (creates new one if None)
            storage_path: Path to trade history storage (if creating new manager)
        """
        # Use provided manager or create new one
        self.trade_history = trade_history_manager or EnhancedTradeHistoryManager(
            storage_path=storage_path,
            enable_async=True
        )
    
    async def record_execution_result(
        self,
        execution_result: Dict[str, Any],
        strategy: str = ''
    ) -> EnhancedTradeMetrics:
        """Record execution result to trade history
        
        Args:
            execution_result: Result from transaction execution
            strategy: Strategy name
            
        Returns:
            Recorded trade metrics
        """
        # Convert execution result to EnhancedTradeMetrics
        metrics = EnhancedTradeMetrics.from_execution_result(
            execution_result,
            strategy=strategy
        )
        
        # Record to trade history
        await self.trade_history.record_trade_async(metrics)
        
        return metrics
    
    async def analyze_gas_strategy_performance(
        self,
        timeframe: str = '7d'
    ) -> Dict[str, Any]:
        """Analyze performance of different gas optimization strategies
        
        Args:
            timeframe: Timeframe for analysis (e.g., '24h', '7d', '30d')
            
        Returns:
            Dictionary with gas strategy performance analysis
        """
        # Get historical data
        df = self.trade_history.get_history(start_time=datetime.now() - pd.Timedelta(timeframe))
        
        if df.empty or 'optimization_mode' not in df.columns:
            return {}
        
        # Calculate net profit
        if 'profit' in df.columns and 'gas_cost_usd' in df.columns:
            df['net_profit'] = df['profit'] - df['gas_cost_usd']
        
        # Group by optimization mode and strategy
        if 'strategy' in df.columns:
            mode_strategy_analysis = df.groupby(['optimization_mode', 'strategy']).agg({
                'profit': ['mean', 'sum'],
                'gas_cost_usd': ['mean', 'sum'] if 'gas_cost_usd' in df.columns else None,
                'net_profit': ['mean', 'sum'] if 'net_profit' in df.columns else None,
                'success': 'mean',
                'execution_time': 'mean',
            }).reset_index()
            
            return {
                'mode_strategy_performance': mode_strategy_analysis.to_dict(),
                'timeframe': timeframe
            }
        
        return {}
    
    def recommend_gas_strategy(
        self,
        strategy: str,
        token_pair: Optional[str] = None,
        timeframe: str = '7d'
    ) -> Dict[str, Any]:
        """Recommend optimal gas strategy based on historical performance
        
        Args:
            strategy: Trading strategy
            token_pair: Optional token pair filter
            timeframe: Timeframe for analysis (e.g., '24h', '7d', '30d')
            
        Returns:
            Dictionary with recommended gas strategy
        """
        # Get historical data
        start_time = datetime.now() - pd.Timedelta(timeframe)
        
        filters = {'start_time': start_time, 'strategy': strategy}
        if token_pair:
            filters['token_pair'] = token_pair
            
        df = self.trade_history.get_history(**filters)
        
        if df.empty or 'optimization_mode' not in df.columns:
            return {
                'recommended_mode': 'normal',  # Default recommendation
                'confidence': 0.0,
                'reason': 'Insufficient historical data'
            }
        
        # Calculate net profit
        if 'profit' in df.columns and 'gas_cost_usd' in df.columns:
            df['net_profit'] = df['profit'] - df['gas_cost_usd']
        else:
            df['net_profit'] = df['profit']
        
        # Group by optimization mode
        mode_performance = df.groupby('optimization_mode').agg({
            'net_profit': ['mean', 'sum'],
            'success': 'mean',
            'execution_time': 'mean',
        }).reset_index()
        
        # Find mode with highest average net profit
        if len(mode_performance) > 0 and 'net_profit' in mode_performance.columns:
            # Sort by average net profit
            mode_performance = mode_performance.sort_values(('net_profit', 'mean'), ascending=False)
            
            best_mode = mode_performance.iloc[0]['optimization_mode']
            best_profit = mode_performance.iloc[0][('net_profit', 'mean')]
            
            # Calculate confidence based on sample size
            mode_counts = df['optimization_mode'].value_counts()
            total_samples = len(df)
            
            if best_mode in mode_counts:
                samples = mode_counts[best_mode]
                confidence = min(samples / 10, 1.0)  # Scale confidence by sample size, max 1.0
            else:
                confidence = 0.0
            
            return {
                'recommended_mode': best_mode,
                'confidence': confidence,
                'average_profit': best_profit,
                'sample_size': mode_counts.get(best_mode, 0),
                'total_samples': total_samples,
                'all_modes': mode_performance.to_dict()
            }
        
        return {
            'recommended_mode': 'normal',  # Default recommendation
            'confidence': 0.0,
            'reason': 'Insufficient performance data'
        }
    
    def close(self) -> None:
        """Close integrator and clean up resources"""
        self.trade_history.close() 