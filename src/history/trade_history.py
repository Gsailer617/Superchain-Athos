from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import structlog
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import asyncio
import aiofiles
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from functools import lru_cache
import time

from .enhanced_trade_metrics import EnhancedTradeMetrics, GasMetrics, ExecutionMetrics, TokenMetrics

logger = structlog.get_logger(__name__)

@dataclass
class TradeMetrics:
    """Metrics for a single trade"""
    timestamp: datetime
    strategy: str
    token_pair: str
    dex: str
    profit: float
    gas_price: float
    execution_time: float
    success: bool
    additional_data: Dict[str, Any]

    @classmethod
    def from_enhanced_metrics(cls, enhanced: EnhancedTradeMetrics) -> 'TradeMetrics':
        """Convert from enhanced metrics to legacy format for backward compatibility"""
        return cls(
            timestamp=enhanced.timestamp,
            strategy=enhanced.strategy,
            token_pair=enhanced.token_pair,
            dex=enhanced.dex,
            profit=enhanced.profit,
            gas_price=enhanced.gas.gas_price,
            execution_time=enhanced.execution.execution_time,
            success=enhanced.success,
            additional_data=enhanced.additional_data
        )

class EnhancedTradeHistoryManager:
    """Enhanced manager for historical trade data with async support and advanced analytics"""
    
    def __init__(
        self,
        storage_path: str = "data/trade_history",
        max_memory_entries: int = 10000,
        flush_interval: int = 100,
        enable_async: bool = True,
        backup_enabled: bool = True,
        backup_interval: int = 1000,
        backup_path: Optional[str] = None,
        thread_pool_size: int = 4
    ):
        """Initialize enhanced trade history manager
        
        Args:
            storage_path: Path to store trade history
            max_memory_entries: Maximum number of entries to keep in memory
            flush_interval: Number of trades before auto-flushing to disk
            enable_async: Whether to enable async operations
            backup_enabled: Whether to enable automatic backups
            backup_interval: Number of trades before creating a backup
            backup_path: Path for backups (defaults to storage_path/backups)
            thread_pool_size: Size of thread pool for parallel operations
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_entries = max_memory_entries
        self.flush_interval = flush_interval
        self.enable_async = enable_async
        
        # Backup settings
        self.backup_enabled = backup_enabled
        self.backup_interval = backup_interval
        self.backup_path = Path(backup_path) if backup_path else self.storage_path / "backups"
        if self.backup_enabled:
            self.backup_path.mkdir(parents=True, exist_ok=True)
        
        # Thread pool for parallel operations
        self.thread_pool = ThreadPoolExecutor(max_workers=thread_pool_size)
        
        # In-memory storage
        self.recent_trades: List[EnhancedTradeMetrics] = []
        self.trade_count = 0
        
        # Performance tracking
        self.last_flush_time = time.time()
        self.flush_durations = []
        
        # Load existing data
        self._load_recent_history()
        
        # Cleanup old files on startup
        self._cleanup_old_files()
    
    def record_trade(self, metrics: Union[EnhancedTradeMetrics, TradeMetrics]) -> None:
        """Record a new trade (synchronous version)
        
        Args:
            metrics: Trade metrics to record (EnhancedTradeMetrics or legacy TradeMetrics)
        """
        # Convert legacy metrics if needed
        if isinstance(metrics, TradeMetrics):
            # Create minimal EnhancedTradeMetrics from legacy format
            gas = GasMetrics(
                gas_used=0,
                gas_price=int(metrics.gas_price),
                gas_cost_wei=0
            )
            
            execution = ExecutionMetrics(
                tx_hash="",
                execution_time=metrics.execution_time
            )
            
            tokens = TokenMetrics(
                token_in="",
                token_out="",
                amount_in=0,
                amount_out=0,
                token_in_symbol="",
                token_out_symbol="",
                token_in_decimals=18,
                token_out_decimals=18
            )
            
            enhanced_metrics = EnhancedTradeMetrics(
                timestamp=metrics.timestamp,
                strategy=metrics.strategy,
                token_pair=metrics.token_pair,
                dex=metrics.dex,
                profit=metrics.profit,
                success=metrics.success,
                gas=gas,
                execution=execution,
                tokens=tokens,
                additional_data=metrics.additional_data
            )
        else:
            enhanced_metrics = metrics
        
        self.recent_trades.append(enhanced_metrics)
        self.trade_count += 1
        
        # Auto-flush if needed
        if self.trade_count % self.flush_interval == 0:
            self.flush_to_disk()
        
        # Auto-backup if needed
        if self.backup_enabled and self.trade_count % self.backup_interval == 0:
            self.create_backup()
        
        # Trim memory if needed
        if len(self.recent_trades) > self.max_memory_entries:
            self._trim_memory()
    
    async def record_trade_async(self, metrics: Union[EnhancedTradeMetrics, TradeMetrics]) -> None:
        """Record a new trade asynchronously
        
        Args:
            metrics: Trade metrics to record (EnhancedTradeMetrics or legacy TradeMetrics)
        """
        if not self.enable_async:
            self.record_trade(metrics)
            return
            
        # Use ThreadPoolExecutor to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.thread_pool,
            self.record_trade,
            metrics
        )
    
    def flush_to_disk(self) -> None:
        """Flush current trades to disk (synchronous version)"""
        start_time = time.time()
        
        try:
            if not self.recent_trades:
                return
                
            # Convert to list of dicts
            trade_dicts = [t.to_dict() for t in self.recent_trades]
                
            # Convert to DataFrame
            df = pd.DataFrame(trade_dicts)
            
            # Convert timestamp to string for storage
            if 'timestamp' in df.columns:
                df['timestamp'] = df['timestamp'].astype(str)
            
            # Save to parquet file with timestamp
            filename = f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            df.to_parquet(self.storage_path / filename)
            
            logger.info(
                "Flushed trades to disk",
                count=len(self.recent_trades),
                filename=filename
            )
            
            # Track flush duration
            duration = time.time() - start_time
            self.flush_durations.append(duration)
            self.last_flush_time = time.time()
            
        except Exception as e:
            logger.error("Error flushing trades to disk", error=str(e))
    
    async def flush_to_disk_async(self) -> None:
        """Flush current trades to disk asynchronously"""
        if not self.enable_async:
            self.flush_to_disk()
            return
            
        # Use ThreadPoolExecutor to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.thread_pool,
            self.flush_to_disk
        )
    
    def create_backup(self) -> str:
        """Create a backup of all trade history
        
        Returns:
            Path to backup file
        """
        try:
            # Get all trades (in-memory and from disk)
            df = self.get_history()
            
            # Save to backup file with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = f"backup_{timestamp}.parquet"
            backup_path = self.backup_path / backup_file
            
            df.to_parquet(backup_path)
            
            logger.info(
                "Created trade history backup",
                file=str(backup_path),
                trades=len(df)
            )
            
            return str(backup_path)
            
        except Exception as e:
            logger.error("Error creating backup", error=str(e))
            return ""
    
    def get_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        strategy: Optional[str] = None,
        token_pair: Optional[str] = None,
        include_memory: bool = True,
        as_enhanced: bool = False
    ) -> Union[pd.DataFrame, List[EnhancedTradeMetrics]]:
        """Get historical trade data
        
        Args:
            start_time: Start time for history
            end_time: End time for history
            strategy: Filter by strategy
            token_pair: Filter by token pair
            include_memory: Whether to include in-memory trades
            as_enhanced: Whether to return EnhancedTradeMetrics objects instead of DataFrame
            
        Returns:
            DataFrame with historical trade data or list of EnhancedTradeMetrics
        """
        # Load all parquet files in date range
        dfs = []
        
        for file in self.storage_path.glob("*.parquet"):
            df = pd.read_parquet(file)
            
            # Convert timestamp strings back to datetime if needed
            if 'timestamp' in df.columns and df['timestamp'].dtype == 'object':
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Apply filters
            if start_time:
                df = df[df['timestamp'] >= start_time]
            if end_time:
                df = df[df['timestamp'] <= end_time]
            if strategy:
                df = df[df['strategy'] == strategy]
            if token_pair:
                df = df[df['token_pair'] == token_pair]
                
            dfs.append(df)
        
        # Add in-memory data if requested
        if include_memory and self.recent_trades:
            # Convert to list of dicts
            trade_dicts = [t.to_dict() for t in self.recent_trades]
            
            # Convert to DataFrame
            memory_df = pd.DataFrame(trade_dicts)
            
            # Apply filters
            if start_time:
                memory_df = memory_df[memory_df['timestamp'] >= start_time]
            if end_time:
                memory_df = memory_df[memory_df['timestamp'] <= end_time]
            if strategy:
                memory_df = memory_df[memory_df['strategy'] == strategy]
            if token_pair:
                memory_df = memory_df[memory_df['token_pair'] == token_pair]
                
            dfs.append(memory_df)
        
        if not dfs:
            return [] if as_enhanced else pd.DataFrame()
            
        # Combine all data
        result_df = pd.concat(dfs, ignore_index=True)
        
        # Sort by timestamp
        if 'timestamp' in result_df.columns:
            result_df = result_df.sort_values('timestamp')
        
        if as_enhanced:
            # Convert DataFrame rows to EnhancedTradeMetrics objects
            return [EnhancedTradeMetrics.from_dict(row.to_dict()) for _, row in result_df.iterrows()]
        
        return result_df
    
    def analyze_performance(
        self,
        timeframe: str = '24h',
        group_by: Optional[str] = None,
        include_gas_metrics: bool = True,
        include_charts: bool = False
    ) -> Dict[str, Any]:
        """Analyze trading performance with enhanced metrics
        
        Args:
            timeframe: Timeframe for analysis (e.g., '24h', '7d', '30d')
            group_by: Field to group results by (strategy, token_pair, dex, etc.)
            include_gas_metrics: Whether to include detailed gas metrics
            include_charts: Whether to generate and include chart data
            
        Returns:
            Dictionary with performance metrics
        """
        # Calculate start time based on timeframe
        start_time = datetime.now() - pd.Timedelta(timeframe)
        
        # Get historical data
        df = self.get_history(start_time=start_time)
        
        if df.empty:
            return {}
        
        # Basic metrics
        metrics = {
            'total_trades': len(df),
            'successful_trades': df['success'].sum(),
            'total_profit': df['profit'].sum(),
            'average_profit': df['profit'].mean(),
            'profit_std': df['profit'].std(),
            'success_rate': df['success'].mean() * 100,
            'average_execution_time': df['execution_time'].mean(),
            'timeframe': timeframe,
            'start_time': start_time,
            'end_time': datetime.now(),
        }
        
        # Gas metrics if requested
        if include_gas_metrics:
            metrics.update({
            'average_gas_price': df['gas_price'].mean(),
                'average_gas_used': df['gas_used'].mean(),
                'total_gas_cost_eth': df['gas_cost_eth'].sum(),
                'total_gas_cost_usd': df['gas_cost_usd'].sum(),
                'average_gas_cost_usd': df['gas_cost_usd'].mean(),
                'gas_optimization_savings': df['optimization_savings'].mean(),
                'average_network_congestion': df['network_congestion'].mean(),
            })
            
            # Gas optimization mode distribution
            if 'optimization_mode' in df.columns:
                mode_counts = df['optimization_mode'].value_counts().to_dict()
                metrics['optimization_mode_distribution'] = mode_counts
        
        # Group by analysis if requested
        if group_by and group_by in df.columns:
            group_metrics = df.groupby(group_by).agg({
                'profit': ['sum', 'mean', 'count'],
                'success': 'mean',
                'execution_time': 'mean',
                'gas_cost_usd': 'sum' if 'gas_cost_usd' in df.columns else None,
            }).to_dict()
            
            metrics[f'{group_by}_performance'] = group_metrics
        else:
            # Default groupings
            for field in ['strategy', 'token_pair', 'dex']:
                if field in df.columns:
                    group_metrics = df.groupby(field).agg({
                'profit': ['sum', 'mean', 'count'],
                        'success': 'mean',
                    }).to_dict()
                    
                    metrics[f'{field}_performance'] = group_metrics
        
        # Generate chart data if requested
        if include_charts:
            # Profit over time
            df_time = df.set_index('timestamp')
            hourly_profit = df_time.resample('1H')['profit'].sum().reset_index()
            
            metrics['charts'] = {
                'hourly_profit': {
                    'x': hourly_profit['timestamp'].tolist(),
                    'y': hourly_profit['profit'].tolist(),
                }
            }
            
            # Gas price vs. profit correlation
            if 'gas_price' in df.columns and 'profit' in df.columns:
                metrics['charts']['gas_profit_correlation'] = {
                    'x': df['gas_price'].tolist(),
                    'y': df['profit'].tolist(),
                    'correlation': df['gas_price'].corr(df['profit'])
                }
        
        return metrics
    
    def analyze_gas_performance(
        self,
        timeframe: str = '24h'
    ) -> Dict[str, Any]:
        """Analyze gas optimization performance
        
        Args:
            timeframe: Timeframe for analysis (e.g., '24h', '7d', '30d')
            
        Returns:
            Dictionary with gas performance metrics
        """
        # Calculate start time based on timeframe
        start_time = datetime.now() - pd.Timedelta(timeframe)
        
        # Get historical data
        df = self.get_history(start_time=start_time)
        
        if df.empty or 'gas_cost_usd' not in df.columns:
            return {}
        
        # Calculate gas metrics
        metrics = {
            'total_gas_cost_eth': df['gas_cost_eth'].sum(),
            'total_gas_cost_usd': df['gas_cost_usd'].sum(),
            'average_gas_cost_usd': df['gas_cost_usd'].mean(),
            'average_gas_price': df['gas_price'].mean(),
            'average_gas_used': df['gas_used'].mean(),
            'total_estimated_savings_usd': (df['optimization_savings'] * df['gas_cost_usd']).sum(),
            'average_network_congestion': df['network_congestion'].mean(),
        }
        
        # Gas optimization mode analysis
        if 'optimization_mode' in df.columns:
            mode_df = df.groupby('optimization_mode').agg({
                'gas_cost_usd': ['mean', 'sum'],
                'profit': ['mean', 'sum'],
                'execution_time': 'mean',
                'confirmation_time': 'mean' if 'confirmation_time' in df.columns else None,
                'success': 'mean',
                'gas_used': 'mean',
            }).reset_index()
            
            metrics['optimization_mode_performance'] = mode_df.set_index('optimization_mode').to_dict()
        
        # Network congestion correlation with gas price
        if 'network_congestion' in df.columns and 'gas_price' in df.columns:
            metrics['congestion_gas_correlation'] = df['network_congestion'].corr(df['gas_price'])
        
        # Gas price trend over time
        df_time = df.set_index('timestamp')
        hourly_gas = df_time.resample('1H')['gas_price'].mean().reset_index()
        
        metrics['gas_price_trend'] = {
            'timestamps': hourly_gas['timestamp'].tolist(),
            'gas_prices': hourly_gas['gas_price'].tolist(),
        }
        
        return metrics
    
    def get_learning_features(
        self,
        lookback_period: str = '7d',
        include_gas_features: bool = True,
        include_execution_features: bool = True
    ) -> pd.DataFrame:
        """Get enhanced features for AI learning
        
        Args:
            lookback_period: Period to look back for features
            include_gas_features: Whether to include detailed gas features
            include_execution_features: Whether to include execution features
            
        Returns:
            DataFrame with features for AI learning
        """
        start_time = datetime.now() - pd.Timedelta(lookback_period)
        df = self.get_history(start_time=start_time)
        
        if df.empty:
            return pd.DataFrame()
        
        # Create features
        features = pd.DataFrame()
        features['timestamp'] = df['timestamp']
        features['hour'] = df['timestamp'].dt.hour
        features['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Profit features
        features['profit'] = df['profit']
        features['rolling_mean_profit'] = df['profit'].rolling('1h').mean()
        features['rolling_std_profit'] = df['profit'].rolling('1h').std()
        
        # Gas features
        features['gas_price'] = df['gas_price']
        features['rolling_mean_gas'] = df['gas_price'].rolling('1h').mean()
        
        # Enhanced gas features
        if include_gas_features and 'gas_cost_usd' in df.columns:
            features['gas_cost_usd'] = df['gas_cost_usd']
            features['gas_used'] = df['gas_used']
            features['network_congestion'] = df['network_congestion']
            features['optimization_savings'] = df['optimization_savings']
            
            if 'optimization_mode' in df.columns:
                # One-hot encode optimization mode
                mode_dummies = pd.get_dummies(df['optimization_mode'], prefix='opt_mode')
                features = pd.concat([features, mode_dummies], axis=1)
        
        # Execution features
        if include_execution_features:
            features['execution_time'] = df['execution_time']
            
            if 'confirmation_time' in df.columns:
                features['confirmation_time'] = df['confirmation_time']
            
            if 'retry_count' in df.columns:
                features['retry_count'] = df['retry_count']
        
        # Success features
        features['success'] = df['success'].astype(int)
        features['rolling_success_rate'] = df['success'].rolling('1h').mean()
        
        # Strategy and token features
        features = pd.concat([
            features,
            pd.get_dummies(df['strategy'], prefix='strategy'),
            pd.get_dummies(df['token_pair'], prefix='token_pair'),
            pd.get_dummies(df['dex'], prefix='dex')
        ], axis=1)
        
        return features
    
    def export_to_csv(self, filepath: str, timeframe: str = 'all') -> str:
        """Export trade history to CSV file
        
        Args:
            filepath: Path to export CSV file
            timeframe: Timeframe to export (e.g., '24h', '7d', '30d', 'all')
            
        Returns:
            Path to exported file
        """
        # Get data based on timeframe
        if timeframe == 'all':
            df = self.get_history()
        else:
            start_time = datetime.now() - pd.Timedelta(timeframe)
            df = self.get_history(start_time=start_time)
        
        if df.empty:
            return ""
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Export to CSV
        df.to_csv(filepath, index=False)
        
        logger.info(
            "Exported trade history to CSV",
            filepath=filepath,
            rows=len(df)
        )
        
        return filepath
    
    def import_from_csv(self, filepath: str, replace_existing: bool = False) -> int:
        """Import trade history from CSV file
        
        Args:
            filepath: Path to CSV file
            replace_existing: Whether to replace existing data
            
        Returns:
            Number of trades imported
        """
        try:
            # Read CSV file
            df = pd.read_csv(filepath)
            
            if df.empty:
                return 0
            
            # Convert timestamp strings to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Clear existing data if requested
            if replace_existing:
                self.recent_trades = []
                
                # Remove existing parquet files
                for file in self.storage_path.glob("*.parquet"):
                    file.unlink()
            
            # Convert rows to EnhancedTradeMetrics
            for _, row in df.iterrows():
                metrics = EnhancedTradeMetrics.from_dict(row.to_dict())
                self.record_trade(metrics)
            
            logger.info(
                "Imported trade history from CSV",
                filepath=filepath,
                trades_imported=len(df)
            )
            
            return len(df)
            
        except Exception as e:
            logger.error("Error importing from CSV", error=str(e), filepath=filepath)
            return 0
    
    def _load_recent_history(self) -> None:
        """Load recent history from disk"""
        try:
            files = sorted(self.storage_path.glob("*.parquet"))[-5:]  # Load last 5 files
            
            for file in files:
                df = pd.read_parquet(file)
                
                # Convert timestamp strings to datetime if needed
                if 'timestamp' in df.columns and df['timestamp'].dtype == 'object':
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                for _, row in df.iterrows():
                    # Convert row to dict and create EnhancedTradeMetrics
                    try:
                        metrics = EnhancedTradeMetrics.from_dict(row.to_dict())
                        self.recent_trades.append(metrics)
                    except Exception as e:
                        logger.error("Error converting row to EnhancedTradeMetrics", error=str(e))
                    
            logger.info(
                "Loaded recent history",
                trades_loaded=len(self.recent_trades)
            )
            
        except Exception as e:
            logger.error("Error loading recent history", error=str(e))
    
    def _trim_memory(self) -> None:
        """Trim in-memory storage to max entries"""
        if len(self.recent_trades) > self.max_memory_entries:
            # Keep most recent entries
            self.recent_trades = self.recent_trades[-self.max_memory_entries:] 
    
    def _cleanup_old_files(self, max_age_days: int = 30) -> int:
        """Clean up old trade history files
        
        Args:
            max_age_days: Maximum age of files to keep (in days)
            
        Returns:
            Number of files deleted
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            deleted_count = 0
            
            for file in self.storage_path.glob("*.parquet"):
                # Get file modification time
                mtime = datetime.fromtimestamp(file.stat().st_mtime)
                
                if mtime < cutoff_date:
                    # Delete old file
                    file.unlink()
                    deleted_count += 1
            
            if deleted_count > 0:
                logger.info(
                    "Cleaned up old trade history files",
                    deleted_count=deleted_count,
                    max_age_days=max_age_days
                )
            
            return deleted_count
            
        except Exception as e:
            logger.error("Error cleaning up old files", error=str(e))
            return 0
    
    def close(self) -> None:
        """Close the manager and clean up resources"""
        try:
            # Flush any remaining trades
            if self.recent_trades:
                self.flush_to_disk()
            
            # Shutdown thread pool
            self.thread_pool.shutdown()
            
            logger.info("Trade history manager closed")
            
        except Exception as e:
            logger.error("Error closing trade history manager", error=str(e))

# For backward compatibility
TradeHistoryManager = EnhancedTradeHistoryManager 