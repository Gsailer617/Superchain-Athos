from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import structlog
from dataclasses import dataclass
import numpy as np
from pathlib import Path

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

class TradeHistoryManager:
    """Manages historical trade data for AI learning"""
    
    def __init__(
        self,
        storage_path: str = "data/trade_history",
        max_memory_entries: int = 10000,
        flush_interval: int = 100
    ):
        """Initialize trade history manager
        
        Args:
            storage_path: Path to store trade history
            max_memory_entries: Maximum number of entries to keep in memory
            flush_interval: Number of trades before auto-flushing to disk
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_entries = max_memory_entries
        self.flush_interval = flush_interval
        
        # In-memory storage
        self.recent_trades: List[TradeMetrics] = []
        self.trade_count = 0
        
        # Load existing data
        self._load_recent_history()
        
    def record_trade(self, metrics: TradeMetrics) -> None:
        """Record a new trade
        
        Args:
            metrics: Trade metrics to record
        """
        self.recent_trades.append(metrics)
        self.trade_count += 1
        
        # Auto-flush if needed
        if self.trade_count % self.flush_interval == 0:
            self.flush_to_disk()
        
        # Trim memory if needed
        if len(self.recent_trades) > self.max_memory_entries:
            self._trim_memory()
    
    def flush_to_disk(self) -> None:
        """Flush current trades to disk"""
        try:
            if not self.recent_trades:
                return
                
            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    'timestamp': t.timestamp,
                    'strategy': t.strategy,
                    'token_pair': t.token_pair,
                    'dex': t.dex,
                    'profit': t.profit,
                    'gas_price': t.gas_price,
                    'execution_time': t.execution_time,
                    'success': t.success,
                    'additional_data': json.dumps(t.additional_data)
                }
                for t in self.recent_trades
            ])
            
            # Save to parquet file with timestamp
            filename = f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            df.to_parquet(self.storage_path / filename)
            
            logger.info(
                "Flushed trades to disk",
                count=len(self.recent_trades),
                filename=filename
            )
            
        except Exception as e:
            logger.error("Error flushing trades to disk", error=str(e))
    
    def get_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        strategy: Optional[str] = None,
        token_pair: Optional[str] = None
    ) -> pd.DataFrame:
        """Get historical trade data
        
        Args:
            start_time: Start time for history
            end_time: End time for history
            strategy: Filter by strategy
            token_pair: Filter by token pair
            
        Returns:
            DataFrame with historical trade data
        """
        # Load all parquet files in date range
        dfs = []
        
        for file in self.storage_path.glob("*.parquet"):
            df = pd.read_parquet(file)
            
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
        
        # Add in-memory data
        if self.recent_trades:
            memory_df = pd.DataFrame([
                {
                    'timestamp': t.timestamp,
                    'strategy': t.strategy,
                    'token_pair': t.token_pair,
                    'dex': t.dex,
                    'profit': t.profit,
                    'gas_price': t.gas_price,
                    'execution_time': t.execution_time,
                    'success': t.success,
                    'additional_data': json.dumps(t.additional_data)
                }
                for t in self.recent_trades
            ])
            dfs.append(memory_df)
        
        if not dfs:
            return pd.DataFrame()
            
        return pd.concat(dfs, ignore_index=True)
    
    def analyze_performance(
        self,
        timeframe: str = '24h'
    ) -> Dict[str, Any]:
        """Analyze trading performance
        
        Args:
            timeframe: Timeframe for analysis (e.g., '24h', '7d', '30d')
            
        Returns:
            Dictionary with performance metrics
        """
        # Calculate start time based on timeframe
        start_time = datetime.now() - pd.Timedelta(timeframe)
        
        # Get historical data
        df = self.get_history(start_time=start_time)
        
        if df.empty:
            return {}
        
        # Calculate metrics
        metrics = {
            'total_trades': len(df),
            'successful_trades': df['success'].sum(),
            'total_profit': df['profit'].sum(),
            'average_profit': df['profit'].mean(),
            'profit_std': df['profit'].std(),
            'success_rate': df['success'].mean() * 100,
            'average_execution_time': df['execution_time'].mean(),
            'average_gas_price': df['gas_price'].mean(),
            
            # Strategy performance
            'strategy_performance': df.groupby('strategy').agg({
                'profit': ['sum', 'mean', 'count'],
                'success': 'mean'
            }).to_dict(),
            
            # Token pair performance
            'token_pair_performance': df.groupby('token_pair').agg({
                'profit': ['sum', 'mean', 'count'],
                'success': 'mean'
            }).to_dict(),
            
            # DEX performance
            'dex_performance': df.groupby('dex').agg({
                'profit': ['sum', 'mean', 'count'],
                'success': 'mean'
            }).to_dict()
        }
        
        return metrics
    
    def get_learning_features(
        self,
        lookback_period: str = '7d'
    ) -> pd.DataFrame:
        """Get features for AI learning
        
        Args:
            lookback_period: Period to look back for features
            
        Returns:
            DataFrame with features for AI learning
        """
        start_time = datetime.now() - pd.Timedelta(lookback_period)
        df = self.get_history(start_time=start_time)
        
        if df.empty:
            return pd.DataFrame()
        
        # Extract features from additional_data
        df['additional_data'] = df['additional_data'].apply(json.loads)
        
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
    
    def _load_recent_history(self) -> None:
        """Load recent history from disk"""
        try:
            files = sorted(self.storage_path.glob("*.parquet"))[-5:]  # Load last 5 files
            
            for file in files:
                df = pd.read_parquet(file)
                
                for _, row in df.iterrows():
                    self.recent_trades.append(TradeMetrics(
                        timestamp=row['timestamp'],
                        strategy=row['strategy'],
                        token_pair=row['token_pair'],
                        dex=row['dex'],
                        profit=row['profit'],
                        gas_price=row['gas_price'],
                        execution_time=row['execution_time'],
                        success=row['success'],
                        additional_data=json.loads(row['additional_data'])
                    ))
                    
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