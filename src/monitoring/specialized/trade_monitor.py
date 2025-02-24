"""
Specialized trade monitoring system that extends base monitoring capabilities
with integrated historical data management
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import structlog
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path
import json
from ..core.base_monitor import BaseMonitor
from ..core.metrics_manager import metrics_manager, MetricConfig, MetricType
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from ...utils.unified_metrics import UnifiedMetricsSystem

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

@dataclass
class PerformanceAlert:
    """Alert for performance issues"""
    timestamp: datetime
    metric: str
    value: float
    threshold: float
    severity: str
    message: str

@dataclass
class MLPrediction:
    """Machine learning prediction results"""
    predicted_value: float
    confidence: float
    features_importance: Dict[str, float]
    model_metrics: Dict[str, List[float]]

class TradeMonitor(BaseMonitor):
    """Specialized monitoring system for trading operations with integrated historical data management"""
    
    def __init__(
        self,
        storage_path: str = "data/monitoring/trades",
        max_memory_entries: int = 10000,
        flush_interval: int = 100,
        enable_prometheus: bool = True,
        performance_thresholds: Optional[Dict[str, float]] = None,
        ml_config: Optional[Dict[str, Any]] = None,
        historical_storage_path: Optional[str] = None,
        metrics_system: Optional[UnifiedMetricsSystem] = None
    ):
        """Initialize trade monitoring system
        
        Args:
            storage_path: Path for storing trade monitoring data
            max_memory_entries: Maximum entries to keep in memory
            flush_interval: Interval for flushing to disk
            enable_prometheus: Whether to enable Prometheus metrics
            performance_thresholds: Custom thresholds for performance metrics
            ml_config: Configuration for machine learning models
            historical_storage_path: Path for historical data storage (parquet files)
            metrics_system: Optional unified metrics system
        """
        super().__init__(
            storage_path=storage_path,
            enable_prometheus=enable_prometheus
        )
        
        self.max_memory_entries = max_memory_entries
        self.flush_interval = flush_interval
        
        # Historical storage setup
        self.historical_storage_path = Path(historical_storage_path or storage_path) / "historical"
        self.historical_storage_path.mkdir(parents=True, exist_ok=True)
        
        # Trade-specific storage
        self.recent_trades: List[TradeMetrics] = []
        self.trade_count = 0
        
        # Performance monitoring
        self._performance_thresholds = performance_thresholds or {
            'min_profit': -0.1,
            'max_gas_price': 100.0,
            'max_execution_time': 5.0,
            'min_success_rate': 95.0
        }
        
        # ML configuration
        self._ml_config = ml_config or {
            'profit_prediction': {
                'window_size': 24,  # Hours
                'train_size': 0.8,
                'feature_importance_threshold': 0.05
            },
            'anomaly_detection': {
                'contamination': 0.1,
                'n_estimators': 100,
                'max_features': 0.8
            },
            'risk_assessment': {
                'confidence_threshold': 0.8,
                'risk_window': '7d'
            }
        }
        
        # Initialize ML models
        self._profit_predictor = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        self._risk_classifier = GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        )
        self._anomaly_detector = IsolationForest(
            contamination=self._ml_config['anomaly_detection']['contamination'],
            n_estimators=self._ml_config['anomaly_detection']['n_estimators'],
            max_features=self._ml_config['anomaly_detection']['max_features'],
            random_state=42
        )
        
        # Feature preprocessing
        self._scaler = StandardScaler()
        
        # Model performance tracking
        self._model_metrics = {
            'profit_prediction': {'mse': [], 'mae': [], 'r2': []},
            'risk_assessment': {'accuracy': [], 'precision': [], 'recall': []},
            'anomaly_detection': {'anomaly_ratio': []}
        }
        
        # Performance alerts
        self._recent_alerts: List[PerformanceAlert] = []
        
        # Initialize trade-specific metrics
        self._setup_trade_metrics()
        
        # Load historical data
        self._load_recent_history()
        
        # Use provided metrics system or create new one
        self.metrics = metrics_system or UnifiedMetricsSystem(
            enable_prometheus=enable_prometheus
        )
        
        # Initialize storage
        self._initialize_storage()
    
    def _setup_trade_metrics(self):
        """Setup trade-specific metrics"""
        # Trade metrics
        self._metrics['trade_profit'] = metrics_manager.create_metric(MetricConfig(
            name="trade_profit",
            description="Profit from trades",
            type=MetricType.GAUGE,
            labels=['strategy', 'token_pair', 'dex']
        ))
        
        self._metrics['gas_efficiency'] = metrics_manager.create_metric(MetricConfig(
            name="gas_efficiency",
            description="Profit per unit of gas",
            type=MetricType.GAUGE,
            labels=['strategy', 'dex']
        ))
        
        self._metrics['execution_speed'] = metrics_manager.create_metric(MetricConfig(
            name="execution_speed",
            description="Trade execution time",
            type=MetricType.HISTOGRAM,
            labels=['strategy', 'dex'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        ))
        
        self._metrics['success_rate'] = metrics_manager.create_metric(MetricConfig(
            name="success_rate",
            description="Trade success rate",
            type=MetricType.GAUGE,
            labels=['strategy', 'token_pair']
        ))
        
        # ML metrics
        self._metrics['prediction_error'] = metrics_manager.create_metric(MetricConfig(
            name="prediction_error",
            description="ML prediction error",
            type=MetricType.GAUGE,
            labels=['model', 'metric']
        ))
        
        self._metrics['anomaly_score'] = metrics_manager.create_metric(MetricConfig(
            name="anomaly_score",
            description="Trade anomaly score",
            type=MetricType.GAUGE,
            labels=['strategy']
        ))
    
    async def record_trade(self, metrics: TradeMetrics) -> None:
        """Record a new trade
        
        Args:
            metrics: Trade metrics to record
        """
        self.recent_trades.append(metrics)
        self.trade_count += 1
        
        # Update metrics
        await self._update_trade_metrics(metrics)
        
        # Auto-flush if needed
        if self.trade_count % self.flush_interval == 0:
            await self.flush_to_disk()
        
        # Trim memory if needed
        if len(self.recent_trades) > self.max_memory_entries:
            await self._trim_memory()
    
    async def _update_trade_metrics(self, metrics: TradeMetrics):
        """Update trade metrics
        
        Args:
            metrics: Trade metrics to update
        """
        labels = {
            'strategy': metrics.strategy,
            'token_pair': metrics.token_pair,
            'dex': metrics.dex
        }
        
        # Update trade profit
        await metrics_manager.record_metric(
            'trade_profit',
            metrics.profit,
            labels
        )
        
        # Update gas efficiency
        if metrics.gas_price > 0:
            await metrics_manager.record_metric(
                'gas_efficiency',
                metrics.profit / metrics.gas_price,
                {'strategy': metrics.strategy, 'dex': metrics.dex}
            )
        
        # Update execution speed
        await metrics_manager.record_metric(
            'execution_speed',
            metrics.execution_time,
            {'strategy': metrics.strategy, 'dex': metrics.dex}
        )
        
        # Update success rate
        total_trades = len([t for t in self.recent_trades if t.strategy == metrics.strategy])
        successful_trades = len([t for t in self.recent_trades if t.strategy == metrics.strategy and t.success])
        success_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0
        
        await metrics_manager.record_metric(
            'success_rate',
            success_rate,
            {'strategy': metrics.strategy, 'token_pair': metrics.token_pair}
        )
    
    def get_trade_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        strategy: Optional[str] = None,
        token_pair: Optional[str] = None,
        include_memory: bool = True
    ) -> pd.DataFrame:
        """Get historical trade data
        
        Args:
            start_time: Start time for history
            end_time: End time for history
            strategy: Filter by strategy
            token_pair: Filter by token pair
            include_memory: Whether to include in-memory trades
            
        Returns:
            DataFrame with historical trade data
        """
        # Load all parquet files in date range
        dfs = []
        
        for file in self.historical_storage_path.glob("*.parquet"):
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
        
        # Add in-memory data if requested
        if include_memory and self.recent_trades:
            memory_df = pd.DataFrame([
                {
                    'timestamp': t.timestamp,
                    'strategy': t.strategy,
                    'token_pair': t.token_pair,
                    'dex': t.dex,
                    'profit': float(t.profit),
                    'gas_price': float(t.gas_price),
                    'execution_time': float(t.execution_time),
                    'success': t.success,
                    'additional_data': json.dumps(t.additional_data)
                }
                for t in self.recent_trades
            ])
            
            # Apply filters to memory data
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
            return pd.DataFrame()
            
        return pd.concat(dfs, ignore_index=True)
    
    def analyze_performance(
        self,
        timeframe: str = '24h',
        strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze trading performance with enhanced metrics
        
        Args:
            timeframe: Timeframe for analysis (e.g., '24h', '7d', '30d')
            strategy: Optional strategy to filter by
            
        Returns:
            Dictionary with comprehensive performance metrics
        """
        # Get historical data
        start_time = datetime.now() - pd.Timedelta(timeframe)
        df = self.get_trade_history(start_time=start_time, strategy=strategy)
        
        if df.empty:
            return {}
        
        # Basic metrics
        metrics = {
            'total_trades': len(df),
            'successful_trades': df['success'].sum(),
            'total_profit': float(df['profit'].sum()),
            'average_profit': float(df['profit'].mean()),
            'profit_std': float(df['profit'].std()),
            'success_rate': float(df['success'].mean() * 100),
            'average_execution_time': float(df['execution_time'].mean()),
            'average_gas_price': float(df['gas_price'].mean())
        }
        
        # Strategy performance
        strategy_metrics = df.groupby('strategy').agg({
            'profit': ['sum', 'mean', 'count', 'std'],
            'success': 'mean',
            'execution_time': 'mean',
            'gas_price': 'mean'
        })
        metrics['strategy_performance'] = {
            strategy: {
                'total_profit': float(row['profit']['sum']),
                'avg_profit': float(row['profit']['mean']),
                'trade_count': int(row['profit']['count']),
                'profit_std': float(row['profit']['std']),
                'success_rate': float(row['success']['mean'] * 100),
                'avg_execution_time': float(row['execution_time']['mean']),
                'avg_gas_price': float(row['gas_price']['mean'])
            }
            for strategy, row in strategy_metrics.iterrows()
        }
        
        # Token pair performance
        token_metrics = df.groupby('token_pair').agg({
            'profit': ['sum', 'mean', 'count', 'std'],
            'success': 'mean'
        })
        metrics['token_pair_performance'] = {
            token: {
                'total_profit': float(row['profit']['sum']),
                'avg_profit': float(row['profit']['mean']),
                'trade_count': int(row['profit']['count']),
                'profit_std': float(row['profit']['std']),
                'success_rate': float(row['success']['mean'] * 100)
            }
            for token, row in token_metrics.iterrows()
        }
        
        # DEX performance
        dex_metrics = df.groupby('dex').agg({
            'profit': ['sum', 'mean', 'count', 'std'],
            'success': 'mean',
            'gas_price': 'mean'
        })
        metrics['dex_performance'] = {
            dex: {
                'total_profit': float(row['profit']['sum']),
                'avg_profit': float(row['profit']['mean']),
                'trade_count': int(row['profit']['count']),
                'profit_std': float(row['profit']['std']),
                'success_rate': float(row['success']['mean'] * 100),
                'avg_gas_price': float(row['gas_price']['mean'])
            }
            for dex, row in dex_metrics.iterrows()
        }
        
        # Advanced metrics
        metrics.update({
            'profit_volatility': float(df['profit'].std() / abs(df['profit'].mean())) if df['profit'].mean() != 0 else 0.0,
            'gas_efficiency': float((df['profit'] / df['gas_price']).mean()),
            'execution_efficiency': float(df['profit'].sum() / df['execution_time'].sum()) if df['execution_time'].sum() > 0 else 0.0
        })
        
        # Time-based analysis
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        hourly_stats = df.groupby('hour').agg({
            'profit': ['mean', 'sum', 'count'],
            'success': 'mean',
            'gas_price': 'mean'
        })
        metrics['hourly_performance'] = {
            str(hour): {
                'avg_profit': float(row['profit']['mean']),
                'total_profit': float(row['profit']['sum']),
                'trade_count': int(row['profit']['count']),
                'success_rate': float(row['success']['mean'] * 100),
                'avg_gas_price': float(row['gas_price']['mean'])
            }
            for hour, row in hourly_stats.iterrows()
        }
        
        weekly_stats = df.groupby('day_of_week').agg({
            'profit': ['mean', 'sum', 'count'],
            'success': 'mean',
            'gas_price': 'mean'
        })
        metrics['weekly_performance'] = {
            str(day): {
                'avg_profit': float(row['profit']['mean']),
                'total_profit': float(row['profit']['sum']),
                'trade_count': int(row['profit']['count']),
                'success_rate': float(row['success']['mean'] * 100),
                'avg_gas_price': float(row['gas_price']['mean'])
            }
            for day, row in weekly_stats.iterrows()
        }
        
        return metrics
    
    def predict_profit(
        self,
        strategy: str,
        token_pair: str,
        market_conditions: Dict[str, float]
    ) -> MLPrediction:
        """Predict profit for a potential trade
        
        Args:
            strategy: Trading strategy
            token_pair: Token pair to trade
            market_conditions: Current market conditions
            
        Returns:
            MLPrediction with predicted profit and confidence
        """
        # Prepare features
        features = self._prepare_prediction_features(
            strategy,
            token_pair,
            market_conditions
        )
        
        # Make prediction
        prediction = float(self._profit_predictor.predict([features])[0])
        
        # Calculate prediction confidence using bootstrap
        predictions = []
        n_predictions = 10
        for _ in range(n_predictions):
            bootstrap_idx = np.random.choice(
                len(features),
                size=len(features),
                replace=True
            )
            bootstrap_features = features[bootstrap_idx]
            predictions.append(
                float(self._profit_predictor.predict([bootstrap_features])[0])
            )
        
        # Get feature importance
        importance = dict(zip(
            self._profit_predictor.feature_names_in_,
            self._profit_predictor.feature_importances_
        ))
        
        # Filter important features
        importance = {
            k: float(v) for k, v in importance.items()
            if v >= self._ml_config['profit_prediction']['feature_importance_threshold']
        }
        
        return MLPrediction(
            predicted_value=prediction,
            confidence=float(1.0 / (1.0 + np.std(predictions))),
            features_importance=importance,
            model_metrics=self._model_metrics['profit_prediction']
        )
    
    def detect_anomalies(
        self,
        timeframe: str = '24h'
    ) -> Dict[str, Any]:
        """Detect anomalies in trading patterns
        
        Args:
            timeframe: Time window for anomaly detection
            
        Returns:
            Dictionary with anomaly detection results
        """
        # Get recent trading data
        df = self.get_trade_history(
            start_time=datetime.now() - pd.Timedelta(timeframe)
        )
        
        if df.empty:
            return {}
        
        # Prepare features for anomaly detection
        features = self._prepare_anomaly_features(df)
        
        # Detect anomalies
        anomaly_scores = self._anomaly_detector.score_samples(features)
        anomalies = self._anomaly_detector.predict(features)
        
        # Calculate anomaly metrics
        anomaly_results = {
            'anomaly_scores': [float(score) for score in anomaly_scores],
            'anomaly_indices': [int(idx) for idx in np.where(anomalies == -1)[0]],
            'anomaly_ratio': float(len(anomalies[anomalies == -1]) / len(anomalies)),
            'timestamp': [ts.isoformat() for ts in df.iloc[anomalies == -1]['timestamp']],
            'metrics': {
                'mean_score': float(np.mean(anomaly_scores)),
                'std_score': float(np.std(anomaly_scores)),
                'min_score': float(np.min(anomaly_scores)),
                'max_score': float(np.max(anomaly_scores))
            }
        }
        
        # Update metrics
        self._model_metrics['anomaly_detection']['anomaly_ratio'].append(
            anomaly_results['anomaly_ratio']
        )
        
        return anomaly_results
    
    def _prepare_prediction_features(
        self,
        strategy: str,
        token_pair: str,
        market_conditions: Dict[str, float]
    ) -> np.ndarray:
        """Prepare features for profit prediction
        
        Args:
            strategy: Trading strategy
            token_pair: Token pair
            market_conditions: Market conditions
            
        Returns:
            Feature array for prediction
        """
        # Get historical data for the strategy and token pair
        recent_data = self.get_trade_history(
            strategy=strategy,
            token_pair=token_pair,
            start_time=datetime.now() - pd.Timedelta(
                hours=self._ml_config['profit_prediction']['window_size']
            )
        )
        
        features = []
        
        # Historical performance features
        if not recent_data.empty:
            features.extend([
                float(recent_data['profit'].mean()),
                float(recent_data['profit'].std()),
                float(recent_data['success'].mean()),
                float(recent_data['gas_price'].mean()),
                float(recent_data['execution_time'].mean())
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Market condition features
        features.extend([
            float(market_conditions.get('volume', 0)),
            float(market_conditions.get('volatility', 0)),
            float(market_conditions.get('liquidity', 0)),
            float(market_conditions.get('gas_price', 0))
        ])
        
        return np.array(features)
    
    def _prepare_anomaly_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for anomaly detection
        
        Args:
            df: Trading data DataFrame
            
        Returns:
            Feature array for anomaly detection
        """
        # Calculate rolling statistics
        df['rolling_profit_mean'] = df['profit'].rolling('1h').mean()
        df['rolling_profit_std'] = df['profit'].rolling('1h').std()
        df['rolling_gas_mean'] = df['gas_price'].rolling('1h').mean()
        df['rolling_execution_mean'] = df['execution_time'].rolling('1h').mean()
        
        # Prepare feature matrix
        features = df[[
            'profit',
            'gas_price',
            'execution_time',
            'rolling_profit_mean',
            'rolling_profit_std',
            'rolling_gas_mean',
            'rolling_execution_mean'
        ]].fillna(0)
        
        # Scale features
        return self._scaler.fit_transform(features)
    
    async def flush_to_disk(self) -> None:
        """Flush current trades to disk in both monitoring and historical formats"""
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
                    'profit': float(t.profit),  # Convert numpy float to Python float
                    'gas_price': float(t.gas_price),
                    'execution_time': float(t.execution_time),
                    'success': t.success,
                    'additional_data': json.dumps(t.additional_data)
                }
                for t in self.recent_trades
            ])
            
            # Save to parquet file with timestamp
            filename = f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            df.to_parquet(self.historical_storage_path / filename)
            
            logger.info(
                "Flushed trades to disk",
                count=len(self.recent_trades),
                filename=filename
            )
            
            # Also save monitoring-specific data
            await self._save_monitoring_data()
            
        except Exception as e:
            logger.error("Error flushing trades to disk", error=str(e))
    
    async def _trim_memory(self) -> None:
        """Trim in-memory storage to max entries"""
        if len(self.recent_trades) > self.max_memory_entries:
            # Keep most recent entries
            self.recent_trades = self.recent_trades[-self.max_memory_entries:] 
    
    def _load_recent_history(self) -> None:
        """Load recent history from disk"""
        try:
            files = sorted(self.historical_storage_path.glob("*.parquet"))[-5:]  # Load last 5 files
            
            for file in files:
                df = pd.read_parquet(file)
                
                for _, row in df.iterrows():
                    self.recent_trades.append(TradeMetrics(
                        timestamp=row['timestamp'],
                        strategy=row['strategy'],
                        token_pair=row['token_pair'],
                        dex=row['dex'],
                        profit=float(row['profit']),
                        gas_price=float(row['gas_price']),
                        execution_time=float(row['execution_time']),
                        success=row['success'],
                        additional_data=json.loads(row['additional_data'])
                    ))
                    
            logger.info(
                "Loaded recent history",
                trades_loaded=len(self.recent_trades)
            )
            
        except Exception as e:
            logger.error("Error loading recent history", error=str(e))

    async def _save_monitoring_data(self) -> None:
        """Save monitoring-specific data"""
        try:
            # Save ML model states
            model_path = self.storage_path / "ml_models"
            model_path.mkdir(exist_ok=True)
            
            # Save performance metrics
            metrics_path = self.storage_path / "metrics"
            metrics_path.mkdir(exist_ok=True)
            
            # Save alerts
            alerts_path = self.storage_path / "alerts"
            alerts_path.mkdir(exist_ok=True)
            
        except Exception as e:
            logger.error("Error saving monitoring data", error=str(e))

    def _initialize_storage(self):
        """Initialize trade data storage"""
        self.trades_file = self.storage_path / "trades.parquet"
        if not self.trades_file.exists():
            pd.DataFrame().to_parquet(self.trades_file)
            
    def record_trade(self, trade_data: Dict[str, Any]):
        """Record trade data and update metrics
        
        Args:
            trade_data: Trade information
        """
        try:
            # Update metrics
            self.metrics.record_trade(trade_data)
            
            # Check performance thresholds
            self._check_thresholds(trade_data)
            
            # Store trade data
            self._store_trade(trade_data)
            
        except Exception as e:
            logger.error(f"Error recording trade: {str(e)}")
            
    def _check_thresholds(self, trade_data: Dict[str, Any]):
        """Check trade against performance thresholds"""
        try:
            profit = trade_data.get('profit', 0)
            gas_price = trade_data.get('gas_price', 0)
            execution_time = trade_data.get('execution_time', 0)
            
            # Check profit threshold
            min_profit = self._performance_thresholds.get('min_profit', -0.1)
            if profit < min_profit:
                logger.warning(
                    "Trade profit below threshold",
                    profit=profit,
                    threshold=min_profit
                )
                
            # Check gas price threshold
            max_gas = self._performance_thresholds.get('max_gas_price', 100)
            if gas_price > max_gas:
                logger.warning(
                    "Gas price above threshold",
                    gas_price=gas_price,
                    threshold=max_gas
                )
                
            # Check execution time threshold
            max_time = self._performance_thresholds.get('max_execution_time', 5)
            if execution_time > max_time:
                logger.warning(
                    "Execution time above threshold",
                    execution_time=execution_time,
                    threshold=max_time
                )
                
        except Exception as e:
            logger.error(f"Error checking thresholds: {str(e)}")
            
    def _store_trade(self, trade_data: Dict[str, Any]):
        """Store trade data to disk"""
        try:
            # Add timestamp
            trade_data['timestamp'] = datetime.now()
            
            # Convert to DataFrame
            trade_df = pd.DataFrame([trade_data])
            
            # Append to storage
            if self.trades_file.exists():
                existing_df = pd.read_parquet(self.trades_file)
                updated_df = pd.concat([existing_df, trade_df], ignore_index=True)
            else:
                updated_df = trade_df
                
            # Keep only max entries
            if len(updated_df) > self.max_memory_entries:
                updated_df = updated_df.tail(self.max_memory_entries)
                
            # Save to disk
            updated_df.to_parquet(self.trades_file)
            
        except Exception as e:
            logger.error(f"Error storing trade data: {str(e)}")
            
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.metrics.get_performance_summary()
        
    def get_token_analytics(self, token: Optional[str] = None) -> Dict[str, Any]:
        """Get token-specific analytics"""
        return self.metrics.get_token_analytics(token)
        
    def get_dex_analytics(self, dex: Optional[str] = None) -> Dict[str, Any]:
        """Get DEX-specific analytics"""
        return self.metrics.get_dex_analytics(dex)
        
    def get_trade_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get historical trade data
        
        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            DataFrame with trade history
        """
        try:
            if not self.trades_file.exists():
                return pd.DataFrame()
                
            df = pd.read_parquet(self.trades_file)
            
            if start_time:
                df = df[df['timestamp'] >= start_time]
            if end_time:
                df = df[df['timestamp'] <= end_time]
                
            return df
            
        except Exception as e:
            logger.error(f"Error getting trade history: {str(e)}")
            return pd.DataFrame() 