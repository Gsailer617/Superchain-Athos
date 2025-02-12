"""Advanced data preprocessing and feature engineering"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import torch
from torch.utils.data import Dataset
import logging
from pathlib import Path
import ray
from concurrent.futures import ThreadPoolExecutor
import asyncio
from river import preprocessing, feature_extraction, stats
import joblib

logger = logging.getLogger(__name__)

@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    window_sizes: List[int] = (5, 10, 20, 50, 100)
    use_ta_features: bool = True
    use_volume_features: bool = True
    use_orderbook_features: bool = True
    use_sentiment_features: bool = True
    pca_components: Optional[int] = None

class OnlineFeatureProcessor:
    def __init__(self):
        self.scalers = {}
        self.stats = {}
        self.feature_extractors = {}
    
    def _init_processors(self, feature_names: List[str]):
        for name in feature_names:
            self.scalers[name] = preprocessing.StandardScaler()
            self.stats[name] = {
                'mean': stats.Mean(),
                'var': stats.Var(),
                'max': stats.Max(),
                'min': stats.Min()
            }
            self.feature_extractors[name] = feature_extraction.TargetAgg()
    
    def update(self, features: Dict[str, float]):
        for name, value in features.items():
            if name not in self.scalers:
                self._init_processors([name])
            self.scalers[name].learn_one(value)
            for stat in self.stats[name].values():
                stat.update(value)
            self.feature_extractors[name].learn_one(value)
    
    def transform(self, features: Dict[str, float]) -> Dict[str, float]:
        result = {}
        for name, value in features.items():
            scaled = self.scalers[name].transform_one(value)
            result[f"{name}_scaled"] = scaled
            
            stats_dict = {
                f"{name}_{stat_name}": stat.get()
                for stat_name, stat in self.stats[name].items()
            }
            result.update(stats_dict)
            
            extracted = self.feature_extractors[name].transform_one(value)
            result.update({
                f"{name}_{k}": v
                for k, v in extracted.items()
            })
        return result

class FeatureEngineer:
    def __init__(self, config: FeatureConfig, cache_dir: Optional[Path] = None):
        self.config = config
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.scalers = {}
        self.pca = None
        self.online_processor = OnlineFeatureProcessor()
    
    def _compute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        # Moving averages
        for window in self.config.window_sizes:
            result[f'ma_{window}'] = df['price'].rolling(window).mean()
            result[f'std_{window}'] = df['price'].rolling(window).std()
            
            if self.config.use_volume_features:
                result[f'volume_ma_{window}'] = df['volume'].rolling(window).mean()
        
        # Momentum indicators
        result['rsi'] = self._compute_rsi(df['price'])
        result['macd'] = self._compute_macd(df['price'])
        
        # Volatility indicators
        result['atr'] = self._compute_atr(df)
        result['bollinger_upper'], result['bollinger_lower'] = self._compute_bollinger_bands(df['price'])
        
        return result
    
    def _compute_orderbook_features(self, orderbook: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame()
        
        # Spread and depth
        features['spread'] = orderbook['ask_price_1'] - orderbook['bid_price_1']
        features['depth_imbalance'] = (
            orderbook['bid_size_1'] - orderbook['ask_size_1']
        ) / (orderbook['bid_size_1'] + orderbook['ask_size_1'])
        
        # Price impact
        for level in range(1, 6):
            features[f'price_impact_bid_{level}'] = (
                orderbook[f'bid_price_{level}'] * orderbook[f'bid_size_{level}']
            ).cumsum()
            features[f'price_impact_ask_{level}'] = (
                orderbook[f'ask_price_{level}'] * orderbook[f'ask_size_{level}']
            ).cumsum()
        
        return features
    
    def _compute_sentiment_features(self, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame()
        
        # Aggregate sentiment
        for window in self.config.window_sizes:
            features[f'sentiment_ma_{window}'] = sentiment_data['sentiment'].rolling(window).mean()
            features[f'sentiment_std_{window}'] = sentiment_data['sentiment'].rolling(window).std()
        
        # Sentiment momentum
        features['sentiment_momentum'] = sentiment_data['sentiment'].diff()
        
        # Sentiment extremes
        features['sentiment_max'] = sentiment_data['sentiment'].rolling(50).max()
        features['sentiment_min'] = sentiment_data['sentiment'].rolling(50).min()
        
        return features
    
    @staticmethod
    def _compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line
    
    @staticmethod
    def _compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window).mean()
    
    @staticmethod
    def _compute_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series]:
        ma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = ma + (std * num_std)
        lower = ma - (std * num_std)
        return upper, lower
    
    def fit_transform(self, data: Dict[str, pd.DataFrame], is_train: bool = True) -> pd.DataFrame:
        features = []
        
        # Technical indicators
        if 'market_data' in data:
            tech_features = self._compute_technical_indicators(data['market_data'])
            features.append(tech_features)
        
        # Orderbook features
        if self.config.use_orderbook_features and 'orderbook' in data:
            ob_features = self._compute_orderbook_features(data['orderbook'])
            features.append(ob_features)
        
        # Sentiment features
        if self.config.use_sentiment_features and 'sentiment' in data:
            sent_features = self._compute_sentiment_features(data['sentiment'])
            features.append(sent_features)
        
        # Combine features
        combined = pd.concat(features, axis=1)
        
        # Scale features
        if is_train:
            for col in combined.columns:
                self.scalers[col] = RobustScaler()
                combined[col] = self.scalers[col].fit_transform(combined[[col]])
        else:
            for col in combined.columns:
                if col in self.scalers:
                    combined[col] = self.scalers[col].transform(combined[[col]])
        
        # Dimensionality reduction
        if self.config.pca_components and is_train:
            self.pca = PCA(n_components=self.config.pca_components)
            combined = pd.DataFrame(
                self.pca.fit_transform(combined),
                index=combined.index
            )
        elif self.pca is not None:
            combined = pd.DataFrame(
                self.pca.transform(combined),
                index=combined.index
            )
        
        return combined
    
    def save(self, path: Path):
        state = {
            'config': self.config,
            'scalers': self.scalers,
            'pca': self.pca
        }
        joblib.dump(state, path)
    
    @classmethod
    def load(cls, path: Path) -> 'FeatureEngineer':
        state = joblib.load(path)
        engineer = cls(state['config'])
        engineer.scalers = state['scalers']
        engineer.pca = state['pca']
        return engineer

class StreamingDataset(Dataset):
    def __init__(self, feature_engineer: FeatureEngineer, window_size: int = 100, batch_size: int = 32):
        self.feature_engineer = feature_engineer
        self.window_size = window_size
        self.batch_size = batch_size
        
        self.buffer = []
        self.online_processor = OnlineFeatureProcessor()
    
    def update(self, new_data: Dict[str, Any]):
        # Process new data
        features = self.feature_engineer.fit_transform(
            {'market_data': pd.DataFrame([new_data])},
            is_train=False
        )
        
        # Update online processors
        self.online_processor.update(features.iloc[0].to_dict())
        
        # Add to buffer
        self.buffer.append(features.iloc[0].to_dict())
        
        # Maintain window size
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
    
    def __len__(self) -> int:
        return len(self.buffer) - self.batch_size + 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.buffer[idx:idx + self.batch_size]
        features = []
        for item in sequence:
            # Get online features
            online_features = self.online_processor.transform(item)
            # Combine with original features
            combined = {**item, **online_features}
            features.append(list(combined.values()))
        
        return (
            torch.tensor(features[:-1], dtype=torch.float32),
            torch.tensor(features[-1], dtype=torch.float32)
        ) 