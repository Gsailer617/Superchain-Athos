"""Parallel data loading and preprocessing for market data"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import ray
from pathlib import Path
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiohttp
from functools import partial

logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Data loading configuration"""
    batch_size: int = 32
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    shuffle: bool = True
    cache_dir: Optional[Path] = None

class MarketDataset(Dataset):
    """Dataset for market data with parallel preprocessing"""
    
    def __init__(
        self,
        data_config: DataConfig,
        feature_columns: List[str],
        target_column: str
    ):
        self.config = data_config
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.data: Optional[pd.DataFrame] = None
        self.cache = {}
        
        if self.config.cache_dir:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self, data: pd.DataFrame):
        """Load and preprocess data"""
        self.data = data
        
        # Parallel feature preprocessing
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = []
            for col in self.feature_columns:
                futures.append(
                    executor.submit(self._preprocess_feature, col)
                )
            
            # Wait for all preprocessing to complete
            for future in futures:
                future.result()
    
    def _preprocess_feature(self, column: str):
        """Preprocess a single feature column"""
        if column not in self.data.columns:
            raise ValueError(f"Column {column} not found in data")
            
        values = self.data[column].values
        
        # Handle missing values
        if np.isnan(values).any():
            values = np.nan_to_num(values, nan=0.0)
        
        # Normalize
        values = (values - np.mean(values)) / (np.std(values) + 1e-8)
        
        # Cache preprocessed values
        cache_key = f"feature_{column}"
        if self.config.cache_dir:
            cache_path = self.config.cache_dir / f"{cache_key}.npy"
            np.save(cache_path, values)
        else:
            self.cache[cache_key] = values
    
    def __len__(self) -> int:
        return len(self.data) if self.data is not None else 0
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single item with cached features"""
        features = []
        for col in self.feature_columns:
            cache_key = f"feature_{col}"
            if self.config.cache_dir:
                cache_path = self.config.cache_dir / f"{cache_key}.npy"
                values = np.load(cache_path)
            else:
                values = self.cache[cache_key]
            features.append(values[idx])
        
        target = self.data[self.target_column].values[idx]
        
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32)
        )

class LiquidityGraphDataset(Dataset):
    """Dataset for liquidity graph data"""
    
    def __init__(
        self,
        data_config: DataConfig,
        node_features: List[str],
        edge_features: List[str]
    ):
        self.config = data_config
        self.node_features = node_features
        self.edge_features = edge_features
        self.graphs: List[Data] = []
    
    async def load_graphs(self, token_addresses: List[str]):
        """Load liquidity graphs for tokens"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for address in token_addresses:
                tasks.append(self._fetch_graph_data(session, address))
            
            graphs = await asyncio.gather(*tasks)
            self.graphs.extend([g for g in graphs if g is not None])
    
    async def _fetch_graph_data(
        self,
        session: aiohttp.ClientSession,
        token_address: str
    ) -> Optional[Data]:
        """Fetch graph data for a single token"""
        try:
            # Fetch liquidity pool data
            async with session.get(
                f"https://api.dex.com/pools/{token_address}"
            ) as response:
                if response.status != 200:
                    return None
                data = await response.json()
            
            # Convert to graph format
            nodes, edges = self._create_graph(data)
            
            return Data(
                x=torch.tensor(nodes, dtype=torch.float32),
                edge_index=torch.tensor(edges['index'], dtype=torch.long),
                edge_attr=torch.tensor(edges['attr'], dtype=torch.float32)
            )
            
        except Exception as e:
            logger.error(f"Error fetching graph data: {str(e)}")
            return None
    
    def _create_graph(self, data: Dict) -> Tuple[np.ndarray, Dict]:
        """Create graph representation from pool data"""
        # Implementation depends on specific data format
        raise NotImplementedError
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> Data:
        return self.graphs[idx]

class ParallelDataLoader:
    """Parallel data loading and preprocessing"""
    
    def __init__(
        self,
        data_config: DataConfig,
        num_parallel_loads: int = 2
    ):
        self.config = data_config
        self.num_parallel_loads = num_parallel_loads
        
        # Initialize Ray for parallel processing
        if not ray.is_initialized():
            ray.init()
    
    @ray.remote
    def _load_partition(
        self,
        file_path: Path,
        feature_columns: List[str],
        target_column: str
    ) -> MarketDataset:
        """Load and preprocess a data partition"""
        data = pd.read_parquet(file_path)
        dataset = MarketDataset(
            self.config,
            feature_columns,
            target_column
        )
        dataset.load_data(data)
        return dataset
    
    def create_market_loader(
        self,
        file_paths: List[Path],
        feature_columns: List[str],
        target_column: str
    ) -> DataLoader:
        """Create DataLoader for market data"""
        # Load partitions in parallel
        partition_refs = [
            self._load_partition.remote(
                path,
                feature_columns,
                target_column
            )
            for path in file_paths
        ]
        
        # Get results
        datasets = ray.get(partition_refs)
        
        # Combine datasets
        combined_dataset = ConcatDataset(datasets)
        
        return DataLoader(
            combined_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=self.config.pin_memory,
            shuffle=self.config.shuffle
        )
    
    async def create_graph_loader(
        self,
        token_addresses: List[str],
        node_features: List[str],
        edge_features: List[str]
    ) -> DataLoader:
        """Create DataLoader for graph data"""
        dataset = LiquidityGraphDataset(
            self.config,
            node_features,
            edge_features
        )
        
        # Load graphs
        await dataset.load_graphs(token_addresses)
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=self.config.pin_memory,
            shuffle=self.config.shuffle,
            collate_fn=Batch.from_data_list
        ) 