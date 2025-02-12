"""Parallel processing manager for distributed ML computations"""

import ray
from typing import List, Dict, Any, Optional, Union, Callable, TypeVar
import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import horovod.torch as hvd
from mpi4py import MPI
import dask.dataframe as dd
import dask.array as da
from dask.distributed import Client, LocalCluster
import modin.pandas as mpd

logger = logging.getLogger(__name__)
T = TypeVar('T')
U = TypeVar('U')

@dataclass
class ParallelConfig:
    """Configuration for parallel processing"""
    num_cpus: int = 4
    num_gpus: int = 0
    memory_limit: str = "8GB"
    use_horovod: bool = False
    use_dask: bool = True
    use_modin: bool = True
    batch_size: int = 1000

class DistributedProcessor:
    """Manager for distributed processing"""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self._init_distributed()
    
    def _init_distributed(self):
        """Initialize distributed computing frameworks"""
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(
                num_cpus=self.config.num_cpus,
                num_gpus=self.config.num_gpus,
                _memory=self.config.memory_limit
            )
        
        # Initialize Horovod if needed
        if self.config.use_horovod:
            hvd.init()
        
        # Initialize Dask if needed
        if self.config.use_dask:
            cluster = LocalCluster(
                n_workers=self.config.num_cpus,
                memory_limit=self.config.memory_limit
            )
            self.dask_client = Client(cluster)
    
    @ray.remote
    def _process_partition(
        self,
        func: Callable[[T], U],
        data: T,
        *args,
        **kwargs
    ) -> U:
        """Process a single data partition"""
        return func(data, *args, **kwargs)
    
    def parallel_map(
        self,
        func: Callable[[T], U],
        data: List[T],
        *args,
        **kwargs
    ) -> List[U]:
        """Map function over data in parallel"""
        # Split data into batches
        batches = [
            data[i:i + self.config.batch_size]
            for i in range(0, len(data), self.config.batch_size)
        ]
        
        # Process batches in parallel
        futures = [
            self._process_partition.remote(func, batch, *args, **kwargs)
            for batch in batches
        ]
        
        return ray.get(futures)
    
    def parallel_pandas(
        self,
        func: Callable[[pd.DataFrame], pd.DataFrame],
        df: pd.DataFrame,
        *args,
        **kwargs
    ) -> pd.DataFrame:
        """Process pandas DataFrame in parallel"""
        if self.config.use_modin:
            # Use Modin for parallel pandas operations
            mdf = mpd.DataFrame(df)
            result = func(mdf, *args, **kwargs)
            return result._to_pandas()
        elif self.config.use_dask:
            # Use Dask for parallel pandas operations
            ddf = dd.from_pandas(df, npartitions=self.config.num_cpus)
            result = func(ddf, *args, **kwargs)
            return result.compute()
        else:
            # Use Ray for parallel pandas operations
            partitions = np.array_split(df, self.config.num_cpus)
            futures = [
                self._process_partition.remote(func, part, *args, **kwargs)
                for part in partitions
            ]
            results = ray.get(futures)
            return pd.concat(results)
    
    def parallel_numpy(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        arr: np.ndarray,
        *args,
        **kwargs
    ) -> np.ndarray:
        """Process numpy array in parallel"""
        if self.config.use_dask:
            # Use Dask for parallel array operations
            darr = da.from_array(arr, chunks='auto')
            result = func(darr, *args, **kwargs)
            return result.compute()
        else:
            # Use Ray for parallel array operations
            splits = np.array_split(arr, self.config.num_cpus)
            futures = [
                self._process_partition.remote(func, split, *args, **kwargs)
                for split in splits
            ]
            results = ray.get(futures)
            return np.concatenate(results)
    
    def distributed_training(
        self,
        model: torch.nn.Module,
        train_func: Callable,
        *args,
        **kwargs
    ):
        """Distributed model training"""
        if self.config.use_horovod:
            # Horovod distributed training
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            
            # Wrap optimizer with Horovod
            optimizer = kwargs.get('optimizer')
            if optimizer:
                optimizer = hvd.DistributedOptimizer(
                    optimizer,
                    named_parameters=model.named_parameters()
                )
                kwargs['optimizer'] = optimizer
            
            # Train on this rank's data
            return train_func(model, *args, **kwargs)
        else:
            # PyTorch DDP training
            if torch.cuda.is_available():
                dist.init_process_group('nccl')
                local_rank = dist.get_rank()
                torch.cuda.set_device(local_rank)
                model = model.cuda()
                model = DistributedDataParallel(
                    model,
                    device_ids=[local_rank]
                )
            
            return train_func(model, *args, **kwargs)
    
    def parallel_preprocessing(
        self,
        preprocessor: Any,
        data: Union[pd.DataFrame, np.ndarray],
        *args,
        **kwargs
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Parallel data preprocessing"""
        if isinstance(data, pd.DataFrame):
            return self.parallel_pandas(
                preprocessor.transform,
                data,
                *args,
                **kwargs
            )
        else:
            return self.parallel_numpy(
                preprocessor.transform,
                data,
                *args,
                **kwargs
            )
    
    def parallel_feature_engineering(
        self,
        feature_engineer: Any,
        data: Dict[str, pd.DataFrame],
        *args,
        **kwargs
    ) -> pd.DataFrame:
        """Parallel feature engineering"""
        results = []
        
        # Process each data source in parallel
        for source, df in data.items():
            if hasattr(feature_engineer, f'_compute_{source}_features'):
                func = getattr(feature_engineer, f'_compute_{source}_features')
                result = self.parallel_pandas(func, df, *args, **kwargs)
                results.append(result)
        
        # Combine results
        if results:
            return pd.concat(results, axis=1)
        return pd.DataFrame()
    
    def cleanup(self):
        """Clean up distributed resources"""
        if self.config.use_dask:
            self.dask_client.close()
        
        if self.config.use_horovod:
            hvd.shutdown()
        
        if ray.is_initialized():
            ray.shutdown()
            
class BatchProcessor:
    """Process data in batches with progress tracking"""
    
    def __init__(
        self,
        processor: DistributedProcessor,
        batch_size: int = 1000
    ):
        self.processor = processor
        self.batch_size = batch_size
    
    def process(
        self,
        func: Callable[[T], U],
        data: List[T],
        *args,
        show_progress: bool = True,
        **kwargs
    ) -> List[U]:
        """Process data in batches"""
        from tqdm import tqdm
        
        results = []
        batches = range(0, len(data), self.batch_size)
        
        if show_progress:
            batches = tqdm(batches)
        
        for i in batches:
            batch = data[i:i + self.batch_size]
            batch_results = self.processor.parallel_map(
                func,
                batch,
                *args,
                **kwargs
            )
            results.extend(batch_results)
        
        return results 