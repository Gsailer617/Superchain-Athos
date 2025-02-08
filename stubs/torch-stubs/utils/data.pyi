from typing import (
    Any, Callable, Generic, Iterator, List, Optional,
    Sequence, Tuple, TypeVar, Union, Dict, Iterable
)
from torch import Tensor

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

class Dataset(Generic[T_co]):
    """Abstract base class for all datasets.
    
    All datasets that represent a map from keys to data samples should subclass it.
    All subclasses should overwrite __getitem__ and __len__.
    """
    def __getitem__(self, index: int) -> T_co: ...
    def __len__(self) -> int: ...
    def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]': ...

class ConcatDataset(Dataset[T_co]):
    def __init__(self, datasets: Sequence[Dataset[T_co]]) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> T_co: ...

class IterableDataset(Generic[T_co]):
    """Base class for iterable datasets.
    
    All datasets that represent an iterable of data samples should subclass it.
    All subclasses should overwrite __iter__.
    """
    def __iter__(self) -> Iterator[T_co]: ...

class DataLoader(Generic[T]):
    """Data loader combining a dataset and a sampler to provide iterative data access.
    
    Args:
        dataset: Dataset from which to load data
        batch_size: How many samples per batch
        shuffle: Set to True to have data reshuffled at every epoch
        sampler: Defines strategy to draw samples from dataset
        batch_sampler: Like sampler, but returns a batch of indices at a time
        num_workers: How many subprocesses to use for data loading
        collate_fn: Merges a list of samples to form a mini-batch
        pin_memory: If True, copies Tensors into CUDA pinned memory
        drop_last: If True, drops the last incomplete batch
        timeout: If positive, the timeout value for collecting a batch
        worker_init_fn: If not None, called on each worker subprocess
        prefetch_factor: Number of batches loaded in advance by each worker
        persistent_workers: If True, the data loader will not shutdown the worker processes
    """
    def __init__(
        self,
        dataset: Union[Dataset[T], IterableDataset[T]],
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler[int]] = None,
        batch_sampler: Optional[Sampler[List[int]]] = None,
        num_workers: int = 0,
        collate_fn: Optional[Callable[[List[T]], Any]] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False
    ) -> None: ...
    def __iter__(self) -> Iterator[T]: ...
    def __len__(self) -> int: ...

class Sampler(Generic[T]):
    """Base class for all Samplers.
    
    Every Sampler subclass has to provide an __iter__ method and a __len__ method.
    """
    def __init__(self, data_source: Optional[Dataset[Any]]) -> None: ...
    def __iter__(self) -> Iterator[T]: ...
    def __len__(self) -> int: ...

class RandomSampler(Sampler[int]):
    """Samples elements randomly, without replacement.
    
    Args:
        data_source: Dataset to sample from
        replacement: If True, samples are drawn with replacement
        num_samples: Number of samples to draw
        generator: Generator used for random sampling
    """
    def __init__(
        self,
        data_source: Dataset[Any],
        replacement: bool = False,
        num_samples: Optional[int] = None,
        generator: Optional[Any] = None
    ) -> None: ...

class SubsetRandomSampler(Sampler[int]):
    """Samples elements randomly from a given list of indices, without replacement.
    
    Args:
        indices: List of indices to sample from
        generator: Generator used for random sampling
    """
    def __init__(
        self,
        indices: Sequence[int],
        generator: Optional[Any] = None
    ) -> None: ...

class WeightedRandomSampler(Sampler[int]):
    """Samples elements from [0,..,len(weights)-1] with given probabilities (weights).
    
    Args:
        weights: List of weights for each index
        num_samples: Number of samples to draw
        replacement: If True, samples are drawn with replacement
        generator: Generator used for random sampling
    """
    def __init__(
        self,
        weights: Sequence[float],
        num_samples: int,
        replacement: bool = True,
        generator: Optional[Any] = None
    ) -> None: ... 