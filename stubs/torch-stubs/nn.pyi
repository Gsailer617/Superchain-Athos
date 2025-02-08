from typing import Any, Optional, Union, List, Tuple, Dict, Callable, TypeVar, Generic, overload, Iterator
import torch
from torch import Tensor

T = TypeVar('T', bound=Tensor)
Module = TypeVar('Module', bound='_Module')

class _Module:
    """Base class for all neural network modules.
    
    Your models should also subclass this class.
    Modules can also contain other Modules, allowing to nest them in a tree structure.
    
    Attributes:
        training: Boolean indicating if the module is in training mode
    """
    training: bool
    def __init__(self) -> None: ...
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def parameters(self) -> Iterator[Parameter]: ...
    def named_parameters(self) -> Iterator[Tuple[str, Parameter]]: ...
    def train(self: Module, mode: bool = True) -> Module: ...
    def eval(self: Module) -> Module: ...
    def to(self: Module, device: Union[str, torch.device]) -> Module: ...

class Linear(_Module):
    """Applies a linear transformation to the incoming data: y = xA^T + b
    
    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        bias: If set to False, the layer will not learn an additive bias
        
    Shape:
        - Input: (N, *, in_features) where * means any number of additional dimensions
        - Output: (N, *, out_features) where all but the last dimension are the same shape as the input
    
    Attributes:
        weight: The learnable weights of the module of shape (out_features, in_features)
        bias: The learnable bias of the module of shape (out_features)
    """
    in_features: int
    out_features: int
    weight: Parameter
    bias: Optional[Parameter]
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class Sequential(_Module):
    """A sequential container of modules.
    
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an OrderedDict of modules can be passed in.
    
    Example:
        model = Sequential(
            Conv2d(1,20,5),
            ReLU(),
            Conv2d(20,64,5),
            ReLU()
        )
    """
    _modules: Dict[str, _Module]
    def __init__(self, *args: _Module) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def add_module(self, name: str, module: Optional[_Module]) -> None: ...

class ReLU(_Module):
    """Applies the rectified linear unit function element-wise: ReLU(x) = max(0, x)
    
    Args:
        inplace: If True, modifies the input tensor in-place
        
    Shape:
        - Input: (N, *) where * means any number of additional dimensions
        - Output: (N, *), same shape as the input
    """
    inplace: bool
    def __init__(self, inplace: bool = False) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class Dropout(_Module):
    """Randomly zeroes some of the elements of the input tensor with probability p.
    
    Args:
        p: Probability of an element to be zeroed. Default: 0.5
        inplace: If True, will do this operation in-place
        
    Shape:
        - Input: (N, *) where * means any number of additional dimensions
        - Output: (N, *), same shape as the input
    """
    p: float
    inplace: bool
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class BatchNorm1d(_Module):
    """Applies Batch Normalization over a 2D or 3D input.
    
    Args:
        num_features: Number of features or channels
        eps: Small constant added to denominator for numerical stability
        momentum: Value used for running_mean and running_var computation
        affine: If True, learnable affine parameters are used
        track_running_stats: If True, running statistics are tracked
        
    Shape:
        - Input: (N, C) or (N, C, L)
        - Output: Same shape as input
        
    Attributes:
        running_mean: Running mean of batches
        running_var: Running variance of batches
    """
    num_features: int
    eps: float
    momentum: float
    affine: bool
    track_running_stats: bool
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class Parameter(Tensor):
    """A kind of Tensor that is to be considered a module parameter.
    
    Parameters are Tensor subclasses, that have a very special property when used with Module s
    - when they're assigned as Module attributes they are automatically added to the list of
    its parameters, and will appear e.g. in parameters() iterator.
    """
    def __init__(self, data: Tensor, requires_grad: bool = True) -> None: ...

class MultiheadAttention(_Module):
    """Multi-head attention layer.
    
    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of parallel attention heads
        dropout: Dropout probability on attention weights
        bias: Add bias as module parameter
        add_bias_kv: Add bias to the key and value sequences at dim=0
        add_zero_attn: Add a new batch of zeros to the key and value sequences
        kdim: Total number of features for keys (default: embed_dim)
        vdim: Total number of features for values (default: embed_dim)
        
    Shape:
        - query: (L, N, E) where L is sequence length, N is batch size, E is embedding dimension
        - key: (S, N, E) where S is source sequence length
        - value: (S, N, E)
        - Output: (L, N, E)
    """
    embed_dim: int
    num_heads: int
    dropout: float
    bias: bool
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None
    ) -> None: ...
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]: ... 