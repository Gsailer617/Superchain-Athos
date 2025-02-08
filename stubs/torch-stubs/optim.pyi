from typing import Any, Optional, Union, List, Dict, Iterator, Callable, Tuple
from torch import Tensor
from torch.nn import Parameter

class Optimizer:
    param_groups: List[Dict[str, Any]]
    state: Dict[Parameter, Dict[str, Any]]
    defaults: Dict[str, Any]

    def __init__(self, params: Iterator[Parameter], defaults: Dict[str, Any]) -> None: ...
    def zero_grad(self, set_to_none: bool = False) -> None: ...
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]: ...
    def add_param_group(self, param_group: Dict[str, Any]) -> None: ...
    def state_dict(self) -> Dict[str, Any]: ...
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None: ...

class Adam(Optimizer):
    def __init__(
        self,
        params: Iterator[Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False
    ) -> None: ...

class SGD(Optimizer):
    def __init__(
        self,
        params: Iterator[Parameter],
        lr: float,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False
    ) -> None: ...

class RMSprop(Optimizer):
    def __init__(
        self,
        params: Iterator[Parameter],
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum: float = 0,
        centered: bool = False
    ) -> None: ...

class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterator[Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False
    ) -> None: ...

class Adagrad(Optimizer):
    def __init__(
        self,
        params: Iterator[Parameter],
        lr: float = 1e-2,
        lr_decay: float = 0,
        weight_decay: float = 0,
        initial_accumulator_value: float = 0,
        eps: float = 1e-10
    ) -> None: ... 