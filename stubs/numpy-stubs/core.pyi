from typing import (
    Any, Callable, Iterator, List, Optional, Sequence,
    Tuple, TypeVar, Union, overload, Generic
)
from typing_extensions import Protocol, Literal

Shape = Tuple[int, ...]
DType = TypeVar('DType')
ShapeType = TypeVar('ShapeType', bound=Tuple[int, ...])
ArrayLike = Union['ndarray[Any, Any]', float, int, bool, complex, str, bytes, List[Any]]

class ndarray(Generic[DType, ShapeType]):
    """N-dimensional array class.
    
    Attributes:
        shape: Tuple of array dimensions
        dtype: Data-type of the array's elements
        size: Number of elements in the array
        ndim: Number of array dimensions
        T: Same array with axes transposed
    """
    shape: ShapeType
    dtype: DType
    size: int
    ndim: int
    T: 'ndarray[DType, ShapeType]'
    
    def __init__(self, shape: Shape, dtype: Optional[DType] = None) -> None: ...
    
    @overload
    def __getitem__(self, key: int) -> Union[Any, 'ndarray[DType, ShapeType]']: ...
    @overload
    def __getitem__(self, key: slice) -> 'ndarray[DType, ShapeType]': ...
    @overload
    def __getitem__(self, key: Tuple[Union[int, slice], ...]) -> 'ndarray[DType, ShapeType]': ...
    
    def __setitem__(self, key: Union[int, slice, Tuple[Union[int, slice], ...]], value: ArrayLike) -> None: ...
    
    # Array operations
    def reshape(self, shape: Shape) -> 'ndarray[DType, ShapeType]': ...
    def transpose(self, *axes: int) -> 'ndarray[DType, ShapeType]': ...
    def astype(self, dtype: Any, copy: bool = True) -> 'ndarray[Any, ShapeType]': ...
    
    # Math operations
    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'ndarray[DType, ShapeType]': ...
    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'ndarray[DType, ShapeType]': ...
    def std(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'ndarray[DType, ShapeType]': ...
    def min(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'ndarray[DType, ShapeType]': ...
    def max(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'ndarray[DType, ShapeType]': ...
    
    # Comparison operations
    def __lt__(self, other: ArrayLike) -> 'ndarray[bool, ShapeType]': ...
    def __le__(self, other: ArrayLike) -> 'ndarray[bool, ShapeType]': ...
    def __gt__(self, other: ArrayLike) -> 'ndarray[bool, ShapeType]': ...
    def __ge__(self, other: ArrayLike) -> 'ndarray[bool, ShapeType]': ...
    def __eq__(self, other: ArrayLike) -> 'ndarray[bool, ShapeType]': ...  # type: ignore
    def __ne__(self, other: ArrayLike) -> 'ndarray[bool, ShapeType]': ...  # type: ignore

# Array creation functions
def array(
    object: ArrayLike,
    dtype: Optional[Any] = None,
    copy: bool = True,
    order: str = 'K',
    subok: bool = False,
    ndmin: int = 0
) -> 'ndarray[Any, Any]': ...

def zeros(
    shape: Union[int, Shape],
    dtype: Optional[Any] = None,
    order: str = 'C'
) -> 'ndarray[Any, Any]': ...

def ones(
    shape: Union[int, Shape],
    dtype: Optional[Any] = None,
    order: str = 'C'
) -> 'ndarray[Any, Any]': ...

def arange(
    start: Union[int, float],
    stop: Optional[Union[int, float]] = None,
    step: Union[int, float] = 1,
    dtype: Optional[Any] = None
) -> 'ndarray[Any, Any]': ...

def linspace(
    start: Union[int, float],
    stop: Union[int, float],
    num: int = 50,
    endpoint: bool = True,
    retstep: bool = False,
    dtype: Optional[Any] = None,
    axis: int = 0
) -> Union['ndarray[Any, Any]', Tuple['ndarray[Any, Any]', float]]: ... 