from typing import Any, List, Tuple, Iterable, Optional

from numpy import ndarray

class TensorIterator:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...
    def __next__(self) -> Any: ...

class Tensor:
    T: Tensor
    axis: int
    dtype: int
    data: ndarray
    depth: int
    dim: int
    grad: ndarray
    grad_fn: str
    graph: Any
    has_conv: int
    require_grad: bool
    x: Tensor
    y: Any

    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...

    def astype(self, *args, **kwargs) -> Tensor: ...

    def backward(self, *args, **kwargs) -> Any: ...

    def permute(self, *args, **kwargs) -> Tensor: ...

    def reshape(self, *args, **kwargs) -> Tensor: ...

    def transpose(self, *args, **kwargs) -> Tensor: ...

    def __abs__(self) -> Tensor: ...
    def __add__(self, other) -> Tensor: ...
    def __and__(self, other) -> Tensor: ...
    def __divmod__(self, other) -> Tensor: ...
    def __eq__(self, other) -> Tensor: ...
    def __float__(self) -> float: ...
    def __floordiv__(self, other) -> Tensor: ...
    def __ge__(self, other) -> Tensor: ...
    def __getitem__(self, index) -> Tensor: ...
    def __gt__(self, other) -> Tensor: ...
    def __hash__(self) -> int: ...
    def __iadd__(self, other) -> Tensor: ...
    def __iand__(self, other) -> Tensor: ...
    def __ifloordiv__(self, other) -> Tensor: ...
    def __ilshift__(self, other) -> Tensor: ...
    def __imatmul__(self, *args, **kwargs) -> Tensor: ...
    def __imod__(self, other) -> Tensor: ...
    def __imul__(self, other) -> Tensor: ...
    def __int__(self) -> int: ...
    def __invert__(self) -> Tensor: ...
    def __ior__(self, other) -> Tensor: ...
    def __ipow__(self, other) -> Tensor: ...
    def __irshift__(self, other) -> Tensor: ...
    def __isub__(self, other) -> Tensor: ...
    def __iter__(self) -> Any: ...
    def __itruediv__(self, other) -> Tensor: ...
    def __ixor__(self, other) -> Tensor: ...
    def __le__(self, other) -> Tensor: ...
    def __len__(self) -> int: ...
    def __lshift__(self, other) -> Tensor: ...
    def __lt__(self, other) -> Tensor: ...
    def __matmul__(self, *args, **kwargs) -> Tensor: ...
    def __mod__(self, other) -> Tensor: ...
    def __mul__(self, other) -> Tensor: ...
    def __ne__(self, other) -> Tensor: ...
    def __neg__(self) -> Tensor: ...
    def __or__(self, other) -> Tensor: ...
    def __pos__(self) -> Tensor: ...
    def __pow__(self, other) -> Tensor: ...
    def __radd__(self, other) -> Tensor: ...
    def __rand__(self, other) -> Tensor: ...
    def __rdivmod__(self, other) -> Tensor: ...
    def __rfloordiv__(self, other) -> Tensor: ...
    def __rlshift__(self, other) -> Tensor: ...
    def __rmatmul__(self, *args, **kwargs) -> Tensor: ...
    def __rmod__(self, other) -> Tensor: ...
    def __rmul__(self, other) -> Tensor: ...
    def __ror__(self, other) -> Tensor: ...
    def __rpow__(self, other) -> Tensor: ...
    def __rrshift__(self, other) -> Tensor: ...
    def __rshift__(self, other) -> Tensor: ...
    def __rsub__(self, other) -> Tensor: ...
    def __rtruediv__(self, other) -> Tensor: ...
    def __rxor__(self, other) -> Tensor: ...
    def __sub__(self, other) -> Tensor: ...
    def __truediv__(self, other) -> Tensor: ...
    def __xor__(self, other) -> Tensor: ...
    def __next__(self) -> TensorIterator: ...


def where(mask: Tensor, x = None,
          y = None, exact_indice=False) -> Tensor: ...
def arange(start: Optional[int]=None, *args: Any, **kwargs: Any) -> Tensor: ...
def add(a:Tensor, b:Tensor, out:Tensor) -> Tensor: ...
def sub(a:Tensor, b:Tensor, out:Tensor) -> Tensor: ...
def mul(a:Tensor, b:Tensor, out:Tensor) -> Tensor: ...
def div(a:Tensor, b:Tensor, out:Tensor) -> Tensor: ...
def mod(a:Tensor, out:Tensor) -> Tensor: ...
def fdiv(a:Tensor, out:Tensor) -> Tensor: ...
def abs(a:Tensor, out:Tensor) -> Tensor: ...
def arccos(a:Tensor, out:Tensor) -> Tensor: ...
def arccosh(a:Tensor, out:Tensor) -> Tensor: ...
def arcsin(a:Tensor, out:Tensor) -> Tensor: ...
def arcsinh(a:Tensor, out:Tensor) -> Tensor: ...
def arctan(a:Tensor, out:Tensor) -> Tensor: ...
def arctanh(a:Tensor, out:Tensor) -> Tensor: ...
def argmax(a:Tensor, axis:int|Iterable|tuple, keepdims:Optional[bool]=None, out:Optional[Tensor]=None) -> Tensor: ...
def argmin(a:Tensor, axis:int|Iterable|tuple, keepdims:Optional[bool]=None, out:Optional[Tensor]=None) -> Tensor: ...
def cos(a:Tensor, out:Tensor) -> Tensor: ...
def cosh(a:Tensor, out:Tensor) -> Tensor: ...
def exp(a:Tensor, out:Tensor) -> Tensor: ...
def log(a:Tensor, out:Tensor) -> Tensor: ...
def log10(a:Tensor, out:Tensor) -> Tensor: ...
def max(a:Tensor, axis:int|Iterable|tuple, keepdims:Optional[bool]=None, out:Optional[Tensor]=None) -> Tensor: ...
def mean(a:Tensor, axis:int|Iterable|tuple, keepdims:Optional[bool]=None, out:Optional[Tensor]=None) -> Tensor: ...
def min(a:Tensor, axis:int|Iterable|tuple, keepdims:Optional[bool]=None, out:Optional[Tensor]=None) -> Tensor: ...
def power(*args, **kwargs) -> Tensor: ...
def reshape(*args, **kwargs) -> Tensor: ...
def sin(a:Tensor, out:Tensor) -> Tensor: ...
def sinh(a:Tensor, out:Tensor) -> Tensor: ...
def sqrt(a:Tensor, out:Tensor) -> Tensor: ...
def sum(a:Tensor, axis:int|Iterable|tuple, keepdims:Optional[bool]=None, out:Optional[Tensor]=None) -> Tensor: ...
def tan(a:Tensor, out:Tensor) -> Tensor: ...
def tanh(a:Tensor, out:Tensor) -> Tensor: ...
def transpose(*args, **kwargs) -> Tensor: ...
def tensordot(*args, **kwargs) -> Tensor: ...
def set_track(val: int) -> None: ...
def jit_wrapper(*args, **kwargs) -> Any: ...
def to_dict(*args, **kwargs) -> Any: ...
def global_float_type(*args, **kwargs) -> Any: ...
def result_type(op: int, a_dtype: int, a_elsize: int, b_dtype: int, b_elsize: int) -> int: ...
def tensor(array: list|tuple|ndarray, requires_grad: bool = False) -> Tensor: ...