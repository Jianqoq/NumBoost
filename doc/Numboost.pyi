from typing import Any, Iterable, List, Tuple

from numpy import ndarray


class Tensor:
    T: Tensor
    axis: int
    dtype: ndarray.dtype
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
    def __init__(self, data: List|Tuple|ndarray, requires_grad=False) -> None: ...
    def backward(self, grad: Tensor|ndarray) -> None: ...
    def permute(self, *axis) -> Tensor: ...
    def transpose(self, *axis) -> Tensor: ...
    def reshape(self, *axis, order) -> Tensor: ...
    def __abs__(self) -> Tensor: ...
    def __add__(self, other) -> Tensor: ...
    def __and__(self, other) -> Tensor: ...
    def __divmod__(self, other) -> Any: ...
    def __float__(self) -> float: ...
    def __floordiv__(self, other) -> Tensor: ...
    def __iadd__(self, other) -> Tensor: ...
    def __iand__(self, other) -> Tensor: ...
    def __ifloordiv__(self, other) -> Tensor: ...
    def __ilshift__(self, other) -> Tensor: ...
    def __imatmul__(self, *args, **kwargs) -> Tensor: ...
    def __imod__(self, other) -> Tensor: ...
    def __imul__(self, other) -> Tensor: ...
    def __int__(self) -> int: ...
    def __invert__(self) -> Tensor: ...
    def __ior__(self, other) -> Any: ...
    def __ipow__(self, other) -> Any: ...
    def __irshift__(self, other) -> Any: ...
    def __isub__(self, other) -> Any: ...
    def __itruediv__(self, other) -> Any: ...
    def __ixor__(self, other) -> Any: ...
    def __lshift__(self, other) -> Tensor: ...
    def __matmul__(self, *args, **kwargs) -> Tensor: ...
    def __mod__(self, other) -> Any: ...
    def __mul__(self, other) -> Tensor: ...
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

def abs(*args, **kwargs) -> Tensor: ...
def arccos(*args, **kwargs) -> Tensor: ...
def arccosh(*args, **kwargs) -> Tensor: ...
def arcsin(*args, **kwargs) -> Tensor: ...
def arcsinh(*args, **kwargs) -> Tensor: ...
def arctan(*args, **kwargs) -> Tensor: ...
def arctanh(*args, **kwargs) -> Tensor: ...
def argmax(*args, **kwargs) -> Tensor: ...
def argmin(*args, **kwargs) -> Tensor: ...
def cos(*args, **kwargs) -> Tensor: ...
def cosh(*args, **kwargs) -> Tensor: ...
def exp(*args, **kwargs) -> Tensor: ...
def log(*args, **kwargs) -> Tensor: ...
def log10(*args, **kwargs) -> Tensor: ...
def max(*args, **kwargs) -> Tensor: ...
def mean(*args, **kwargs) -> Tensor: ...
def min(*args, **kwargs) -> Tensor: ...
def power(*args, **kwargs) -> Tensor: ...
def reshape(*args, **kwargs) -> Tensor: ...
def sin(*args, **kwargs) -> Tensor: ...
def sinh(*args, **kwargs) -> Tensor: ...
def sqrt(*args, **kwargs) -> Tensor: ...
def sum(*args, **kwargs) -> Tensor: ...
def tan(*args, **kwargs) -> Tensor: ...
def tanh(*args, **kwargs) -> Tensor: ...
def transpose(*args, **kwargs) -> Tensor: ...
def tensordot(*args, **kwargs) -> Tensor: ...
def set_track(val: int) -> None: ...
def jit_wrapper(*args, **kwargs) -> Any: ...
def to_dict(*args, **kwargs) -> Any: ...