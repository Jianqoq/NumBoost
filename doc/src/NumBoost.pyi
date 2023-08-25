from typing import Any, List, Tuple

from numpy import ndarray

class Tensor:
    T: Tensor
    axis: int
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

    def astype(self, *args, **kwargs) -> Any: ...

    def backward(self, *args, **kwargs) -> Any: ...

    def permute(self, *args, **kwargs) -> Any: ...

    def reshape(self, *args, **kwargs) -> Any: ...

    def transpose(self, *args, **kwargs) -> Any: ...

    def __abs__(self) -> Any: ...

    def __add__(self, other) -> Any: ...

    def __and__(self, other) -> Any: ...

    def __divmod__(self, other) -> Any: ...

    def __float__(self) -> float: ...

    def __floordiv__(self, other) -> Any: ...

    def __getitem__(self, index) -> Any: ...

    def __iadd__(self, other) -> Any: ...

    def __iand__(self, other) -> Any: ...

    def __ifloordiv__(self, other) -> Any: ...

    def __ilshift__(self, other) -> Any: ...

    def __imatmul__(self, *args, **kwargs) -> Any: ...

    def __imod__(self, other) -> Any: ...

    def __imul__(self, other) -> Any: ...

    def __int__(self) -> int: ...

    def __invert__(self) -> Any: ...

    def __ior__(self, other) -> Any: ...

    def __ipow__(self, other) -> Any: ...

    def __irshift__(self, other) -> Any: ...

    def __isub__(self, other) -> Any: ...

    def __iter__(self) -> Any: ...

    def __itruediv__(self, other) -> Any: ...

    def __ixor__(self, other) -> Any: ...

    def __len__(self) -> int: ...

    def __lshift__(self, other) -> Any: ...

    def __matmul__(self, *args, **kwargs) -> Any: ...

    def __mod__(self, other) -> Any: ...

    def __mul__(self, other) -> Any: ...

    def __neg__(self) -> Any: ...

    def __next__(self) -> Any: ...

    def __or__(self, other) -> Any: ...

    def __pos__(self) -> Any: ...

    def __pow__(self, other) -> Any: ...

    def __radd__(self, other) -> Any: ...

    def __rand__(self, other) -> Any: ...

    def __rdivmod__(self, other) -> Any: ...

    def __rfloordiv__(self, other) -> Any: ...

    def __rlshift__(self, other) -> Any: ...

    def __rmatmul__(self, *args, **kwargs) -> Any: ...

    def __rmod__(self, other) -> Any: ...

    def __rmul__(self, other) -> Any: ...

    def __ror__(self, other) -> Any: ...

    def __rpow__(self, other) -> Any: ...

    def __rrshift__(self, other) -> Any: ...

    def __rshift__(self, other) -> Any: ...

    def __rsub__(self, other) -> Any: ...

    def __rtruediv__(self, other) -> Any: ...

    def __rxor__(self, other) -> Any: ...

    def __sub__(self, other) -> Any: ...

    def __truediv__(self, other) -> Any: ...

    def __xor__(self, other) -> Any: ...

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
def tensordot(a, b, axes: int|list|tuple) -> Tensor: ...
def set_track(val: int) -> None: ...
def jit_wrapper(*args, **kwargs) -> Any: ...
def to_dict(*args, **kwargs) -> Any: ...
def global_float_type(*args, **kwargs) -> Any: ...
def result_type(op: int, a_dtype: int, a_elsize: int, b_dtype: int, b_elsize: int) -> int: ...
def tensor(array: list|tuple|ndarray, requires_grad: bool = False) -> Tensor: ...