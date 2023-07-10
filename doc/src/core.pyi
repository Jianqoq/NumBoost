from typing import Tuple, Iterable

import numpy as np

from tensor import Tensor

def abs(a: Tensor, out: Tensor|None=None) -> Tensor:
    pass
def arccos(a: Tensor, out: Tensor|None=None) -> Tensor:
    pass
def arccosh(a: Tensor, out: Tensor|None=None) -> Tensor:
    pass
def arcsin(a: Tensor, out: Tensor|None=None) -> Tensor:
    pass
def arcsinh(a: Tensor, out: Tensor|None=None) -> Tensor:
    pass
def arctan(a: Tensor, out: Tensor|None=None) -> Tensor:
    pass
def arctanh(a: Tensor, out: Tensor|None=None) -> Tensor:
    pass
def argmax(a: Tensor, axis:int, keepdims=False, out: Tensor|None=None) -> Tensor:
    pass
def argmin(a: Tensor, axis:int, keepdims=False, out: Tensor|None=None) -> Tensor:
    pass
def cos(a: Tensor, out: Tensor|None=None) -> Tensor:
    pass
def cosh(a: Tensor, out: Tensor|None=None) -> Tensor:
    pass
def exp(a: Tensor, out: Tensor|None=None) -> Tensor:
    pass
def log(a: Tensor, out: Tensor|None=None) -> Tensor:
    pass
def log10(a: Tensor, out: Tensor|None=None) -> Tensor:
    pass
def max(a: Tensor, axis:int, keepdims=False, out: Tensor|None=None) -> Tensor:
    pass
def mean(a: Tensor, axis:int, out: Tensor|None=None, dtype: np.dtype = None, keepdims=False) -> Tensor:
    pass
def min(a: Tensor, axis:int, keepdims=False, out: Tensor|None=None) -> Tensor:
    pass
def power(a: Tensor, out: Tensor|None=None) -> Tensor:
    pass
def reshape(a: Tensor, out: Tensor|None=None) -> Tensor:
    pass
def sin(a: Tensor, out: Tensor|None=None) -> Tensor:
    pass
def sinh(a: Tensor, out: Tensor|None=None) -> Tensor:
    pass
def sqrt(a: Tensor, out: Tensor|None=None) -> Tensor:
    pass
def sum(a: Tensor,
        axis: None | int | Iterable | tuple[int] = None,
        dtype: object | None = None,
        out: Tensor | None = None,
        keepdims: bool | None = None) -> Tensor:
    pass
def tan(a: Tensor, out: Tensor|None=None) -> Tensor:
    pass
def tanh(a: Tensor, out: Tensor|None=None) -> Tensor:
    pass
def transpose(a: Tensor, axis: Tuple) -> Tensor:
    pass
