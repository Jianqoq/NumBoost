from src.Numboost import *
from typing import Callable, Union, Sequence, Iterable, Optional, Any
import jax
from jax._src import sharding_impls
from jax._src.lax.lax import xc
from contextlib import contextmanager
from NumBoost_type import *
import numpy as _np
from NumBoost_openum import *

cache = {}

nb_type_2_np = {uint: _np.dtype(_np.uint), bool_: _np.dtype(_np.bool_), byte: _np.dtype(_np.byte),
                ubyte: _np.dtype(_np.ubyte), short: _np.dtype(_np.short), ushort: _np.dtype(_np.ushort),
                int_: _np.dtype(_np.int_), uint32: _np.dtype(_np.uint32), int32: _np.dtype(_np.int32),
                uint64: _np.dtype(_np.uint64), int64: _np.dtype(_np.int64), float32: _np.dtype(_np.float32),
                float64: _np.dtype(_np.float64), float16: _np.dtype(_np.float16), double: _np.dtype(_np.double),
                longdouble: _np.dtype(_np.longdouble), longlong: _np.dtype(_np.longlong),
                ulonglong: _np.dtype(_np.ulonglong)}

np_type_2_nb = {_np.dtype(_np.uint): uint, _np.dtype(_np.bool_): bool_, _np.dtype(_np.byte): byte,
                _np.dtype(_np.ubyte): ubyte, _np.dtype(_np.short): short, _np.dtype(_np.ushort): ushort,
                _np.dtype(_np.int_): int_, _np.dtype(_np.uint32): uint32, _np.dtype(_np.int32): int32,
                _np.dtype(_np.uint64): uint64, _np.dtype(_np.int64): int64, _np.dtype(_np.float32): float32,
                _np.dtype(_np.float64): float64, _np.dtype(_np.float16): float16, _np.dtype(_np.double): double,
                _np.dtype(_np.longdouble): longdouble, _np.dtype(_np.longlong): longlong}

pass
def jit(fun: Callable,
        in_shardings=sharding_impls.UNSPECIFIED,
        out_shardings=sharding_impls.UNSPECIFIED,
        static_argnums: Union[int, Sequence[int], None] = None,
        static_argnames: Union[str, Iterable[str], None] = None,
        donate_argnums: Union[int, Sequence[int]] = (),
        keep_unused: bool = False,
        device: Optional[xc.Device] = None,
        backend: Optional[str] = None,
        inline: bool = False,
        abstracted_axes: Optional[Any] = None):
    """Wrapper for jax.jit to support NumBoost.Tensor"""

    def wrapper(*args):
        if fun not in cache:
            cache[fun] = jax.jit(fun, in_shardings, out_shardings,
                                 static_argnums, static_argnames,
                                 donate_argnums, keep_unused, device,
                                 backend, inline, abstracted_axes)
        set_track(1)
        args = [arg.data if isinstance(arg, Tensor) else arg for arg in args]
        result = cache[fun](*args)
        set_track(0)
        return result

    return wrapper


def track(fun: Callable):
    """Enable tracking for Tensor in the function(Normally for jax jit or make_jaxpr)"""

    def wrapper(*args):
        set_track(1)
        args = tuple(arg.data if isinstance(
            arg, Tensor) else arg for arg in args)
        result = fun(*args)
        set_track(0)
        return result

    return wrapper


@contextmanager
def enable_track():
    set_track(1)
    try:
        yield
    finally:
        set_track(0)
