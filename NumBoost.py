from src.Numboost import *
from typing import Callable, Union, Sequence, Iterable, Optional, Any
import jax
from jax._src import sharding_impls
from jax._src.lax.lax import xc
from contextlib import contextmanager
cache = {}


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
        args = tuple(arg.data if isinstance(arg, Tensor) else arg for arg in args)
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