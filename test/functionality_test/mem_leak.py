import sys

import jax
import numpy as np
import os
import pytest
import platform

if platform.system() == 'Windows':
    os.add_dll_directory(r'C:\Program Files (x86)\Intel\oneAPI\mkl\2023.1.0\redist\intel64')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from num_boost import Tensor, set_track, to_dict
import num_boost as core


@pytest.mark.parametrize("array, array2, grad", [(np.array([1., 2., 3.]), np.array([5., 7., 8.]), np.array([10., 9., 5.])),])
def test_C_Tensor_addition_backward(array, array2, grad):
    for i in range(10000000):
        autograd_grad_operands1 = Tensor(array, True)
        autograd_grad_operands2 = Tensor(array2)
        autograd_grad = grad
        result1 = autograd_grad_operands1 + autograd_grad_operands2
        set_track(1)
        res = jax.jit(result1.backward)(autograd_grad)
        set_track(0)


@pytest.mark.parametrize("array, array2, grad", [(np.array([1., 2., 3.]), np.array([5., 7., 8.]), np.array([10., 9., 5.])),])
def test_C_Tensor_subtraction_backward(array, array2, grad):
    for i in range(10000000):
        autograd_grad_operands1 = Tensor(array, True)
        autograd_grad_operands2 = Tensor(array2)
        autograd_grad = grad
        result1 = autograd_grad_operands1 - autograd_grad_operands2
        set_track(1)
        res = jax.jit(result1.backward)(autograd_grad)
        set_track(0)


@pytest.mark.parametrize("array, array2, grad", [(np.array([1., 2., 3.]), np.array([5., 7., 8.]), np.array([10., 9., 5.])),])
def test_C_Tensor_division_backward(array, array2, grad):
    for i in range(10000000):
        autograd_grad_operands1 = Tensor(array, True)
        autograd_grad_operands2 = Tensor(array2)
        autograd_grad = grad
        result1 = autograd_grad_operands1 / autograd_grad_operands2
        set_track(1)
        res = jax.jit(result1.backward)(autograd_grad)
        set_track(0)


@pytest.mark.parametrize("array, array2, grad", [(np.array([[1.], [2.], [3.]]), np.array([[5., 7., 8.]]), np.random.random((3, 3)))])
def test_C_Tensor_matmul_backward(array, array2, grad):
    for i in range(10000000):
        autograd_grad_operands1 = Tensor(array, True)
        autograd_grad_operands2 = Tensor(array2)
        autograd_grad = grad
        result1 = autograd_grad_operands1 @ autograd_grad_operands2
        set_track(1)
        res = jax.jit(result1.backward)(autograd_grad)
        set_track(0)


@pytest.mark.parametrize("array, grad", [(np.array([1., 2., 3.]), np.random.random((3, )))])
def test_C_Tensor_negative_backward(array, grad):
    for i in range(10000000):
        p = Tensor(array, True)
        result1 = -p
        set_track(1)
        to_dict(jax.jit(result1.backward)(grad))
        set_track(0)


@pytest.mark.parametrize("array, array2, grad", [(np.array([[1.], [2.], [3.]]), np.array([[5., 7., 8.]]), np.random.random((3, 3)))])
def test_C_Tensor_mul_backward(array, array2, grad):
    for i in range(10000000):
        autograd_grad_operands1 = Tensor(array, True)
        autograd_grad_operands2 = Tensor(array2)
        autograd_grad = grad
        result1 = autograd_grad_operands1 * autograd_grad_operands2
        set_track(1)
        res = jax.jit(result1.backward)(autograd_grad)
        set_track(0)


@pytest.mark.parametrize("array, grad", [(np.array([[1.], [2.], [3.]]), np.random.random((3, 1)))])
def test_C_Tensor_sin_backward(array, grad):
    for i in range(10000000):
        autograd_grad_operands1 = Tensor(array, True)
        autograd_grad = grad
        result1 = core.sin(autograd_grad_operands1)
        set_track(1)
        res = jax.jit(result1.backward)(autograd_grad)
        set_track(0)


@pytest.mark.parametrize("array, grad", [(np.array([[1.], [2.], [3.]]), np.random.random((3, 1)))])
def test_C_Tensor_cos_backward(array, grad):
    for i in range(10000000):
        autograd_grad_operands1 = Tensor(array, True)
        autograd_grad = grad
        result1 = core.cos(autograd_grad_operands1)
        set_track(1)
        res = jax.jit(result1.backward)(autograd_grad)
        set_track(0)


@pytest.mark.parametrize("array, grad", [(np.array([[1.], [2.], [3.]]), np.random.random((3, 1)))])
def test_C_Tensor_tan_backward(array, grad):
    for i in range(10000000):
        autograd_grad_operands1 = Tensor(array, True)
        autograd_grad = grad
        result1 = core.tan(autograd_grad_operands1)
        set_track(1)
        res = jax.jit(result1.backward)(autograd_grad)
        set_track(0)


@pytest.mark.parametrize("array, grad", [(np.array([[1.], [2.], [3.]]), np.random.random((3, 1)))])
def test_C_Tensor_sinh_backward(array, grad):
    for i in range(10000000):
        autograd_grad_operands1 = Tensor(array, True)
        autograd_grad = grad
        result1 = core.sinh(autograd_grad_operands1)
        set_track(1)
        res = jax.jit(result1.backward)(autograd_grad)
        set_track(0)


@pytest.mark.parametrize("array, grad", [(np.array([[1.], [2.], [3.]]), np.random.random((3, 1)))])
def test_C_Tensor_cosh_backward(array, grad):
    for i in range(10000000):
        autograd_grad_operands1 = Tensor(array, True)
        autograd_grad = grad
        result1 = core.cosh(autograd_grad_operands1)
        set_track(1)
        res = jax.jit(result1.backward)(autograd_grad)
        set_track(0)


@pytest.mark.parametrize("array, grad", [(np.array([[1.], [2.], [3.]]), np.random.random((3, 1)))])
def test_C_Tensor_tanh_backward(array, grad):
    for i in range(10000000):
        autograd_grad_operands1 = Tensor(array, True)
        autograd_grad = grad
        result1 = core.tanh(autograd_grad_operands1)
        set_track(1)
        res = jax.jit(result1.backward)(autograd_grad)
        set_track(0)


@pytest.mark.parametrize("array, grad", [(np.array([[1.], [2.], [3.]]), np.random.random((3, 1)))])
def test_C_Tensor_arcsin_backward(array, grad):
    for i in range(10000000):
        autograd_grad_operands1 = Tensor(array, True)
        autograd_grad = grad
        result1 = core.arcsin(autograd_grad_operands1)
        set_track(1)
        res = jax.jit(result1.backward)(autograd_grad)
        set_track(0)


@pytest.mark.parametrize("array, grad", [(np.array([[1.], [2.], [3.]]), np.random.random((3, 1)))])
def test_C_Tensor_arctan_backward(array, grad):
    for i in range(10000000):
        autograd_grad_operands1 = Tensor(array, True)
        autograd_grad = grad
        result1 = core.arctan(autograd_grad_operands1)
        set_track(1)
        res = jax.jit(result1.backward)(autograd_grad)
        set_track(0)


@pytest.mark.parametrize("array, grad", [(np.array([[1.], [2.], [3.]]), np.random.random((3, 1)))])
def test_C_Tensor_arcsinh_backward(array, grad):
    for i in range(10000000):
        autograd_grad_operands1 = Tensor(array, True)
        autograd_grad = grad
        result1 = core.arcsinh(autograd_grad_operands1)
        set_track(1)
        res = jax.jit(result1.backward)(autograd_grad)
        set_track(0)


@pytest.mark.parametrize("array, grad", [(np.array([[1.], [2.], [3.]]), np.random.random((3, 1)))])
def test_C_Tensor_power_backward(array, grad):
    for i in range(10000000):
        autograd_grad_operands1 = Tensor(array, True)
        autograd_grad = grad
        result1 = core.power(autograd_grad_operands1, 3)
        result1.backward(autograd_grad)


@pytest.mark.parametrize("array, grad", [(np.array([[1.], [2.], [3.]]), np.random.random((3, 1)))])
def test_C_Tensor_sqrt_backward(array, grad):
    for i in range(10000000):
        autograd_grad_operands1 = Tensor(array, True)
        autograd_grad = grad
        result1 = core.sqrt(autograd_grad_operands1)
        set_track(1)
        res = jax.jit(result1.backward)(autograd_grad)
        set_track(0)


@pytest.mark.parametrize("array, grad", [(np.array([[1.], [2.], [3.]]), np.random.random((3,)))])
def test_C_Tensor_reshape_backward(array, grad):
    for i in range(10000000):
        autograd_grad_operands1 = Tensor(array, True)
        autograd_grad = grad
        result1 = core.reshape(autograd_grad_operands1, (3, ))
        set_track(1)
        res = jax.jit(result1.backward)(autograd_grad)
        set_track(0)


@pytest.mark.parametrize("array, grad", [(np.array([[1.], [2.], [3.]]), np.random.random((3, 1)))])
def test_C_Tensor_log_backward(array, grad):
    for i in range(10000000):
        autograd_grad_operands1 = Tensor(array, True)
        autograd_grad = grad
        result1 = core.log(autograd_grad_operands1)
        set_track(1)
        res = jax.jit(result1.backward)(autograd_grad)
        set_track(0)

