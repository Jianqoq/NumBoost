import numpy as np
import torch
import os
import sys
import pytest
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.tensor import Tensor
import src.core as core

@pytest.mark.parametrize("array, array2, grad", [(np.array([1., 2., 3.]), np.array([5., 7., 8.]), np.array([10., 9., 5.])),])
def test_C_Tensor_addition_backward(array, array2, grad):
    autograd_grad_operands1 = Tensor(array, True)
    autograd_grad_operands2 = Tensor(array2)
    autograd_grad = grad
    torch_operands1 = torch.tensor(array, requires_grad=True)
    torch_operands2 = torch.tensor(array2)
    torch_grad = torch.tensor(grad)
    result1 = autograd_grad_operands1 + autograd_grad_operands2
    result2 = torch_operands1 + torch_operands2
    result1.backward(autograd_grad)
    result2.backward(torch_grad)
    assert np.allclose(autograd_grad_operands1.grad, torch_operands1.grad.numpy()),\
        f"correct: {torch_operands1.grad.numpy()} | got: {autograd_grad_operands1.grad}"


@pytest.mark.parametrize("array, array2, grad", [(np.array([1., 2., 3.]), np.array([5., 7., 8.]), np.array([10., 9., 5.])),])
def test_C_Tensor_subtraction_backward(array, array2, grad):
    autograd_grad_operands1 = Tensor(array, True)
    autograd_grad_operands2 = Tensor(array2)
    autograd_grad = grad
    torch_operands1 = torch.tensor(array, requires_grad=True)
    torch_operands2 = torch.tensor(array2)
    torch_grad = torch.tensor(grad)
    result1 = autograd_grad_operands1 - autograd_grad_operands2
    result2 = torch_operands1 - torch_operands2
    result1.backward(autograd_grad)
    result2.backward(torch_grad)
    assert np.allclose(autograd_grad_operands1.grad, torch_operands1.grad.numpy()),\
        f"correct: {torch_operands1.grad.numpy()} | got: {autograd_grad_operands1.grad}"


@pytest.mark.parametrize("array, array2, grad", [(np.array([1., 2., 3.]), np.array([5., 7., 8.]), np.array([10., 9., 5.])),])
def test_C_Tensor_division_backward(array, array2, grad):
    autograd_grad_operands1 = Tensor(array, True)
    autograd_grad_operands2 = Tensor(array2)
    autograd_grad = grad
    torch_operands1 = torch.tensor(array, requires_grad=True)
    torch_operands2 = torch.tensor(array2)
    torch_grad = torch.tensor(grad)
    result1 = autograd_grad_operands1 / autograd_grad_operands2
    result2 = torch_operands1 / torch_operands2
    result1.backward(autograd_grad)
    result2.backward(torch_grad)
    assert np.allclose(autograd_grad_operands1.grad, torch_operands1.grad.numpy()),\
        f"correct: {torch_operands1.grad.numpy()} | got: {autograd_grad_operands1.grad}"


@pytest.mark.parametrize("array, array2, grad", [(np.array([[1.], [2.], [3.]]), np.array([[5., 7., 8.]]), np.random.random((3, 3)))])
def test_C_Tensor_matmul_backward(array, array2, grad):
    autograd_grad_operands1 = Tensor(array, True)
    autograd_grad_operands2 = Tensor(array2)
    autograd_grad = grad
    torch_operands1 = torch.tensor(array, requires_grad=True)
    torch_operands2 = torch.tensor(array2)
    torch_grad = torch.tensor(grad)
    result1 = autograd_grad_operands1 @ autograd_grad_operands2
    result2 = torch_operands1 @ torch_operands2
    result1.backward(autograd_grad)
    result2.backward(torch_grad)
    assert np.allclose(autograd_grad_operands1.grad, torch_operands1.grad.numpy()),\
        f"correct: {torch_operands1.grad.numpy()} | got: {autograd_grad_operands1.grad}"


@pytest.mark.parametrize("array, grad", [(np.array([1., 2., 3.]), np.random.random((3, )))])
def test_C_Tensor_negative_backward(array, grad):
    p = Tensor(array, True)
    result1 = -p
    torch_operands1 = torch.tensor(array, requires_grad=True)
    result2 = -torch_operands1
    torch_grad = torch.tensor(grad)
    result1.backward(grad)
    result2.backward(torch_grad)
    assert np.allclose(p.grad, torch_operands1.grad.numpy()), f"correct: {torch_operands1.grad.numpy()} | got: {p.grad}"


@pytest.mark.parametrize("array, array2, grad", [(np.array([[1.], [2.], [3.]]), np.array([[5., 7., 8.]]), np.random.random((3, 3)))])
def test_C_Tensor_mul_backward(array, array2, grad):
    autograd_grad_operands1 = Tensor(array, True)
    autograd_grad_operands2 = Tensor(array2)
    autograd_grad = grad
    torch_operands1 = torch.tensor(array, requires_grad=True)
    torch_operands2 = torch.tensor(array2)
    torch_grad = torch.tensor(grad)
    result1 = autograd_grad_operands1 * autograd_grad_operands2
    result2 = torch_operands1 * torch_operands2
    result1.backward(autograd_grad)
    result2.backward(torch_grad)
    assert np.allclose(autograd_grad_operands1.grad, torch_operands1.grad.numpy()),\
        f"correct: {torch_operands1.grad.numpy()} | got: {autograd_grad_operands1.grad}"


@pytest.mark.parametrize("array, grad", [(np.array([[1.], [2.], [3.]]), np.random.random((3, 1)))])
def test_C_Tensor_sin_backward(array, grad):
    autograd_grad_operands1 = Tensor(array, True)
    autograd_grad = grad
    torch_operands1 = torch.tensor(array, requires_grad=True)
    torch_grad = torch.tensor(grad)
    result1 = core.sin(autograd_grad_operands1)
    result2 = torch.sin(torch_operands1)
    result1.backward(autograd_grad)
    result2.backward(torch_grad)
    assert np.allclose(autograd_grad_operands1.grad, torch_operands1.grad.numpy()),\
        f"correct: {torch_operands1.grad.numpy()} | got: {autograd_grad_operands1.grad}"


@pytest.mark.parametrize("array, grad", [(np.array([[1.], [2.], [3.]]), np.random.random((3, 1)))])
def test_C_Tensor_cos_backward(array, grad):
    autograd_grad_operands1 = Tensor(array, True)
    autograd_grad = grad
    torch_operands1 = torch.tensor(array, requires_grad=True)
    torch_grad = torch.tensor(grad)
    result1 = core.cos(autograd_grad_operands1)
    result2 = torch.cos(torch_operands1)
    result1.backward(autograd_grad)
    result2.backward(torch_grad)
    assert np.allclose(autograd_grad_operands1.grad, torch_operands1.grad.numpy()),\
        f"correct: {torch_operands1.grad.numpy()} | got: {autograd_grad_operands1.grad}"


@pytest.mark.parametrize("array, grad", [(np.array([[1.], [2.], [3.]]), np.random.random((3, 1)))])
def test_C_Tensor_tan_backward(array, grad):
    autograd_grad_operands1 = Tensor(array, True)
    autograd_grad = grad
    torch_operands1 = torch.tensor(array, requires_grad=True)
    torch_grad = torch.tensor(grad)
    result1 = core.tan(autograd_grad_operands1)
    result2 = torch.tan(torch_operands1)
    result1.backward(autograd_grad)
    result2.backward(torch_grad)
    assert np.allclose(autograd_grad_operands1.grad, torch_operands1.grad.numpy()),\
        f"correct: {torch_operands1.grad.numpy()} | got: {autograd_grad_operands1.grad}"


@pytest.mark.parametrize("array, grad", [(np.array([[1.], [2.], [3.]]), np.random.random((3, 1)))])
def test_C_Tensor_sinh_backward(array, grad):
    autograd_grad_operands1 = Tensor(array, True)
    autograd_grad = grad
    torch_operands1 = torch.tensor(array, requires_grad=True)
    torch_grad = torch.tensor(grad)
    result1 = core.sinh(autograd_grad_operands1)
    result2 = torch.sinh(torch_operands1)
    result1.backward(autograd_grad)
    result2.backward(torch_grad)
    assert np.allclose(autograd_grad_operands1.grad, torch_operands1.grad.numpy()),\
        f"correct: {torch_operands1.grad.numpy()} | got: {autograd_grad_operands1.grad}"


@pytest.mark.parametrize("array, grad", [(np.array([[1.], [2.], [3.]]), np.random.random((3, 1)))])
def test_C_Tensor_cosh_backward(array, grad):
    autograd_grad_operands1 = Tensor(array, True)
    autograd_grad = grad
    torch_operands1 = torch.tensor(array, requires_grad=True)
    torch_grad = torch.tensor(grad)
    result1 = core.cosh(autograd_grad_operands1)
    result2 = torch.cosh(torch_operands1)
    result1.backward(autograd_grad)
    result2.backward(torch_grad)
    assert np.allclose(autograd_grad_operands1.grad, torch_operands1.grad.numpy()),\
        f"correct: {torch_operands1.grad.numpy()} | got: {autograd_grad_operands1.grad}"


@pytest.mark.parametrize("array, grad", [(np.array([[1.], [2.], [3.]]), np.random.random((3, 1)))])
def test_C_Tensor_tanh_backward(array, grad):
    autograd_grad_operands1 = Tensor(array, True)
    autograd_grad = grad
    torch_operands1 = torch.tensor(array, requires_grad=True)
    torch_grad = torch.tensor(grad)
    result1 = core.tanh(autograd_grad_operands1)
    result2 = torch.tanh(torch_operands1)
    result1.backward(autograd_grad)
    result2.backward(torch_grad)
    assert np.allclose(autograd_grad_operands1.grad, torch_operands1.grad.numpy()),\
        f"correct: {torch_operands1.grad.numpy()} | got: {autograd_grad_operands1.grad}"


@pytest.mark.parametrize("array, grad", [(np.array([[1.], [2.], [3.]]), np.random.random((3, 1)))])
def test_C_Tensor_arcsin_backward(array, grad):
    autograd_grad_operands1 = Tensor(array, True)
    autograd_grad = grad
    torch_operands1 = torch.tensor(array, requires_grad=True)
    torch_grad = torch.tensor(grad)
    result1 = core.arcsin(autograd_grad_operands1)
    result2 = torch.arcsin(torch_operands1)
    result1.backward(autograd_grad)
    result2.backward(torch_grad)
    assert np.allclose(autograd_grad_operands1.grad, torch_operands1.grad.numpy(), equal_nan=True),\
        f"correct: {torch_operands1.grad.numpy()} | got: {autograd_grad_operands1.grad}"


@pytest.mark.parametrize("array, grad", [(np.array([[1.], [2.], [3.]]), np.random.random((3, 1)))])
def test_C_Tensor_arccos_backward(array, grad):
    autograd_grad_operands1 = Tensor(array, True)
    autograd_grad = grad
    torch_operands1 = torch.tensor(array, requires_grad=True)
    torch_grad = torch.tensor(grad)
    result1 = core.arccos(autograd_grad_operands1)
    result2 = torch.arccos(torch_operands1)
    result1.backward(autograd_grad)
    result2.backward(torch_grad)
    assert np.allclose(autograd_grad_operands1.grad, torch_operands1.grad.numpy(), equal_nan=True),\
        f"correct: {torch_operands1.grad.numpy()} | got: {autograd_grad_operands1.grad}"


@pytest.mark.parametrize("array, grad", [(np.array([[1.], [2.], [3.]]), np.random.random((3, 1)))])
def test_C_Tensor_arctan_backward(array, grad):
    autograd_grad_operands1 = Tensor(array, True)
    autograd_grad = grad
    torch_operands1 = torch.tensor(array, requires_grad=True)
    torch_grad = torch.tensor(grad)
    result1 = core.arctan(autograd_grad_operands1)
    result2 = torch.arctan(torch_operands1)
    result1.backward(autograd_grad)
    result2.backward(torch_grad)
    assert np.allclose(autograd_grad_operands1.grad, torch_operands1.grad.numpy(), equal_nan=True), \
        f"correct: {torch_operands1.grad.numpy()} | got: {autograd_grad_operands1.grad}"


@pytest.mark.parametrize("array, grad", [(np.array([[1.], [2.], [3.]]), np.random.random((3, 1)))])
def test_C_Tensor_arcsinh_backward(array, grad):
    autograd_grad_operands1 = Tensor(array, True)
    autograd_grad = grad
    torch_operands1 = torch.tensor(array, requires_grad=True)
    torch_grad = torch.tensor(grad)
    result1 = core.arcsinh(autograd_grad_operands1)
    result2 = torch.arcsinh(torch_operands1)
    result1.backward(autograd_grad)
    result2.backward(torch_grad)
    assert np.allclose(autograd_grad_operands1.grad, torch_operands1.grad.numpy(), equal_nan=True), \
        f"correct: {torch_operands1.grad.numpy()} | got: {autograd_grad_operands1.grad}"


@pytest.mark.parametrize("array, grad", [(np.array([[1.], [2.], [3.]]), np.random.random((3, 1)))])
def test_C_Tensor_power_backward(array, grad):
    autograd_grad_operands1 = Tensor(array, True)
    autograd_grad = grad
    torch_operands1 = torch.tensor(array, requires_grad=True)
    torch_grad = torch.tensor(grad)
    result1 = core.power(autograd_grad_operands1, 3)
    result2 = torch.pow(torch_operands1, 3)
    result1.backward(autograd_grad)
    result2.backward(torch_grad)
    assert np.allclose(autograd_grad_operands1.grad, torch_operands1.grad.numpy(), equal_nan=True), \
        f"correct: {torch_operands1.grad.numpy()} | got: {autograd_grad_operands1.grad}"


@pytest.mark.parametrize("array, grad", [(np.array([[1.], [2.], [3.]]), np.random.random((3, 1)))])
def test_C_Tensor_power_backward(array, grad):
    autograd_grad_operands1 = Tensor(array, True)
    autograd_grad = grad
    torch_operands1 = torch.tensor(array, requires_grad=True)
    torch_grad = torch.tensor(grad)
    result1 = core.power(autograd_grad_operands1, 3)
    result2 = torch.pow(torch_operands1, 3)
    result1.backward(autograd_grad)
    result2.backward(torch_grad)
    assert np.allclose(autograd_grad_operands1.grad, torch_operands1.grad.numpy(), equal_nan=True), \
        f"correct: {torch_operands1.grad.numpy()} | got: {autograd_grad_operands1.grad}"


@pytest.mark.parametrize("array, grad", [(np.array([[1.], [2.], [3.]]), np.random.random((3, 1)))])
def test_C_Tensor_sqrt_backward(array, grad):
    autograd_grad_operands1 = Tensor(array, True)
    autograd_grad = grad
    torch_operands1 = torch.tensor(array, requires_grad=True)
    torch_grad = torch.tensor(grad)
    result1 = core.sqrt(autograd_grad_operands1)
    result2 = torch.sqrt(torch_operands1)
    result1.backward(autograd_grad)
    result2.backward(torch_grad)
    assert np.allclose(autograd_grad_operands1.grad, torch_operands1.grad.numpy(), equal_nan=True), \
        f"correct: {torch_operands1.grad.numpy()} | got: {autograd_grad_operands1.grad}"


@pytest.mark.parametrize("array, grad", [(np.array([[1.], [2.], [3.]]), np.random.random((3,)))])
def test_C_Tensor_reshape_backward(array, grad):
    autograd_grad_operands1 = Tensor(array, True)
    autograd_grad = grad
    torch_operands1 = torch.tensor(array, requires_grad=True)
    torch_grad = torch.tensor(grad)
    result1 = core.reshape(autograd_grad_operands1, (3, ))
    result2 = torch.reshape(torch_operands1, (3, ))
    result1.backward(autograd_grad)
    result2.backward(torch_grad)
    assert np.allclose(autograd_grad_operands1.grad, torch_operands1.grad.numpy(), equal_nan=True), \
        f"correct: {torch_operands1.grad.numpy()} | got: {autograd_grad_operands1.grad}"


@pytest.mark.parametrize("array, grad", [(np.array([[1.], [2.], [3.]]), np.random.random((3, 1)))])
def test_C_Tensor_log_backward(array, grad):
    autograd_grad_operands1 = Tensor(array, True)
    autograd_grad = grad
    torch_operands1 = torch.tensor(array, requires_grad=True)
    torch_grad = torch.tensor(grad)
    result1 = core.log(autograd_grad_operands1)
    result2 = torch.log(torch_operands1)
    result1.backward(autograd_grad)
    result2.backward(torch_grad)
    assert np.allclose(autograd_grad_operands1.grad, torch_operands1.grad.numpy(), equal_nan=True), \
        f"correct: {torch_operands1.grad.numpy()} | got: {autograd_grad_operands1.grad}"



