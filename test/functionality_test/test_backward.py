import numpy as np
import torch
import os
import sys
import pytest
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tensor import Tensor


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


