import numpy as np
import os
import sys
import pytest
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tensor import Tensor


class MyTensor:
    def __init__(self, data, require_grad=False):
        self.data = data
        self.x = None
        self.y = None
        self.has_conv = 0
        self.vars = 0
        self.require_grad = require_grad
        self.grad_fn = None
        self.graph = None
        self.axis = None
        self.grad = None
        self.dim = 0
        self.base = None

    def __add__(self, other):
        new_result = self.data + other.data
        new_tensor = MyTensor(new_result)
        new_tensor.x = self.data
        new_tensor.y = other.data
        return new_tensor

    def __iadd__(self, other):
        new_result = self.data + other.data
        self.x = self
        self.y = other
        self.data = new_result
        return self


def C_Tensor(array, require_grad):
    Tensor(array, require_grad)


def Python_Tensor(array, require_grad):
    MyTensor(array, require_grad)


@pytest.mark.benchmark
def test_C_Tensor(benchmark):
    array = np.array([[1, 2, 3], [4, 5, 6]])
    benchmark(C_Tensor, array, True)


@pytest.mark.benchmark
def test_Python_Tensor(benchmark):
    array = np.array([[1, 2, 3], [4, 5, 6]])
    benchmark(Python_Tensor, array, True)

