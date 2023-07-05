import numpy as np
import os
import sys
import timeit
import statistics

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tensor import Tensor


class MyTensor:
    def __init__(self, data, require_grad=False):
        self.data = np.asarray(data)
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


def C_Tensor(arr):
    Tensor(arr, True)


def Python_Tensor(arr):
    MyTensor(arr, True)


def torch_Tensor(arr):
    torch.tensor(arr, requires_grad=True)


array = [1., 2., 3.]
stmt_code = """
result = C_Tensor(array)
"""

stmt_code2 = """
result = Python_Tensor(array)
"""

stmt_code3 = """
result = torch_Tensor(array)
"""

print("Creating Objects")
time = [timeit.timeit(stmt_code, globals=globals(), number=1000000) for _ in range(10)]
print(''.join(format(i, ".5f") + " s\t\t" for i in time), end="")
print("stdev:", statistics.stdev(time))
time2 = [timeit.timeit(stmt_code2, globals=globals(), number=1000000) for _ in range(10)]
print(''.join(format(i, ".5f") + " s\t\t" for i in time2), end="")
print("stdev:", statistics.stdev(time2))
time3 = [timeit.timeit(stmt_code3, globals=globals(), number=1000000) for _ in range(10)]
print(''.join(format(i, ".5f") + " s\t\t" for i in time3), end="")
mean = statistics.mean(time)
mean2 = statistics.mean(time2)
mean3 = statistics.mean(time3)
print("\nDiff with python class:", (mean2 - mean)/mean * 100, "%")
print("\nDiff with torch tensor:", (mean3 - mean)/mean * 100, "%")

