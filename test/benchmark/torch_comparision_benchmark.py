import numpy as np
import os
import sys
import timeit
import statistics

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tensor import Tensor
from torch import tensor

def C_Tensor_addition(arr):
    a = Tensor(arr)
    p = a + a


def torch_Tensor_addition(arr):
    a = tensor(arr)
    p = a + a


def C_Tensor_subtraction(arr):
    a = Tensor(arr)
    p = a - a


def torch_Tensor_subtraction(arr):
    a = tensor(arr)
    p = a - a


def C_Tensor_division(arr):
    a = Tensor(arr)
    p = a / a


def torch_Tensor_division(arr):
    a = tensor(arr)
    p = a / a


def C_Tensor_matmul(arr):
    a = Tensor(arr)
    p = a @ a


def torch_Tensor_matmul(arr):
    a = tensor(arr)
    p = a @ a


def C_Tensor_mul(arr):
    a = Tensor(arr)
    p = a * a


def torch_Tensor_mul(arr):
    a = tensor(arr)
    p = a * a



array = np.array([1, 2, 3])
ten = torch.Tensor([1, 2, 3])
stmt_code = """
result = C_Tensor_addition(array)
"""

stmt_code2 = """
result = torch_Tensor_addition(ten)
"""

print("Calculation benchmark")
time = [timeit.timeit(stmt_code, globals=globals(), number=1000000) for _ in range(10)]
print(''.join(format(i, ".5f") + " s\t\t" for i in time), end="")
print("stdev:", statistics.stdev(time))
time2 = [timeit.timeit(stmt_code2, globals=globals(), number=1000000) for _ in range(10)]
print(''.join(format(i, ".5f") + " s\t\t" for i in time2), end="")
print("stdev:", statistics.stdev(time2))
mean = statistics.mean(time)
mean2 = statistics.mean(time2)
print("\nDiff:", (mean2 - mean)/mean * 100, "%")


