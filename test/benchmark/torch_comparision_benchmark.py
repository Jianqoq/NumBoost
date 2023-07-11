import platform

import numpy as np
import os
import sys
import timeit
import statistics
import torch
if platform.system() == 'Windows':
    os.add_dll_directory(r'C:\Program Files (x86)\Intel\oneAPI\mkl\2023.1.0\redist\intel64')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.tensor import Tensor


def C_Tensor_addition(arr):
    p = arr + arr


def torch_Tensor_addition(arr):
    p = arr + arr


def C_Tensor_subtraction(arr):
    p = arr - arr


def torch_Tensor_subtraction(arr):
    p = arr - arr


def C_Tensor_division(arr):
    p = arr / arr


def torch_Tensor_division(arr):
    p = arr / arr


def C_Tensor_matmul(arr):
    p = arr @ arr


def torch_Tensor_matmul(arr):
    p = arr @ arr


def C_Tensor_mul(arr):
    p = arr * arr


def torch_Tensor_mul(arr):
    p = arr * arr


array = Tensor([1, 2, 3])
ten = torch.tensor([1, 2, 3])
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


