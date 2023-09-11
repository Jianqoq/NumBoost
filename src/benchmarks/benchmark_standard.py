import os
import platform
import time

import numpy as np
import torch

np.seterr(invalid='ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'
if platform.system() == 'Windows':
    os.add_dll_directory(r'C:\Program Files (x86)\Intel\oneAPI\mkl\2023.1.0\redist\intel64')
    os.add_dll_directory(r'C:\Users\123\autograd-C\Autograd-C\src\libraries\jemalloc-5.3.0\jemalloc-5.3.0\msvc\x64\Release')
from num_boost import Tensor

benchmark_cases = [
    (lambda x, y: x + y, "add"),
    (lambda x, y: x - y, "sub"),
    (lambda x, y: x * y, "mul"),
    (lambda x, y: x / y, "div"),
    (lambda x, y: x ** y, "pow")]
shape_cases = [
    (2048, 2048),
    (5120, 5120),
    (10240, 10240),
    (20480, 20480)]

for shape in shape_cases:
    shape1 = (shape[0], 1)
    shape2 = (1, shape[1])
    a = np.random.randn(*shape1)
    b = np.random.randn(*shape2)
    for case in benchmark_cases:
        print(case[1], shape)
        a_nb = Tensor(a)
        b_nb = Tensor(b)
        a_pt = torch.tensor(a)
        b_pt = torch.tensor(b)
        start = time.time()
        res_nb = case[0](a_nb, b_nb)
        end = time.time()
        print(end - start)
        start = time.time()
        res_pt = case[0](a_pt, b_pt)
        end = time.time()
        print(end - start)
        start = time.time()
        res = case[0](a, b)
        end = time.time()
        print(end - start)

