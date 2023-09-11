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
import num_boost as nb
from num_boost import Tensor

t1 = time.time()
for i in range(1):
    c = np.random.randn(10000, 1000)
    a = Tensor(c)
t2 = time.time()
print(t2 - t1)

t1 = time.time()
for i in range(1):
    c = torch.randn(10000, 1000)
t2 = time.time()
print(t2 - t1)
