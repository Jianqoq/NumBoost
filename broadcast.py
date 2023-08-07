import numpy as np
# need to make it to the same shape first
a = np.random.randn(1, 5)
b = np.random.randn(5, 1)

data_ptr = a.ctypes.data
strides = a.strides
shape = a.shape
shape2 = b.shape
axis = []
steps = []
stride_ = []
for idx, i in enumerate(shape2):
    if i == 1:
        axis.append(str(idx))
        steps.append(shape[idx])
        stride_.append(strides[idx])
prod = np.prod(shape2)
string = ""
o = prod
for idx, y in enumerate(axis):
    if idx != len(axis) - 1:
        o *= steps[idx]
        string += f"{y} for {steps[idx]} steps with {stride_[idx]} strides, got {o} ptrs.\nThen along axis "
    else:
        o *= steps[idx]
        string += f"{y} for {steps[idx]} steps with {stride_[idx]} strides."

print("a = shape", a.shape, "strides =", a.strides)
print("b = shape", b.shape, "strides =", b.strides)
print(f"b first has %d ptrs moving along axis %s" % (int(prod), string))
length = len(axis)
cnt = 0
for idx in range(length):
    cnt = 0
    step = steps[idx]
    for i in range(prod):
        for k in range(step):
            cnt += 1
    prod *= step

print("Finally has %d ptrs." % cnt)

import os
import platform
import sys
import time

import jax
import psutil
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
process = psutil.Process(os.getpid())
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'

if platform.system() == 'Windows':
    os.add_dll_directory(r'C:\Program Files (x86)\Intel\oneAPI\mkl\2023.1.0\redist\intel64')
from num_boost import Tensor


a = Tensor(np.arange(10).reshape((1, 2, 5)))
b = Tensor(np.arange(5).reshape((5, 1, 1)))
p = a + b
print(p)
# print(a0.strides)
# print(b0.strides)
# print(a0)
# print(b0)
# print(a0 + b0)
# print(a.data.strides)
# print(b.data.strides)
# print("=====================================")
# print(a.data)
# print("=====================================")
# print(b.data)
# print("=====================================")
# print(a.data.strides)
# print(a.data)
# print(a0.data.shape)
# print(b0.data.shape)
# print(a0.data.strides)
# print(b0.data.strides)
# print(a0.data)
# print(b0.data)
# print(a0.data + b0.data)
# print((a0.data + b0.data).strides)
