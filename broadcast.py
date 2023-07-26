import numpy as np

a = np.random.randn(4, 4)
b = np.random.randn(4, 1)

data_ptr = a.ctypes.data
strides = a.strides
shape = a.shape
s = 1
shape2 = b.shape
axis = []
steps = []
stride_ = []
for idx, i in enumerate(shape2):
    if i == 1:
        s *= shape[idx]
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

