import numpy as np

a = np.arange(72).reshape((4, 3, 2, 3))
b = np.random.randn(1, 3, 2, 3)
data_ptr = a.ctypes.data
strides = a.strides
shape = a.shape
cnt = 0
s = 1
shape2 = (4, 1, 1, 1)
axis = []
steps = []
idx2 = None
for idx, i in enumerate(shape2):
    if i == 1:
        idx2 = idx
        s *= shape[idx]
        axis.append(str(idx))
        steps.append(shape[idx])
idx1 = idx2 - 1
prod = np.prod(shape2)
string = ""
for idx, y in enumerate(axis):
    if idx != len(axis) - 1:
        string += f"{y} for {steps[idx] - 1} steps then axis "
    else:
        string += f"{y} for {steps[idx] - 1} steps."
print(f"%d ptrs moving along axis %s" % (int(prod), string), end=" ")
for i in range(prod):
    for k in range(s):
        cnt += 1

print("Finally has %d ptrs." % cnt)

