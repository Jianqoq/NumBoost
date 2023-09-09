import numpy as np
import sys
import pytest
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import platform

if platform.system() == 'Windows':
    os.add_dll_directory(
        r'C:\Program Files (x86)\Intel\oneAPI\mkl\2023.1.0\redist\intel64')
    os.add_dll_directory(
        r'C:\Users\123\autograd-C\Autograd-C\src\libraries\jemalloc-5.3.0\jemalloc-5.3.0\msvc\x64\Release')
import num_boost as nb

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

test_cases1 = [np.array([-1, 2, 3]).astype(np.int8),
               np.array([-1, 2, 3]).astype(np.int16),
               np.array([-1, 2, 3]).astype(np.int32),
               np.array([-1, 2, 3]).astype(np.int64),
               np.array([-1, 2, 3]).astype(np.uint8),
               np.array([-1, 2, 3]).astype(np.uint16),
               np.array([-1, 2, 3]).astype(np.uint32),
               np.array([-1, 2, 3]).astype(np.uint64),
               np.array([-1, 2, 3]).astype(np.float16),
               np.array([-1, 2, 3]).astype(np.float32),
               np.array([-1, 2, 3]).astype(np.float64)]

test_cases2 = [(np.array([1, 2, 3])).astype(np.int8),
               (np.array([1, 2, 3])).astype(np.int16),
               (np.array([1, 2, 3])).astype(np.int32),
               (np.array([1, 2, 3])).astype(np.int64)]

test_cases3 = [(np.array([1, 2, 3])).astype(np.uint8),
               (np.array([1, 2, 3])).astype(np.uint16),
               (np.array([1, 2, 3])).astype(np.uint32),
               (np.array([1, 2, 3])).astype(np.uint64)]


def test_C_Tensor_addition():
    ops = {"add": (nb.add, nb.Add),
           "sub": (nb.sub, nb.Sub),
           "mul": (nb.mul, nb.Mul),
           "div": (nb.div, nb.Div),
           "mod": (nb.mod, nb.Mod),
           "pow": (nb.power, nb.Pow),
           "floor div": (nb.fdiv, nb.Floor_Div)}

    ops2 = {"add": (lambda x, y: x + y, nb.Add),
            "sub": (lambda x, y: x - y, nb.Sub),
            "mul": (lambda x, y: x * y, nb.Mul),
            "div": (lambda x, y: x / y, nb.Div),
            "mod": (lambda x, y: x % y, nb.Mod),
            "pow": (lambda x, y: x ** y, nb.Pow),
            "floor div": (lambda x, y: x // y, nb.Floor_Div)}

    for op in ops:
        for j in nb.nb_type_2_np:
            if j not in (nb.float_, nb.float16, nb.float32, nb.float64):
                continue
            nb.global_float_type(j)
            type_ = [np.int8, np.ubyte, np.short, np.ushort,
                     np.int_, np.uint, np.int32, np.uint32, np.longlong, np.ulonglong,
                     np.float32, np.double, np.longdouble, np.float16]
            for k in type_:
                a = np.random.randn(10, 10).astype(k)
                a_ = nb.Tensor(a).astype(nb.np_type_2_nb[k])
                for i in type_:
                    u = np.random.randn(10, 10)
                    b_ = nb.Tensor(u)
                    predicted_type = nb.result_type(ops[op][1], a_.dtype,
                                                    a_.data.itemsize, b_.dtype, b_.data.itemsize)
                    q = ops2[op][0](a.astype(nb.nb_type_2_np[predicted_type]),
                                    u.astype(nb.nb_type_2_np[predicted_type]))
                    c = ops[op][0](a_, b_, out=a_)
                    if c.dtype == a_.dtype:
                        assert np.allclose(c.data, a_.data, equal_nan=True)
                    l = q.astype(c.data.dtype)
                    m = np.where(c.data != l)
                    assert np.allclose(l, c.data, equal_nan=True), (
                        f"{op}({nb.nb_type_2_np[j]}):\n{c}{l}(dtype: {l.dtype})\n"
                        f"correct: {l[m]}\n got: {c.data[m]}\n")
