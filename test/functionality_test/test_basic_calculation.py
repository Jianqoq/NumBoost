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
from src.Numboost import result_type
from NumBoost_openum import Add, LShift, RShift
from num_boost import Tensor

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
    ops = {"add": (lambda x, y: x + y, nb.Add),
           "sub": (lambda x, y: x - y, nb.Sub),
           "mul": (lambda x, y: x * y, nb.Mul),
           "div": (lambda x, y: x / y, nb.Div),
           "mod": (lambda x, y: x % y, nb.Mod),
           "pow": (lambda x, y: x ** y, nb.Pow),
           "floor div": (lambda x, y: x // y, nb.Floor_Div)}

    for op in ops:
        for j in nb.nb_type_2_np:
            if j not in (nb.float, nb.float16, nb.float32, nb.float64):
                continue
            nb.global_float_type(j)
            type_ = [np.byte, np.ubyte, np.short, np.ushort,
                     np.int_, np.uint, np.int32, np.uint32, np.longlong, np.ulonglong,
                     np.float32, np.double, np.longdouble, np.float16]
            for k in type_:
                a = np.random.randn(10, 10)
                a_ = nb.Tensor(a)
                for i in type_:
                    u = np.random.randn(10, 10)
                    b_ = nb.Tensor(u)
                    predicted_type = nb.result_type(ops[op][1], nb.np_type_2_nb[a_.data.dtype],
                                                    a_.data.itemsize, nb.np_type_2_nb[b_.data.dtype], b_.data.itemsize)
                    q = ops[op][0](a.astype(nb.nb_type_2_np[predicted_type]), u.astype(nb.nb_type_2_np[predicted_type]))
                    c = ops[op][0](a_, b_)
                    l = q.astype(c.data.dtype)
                    m = np.where(c.data != l)
                    assert np.allclose(l, c.data, equal_nan=True), (
                        f"{op}({nb.nb_type_2_np[j]}):\n{c}{l}(dtype: {l.dtype})\n"
                        f"correct: {l[m]}\n got: {c.data[m]}\n")


@pytest.mark.parametrize("array", test_cases1)
def test_C_Tensor_matmul(array):
    p = Tensor(array)
    l = p @ p
    o = array @ array
    assert np.allclose(o, l.data, equal_nan=True), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array", test_cases1)
def test_C_Tensor_negative(array):
    p = Tensor(array)
    l = -p
    o = -array
    assert np.allclose(o, l.data, equal_nan=True), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array", test_cases1)
def test_C_Tensor_negative(array):
    p = Tensor(array)
    i = array.copy()
    l = -p
    o = -i
    assert np.allclose(o, l.data, equal_nan=True), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array", test_cases1)
def test_C_Tensor_positive(array):
    p = Tensor(array)
    l = +p
    o = +array
    assert np.allclose(o, l.data, equal_nan=True), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array", test_cases1)
def test_C_Tensor_absolute(array):
    p = Tensor(array)
    l = abs(p)
    o = abs(array)
    assert np.allclose(o, l.data, equal_nan=True), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array", test_cases2)
def test_C_Tensor_invert(array):
    p = Tensor(array)
    l = ~p
    o = ~array
    assert np.allclose(o, l.data, equal_nan=True), f"correct: {o}({o.dtype}) | got: {l}"


@pytest.mark.parametrize("array", test_cases2)
def test_C_Tensor_lshift(array):
    p = Tensor(array)
    result1 = p << 5
    result2 = array << 5
    assert np.allclose(
        result2, result1.data), f"correct: {result2} | got: {result1.data}"
    c = np.array([[1, 2, 3], [4, 5, 6]])
    result1 = p << Tensor(c)
    result2 = array << c
    predict_type = result_type(LShift, nb.np_type_2_nb[p.data.dtype],
                               p.data.itemsize, nb.np_type_2_nb[c.dtype],
                               c.itemsize)
    assert np.allclose(
        result2, result1.data, equal_nan=True), (f"predict type: {predict_type} actual: {result2.dtype}\n"
                                                 f"correct: {result2}({result2.dtype}) | got: {result1}")


@pytest.mark.parametrize("array", test_cases2)
def test_C_Tensor_rshift(array):
    p = Tensor(array)
    l = p >> 5
    o = array >> 5
    assert np.allclose(o, l.data), f"correct: {o} | got: {l.data}"
    c = np.array([[1, 2, 3], [4, 5, 6]])
    l = p >> Tensor(c)
    predict_type = result_type(RShift, nb.np_type_2_nb[p.data.dtype],
                               p.data.itemsize, nb.np_type_2_nb[c.dtype],
                               c.itemsize)
    o = array.astype(nb.nb_type_2_np[predict_type]) >> c.astype(nb.nb_type_2_np[predict_type])
    assert np.allclose(o, l.data, equal_nan=True), (f"predict type: {predict_type} actual: {o.dtype}\n"
                                                    f"correct: {o}({o.dtype}) | got: {l}")


@pytest.mark.parametrize("array", [np.array([-1]), np.array([1])])  # test not perfect
def test_C_Tensor_and(array):
    p = Tensor(array)
    l = p and p
    o = array and array
    assert np.allclose(o, l.data), f"correct: {o}({o.dtype}) | got: {l.data}"


@pytest.mark.parametrize("array", [np.array([-1]), np.array([1])])
def test_C_Tensor_xor(array):
    p = Tensor(array)
    l = p ^ p
    o = array ^ array
    assert np.allclose(o, l.data), f"correct: {o}({o.dtype}) | got: {l.data}"


@pytest.mark.parametrize("array", [np.array([-1]), np.array([1])])
def test_C_Tensor_or(array):
    p = Tensor(array)
    l = p or p
    o = array or array
    assert np.allclose(o, l.data), f"correct: {o}({o.dtype}) | got: {l.data}"


@pytest.mark.parametrize("array", [np.array(-1), np.array(1)])
def test_C_Tensor_float(array):
    p = Tensor(array)
    l = float(p)
    o = float(array)
    assert np.allclose(o, l), f"correct: {o} | got: {l}"


@pytest.mark.parametrize("array", [np.array(-1.), np.array(1)])
def test_C_Tensor_int(array):
    p = Tensor(array)
    l = int(p)
    o = int(array)
    assert np.allclose(o, l), f"correct: {o} | got: {l}"


@pytest.mark.parametrize("array", test_cases1)
def test_C_Tensor_remainder(array):
    p = Tensor(array)
    l = p % p
    o = array % array
    assert np.allclose(o, l.data, equal_nan=True), f"correct: {o}({o.dtype})| got: {l.data}"


@pytest.mark.parametrize("array", test_cases2)
def test_C_Tensor_ixor(array):
    p = Tensor(array)
    array2 = array.copy()
    p ^= p
    array2 ^= array2
    assert np.allclose(array2, p.data, equal_nan=True), f"correct: {array2}({array2.dtype}) | got: {p.data}"


@pytest.mark.parametrize("array", test_cases2)
def test_C_Tensor_ior(array):
    p = Tensor(array)
    array2 = array.copy()
    p |= p
    array2 |= array2
    assert np.allclose(array2, p.data, equal_nan=True), f"correct: {array2}({array2.dtype}) | got: {p.data}"


@pytest.mark.parametrize("array", test_cases2)
def test_C_Tensor_iand(array):
    p = Tensor(array)
    array2 = array.copy()
    p &= p
    array2 &= array2
    assert np.allclose(array2, p.data, equal_nan=True), f"correct: {array2}({array2.dtype}) | got: {p.data}"


@pytest.mark.parametrize("array", test_cases2)
def test_C_Tensor_ilshift(array):
    p = Tensor(array)
    array2 = array.copy()
    p <<= p
    array2 <<= array2
    assert np.allclose(array2, p.data, equal_nan=True), f"correct: {array2}({array2.dtype}) | got: {p.data}"


@pytest.mark.parametrize("array", test_cases1)
def test_C_Tensor_divmod(array):
    p = Tensor(array)
    l = divmod(p, 3)
    o = divmod(array, 3)
    assert np.allclose(o, l.data, equal_nan=True), f"correct: {o}({o.dtype}) | got: {l.data}"


@pytest.mark.parametrize("array", test_cases1)
def test_C_Tensor_iremainder(array):
    p = Tensor(array)
    array2 = array.copy()
    p %= p
    array2 %= array2
    assert np.allclose(array2, p.data, equal_nan=True), f"correct: {array2}({array2.dtype}) | got: {p.data}"


@pytest.mark.parametrize("array", test_cases1)
def test_C_Tensor_floordiv(array):
    p = Tensor(array)
    l = p // p
    o = array // array
    assert np.allclose(o, l.data, equal_nan=True), f"data: {p} correct: {o}({o.dtype}) | got: {l.data}"


@pytest.mark.parametrize("array", test_cases1)
def test_C_Tensor_ifloordiv(array):
    p = Tensor(array)
    array2 = array.copy()
    p //= p
    array2 //= array2
    assert np.allclose(array2, p.data, equal_nan=True), f"correct: {array2}({array2.dtype}) | got: {p.data}"


@pytest.mark.parametrize("array", test_cases1)
def test_C_Tensor_scalar_addition(array):
    p = Tensor(array)
    c = p + 10
    d = array + 10
    assert np.allclose(d, c.data, equal_nan=True), f"correct: {d}({d.dtype}) | got: {c.data}"


@pytest.mark.parametrize("array", test_cases1)
def test_C_Tensor_fscalar_addition(array):
    p = Tensor(array)
    c = p + 10.
    d = array + 10.
    assert np.allclose(d, c.data, equal_nan=True), f"correct: {d}({d.dtype}) | got: {c.data}"


@pytest.mark.parametrize("array", test_cases1)
def test_C_Tensor_scalar_subtraction(array):
    p = Tensor(array)
    c = p * 10
    d = array * np.array(10).astype(np.uint64)
    assert np.allclose(d, c.data, equal_nan=True), f"correct: {d}({d.dtype}) | got: {c.data}"
