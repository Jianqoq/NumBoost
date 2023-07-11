import numpy as np
import sys
import pytest
import os
import platform
if platform.system() == 'Windows':
    os.add_dll_directory(r'C:\Program Files (x86)\Intel\oneAPI\mkl\2023.1.0\redist\intel64')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.tensor import Tensor


@pytest.mark.parametrize("array", [np.array([1, 2, 3])])
def test_C_Tensor_addition(array):
    p = Tensor(array)
    l = p + p
    o = array + array
    assert np.allclose(o, l.data), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array", [np.array([1, 2, 3])])
def test_C_Tensor_subtraction(array):
    p = Tensor(array)
    l = p - p
    o = array - array
    assert np.allclose(o, l.data), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array", [np.array([1, 2, 3])])
def test_C_Tensor_division(array):
    p = Tensor(array)
    l = p / p
    o = array / array
    assert np.allclose(o, l.data), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array", [np.array([1, 2, 3])])
def test_C_Tensor_matmul(array):
    p = Tensor(array)
    l = p @ p
    o = array @ array
    assert np.allclose(o, l.data), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array", [np.array([1, 2, 3])])
def test_C_Tensor_negative(array):
    p = Tensor(array)
    l = -p
    o = -array
    assert np.allclose(o, l.data), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array", [np.array([1, 2, 3])])
def test_C_Tensor_negative(array):
    p = Tensor(array)
    i = array.copy()
    l = -p
    o = -i
    assert np.allclose(o, l.data), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array", [np.array([1, 2, 3])])
def test_C_Tensor_positive(array):
    p = Tensor(array)
    l = +p
    o = +array
    assert np.allclose(o, l.data), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array", [np.array([-1, -2, -3]), np.array([1, 2, 3])])
def test_C_Tensor_absolute(array):
    p = Tensor(array)
    l = abs(p)
    o = abs(array)
    assert np.allclose(o, l.data), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array", [np.array([-1, -2, -3]), np.array([1, 2, 3])])
def test_C_Tensor_invert(array):
    p = Tensor(array)
    l = ~p
    o = ~array
    assert np.allclose(o, l.data), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array", [np.array([-1, -2, -3]), np.array([1, 2, 3])])
def test_C_Tensor_lshift(array):
    p = Tensor(array)
    result1 = p << 5
    result2 = array << 5
    assert np.allclose(result2, result1.data), f"correct: {result2} | got: {result1.data}"
    result1 = p << np.array([[1, 2, 3], [4, 5, 6]])
    result2 = array << np.array([[1, 2, 3], [4, 5, 6]])
    assert np.allclose(result2, result1.data), f"correct: {result2} | got: {result1.data}"


@pytest.mark.parametrize("array", [np.array([-1, -2, -3]), np.array([1, 2, 3])])
def test_C_Tensor_rshift(array):
    p = Tensor(array)
    l = p >> 5
    o = array >> 5
    assert np.allclose(o, l.data), f"correct: {o} | got: {l.data}"
    l = p >> np.array([[1, 2, 3], [4, 5, 6]])
    o = array >> np.array([[1, 2, 3], [4, 5, 6]])
    assert np.allclose(o, l.data), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array", [np.array([-1]), np.array([1])])  # test not perfect
def test_C_Tensor_and(array):
    p = Tensor(array)
    l = p and p
    o = array and array
    assert np.allclose(o, l.data), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array", [np.array([-1]), np.array([1])])
def test_C_Tensor_xor(array):
    p = Tensor(array)
    l = p ^ p
    o = array ^ array
    assert np.allclose(o, l.data), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array", [np.array([-1]), np.array([1])])
def test_C_Tensor_or(array):
    p = Tensor(array)
    l = p or p
    o = array or array
    assert np.allclose(o, l.data), f"correct: {o} | got: {l.data}"


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


@pytest.mark.parametrize("array", [np.array([-1., 2., 3.]), np.array([-1., 2., 3.])])
def test_C_Tensor_remainder(array):
    p = Tensor(array)
    l = p % p
    o = array % array
    assert np.allclose(o, l.data), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array", [np.array([-1, 2, 3]), np.array([1, 3, 2])])
def test_C_Tensor_ixor(array):
    p = Tensor(array)
    array2 = array.copy()
    p ^= p
    array2 ^= array2
    assert np.allclose(array2, p.data), f"correct: {array2} | got: {p.data}"


@pytest.mark.parametrize("array", [np.array([-1, 2, 3]), np.array([-1, 2, 3])])
def test_C_Tensor_ior(array):
    p = Tensor(array)
    array2 = array.copy()
    p |= p
    array2 |= array2
    assert np.allclose(array2, p.data), f"correct: {array2} | got: {p.data}"


@pytest.mark.parametrize("array", [np.array([-1, 2, 3]), np.array([-1, 2, 3])])
def test_C_Tensor_iand(array):
    p = Tensor(array)
    array2 = array.copy()
    p &= p
    array2 &= array2
    assert np.allclose(array2, p.data), f"correct: {array2} | got: {p.data}"


@pytest.mark.parametrize("array", [(np.array([-1, 2, 3])), (np.array([-1, 2, 3]))])
def test_C_Tensor_ilshift(array):
    p = Tensor(array)
    array2 = array.copy()
    p <<= p
    array2 <<= array2
    assert np.allclose(array2, p.data), f"correct: {array2} | got: {p.data}"


@pytest.mark.parametrize("array", [(np.array([-1, 2, 3])), (np.array([-1, 2, 3]))])
def test_C_Tensor_divmod(array):
    p = Tensor(array)
    l = divmod(p, 3)
    o = divmod(array, 3)
    assert np.allclose(o, l.data), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array", [(np.array([-1, 2, 3])), (np.array([-1, 2, 3]))])
def test_C_Tensor_iremainder(array):
    p = Tensor(array)
    array2 = array.copy()
    p %= p
    array2 %= array2
    assert np.allclose(array2, p.data), f"correct: {array2} | got: {p.data}"


@pytest.mark.parametrize("array", [(np.array([-1, 2, 3])), (np.array([-1, 2, 3]))])
def test_C_Tensor_floordiv(array):
    p = Tensor(array)
    l = p // p
    o = array // array
    assert np.allclose(o, l.data), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array", [(np.array([-1, 2, 3])), (np.array([-1, 2, 3]))])
def test_C_Tensor_ifloordiv(array):
    p = Tensor(array)
    array2 = array.copy()
    p //= p
    array2 //= array2
    assert np.allclose(array2, p.data), f"correct: {array2} | got: {p.data}"

