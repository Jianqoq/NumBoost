import numpy as np
import os
import sys
import pytest
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tensor import Tensor


@pytest.mark.parametrize("array, require_grad", [(np.array([1, 2, 3]), True)])
def test_C_Tensor_addition(array, require_grad):
    p = Tensor(array, require_grad)
    l = p + p
    o = array + array
    assert np.allclose(o, l.data), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array, require_grad", [(np.array([1, 2, 3]), True)])
def test_C_Tensor_subtraction(array, require_grad):
    p = Tensor(array, require_grad)
    l = p - p
    o = array - array
    assert np.allclose(o, l.data), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array, require_grad", [(np.array([1, 2, 3]), True)])
def test_C_Tensor_division(array, require_grad):
    p = Tensor(array, require_grad)
    l = p / p
    o = array / array
    assert np.allclose(o, l.data), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array, require_grad", [(np.array([1, 2, 3]), True)])
def test_C_Tensor_matmul(array, require_grad):
    p = Tensor(array, require_grad)
    l = p @ p
    o = array @ array
    assert np.allclose(o, l.data), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array, require_grad", [(np.array([1, 2, 3]), True)])
def test_C_Tensor_negative(array, require_grad):
    p = Tensor(array, require_grad)
    l = -p
    o = -array
    assert np.allclose(o, l.data), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array, require_grad", [(np.array([1, 2, 3]), True)])
def test_C_Tensor_negative(array, require_grad):
    p = Tensor(array, require_grad)
    l = -p
    o = -array
    assert np.allclose(o, l.data), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array, require_grad", [(np.array([1, 2, 3]), True)])
def test_C_Tensor_positive(array, require_grad):
    p = Tensor(array, require_grad)
    l = +p
    o = +array
    assert np.allclose(o, l.data), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array, require_grad", [(np.array([-1, -2, -3]), True), (np.array([1, 2, 3]), True)])
def test_C_Tensor_absolute(array, require_grad):
    p = Tensor(array, require_grad)
    l = abs(p)
    o = abs(array)
    assert np.allclose(o, l.data), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array, require_grad", [(np.array([-1, -2, -3]), True), (np.array([1, 2, 3]), True)])
def test_C_Tensor_invert(array, require_grad):
    p = Tensor(array, require_grad)
    l = ~p
    o = ~array
    assert np.allclose(o, l.data), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array, require_grad", [(np.array([-1, -2, -3]), True), (np.array([1, 2, 3]), True)])
def test_C_Tensor_lshift(array, require_grad):
    p = Tensor(array, require_grad)
    result1 = p << 5
    result2 = array << 5
    assert np.allclose(result2, result1.data), f"correct: {result2} | got: {result1.data}"
    result1 = p << np.array([[1, 2, 3], [4, 5, 6]])
    result2 = array << np.array([[1, 2, 3], [4, 5, 6]])
    assert np.allclose(result2, result1.data), f"correct: {result2} | got: {result1.data}"


@pytest.mark.parametrize("array, require_grad", [(np.array([-1, -2, -3]), True), (np.array([1, 2, 3]), True)])
def test_C_Tensor_rshift(array, require_grad):
    p = Tensor(array, require_grad)
    l = p >> 5
    o = array >> 5
    assert np.allclose(o, l.data), f"correct: {o} | got: {l.data}"
    l = p >> np.array([[1, 2, 3], [4, 5, 6]])
    o = array >> np.array([[1, 2, 3], [4, 5, 6]])
    assert np.allclose(o, l.data), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array, require_grad", [(np.array([-1]), True), (np.array([1]), True)])  # test not perfect
def test_C_Tensor_and(array, require_grad):
    p = Tensor(array, require_grad)
    l = p and p
    o = array and array
    assert np.allclose(o, l.data), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array, require_grad", [(np.array([-1]), True), (np.array([1]), True)])
def test_C_Tensor_xor(array, require_grad):
    p = Tensor(array, require_grad)
    l = p ^ p
    o = array ^ array
    assert np.allclose(o, l.data), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array, require_grad", [(np.array([-1]), True), (np.array([1]), True)])
def test_C_Tensor_or(array, require_grad):
    p = Tensor(array, require_grad)
    l = p or p
    o = array or array
    assert np.allclose(o, l.data), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array, require_grad", [(np.array(-1), True), (np.array(1), True)])
def test_C_Tensor_float(array, require_grad):
    p = Tensor(array, require_grad)
    l = float(p)
    o = float(array)
    assert np.allclose(o, l), f"correct: {o} | got: {l}"


@pytest.mark.parametrize("array, require_grad", [(np.array(-1.), True), (np.array(1), True)])
def test_C_Tensor_int(array, require_grad):
    p = Tensor(array, require_grad)
    l = int(p)
    o = int(array)
    assert np.allclose(o, l), f"correct: {o} | got: {l}"


@pytest.mark.parametrize("array, require_grad", [(np.array([-1., 2., 3.]), True), (np.array([-1., 2., 3.]), True)])
def test_C_Tensor_remainder(array, require_grad):
    p = Tensor(array, require_grad)
    l = p % p
    o = array % array
    assert np.allclose(o, l.data), f"correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array, require_grad", [(np.array([-1, 2, 3]), True), (np.array([1, 3, 2]), True)])
def test_C_Tensor_ixor(array, require_grad):
    p = Tensor(array, require_grad)
    array2 = array.copy()
    p ^= p
    array2 ^= array2
    assert np.allclose(array2, p.data), f"correct: {array2} | got: {p.data}"


@pytest.mark.parametrize("array, require_grad", [(np.array([-1, 2, 3]), True), (np.array([-1, 2, 3]), True)])
def test_C_Tensor_ior(array, require_grad):
    p = Tensor(array, require_grad)
    array2 = array.copy()
    p |= p
    array2 |= array2
    assert np.allclose(array2, p.data), f"correct: {array2} | got: {p.data}"


@pytest.mark.parametrize("array, require_grad", [(np.array([-1, 2, 3]), True), (np.array([-1, 2, 3]), True)])
def test_C_Tensor_iand(array, require_grad):
    p = Tensor(array, require_grad)
    array2 = array.copy()
    p &= p
    array2 &= array2
    assert np.allclose(array2, p.data), f"correct: {array2} | got: {p.data}"

