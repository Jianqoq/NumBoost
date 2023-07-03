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
    assert np.allclose(o, l.data), f"Addition test failed. correct: {o} | got: {l.data}"

@pytest.mark.parametrize("array, require_grad", [(np.array([1, 2, 3]), True)])
def test_C_Tensor_subtraction(array, require_grad):
    p = Tensor(array, require_grad)
    l = p - p
    o = array - array
    assert np.allclose(o, l.data), f"Addition test failed. correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array, require_grad", [(np.array([1, 2, 3]), True)])
def test_C_Tensor_division(array, require_grad):
    p = Tensor(array, require_grad)
    l = p / p
    o = array / array
    assert np.allclose(o, l.data), f"Addition test failed. correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array, require_grad", [(np.array([1, 2, 3]), True)])
def test_C_Tensor_matmul(array, require_grad):
    p = Tensor(array, require_grad)
    l = p @ p
    o = array @ array
    assert np.allclose(o, l.data), f"Addition test failed. correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array, require_grad", [(np.array([1, 2, 3]), True)])
def test_C_Tensor_negative(array, require_grad):
    p = Tensor(array, require_grad)
    l = -p
    o = -array
    assert np.allclose(o, l.data), f"Addition test failed. correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array, require_grad", [(np.array([1, 2, 3]), True)])
def test_C_Tensor_negative(array, require_grad):
    p = Tensor(array, require_grad)
    l = -p
    o = -array
    assert np.allclose(o, l.data), f"Addition test failed. correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array, require_grad", [(np.array([1, 2, 3]), True)])
def test_C_Tensor_positive(array, require_grad):
    p = Tensor(array, require_grad)
    l = +p
    o = +array
    assert np.allclose(o, l.data), f"Addition test failed. correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array, require_grad", [(np.array([-1, -2, -3]), True), (np.array([1, 2, 3]), True)])
def test_C_Tensor_absolute(array, require_grad):
    p = Tensor(array, require_grad)
    l = abs(p)
    o = abs(array)
    assert np.allclose(o, l.data), f"Addition test failed. correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array, require_grad", [(np.array([-1, -2, -3]), True), (np.array([1, 2, 3]), True)])
def test_C_Tensor_invert(array, require_grad):
    p = Tensor(array, require_grad)
    l = ~p
    o = ~array
    assert np.allclose(o, l.data), f"Addition test failed. correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array, require_grad", [(np.array([-1, -2, -3]), True), (np.array([1, 2, 3]), True)])
def test_C_Tensor_lshift(array, require_grad):
    p = Tensor(array, require_grad)
    result1 = p << 5
    result2 = array << 5
    assert np.allclose(result2, result1.data), f"Addition test failed. correct: {result2} | got: {result1.data}"
    result1 = p << np.array([[1, 2, 3], [4, 5, 6]])
    result2 = array << np.array([[1, 2, 3], [4, 5, 6]])
    assert np.allclose(result2, result1.data), f"Addition test failed. correct: {result2} | got: {result1.data}"


@pytest.mark.parametrize("array, require_grad", [(np.array([-1, -2, -3]), True), (np.array([1, 2, 3]), True)])
def test_C_Tensor_rshift(array, require_grad):
    p = Tensor(array, require_grad)
    l = p >> 5
    o = array >> 5
    assert np.allclose(o, l.data), f"Addition test failed. correct: {o} | got: {l.data}"
    l = p >> np.array([[1, 2, 3], [4, 5, 6]])
    o = array >> np.array([[1, 2, 3], [4, 5, 6]])
    assert np.allclose(o, l.data), f"Addition test failed. correct: {o} | got: {l.data}"


@pytest.mark.parametrize("array, require_grad", [(np.array([-1]), True), (np.array([1]), True)])  # test not perfect
def test_C_Tensor_and(array, require_grad):
    p = Tensor(array, require_grad)
    l = p and p
    o = array and array
    assert np.allclose(o, l.data), f"Addition test failed. correct: {o} | got: {l.data}"


