import numpy as np
import sys
import pytest
import os
import platform
if platform.system() == 'Windows':
    os.add_dll_directory(r'C:\Program Files (x86)\Intel\oneAPI\mkl\2023.1.0\redist\intel64')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from num_boost import Tensor, tensordot


@pytest.mark.parametrize("array, array2", [(np.random.randn(2, 5, 3), np.random.randn(2, 5, 3))])
def test_C_Tensor_T(array, array2):
    a = Tensor(array)
    result1 = a.T
    assert np.allclose(array.T, result1.data), f"correct: {array.T} | got: {result1.data}"


@pytest.mark.parametrize("array, array2", [(np.random.randn(2, 5, 3), np.random.randn(2, 5, 3))])
def test_C_Tensor_slice(array, array2):
    a = Tensor(array)[1:, ...]
    b = array[1:, ...]
    assert np.allclose(b, a.data), f"correct: {b} | got: {a.data}"


@pytest.mark.parametrize("array", [np.random.randn(10)])
def test_C_Tensor_max(array):
    a = Tensor(array)
    assert np.allclose(max(array), max(a).data), f"correct: {max(array)} | got: {max(a).data}"


@pytest.mark.parametrize("array", [np.random.randn(10)])
def test_C_Tensor_min(array):
    a = Tensor(array)
    assert np.allclose(min(array), min(a).data), f"correct: {min(array)} | got: {min(a).data}"


@pytest.mark.parametrize("array", [np.random.randn(10, 10)])
def test_C_Tensor_max_raise(array):
    a = Tensor(array)
    with pytest.raises(ValueError):
        max(a)


@pytest.mark.parametrize("array", [np.random.randn(10, 10)])
def test_C_Tensor_min_raise(array):
    a = Tensor(array)
    with pytest.raises(ValueError):
        min(a)


@pytest.mark.parametrize("array", [np.random.randn(10, 100)])
def test_C_Tensor_iter_next(array):
    a = Tensor(array)
    p = iter(a)
    assert np.allclose(next(p).data, array[0]), f"correct: {array[0]} | got: {next(p).data}"


@pytest.mark.parametrize("array", [np.random.randn(100,)])
def test_C_Tensor_iter_next_one_dim(array):
    a = Tensor(array)
    p = iter(a)
    assert np.allclose(next(p).data, array[0]), f"correct: {array[0]} | got: {next(p).data}"

