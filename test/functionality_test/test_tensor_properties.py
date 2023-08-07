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



