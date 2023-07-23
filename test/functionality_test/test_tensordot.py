import numpy as np
import sys
import pytest
import os
import platform
if platform.system() == 'Windows':
    os.add_dll_directory(r'C:\Program Files (x86)\Intel\oneAPI\mkl\2023.1.0\redist\intel64')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from NumBoost import Tensor, tensordot


@pytest.mark.parametrize("array, array2", [(np.random.randn(2, 5, 3), np.random.randn(2, 5, 3))])
def test_C_Tensor_addition1(array, array2):
    autograd_grad_operands1 = Tensor(array)
    autograd_grad_operands2 = Tensor(array2)
    result1 = tensordot(autograd_grad_operands1, autograd_grad_operands2, 3)
    # o = np.tensordot(array, array2, 3)
    # assert np.allclose(o, result1.data), f"correct: {o} | got: {result1.data}"


# @pytest.mark.parametrize("array, array2", [(np.random.randn(2, 5, 3), np.random.randn(2, 5, 3))])
# def test_C_Tensor_addition2(array, array2):
#     autograd_grad_operands1 = Tensor(array, True)
#     autograd_grad_operands2 = Tensor(array2)
#     result1 = tensordot(autograd_grad_operands1, autograd_grad_operands2, [[0, 1], [0, 1]])
#     o = np.tensordot(array, array2, ((0, 1), (0, 1)))
#     assert np.allclose(o, result1.data), f"correct: {o} | got: {result1.data}"
