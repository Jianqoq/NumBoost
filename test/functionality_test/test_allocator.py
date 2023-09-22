import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import platform
# if platform.system() == 'Windows':
#     os.add_dll_directory(
#         r'C:\Program Files (x86)\Intel\oneAPI\mkl\2023.1.0\redist\intel64')
#     os.add_dll_directory(r'C:\Users\123\autograd-C\Autograd-C\src\libraries\jemalloc-5.3.0\jemalloc-5.3.0\msvc\x64\Release')
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def test_normal_malloc():
    a = np.random.randn(100, 100)


def test_malloc_for_existed():
    a = np.random.randn(100, 100)
    b = np.random.randn(100, 100)
    c = np.random.randn(100, 100)