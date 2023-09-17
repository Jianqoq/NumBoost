import os
import platform
import sys

import numpy as np

if platform.system() == 'Windows':
    os.add_dll_directory(r'C:\Program Files (x86)\Intel\oneAPI\mkl\2023.1.0\redist\intel64')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import num_boost as nb


def test_sum():
    array = np.arange(10*10*10*10*10).reshape((10, 10, 10, 10, 10)).astype(np.float64)
    test_cases = [0, 1, 2, 3, (0, 1), (0, 2), (0, 3), (1, 2), (1, 3),
                  (2, 3), (0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3),
                  (0, 2, 4), (0, 3, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4),
                  (2, 3, 4), (0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 3, 4),
                  (0, 2, 3, 4), (1, 2, 3, 4), (0, 1, 2, 3, 4)]
    a = nb.tensor(array)
    for case in test_cases:
        d = np.sum(array, axis=case)
        c = nb.sum(a, axis=case)
        assert np.allclose(c.data, d), f'{c[np.where(c.data != d)], d[np.where(c.data != d)], np.where(c.data != d)}'