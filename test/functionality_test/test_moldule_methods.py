import os
import platform
import sys

import numpy as np

# if platform.system() == 'Windows':
    # os.add_dll_directory(r'C:\Program Files (x86)\Intel\oneAPI\mkl\2023.1.0\redist\intel64')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import num_boost as nb


def test_reduction():
    methods = [nb.sum, nb.mean, nb.max, nb.min]
    np_methods = [np.sum, np.mean, np.max, np.min]
    array = np.arange(20*10*20*10*20).reshape((20, 10, 20, 10, 20)).astype(np.float64)
    test_cases = [0, 1, 2, 3, (0, 1), (0, 2), (0, 3), (1, 2), (1, 3),
                  (2, 3), (0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3),
                  (0, 2, 4), (0, 3, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4),
                  (2, 3, 4), (0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 3, 4),
                  (0, 2, 3, 4), (1, 2, 3, 4), (0, 1, 2, 3, 4)]
    a = nb.tensor(array)
    for idx in range(len(methods)):
        for case in test_cases:
            d = np_methods[idx](array, axis=case)
            c = methods[idx](a, axis=case)
            assert np.allclose(c.data, d), f'len: {len(np.where(c.data != d))}{np.where(c.data != d)}'

def test_arg():
    test_cases = [0, 1, 2, 3, 4]
    a = np.random.randn(32, 32, 32, 32, 32)
    r = nb.tensor(a)
    for case in test_cases:
        d = np.argmax(a, axis=case)
        c = nb.argmax(r, axis=case)
        assert np.allclose(c.data, d), f'{c[np.where(c.data != d)], d[np.where(c.data != d)], np.where(c.data != d)}'
        d = np.argmin(a, axis=case)
        c = nb.argmin(r, axis=case)
        assert np.allclose(c.data, d), f'{c[np.where(c.data != d)], d[np.where(c.data != d)], np.where(c.data != d)}'