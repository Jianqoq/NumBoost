import sys

import numpy as np
from setuptools import setup, Extension
import numpy
import os
from setuptools.command.build_ext import build_ext as _build_ext
import platform


if "-DEBUG" in sys.argv:
    enable_debug = ('DEBUG', '1')
    sys.argv.remove("-DEBUG")
else:
    enable_debug = ('DEBUG',)

current_path = os.getcwd()
files_and_dirs = os.listdir(current_path)
files = [f for f in files_and_dirs if os.path.isfile(
    os.path.join(current_path, f))]

numboost_files = [f for f in files if f.startswith('Numboost') and f.endswith(
    '.so') or f.startswith('Numboost') and f.endswith('.pyd')]

if len(files) > 0:
    for f in numboost_files:
        os.remove(f)
if platform.system() == 'Windows':
    args = ['/Ox', '/openmp']
    extra_link_args = []
else:
    args = ['-O3', '-fopenmp', '-I/mkl-C/mkl/latest/include', '-mavx', '-mavx2']
    extra_link_args = ['-lmkl_rt']
    if os.path.exists('Numboost.cpython-38-x86_64-linux-gnu.so'):
        os.remove('Numboost.cpython-38-x86_64-linux-gnu.so')

mymodule = Extension('Numboost',
                     sources=['utils.c', 'tensor.c', 'operators.c', 'backward_fn.c', 'stack.c',
                              'set_Tensor_properties.c', 'methods.c', 'binaray_backward_fn.c', 'pcg_basic.c',
                              'import_methods.c', 'broadcast.c', 'shape.c', 'binary_func.c', 'type_convertor.c', 'tensor_methods.c'],
                     include_dirs=[
                         numpy.get_include(), 'C:/Program Files (x86)/Intel/oneAPI/mkl/latest/include',
                         'mkl-C/mkl/latest/include', r'C:\Users\123\Downloads\numpy-main\numpy\core\include\numpy'],
                     library_dirs=[
                         'C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib/intel64',
                         r'C:\Program Files (x86)\Intel\oneAPI\mkl\latest\redist\intel64',
                         r'/mkl-C/mkl/latest/lib/intel64', r'C:\Users\123\anaconda3\Lib\site-packages\numpy\core\lib'
                     ],
                     libraries=['mkl_rt', 'npymath'] if platform.system() == 'Windows' else [
                         'mkl_rt', 'gomp'],
                     language='c',
                     extra_compile_args=args,
                     extra_link_args=extra_link_args,
                     define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'), enable_debug])


class build_ext(_build_ext):
    def get_ext_fullpath(self, ext_name):
        filename = _build_ext.get_ext_filename(self, ext_name)
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


setup(name='autograd_C',
      cmdclass={'build_ext': build_ext},
      ext_modules=[mymodule])
