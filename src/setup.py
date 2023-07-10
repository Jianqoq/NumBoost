from setuptools import setup, Extension
import numpy
import os
from setuptools.command.build_ext import build_ext as _build_ext
import platform
if platform.system() == 'Windows':
    args = ['/Ox', '/openmp']
else:
    args = ['-O3', '-fopenmp', '-I/mkl-C/mkl/latest/include']

mymodule = Extension('tensor',
                     sources=['tensor.c', 'operators.c', 'backward_fn.c', 'stack.c',
                              'set_Tensor_properties.c', 'methods.c', 'core.c', 'binaray_backward_fn.c'],
                     include_dirs=[
                         numpy.get_include(), 'C:/Program Files (x86)/Intel/oneAPI/mkl/latest/include',
                         'mkl-C/mkl/latest/include'],
                     # library_dirs=[
                     #     'C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib/intel64',
                     #     r'C:\Program Files (x86)\Intel\oneAPI\mkl\latest\redist\intel64'],  # 添加这一行
                     # libraries=['mkl_rt'],  # 添加这一行
                     language='c',
                     extra_compile_args=args,
                     define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')])

mymodule2 = Extension('core',
                      sources=['tensor.c', 'operators.c', 'backward_fn.c', 'stack.c',
                               'set_Tensor_properties.c', 'methods.c', 'core.c', 'binaray_backward_fn.c'],
                      include_dirs=[
                          numpy.get_include(), 'C:/Program Files (x86)/Intel/oneAPI/mkl/latest/include',
                      'mkl-C/mkl/latest/include'],
                      # library_dirs=[
                      #     'C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib/intel64',
                      #     r'C:\Program Files (x86)\Intel\oneAPI\mkl\latest\redist\intel64'],  # 添加这一行
                      # libraries=['mkl_rt'],  # 添加这一行
                      language='c',
                      extra_compile_args=args,
                      define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')])


class build_ext(_build_ext):
    def get_ext_fullpath(self, ext_name):
        filename = _build_ext.get_ext_filename(self, ext_name)
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


setup(name='autograd_C',
      cmdclass={'build_ext': build_ext},
      ext_modules=[mymodule, mymodule2])
