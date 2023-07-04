from setuptools import setup, Extension
import numpy
import os
from setuptools.command.build_ext import build_ext as _build_ext
import platform
if platform.system() == 'Windows':
    args = ['/Ox']
else:
    args = ['-O3']

mymodule = Extension('tensor',
                     sources=['tensor.c', 'operators.c', 'backward_fn.c', 'stack.c', 'set_Tensor_properties.c'],
                     include_dirs=[numpy.get_include()],
                     language='c',
                     extra_compile_args=args,
                     define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')])

class build_ext(_build_ext):
    def get_ext_fullpath(self, ext_name):
        filename = _build_ext.get_ext_filename(self, ext_name)
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


setup(name='tensor',
      cmdclass={'build_ext': build_ext},
      ext_modules=[mymodule])