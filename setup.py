from setuptools import setup, Extension
import numpy
import os
from setuptools.command.build_ext import build_ext as _build_ext

mymodule = Extension('tensor',
                     sources=['tensor.c', 'operators.c', 'backward_fn.c', 'stack.c'],
                     include_dirs=[numpy.get_include()],
                     extra_compile_args=['/Ox'])
# extra_compile_args=['']

class build_ext(_build_ext):
    def get_ext_fullpath(self, ext_name):
        filename = _build_ext.get_ext_filename(self, ext_name)
        return os.path.join('C:/Users/123/PycharmProjects/Auto-Differentiation/doc_file', filename)

setup(name='tensor',
      cmdclass={'build_ext': build_ext},
      ext_modules=[mymodule])
