import sys
import sysconfig
from setuptools import setup, Extension
import numpy
import os
from setuptools.command.build_ext import build_ext as _build_ext
import platform

available_args = ('-DEBUG', '-Allocator_Debug')
numboost_custom_macro_args = []
for arg in sys.argv:
    if arg in available_args:
        numboost_custom_macro_args.append((arg.split('-')[1], '1'))
for arg in sys.argv:
    if arg in available_args:
        sys.argv.remove(arg)
compiler_info = sysconfig.get_config_var('CC')
if compiler_info is None:
    omp_compile_args = ['/openmp']  # 默认使用MSVC标志
else:
    omp_compile_args = ['-fopenmp'] if 'gcc' in compiler_info else ['/openmp']


class build_ext(_build_ext):
    def get_ext_fullpath(self, ext_name):
        filename = _build_ext.get_ext_filename(self, ext_name)
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


# 定义扩展模块
# extension = Extension(
#     'test',  # 模块名
#     sources=['fine_tune.c'],  # 源文件
#     include_dirs=[numpy.get_include(), 'C:/Program Files (x86)/Intel/oneAPI/mkl/latest/include',
#                          'mkl-C/mkl/latest/include', r'C:\Users\123\Downloads\numpy-main\numpy\core\include\numpy',
#                   r'C:\Users\123\autograd-C\Autograd-C\src\jemalloc-5.3.0\jemalloc-5.3.0\msvc\x64\Release'],  # 包括NumPy的头文件目录
#     extra_compile_args=omp_compile_args,  # 添加OpenMP编译标志
#     extra_link_args=omp_compile_args,  # 添加OpenMP链接标志
#     libraries=['jemalloc', 'mkl_rt', 'npymath'],  # 链接的库文件
# define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
# )
#
# setup(
#     name='test',
#     cmdclass={'build_ext': build_ext},
#     ext_modules=[extension]
# )
current_path = os.getcwd()
files_and_dirs = os.listdir(current_path)
files = [f for f in files_and_dirs if os.path.isfile(os.path.join(current_path, f))]
if platform.system() == 'Windows':
    args = ['/openmp:experimental', '/Ox', '/Zc:preprocessor', '/DUSE_SOFT_INTRINSICS']
    extra_link_args = []
    numboost_files = [f for f in files if f.startswith('Numboost') and f.endswith('.pyd')]
else:
    # args = ['-O3', '-fopenmp', '-I/mkl-C/mkl/latest/include', '-mavx', '-mavx2', '-I/usr/local/include',
    #         '-L/usr/local/lib', '-Wall']
    # extra_link_args = ['-lmkl_rt']
    args = ['-O3', '-fopenmp', '-mavx', '-mavx2', '-I/usr/local/include',
            '-L/usr/local/lib', '-Wall']
    if os.path.exists('Numboost.cpython-38-x86_64-linux-gnu.so'):
        os.remove('Numboost.cpython-38-x86_64-linux-gnu.so')
    numboost_files = [f for f in files if f.startswith('Numboost') and f.endswith('.so')]

if len(files) > 0:
    for f in numboost_files:
        os.remove(f)

mymodule = Extension('Numboost',
                     sources=['element_ops/element_ops_def.c', 'auto_diff/ufunc_backward_def.c', 'tensor.c',
                              'python_magic/python_math_magic.c',
                              'auto_diff/backward_fn.c', 'stack.c', 'element_ops/element_ops_impl.c',
                              'auto_diff/binaray_backward_fn.c', 'random/pcg_basic.c',
                              'import_module_methods.c', 'shape.c',
                              'type_convertor/type_convertor.c', 'tensor_methods.c', 'Iterator/nb_iter.c',
                              'allocator/allocator.c', 'binary_ops/binary_op_def.c',
                              'binary_ops/binary_module_methods.c', 'reduction_ops/reduction_ops_def.c',
                              'allocator/tensor_alloc.c', 'tensor_creation/creation_def.c', 'ufunc_ops/ufunc_def.c'],
                     include_dirs=[
                         numpy.get_include(), 'C:/Program Files (x86)/Intel/oneAPI/mkl/latest/include',
                         'mkl-C/mkl/latest/include', r'C:\Users\123\Downloads\numpy-main\numpy\core\include\numpy',
                         'lib_include/include',
                         r'C:\Users\123\autograd-C\Autograd-C\src\jemalloc-5.3.0\jemalloc-5.3.0\include',
                         r'C:\Users\123\autograd-C\Autograd-C\src\jemalloc-5.3.0\jemalloc-5.3.0\include\msvc_compat'
                     ],
                     library_dirs=[
                         'C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib/intel64',
                         r'C:\Program Files (x86)\Intel\oneAPI\mkl\latest\redist\intel64',
                         r'/mkl-C/mkl/latest/lib/intel64',
                         r'/usr/local/lib',
                         r'C:\Users\123\autograd-C\Autograd-C\src\libraries\jemalloc-5.3.0\jemalloc-5.3.0\msvc\x64\Release',
                         os.path.join(os.path.dirname(numpy.core.__file__), 'lib')
                     ],
                    #  libraries=['mkl_rt', 'npymath']
                    #  if platform.system() == 'Windows' else [
                    #      'mkl_rt', 'gomp', 'npymath'],
                     libraries=['npymath']
                     if platform.system() == 'Windows' else [
                         'gomp', 'npymath'],
                     language='c',
                     extra_compile_args=args,
                     # extra_link_args=extra_link_args,
                     define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_25_API_VERSION')] + numboost_custom_macro_args)

setup(name='Numboost',
      cmdclass={'build_ext': build_ext},
      ext_modules=[mymodule])
