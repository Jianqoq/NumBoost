from multiprocessing import freeze_support
import os
import sys
sys.path.append(r"C:\Users\123\anaconda3\Lib\site-packages\jaxlib")
sys.path.append(r"../")
os.add_dll_directory(r'C:\Program Files (x86)\Intel\oneAPI\mkl\2023.1.0\redist\intel64')

if __name__ == '__main__':
    freeze_support()
    from mypy.stubgen import parse_options, generate_stubs
    args = ['-m', 'Numboost']
    options = parse_options(args)
    generate_stubs(options)
