# NumBoost
Linux, Windows

Python: 3.11+

Developed with Clangd in VSCode, compile with MSVC and GCC

# Feature:
Fast, fully parallelized, vectorisation, JaxJit support, auto differentiation, highly compatible with numpy,
currently only support CPU, GPU support will be added in the future

# Performance:
spec: i5-12600k, 64 GB 3600MHz, 1TB 990 Pro, RTX 4090, Windows 11

Object creation:
```
a = np.array([-4, 2, 3])
start = time.time()
for i in range(10000):
    a_ = Tensor(a)
end = time.time()
print(end - start)
start = time.time()
for i in range(10000):
    a_ = torch.tensor(a)
end = time.time()
print(end - start)

output: 
0.0014982223510742188
0.030042648315429688
```
Elementwise:

![](https://github.com/Jianqoq/NumBoost/blob/allocator_lru_cache/src/benchmarks/img.png)
![](https://github.com/Jianqoq/NumBoost/blob/allocator_lru_cache/src/benchmarks/img_1.png)
![](https://github.com/Jianqoq/NumBoost/blob/allocator_lru_cache/src/benchmarks/img_2.png)
![](https://github.com/Jianqoq/NumBoost/blob/allocator_lru_cache/src/benchmarks/img_3.png)

Broadcast:

![](https://github.com/Jianqoq/NumBoost/blob/allocator_lru_cache/src/benchmarks/img_4.png)
![](https://github.com/Jianqoq/NumBoost/blob/allocator_lru_cache/src/benchmarks/img_5.png)
![](https://github.com/Jianqoq/NumBoost/blob/allocator_lru_cache/src/benchmarks/img_6.png)

# Installation
NumBoost is still in pretty early stage, therefore, there will be a lot of bugs and missing features.

If you want to try it out, you can install it by:

In Linux:

First: Download the source code

Second: Install Dependencies
```
pip install -r requirements.txt
```
Third: Install mkl (currently NumBoost didn't use mkl much, but it will be used in the future)
```
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/cd17b7fe-500e-4305-a89b-bd5b42bfd9f8/l_onemkl_p_2023.1.0.46342_offline.sh

sudo apt-get install -y ncurses-term

sudo sh ./l_onemkl_p_2023.1.0.46342_offline.sh -a -s --eula accept --install-dir /mkl-C/

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mkl-C/mkl/2023.1.0/lib/intel64

source /mkl-C/mkl/latest/env/vars.sh
```
Forth: cd to src folder and run
```
python setup.py build_ext
```
Fifth: You might need to add the path of src folder to PYTHONPATH, example is in benchmarks folder

Windows:

Since NumBoost is developed in Windows, however, it might not as easy as in Linux.

First: Download the source code

Second: Install Dependencies
```
pip install -r requirements.txt
```
Third: Install mkl, go to https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html and download mkl, make sure your system environment variable has MKLROOT

Forth: cd to src folder and run
```
python setup.py build_ext
```

# To do
1. implement methods supports dynamic output
2. ~~add tensordot(forward, backward)~~
3. ~~add slice(forward, backward)~~
4. ~~use parallel in broadcast~~
5. ~~implement add, sub, mul, div operation directly instead of using np built in method(better performance)(include broadcast)~~
6. ~~add debug macro for debug in C. Use python setup.py build_ext -DEBUG to enable~~
7. ~~Add T static method~~
8. ~~add __len__, __min__, __max__, iterator for Tensor~~
9. improve speed when both uncontiguous arrays in same shape
10. develop convenient api for fusion
11. ~~mem pool~~
12. vectorization
13. ~~bool calculation support~~
14. complex calculation support
15. ~~backward node fusion~~
16. Runtime dynamic computation graph analysis and optimization
17. add __setitem__ for Tensor
18. add concat(forward, backward)
19. add stack(forward, backward)
20. add split(forward, backward)
21. add mean(forward, backward)
22. add trace(forward, backward)
23. remove redundant calculation in backward(add require grad check)
24. ~~implement shape prediction instead of doing real calculation in jax tracing~~
25. ~~implement broadcast detection based on shape~~
26. let cache can handle more scenario at the same function need to be jit
27. ~~Proof of concept(POC) Object Pool for Tensor~~
28. Better Tensor dealloc(directly use switch to free dict)
29. To be determined(TBD)
