# NumBoost
Linux, Windows

Python: 3.11+

Developed with Clangd in VSCode, compile with MSVC and GCC

# Installation
NumBoost is still in pretty early stage, therefore, there will be a lot of bugs and missing features.

If you want to try it out, you can install it by:

In Linux:

First: Download the source code

Second: Install Dependencies
```
pip install -r requirements.txt
```
Third: cd to src folder and run
```
python setup.py build_ext
```
Forth: You might need to add the path of src folder to PYTHONPATH, example is in benchmarks folder

Windows:

Since NumBoost is developed in Windows, however, it might not as easy as in Linux.

First: Download the source code

Second: Install Dependencies
```
pip install -r requirements.txt
```
Third: cd to src folder and run
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
