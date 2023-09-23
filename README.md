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
1. Redesign auto diff system
2. implement methods supports dynamic output
3. ~~add tensordot(forward, backward)~~
4. ~~add slice(forward, backward)~~
5. ~~use parallel in broadcast~~
6. ~~implement add, sub, mul, div operation directly instead of using np built in method(better performance)(include broadcast)~~
7. ~~add debug macro for debug in C. Use python setup.py build_ext -DEBUG to enable~~
8. ~~Add T static method~~
9. ~~add __len__, __min__, __max__, iterator for Tensor~~
10. improve speed when both uncontiguous arrays in same shape
11. develop convenient api for fusion
12. ~~mem pool~~
13. vectorization
14. ~~bool calculation support~~
15. complex calculation support
16. ~~backward node fusion~~
17. Runtime dynamic computation graph analysis and optimization
18. add __setitem__ for Tensor
19. add concat(forward, backward)
20. add stack(forward, backward)
21. add split(forward, backward)
22. add mean(forward, backward)
23. add trace(forward, backward)
24. remove redundant calculation in backward(add require grad check)
25. ~~implement shape prediction instead of doing real calculation in jax tracing~~
26. ~~implement broadcast detection based on shape~~
27. let cache can handle more scenario at the same function need to be jit
28. ~~Proof of concept(POC) Object Pool for Tensor~~
29. Better Tensor dealloc(directly use switch to free dict)
30. To be determined(TBD)
