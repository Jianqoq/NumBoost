# NumBoost

Linux, Windows

Python: 3.9+

Compiler: GCC, MSVC

# Feature:
Fast, fully parallelized, vectorisation, JaxJit support, auto differentiation, highly compatible with numpy

# Bug:
1. np.longdouble cast to float16 will cause error even though longdouble is 64bit. np.float64 type directly cast to np.float16 has no issue

# To do
1. ~~add tensordot(forward, backward)~~
2. ~~add slice(forward, backward)~~
3. ~~use parallel in broadcast~~
4. ~~implement add, sub, mul, div operation directly instead of using np built in method(better performance)(include broadcast)~~
5. ~~add debug macro for debug in C. Use python setup.py build_ext -DEBUG to enable~~
6. ~~Add T static method~~
7. ~~add __len__, __min__, __max__, iterator for Tensor~~
8. develop convenient api for fusion
9. mem pool
10. vectorization
11. bool calculation support
12. complex calculation support
13. backward node fusion
14. Runtime dynamic computation graph analysis and optimization
15. add __setitem__ for Tensor
16. add concat(forward, backward)
17. add stack(forward, backward)
18. add split(forward, backward)
19. add mean(forward, backward)
20. add trace(forward, backward)
21. remove redundant calculation in backward(add require grad check)
22. implement shape prediction instead of doing real calculation in jax tracing
23. implement broadcast detection based on shape
24. let cache can handle more scenario at the same function need to be jit
25. Proof of concept(POC) Object Pool for Tensor
26. Better Tensor dealloc(directly use switch to free dict)
27. To be determined(TBD)