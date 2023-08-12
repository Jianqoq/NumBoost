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
8. vectorization
9. bool calculation support
10. complex calculation support
11. backward node fusion
12. Runtime dynamic computation graph analysis and optimization
13. add __setitem__ for Tensor
14. add concat(forward, backward)
15. add stack(forward, backward)
16. add split(forward, backward)
17. add mean(forward, backward)
18. add trace(forward, backward)
19. remove redundant calculation in backward(add require grad check)
20. implement shape prediction instead of doing real calculation in jax tracing
21. implement broadcast detection based on shape
22. let cache can handle more scenario at the same function need to be jit
23. Proof of concept(POC) Object Pool for Tensor
24. Better Tensor dealloc(directly use switch to free dict)
25. To be determined(TBD)