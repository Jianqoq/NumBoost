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
4. vectorization
5. bool calculation support
6. complex calculation support
7. add __len__ for Tensor
8. add __min__ for Tensor
9. add __max__ for Tensor
10. add __setitem__ for Tensor
11. add concat(forward, backward)
12. add stack(forward, backward)
13. add split(forward, backward)
14. add mean(forward, backward)
15. add trace(forward, backward)
16. remove redundant calculation in backward(add require grad check)
17. ~~implement add, sub, mul, div operation directly instead of using np built in method(better performance)(include broadcast)~~
18. implement shape prediction instead of doing real calculation in jax tracing
19. implement broadcast detection based on shape
20. let cache can handle more scenario at the same function need to be jit
21. ~~add debug macro for debug in C. Use python setup.py build_ext -DEBUG to enable~~
22. ~~Add T static method~~
23. Proof of concept(POC) Object Pool for Tensor
24. Better Tensor dealloc(directly use switch to free dict)
25. To be determined(TBD)