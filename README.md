# NumBoost
Platform: Linux, Windows
Python: 3.9+
Compiler: GCC, MSVC
Feature: Fast, fully parallelized, vectorisation, JaxJit support, auto differentiation

Known bug:
1. np.longdouble cast to float16 will cause error even though longdouble is 64bit. np.float64 type directly cast to np.float16 has no issue

# To do
1. ~~add tensordot(forward, backward)~~
2. ~~add slice(forward, backward)~~
3. use parallel in broadcast
4. add __len__ for Tensor
5. add __min__ for Tensor
6. add __max__ for Tensor
7. add __setitem__ for Tensor
8. add concat(forward, backward)
9. add stack(forward, backward)
10. add split(forward, backward)
11. add mean(forward, backward)
12. add trace(forward, backward)
13. remove redundant calculation in backward(add require grad check)
14. implement add, sub, mul, div operation directly instead of using np built in method(better performance)(include broadcast)
15. implement shape prediction instead of doing real calculation in jax tracing
16. implement broadcast detection based on shape
17. let cache can handle more scenario at the same function need to be jit
18. ~~add debug macro for debug in C. Use python setup.py build_ext -DEBUG to enable~~
19. ~~Add T static method~~
20. Proof of concept(POC) Object Pool for Tensor
21. Better Tensor dealloc(directly use switch to free dict)
22. To be determined(TBD)