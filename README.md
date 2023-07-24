# NumBoost

# To do
1. ~~add tensordot(forward, backward)~~
2. ~~add slice(forward, backward)~~
3. add __len__ for Tensor
4. add __min__ for Tensor
5. add __max__ for Tensor
6. add __setitem__ for Tensor
7. remove redundant calculation in backward(add require grad check)
8. implement add, sub, mul, div operation directly instead of using np built in method(better performance)(include broadcast)
9. implement shape prediction instead of doing real calculation in jax tracing
10. implement broadcast detection based on shape
11. let cache can handle more scenario at the same function need to be jit
12. ~~add debug macro for debug in C. Use python setup.py build_ext -DEBUG to enable~~
13. ~~Add T static method~~
14. Proof of concept(POC) Object Pool for Tensor
15. Better Tensor dealloc(directly use switch to free dict)
16. To be determined(TBD)