# NumBoost

# To do
1. ~~add tensordot(forward, backward)~~
2. ~~add slice(forward, backward)~~
3. add __len__ for Tensor
4. add __min__ for Tensor
5. add __max__ for Tensor
6. add __setitem__ for Tensor
7. add concat(forward, backward)
8. add stack(forward, backward)
9. add split(forward, backward)
10. add mean(forward, backward)
11. add trace(forward, backward)
12. remove redundant calculation in backward(add require grad check)
13. implement add, sub, mul, div operation directly instead of using np built in method(better performance)(include broadcast)
14. implement shape prediction instead of doing real calculation in jax tracing
15. implement broadcast detection based on shape
16. let cache can handle more scenario at the same function need to be jit
17. ~~add debug macro for debug in C. Use python setup.py build_ext -DEBUG to enable~~
18. ~~Add T static method~~
19. Proof of concept(POC) Object Pool for Tensor
20. Better Tensor dealloc(directly use switch to free dict)
21. To be determined(TBD)