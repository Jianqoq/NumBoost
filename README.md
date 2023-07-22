# NumBoost

# To do
1. ~~add tensordot(forward, backward)~~
2. add slice(forward, backward)
3. remove redundant calculation in backward(add require grad check)
4. implement add, sub, mul, div operation directly instead of using np built in method(better performance)(include broadcast)
5. implement shape prediction instead of doing real calculation in jax tracing
6. implement broadcast detection based on shape
7. let cache can handle more senario at the same function
8. ~~add debug macro for debug in C. Use python setup.py build_ext -DEBUG to enable~~
