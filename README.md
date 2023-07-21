# NumBoost

# To do
1. add tensordot(forward, backward)
2. remove redundant calculation in backward(add require grad check)
3. implement add, sub, mul, div operation directly instead of using np built in method(better performance)(include broadcast)
4. implement shape prediction instead of doing real calculation in jax tracing
5. implement broadcast detection based on shape
