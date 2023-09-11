# NumBoost

Linux, Windows

Python: 3.11+

Compiler: GCC, MSVC

# Feature:
Fast, fully parallelized, vectorisation, JaxJit support, auto differentiation, highly compatible with numpy

# Performance:
spec: i5-12600k, 64 GB 3600MHz, 1TB 990 Pro, RTX 4090, Windows 11

Object creation:
```
a = np.array([-4, 2, 3])
start = time.time()
for i in range(10000):
    a_ = Tensor(a)
end = time.time()
print(end - start)
start = time.time()
for i in range(10000):
    a_ = torch.tensor(a)
end = time.time()
print(end - start)

output: 
0.0014982223510742188
0.030042648315429688
```
Elementwise:

![](https://github.com/Jianqoq/NumBoost/blob/allocator_lru_cache/src/benchmarks/img.png)
![](https://github.com/Jianqoq/NumBoost/blob/allocator_lru_cache/src/benchmarks/img_1.png)
![](https://github.com/Jianqoq/NumBoost/blob/allocator_lru_cache/src/benchmarks/img_2.png)
![](https://github.com/Jianqoq/NumBoost/blob/allocator_lru_cache/src/benchmarks/img_3.png)

Broadcast:

![](https://github.com/Jianqoq/NumBoost/blob/allocator_lru_cache/src/benchmarks/img_4.png)
![](https://github.com/Jianqoq/NumBoost/blob/allocator_lru_cache/src/benchmarks/img_5.png)
![](https://github.com/Jianqoq/NumBoost/blob/allocator_lru_cache/src/benchmarks/img_6.png)


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
