#ifndef AUTO_DIFF_UFUNC_KERNELS_H_
#define AUTO_DIFF_UFUNC_KERNELS_H_

#define SinBackward_LoopBody(generic_type, type, i, result_ptr, stride_a_last, \
                             stride_b_last, a_ptr, b_ptr)                      \
  generic_type val1 = Map_Method(type, npy_cos, a_ptr[i * stride_a_last]);     \
  generic_type val2 = Promote(type, b_ptr[i * stride_b_last]);                 \
  result_ptr[i] = Demote(type, val1 * val2);

#define CosBackward_LoopBody(generic_type, type, i, result_ptr, stride_a_last, \
                             stride_b_last, a_ptr, b_ptr)                      \
  generic_type val1 = -Map_Method(type, npy_sin, a_ptr[i * stride_a_last]);    \
  generic_type val2 = Promote(type, b_ptr[i * stride_b_last]);                 \
  result_ptr[i] = Demote(type, val1 * val2);

#define TanBackward_LoopBody(generic_type, type, i, result_ptr, stride_a_last, \
                             stride_b_last, a_ptr, b_ptr)                      \
  generic_type cos_x = Map_Method(type, npy_cos, a_ptr[i * stride_a_last]);    \
  generic_type grad = Promote(type, b_ptr[i * stride_b_last]);                 \
  generic_type val3;                                                           \
  Div(1, cos_x, val3, generic_type, generic_type);                             \
  result_ptr[i] = Demote(type, grad * val3 * val3);

#define ArcsinBackward_LoopBody(generic_type, type, i, result_ptr,             \
                                stride_a_last, stride_b_last, a_ptr, b_ptr)    \
  generic_type a_val = Promote(type, a_ptr[i * stride_a_last]);                \
  generic_type square_result = a_val * a_val;                                  \
  generic_type sub_result = 1 - square_result;                                 \
  generic_type sqrt_result = Map_Method(generic_type, npy_sqrt, sub_result);   \
  generic_type reciprocal_sqrt_result;                                         \
  Div2(1, sqrt_result, reciprocal_sqrt_result, generic_type, generic_type);    \
  generic_type b_val = Promote(type, b_ptr[i * stride_b_last]);                \
  result_ptr[i] = Demote(type, b_val * reciprocal_sqrt_result);

#define ArccosBackward_LoopBody(generic_type, type, i, result_ptr,             \
                                stride_a_last, stride_b_last, a_ptr, b_ptr)    \
  generic_type a_val = Promote(type, a_ptr[i * stride_a_last]);                \
  generic_type square_result = a_val * a_val;                                  \
  generic_type sub_result = 1 - square_result;                                 \
  generic_type sqrt_result = Map_Method(generic_type, npy_sqrt, sub_result);   \
  generic_type reciprocal_sqrt_result;                                         \
  Div2(1, sqrt_result, reciprocal_sqrt_result, generic_type, generic_type);    \
  generic_type b_val = Promote(type, b_ptr[i * stride_b_last]);                \
  result_ptr[i] = -Demote(type, b_val * reciprocal_sqrt_result);

#define ArctanBackward_LoopBody(generic_type, type, i, result_ptr,             \
                                stride_a_last, stride_b_last, a_ptr, b_ptr)    \
  generic_type a_val = Promote(type, a_ptr[i * stride_a_last]);                \
  generic_type square_result = a_val * a_val;                                  \
  generic_type add_result = 1 + square_result;                                 \
  generic_type reciprocal_add_result;                                          \
  Div2(1, add_result, reciprocal_add_result, generic_type, generic_type);      \
  generic_type b_val = Promote(type, b_ptr[i * stride_b_last]);                \
  result_ptr[i] = Demote(type, b_val * reciprocal_add_result);

#define SinhBackward_LoopBody(generic_type, type, i, result_ptr,               \
                              stride_a_last, stride_b_last, a_ptr, b_ptr)      \
  generic_type a_val = Map_Method(type, npy_cosh, a_ptr[i * stride_a_last]);   \
  generic_type b_val = Promote(type, b_ptr[i * stride_b_last]);                \
  result_ptr[i] = Demote(type, a_val * b_val);

#define CoshBackward_LoopBody(generic_type, type, i, result_ptr,               \
                              stride_a_last, stride_b_last, a_ptr, b_ptr)      \
  generic_type a_val = Map_Method(type, npy_sinh, a_ptr[i * stride_a_last]);   \
  generic_type b_val = Promote(type, b_ptr[i * stride_b_last]);                \
  result_ptr[i] = Demote(type, a_val * b_val);

#define TanhBackward_LoopBody(generic_type, type, i, result_ptr,               \
                              stride_a_last, stride_b_last, a_ptr, b_ptr)      \
  generic_type a_val = Promote(type, a_ptr[i * stride_a_last]);                \
  generic_type b_val = Promote(type, b_ptr[i * stride_b_last]);                \
  result_ptr[i] = Demote(type, (1 - a_val * a_val) * b_val);

#define ArcsinhBackward_LoopBody(generic_type, type, i, result_ptr,            \
                                 stride_a_last, stride_b_last, a_ptr, b_ptr)   \
  generic_type a_val = Promote(type, a_ptr[i * stride_a_last]);                \
  generic_type square_result = a_val * a_val;                                  \
  generic_type add_result = 1 + square_result;                                 \
  generic_type sqrt_result = Map_Method(generic_type, npy_sqrt, add_result);   \
  generic_type b_val = Promote(type, b_ptr[i * stride_b_last]);                \
  Div(b_val, sqrt_result, result_ptr[i], generic_type, type);

#define ArccoshBackward_LoopBody(generic_type, type, i, result_ptr,            \
                                 stride_a_last, stride_b_last, a_ptr, b_ptr)   \
  generic_type a_val = Promote(type, a_ptr[i * stride_a_last]);                \
  generic_type square_result = a_val * a_val;                                  \
  generic_type sub_result = square_result - 1;                                 \
  generic_type sqrt_result = Map_Method(generic_type, npy_sqrt, sub_result);   \
  generic_type b_val = Promote(type, b_ptr[i * stride_b_last]);                \
  Div(b_val, sqrt_result, result_ptr[i], generic_type, type);

#define ArctanhBackward_LoopBody(generic_type, type, i, result_ptr,            \
                                 stride_a_last, stride_b_last, a_ptr, b_ptr)   \
  generic_type a_val = Promote(type, a_ptr[i * stride_a_last]);                \
  generic_type square_result = a_val * a_val;                                  \
  generic_type sub_result = 1 - square_result;                                 \
  generic_type b_val = Promote(type, b_ptr[i * stride_b_last]);                \
  Div(b_val, sub_result, result_ptr[i], generic_type, type);

#define ExpBackward_LoopBody(generic_type, type, i, result_ptr, stride_a_last, \
                             stride_b_last, a_ptr, b_ptr)                      \
  generic_type a_val = Promote(type, a_ptr[i * stride_a_last]);                \
  generic_type exp_result = Map_Method(generic_type, npy_exp, a_val);          \
  generic_type b_val = Promote(type, b_ptr[i * stride_b_last]);                \
  result_ptr[i] = Demote(type, exp_result * b_val);

#define LogBackward_LoopBody(generic_type, type, i, result_ptr, stride_a_last, \
                             stride_b_last, a_ptr, b_ptr)                      \
  generic_type a_val = Promote(type, a_ptr[i * stride_a_last]);                \
  generic_type b_val = Promote(type, b_ptr[i * stride_b_last]);                \
  Div(a_val, b_val, result_ptr[i], generic_type, type);

#define Log10Backward_LoopBody(generic_type, type, i, result_ptr,              \
                               stride_a_last, stride_b_last, a_ptr, b_ptr)     \
  generic_type a_val = Promote(type, a_ptr[i * stride_a_last]);                \
  generic_type b_val = Promote(type, b_ptr[i * stride_b_last]);                \
  generic_type log_result = Map_Method(type, npy_log, 10);                     \
  generic_type mul_result = Demote(type, log_result * b_val);                  \
  Div(a_val, mul_result, result_ptr[i], generic_type, type);

#define SqrtBackward_LoopBody(generic_type, type, i, result_ptr,               \
                              stride_a_last, stride_b_last, a_ptr, b_ptr)      \
  generic_type a_val = Promote(type, a_ptr[i * stride_a_last]);                \
  generic_type b_val = Promote(type, b_ptr[i * stride_b_last]);                \
  generic_type mul_result = a_val * 2;                                         \
  Div(b_val, mul_result, result_ptr[i], generic_type, type);

#define AbsBackward_LoopBody(generic_type, type, i, result_ptr, stride_a_last, \
                             stride_b_last, a_ptr, b_ptr)                      \
  generic_type a_val = Promote(type, a_ptr[i * stride_a_last]);                \
  generic_type b_val = Promote(type, b_ptr[i * stride_b_last]);                \
  result_ptr[i] = Demote(type, a_val > 0 ? b_val : -b_val);

#define PowerBackward_LoopBody(generic_type, type, i, result_ptr,              \
                               stride_a_last, power_idx, grad_idx, a_ptr,      \
                               power_ptr, grad_ptr)                            \
  generic_type a_val = Promote(type, a_ptr[i * stride_a_last]);                \
  generic_type power_val = Promote(type, power_ptr[power_idx]);                \
  generic_type grad_val = Promote(type, grad_ptr[grad_idx]);                   \
  generic_type tmp = Map_Method(generic_type, npy_pow, a_val, power_val - 1);  \
  result_ptr[i] = Demote(type, tmp * power_val * grad_val);
#endif // AUTO_DIFF_UFUNC_KERNELS_H_