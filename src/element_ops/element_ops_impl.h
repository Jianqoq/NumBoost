#include "../numboost_api.h"

#define Abs_LoopBody(generic_type, type, i, result_ptr, stride_a_last, a_ptr)  \
  generic_type a_val = Promote(type, a_ptr[i * stride_a_last]);                \
  result_ptr[i] = Demote(type, a_val > 0 ? a_val : -a_val);

#define Sin_LoopBody(generic_type, type, i, result_ptr, stride_a_last, a_ptr)  \
  generic_type a_val = Promote(type, a_ptr[i * stride_a_last]);                \
  generic_type result = Map_Method(generic_type, npy_sin, a_val);              \
  result_ptr[i] = Demote(type, result);

#define Cos_LoopBody(generic_type, type, i, result_ptr, stride_a_last, a_ptr)  \
  generic_type a_val = Promote(type, a_ptr[i * stride_a_last]);                \
  generic_type result = Map_Method(generic_type, npy_cos, a_val);              \
  result_ptr[i] = Demote(type, result);

#define Tan_LoopBody(generic_type, type, i, result_ptr, stride_a_last, a_ptr)  \
  generic_type a_val = Promote(type, a_ptr[i * stride_a_last]);                \
  generic_type result = Map_Method(generic_type, npy_tan, a_val);              \
  result_ptr[i] = Demote(type, result);

#define Asin_LoopBody(generic_type, type, i, result_ptr, stride_a_last, a_ptr) \
  generic_type a_val = Promote(type, a_ptr[i * stride_a_last]);                \
  generic_type result = Map_Method(generic_type, npy_asin, a_val);             \
  result_ptr[i] = Demote(type, result);

#define Acos_LoopBody(generic_type, type, i, result_ptr, stride_a_last, a_ptr) \
  generic_type a_val = Promote(type, a_ptr[i * stride_a_last]);                \
  generic_type result = Map_Method(generic_type, npy_acos, a_val);             \
  result_ptr[i] = Demote(type, result);

#define Atan_LoopBody(generic_type, type, i, result_ptr, stride_a_last, a_ptr) \
  generic_type a_val = Promote(type, a_ptr[i * stride_a_last]);                \
  generic_type result = Map_Method(generic_type, npy_atan, a_val);             \
  result_ptr[i] = Demote(type, result);

#define Sinh_LoopBody(generic_type, type, i, result_ptr, stride_a_last, a_ptr) \
  generic_type a_val = Promote(type, a_ptr[i * stride_a_last]);                \
  generic_type result = Map_Method(generic_type, npy_sinh, a_val);             \
  result_ptr[i] = Demote(type, result);

#define Cosh_LoopBody(generic_type, type, i, result_ptr, stride_a_last, a_ptr) \
  generic_type a_val = Promote(type, a_ptr[i * stride_a_last]);                \
  generic_type result = Map_Method(generic_type, npy_cosh, a_val);             \
  result_ptr[i] = Demote(type, result);

#define Tanh_LoopBody(generic_type, type, i, result_ptr, stride_a_last, a_ptr) \
  generic_type a_val = Promote(type, a_ptr[i * stride_a_last]);                \
  generic_type result = Map_Method(generic_type, npy_tanh, a_val);             \
  result_ptr[i] = Demote(type, result);

#define Asinh_LoopBody(generic_type, type, i, result_ptr, stride_a_last,       \
                       a_ptr)                                                  \
  generic_type a_val = Promote(type, a_ptr[i * stride_a_last]);                \
  generic_type result = Map_Method(generic_type, npy_asinh, a_val);            \
  result_ptr[i] = Demote(type, result);

#define Acosh_LoopBody(generic_type, type, i, result_ptr, stride_a_last,       \
                       a_ptr)                                                  \
  generic_type a_val = Promote(type, a_ptr[i * stride_a_last]);                \
  generic_type result = Map_Method(generic_type, npy_acosh, a_val);            \
  result_ptr[i] = Demote(type, result);

#define Atanh_LoopBody(generic_type, type, i, result_ptr, stride_a_last,       \
                       a_ptr)                                                  \
  generic_type a_val = Promote(type, a_ptr[i * stride_a_last]);                \
  generic_type result = Map_Method(generic_type, npy_atanh, a_val);            \
  result_ptr[i] = Demote(type, result);

#define Sqrt_LoopBody(generic_type, type, i, result_ptr, stride_a_last, a_ptr) \
  generic_type a_val = Promote(type, a_ptr[i * stride_a_last]);                \
  generic_type result = Map_Method(generic_type, npy_sqrt, a_val);             \
  result_ptr[i] = Demote(type, result);

#define Log_LoopBody(generic_type, type, i, result_ptr, stride_a_last, a_ptr)  \
  generic_type a_val = Promote(type, a_ptr[i * stride_a_last]);                \
  generic_type result = Map_Method(generic_type, npy_log, a_val);              \
  result_ptr[i] = Demote(type, result);

#define Log10_LoopBody(generic_type, type, i, result_ptr, stride_a_last,       \
                       a_ptr)                                                  \
  generic_type a_val = Promote(type, a_ptr[i * stride_a_last]);                \
  generic_type result = Map_Method(generic_type, npy_log10, a_val);            \
  result_ptr[i] = Demote(type, result);

#define Exp_LoopBody(generic_type, type, i, result_ptr, stride_a_last, a_ptr)  \
  generic_type a_val = Promote(type, a_ptr[i * stride_a_last]);                \
  generic_type result = Map_Method(generic_type, npy_exp, a_val);              \
  result_ptr[i] = Demote(type, result);

#define Negative_LoopBody(generic_type, type, i, result_ptr, stride_a_last,    \
                          a_ptr)                                               \
  generic_type a_val = Promote(type, a_ptr[i * stride_a_last]);                \
  result_ptr[i] = Demote(type, -a_val);