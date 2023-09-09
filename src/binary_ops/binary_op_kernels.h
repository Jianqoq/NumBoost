#ifndef BINARY_FUNC_H
#define BINARY_FUNC_H
#include "../numboost_api.h"
#include "../numboost_math.h"
#include <numpy/npy_math.h>

#define Pow_LoopBody(generic_type, type, i, result_ptr, a_last_stride,         \
                     pow_last_stride, a_ptr, power_ptr)                        \
  generic_type a_val = Promote(type, a_ptr[i * a_last_stride]);                \
  generic_type power_val = Promote(type, power_ptr[i * pow_last_stride]);      \
  generic_type result = Map_Method(generic_type, npy_pow, a_val, power_val);   \
  result_ptr[i] = Demote(type, result);

#define Add_LoopBody(generic_type, type, i, result_ptr, a_last_stride,         \
                     b_last_stride, a_ptr, b_ptr)                              \
  generic_type a_val = Promote(type, a_ptr[i * a_last_stride]);                \
  generic_type b_val = Promote(type, b_ptr[i * b_last_stride]);                \
  generic_type result = a_val + b_val;                                         \
  result_ptr[i] = Demote(type, result);

#define Sub_LoopBody(generic_type, type, i, result_ptr, a_last_stride,         \
                     b_last_stride, a_ptr, b_ptr)                              \
  generic_type a_val = Promote(type, a_ptr[i * a_last_stride]);                \
  generic_type b_val = Promote(type, b_ptr[i * b_last_stride]);                \
  generic_type result = a_val - b_val;                                         \
  result_ptr[i] = Demote(type, result);

#define Mul_LoopBody(generic_type, type, i, result_ptr, a_last_stride,         \
                     b_last_stride, a_ptr, b_ptr)                              \
  generic_type a_val = Promote(type, a_ptr[i * a_last_stride]);                \
  generic_type b_val = Promote(type, b_ptr[i * b_last_stride]);                \
  generic_type result = a_val * b_val;                                         \
  result_ptr[i] = Demote(type, result);

#define Div_LoopBody(generic_type, type, i, result_ptr, a_last_stride,         \
                     b_last_stride, a_ptr, b_ptr)                              \
  generic_type a_val = Promote(type, a_ptr[i * a_last_stride]);                \
  generic_type b_val = Promote(type, b_ptr[i * b_last_stride]);                \
  generic_type result;                                                         \
  Div(a_val, b_val, result, generic_type, generic_type);                       \
  result_ptr[i] = Demote(type, result);

#define LShift_LoopBody(generic_type, type, i, result_ptr, a_last_stride,      \
                        b_last_stride, a_ptr, b_ptr)                           \
  generic_type a_val = Promote(type, a_ptr[i * a_last_stride]);                \
  generic_type b_val = Promote(type, b_ptr[i * b_last_stride]);                \
  generic_type result = (a_val << b_val) * (b_val < sizeof(b_val) * 8);        \
  result_ptr[i] = Demote(type, result);

#define RShift_LoopBody(generic_type, type, i, result_ptr, a_last_stride,      \
                        b_last_stride, a_ptr, b_ptr)                           \
  generic_type a_val = Promote(type, a_ptr[i * a_last_stride]);                \
  generic_type b_val = Promote(type, b_ptr[i * b_last_stride]);                \
  generic_type result = (a_val >> b_val) * (b_val < sizeof(b_val) * 8);        \
  result_ptr[i] = Demote(type, result);

#define Mod_LoopBody(generic_type, type, i, result_ptr, a_last_stride,         \
                     b_last_stride, a_ptr, b_ptr)                              \
  generic_type b_val = Promote(type, b_ptr[i * b_last_stride]);                \
  if (!b_val) {                                                                \
    result_ptr[i] = 0;                                                         \
    continue;                                                                  \
  }                                                                            \
  generic_type a_val = Promote(type, a_ptr[i * a_last_stride]);                \
  generic_type ret = Map_Method(generic_type, nb_mod, a_val, b_val);           \
  ret += ((ret != 0) & ((a_val < 0) != (b_val < 0))) * b_val;                  \
  result_ptr[i] = ret;

#define FloorDiv_LoopBody(generic_type, type, i, result_ptr, a_last_stride,    \
                          b_last_stride, a_ptr, b_ptr)                         \
  generic_type a_val = Promote(type, a_ptr[i * a_last_stride]);                \
  generic_type b_val = Promote(type, b_ptr[i * b_last_stride]);                \
  generic_type result = Map_Method(generic_type, nb_fdiv, a_val, b_val);       \
  result_ptr[i] = Demote(type, result);

#endif