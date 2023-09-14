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
  generic_type a_val = a_ptr[i * a_last_stride];                               \
  generic_type b_val = b_ptr[i * b_last_stride];                               \
  result_ptr[i] = (a_val << b_val) * (b_val < (int)(sizeof(b_val) * 8));

#define RShift_LoopBody(generic_type, type, i, result_ptr, a_last_stride,      \
                        b_last_stride, a_ptr, b_ptr)                           \
  generic_type a_val = a_ptr[i * a_last_stride];                               \
  generic_type b_val = b_ptr[i * b_last_stride];                               \
  result_ptr[i] = (a_val >> b_val) * (b_val < (int)(sizeof(b_val) * 8));

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
  result_ptr[i] = Demote(type, ret);

#define FloorDiv_LoopBody(generic_type, type, i, result_ptr, a_last_stride,    \
                          b_last_stride, a_ptr, b_ptr)                         \
  generic_type a_val = Promote(type, a_ptr[i * a_last_stride]);                \
  generic_type b_val = Promote(type, b_ptr[i * b_last_stride]);                \
  generic_type mod_result = Map_Method(generic_type, nb_mod, a_val, b_val);    \
  generic_type div_res;                                                        \
  Div((a_val - mod_result), b_val, div_res, generic_type, generic_type);       \
  if (mod_result) {                                                            \
    if ((b_val < 0) != (mod_result < 0)) {                                     \
      mod_result += b_val;                                                     \
      div_res -= 1;                                                            \
    }                                                                          \
  } else {                                                                     \
    mod_result = (generic_type)npy_copysign(0.0, (npy_double)b_val);           \
  }                                                                            \
  if (div_res) {                                                               \
    result_ptr[i] = Demote(type, Map_Method(generic_type, nb_floor, div_res)); \
    if (div_res - result_ptr[i] > 0.5) {                                       \
      result_ptr[i] += 1;                                                      \
    }                                                                          \
  } else {                                                                     \
    generic_type div_res2;                                                     \
    Div(a_val, b_val, div_res2, generic_type, generic_type);                   \
    result_ptr[i] =                                                            \
        Demote(type, (generic_type)npy_copysign(0.0, (npy_double)div_res2));   \
  }

/*logic operation don't need to promote data type since it doesn't support float
 * type*/
#define And_LoopBody(generic_type, type, i, result_ptr, a_last_stride,         \
                     b_last_stride, a_ptr, b_ptr)                              \
  result_ptr[i] = a_ptr[i * a_last_stride] & b_ptr[i * b_last_stride];

#define Xor_LoopBody(generic_type, type, i, result_ptr, a_last_stride,         \
                     b_last_stride, a_ptr, b_ptr)                              \
  result_ptr[i] = a_ptr[i * a_last_stride] ^ b_ptr[i * b_last_stride];

#define Or_LoopBody(generic_type, type, i, result_ptr, a_last_stride,          \
                    b_last_stride, a_ptr, b_ptr)                               \
  result_ptr[i] = a_ptr[i * a_last_stride] | b_ptr[i * b_last_stride];

#define DivMod_LoopBody(generic_type, type, i, quotient_ptr, remainder_ptr,    \
                        a_last_stride, b_last_stride, a_ptr, b_ptr)            \
  generic_type b_val = Promote(type, b_ptr[i * b_last_stride]);                \
  generic_type a_val = Promote(type, a_ptr[i * a_last_stride]);                \
  generic_type mod_result = Map_Method(generic_type, nb_mod, a_val, b_val);    \
  if (!b_val) {                                                                \
    remainder_ptr[i] = 0;                                                      \
  } else {                                                                     \
    remainder_ptr[i] = Demote(                                                 \
        type, mod_result +                                                     \
                  ((mod_result != 0) & ((a_val < 0) != (b_val < 0))) * b_val); \
  }                                                                            \
  generic_type div_res;                                                        \
  Div((a_val - mod_result), b_val, div_res, generic_type, generic_type);       \
  if (mod_result) {                                                            \
    if ((b_val < 0) != (mod_result < 0)) {                                     \
      mod_result += b_val;                                                     \
      div_res -= 1;                                                            \
    }                                                                          \
  } else {                                                                     \
    mod_result = (generic_type)npy_copysign(0.0, (npy_double)b_val);           \
  }                                                                            \
  if (div_res) {                                                               \
    quotient_ptr[i] =                                                          \
        Demote(type, Map_Method(generic_type, nb_floor, div_res));             \
    if (div_res - quotient_ptr[i] > 0.5) {                                     \
      quotient_ptr[i] += 1;                                                    \
    }                                                                          \
  } else {                                                                     \
    generic_type div_res2;                                                     \
    Div(a_val, b_val, div_res2, generic_type, generic_type);                   \
    quotient_ptr[i] =                                                          \
        Demote(type, (generic_type)npy_copysign(0.0, (npy_double)div_res2));   \
  }

#endif