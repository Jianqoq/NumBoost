#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "ufunc_backward_def.h"
#include "omp.h"

#define Div(val1, val2, result, input_type, output_type)                       \
  do {                                                                         \
    if (!(val2)) {                                                             \
      if (val1 > 0)                                                            \
        result = Use_Inf(output_type);                                         \
      else if (val1 < 0)                                                       \
        result = -Use_Inf(output_type);                                        \
      else                                                                     \
        result = Use_Nan(output_type);                                         \
      continue;                                                                \
    } else                                                                     \
      result = Demote(output_type,                                             \
                      Demote(input_type, val1) / Demote(input_type, val2));    \
  } while (0)

#define Div2(val1, val2, result, input_type, output_type)                      \
  do {                                                                         \
    if (!(val2)) {                                                             \
      if (val1 > 0)                                                            \
        result = Use_Inf(output_type);                                         \
      else if (val1 < 0)                                                       \
        result = -Use_Inf(output_type);                                        \
      else                                                                     \
        result = Use_Nan(output_type);                                         \
    } else                                                                     \
      result = Demote(output_type,                                             \
                      Demote(input_type, val1) / Demote(input_type, val2));    \
  } while (0)

/*==========================================================================================================================================================*/
/*sin backward fusion*/
#define SinBackward_LoopBody(generic_type, type, i, result_ptr, a_idx, b_idx,  \
                             a_ptr, b_ptr)                                     \
  generic_type val1 = Map_Method(type, npy_cos, a_ptr[a_idx]);                 \
  generic_type val2 = Promote(type, b_ptr[b_idx]);                             \
  result_ptr[i] = Demote(type, val1 * val2);

#define Sin_LoopBody(generic_type, type, i, result_ptr, a_idx, a_ptr)          \
  generic_type a_val = Promote(type, a_ptr[a_idx]);                            \
  generic_type result = Map_Method(generic_type, npy_sin, a_val);              \
  result_ptr[i] = Demote(type, result);

Register_FuseBackward_Operation_FloatingTypes(sin, SinBackward_LoopBody, a, b);
Register_FuseBackward_Operation_Err_Int(sin, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(sin, a, b);
Register_FuseBackward_Operation_Array(sin, a, b)
/*==========================================================================================================================================================*/
/*cos backward fusion*/
#define CosBackward_LoopBody(generic_type, type, i, result_ptr, a_idx, b_idx,  \
                             a_ptr, b_ptr)                                     \
  generic_type val1 = -Map_Method(type, npy_sin, a_ptr[a_idx]);                \
  generic_type val2 = Promote(type, b_ptr[b_idx]);                             \
  result_ptr[i] = Demote(type, val1 * val2);

    Register_FuseBackward_Operation_FloatingTypes(cos, CosBackward_LoopBody, a,
                                                  b);
Register_FuseBackward_Operation_Err_Int(cos, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(cos, a, b);
Register_FuseBackward_Operation_Array(cos, a, b);
/*==========================================================================================================================================================*/
/*tan backward pow(1 / cos(x), 2)*/
#define TanBackward_LoopBody(generic_type, type, i, result_ptr, a_idx, b_idx,  \
                             a_ptr, b_ptr)                                     \
  generic_type cos_x = Map_Method(type, npy_cos, a_ptr[a_idx]);                \
  generic_type grad = Promote(type, b_ptr[b_idx]);                             \
  generic_type val3;                                                           \
  Div(1, cos_x, val3, generic_type, generic_type);                             \
  result_ptr[i] = Demote(type, grad * val3 * val3);

Register_FuseBackward_Operation_FloatingTypes(tan, TanBackward_LoopBody, a, b);
Register_FuseBackward_Operation_Err_Int(tan, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(tan, a, b);
Register_FuseBackward_Operation_Array(tan, a, b);
/*==========================================================================================================================================================*/
/*arcsin backward fusion*/
#define ArcsinBackward_LoopBody(generic_type, type, i, result_ptr, a_idx,      \
                                b_idx, a_ptr, b_ptr)                           \
  generic_type a_val = Promote(type, a_ptr[a_idx]);                            \
  generic_type square_result = a_val * a_val;                                  \
  generic_type sub_result = 1 - square_result;                                 \
  generic_type sqrt_result = Map_Method(generic_type, npy_sqrt, sub_result);           \
  generic_type reciprocal_sqrt_result;                                         \
  Div2(1, sqrt_result, reciprocal_sqrt_result, generic_type, generic_type);    \
  generic_type b_val = Promote(type, b_ptr[b_idx]);                            \
  result_ptr[i] = Demote(type, b_val * reciprocal_sqrt_result);

Register_FuseBackward_Operation_FloatingTypes(arcsin, ArcsinBackward_LoopBody,
                                              a, b);
Register_FuseBackward_Operation_Err_Int(arcsin, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(arcsin, a, b);
Register_FuseBackward_Operation_Array(arcsin, a, b);
/*==========================================================================================================================================================*/
/*arccos backward fusion*/
#define ArccosBackward_LoopBody(generic_type, type, i, result_ptr, a_idx,      \
                                b_idx, a_ptr, b_ptr)                           \
  generic_type a_val = Promote(type, a_ptr[a_idx]);                            \
  generic_type square_result = a_val * a_val;                                  \
  generic_type sub_result = 1 - square_result;                                 \
  generic_type sqrt_result = Map_Method(generic_type, npy_sqrt, sub_result);           \
  generic_type reciprocal_sqrt_result;                                         \
  Div2(1, sqrt_result, reciprocal_sqrt_result, generic_type, generic_type);    \
  generic_type b_val = Promote(type, b_ptr[b_idx]);                            \
  result_ptr[i] = -Demote(type, b_val * reciprocal_sqrt_result);

Register_FuseBackward_Operation_FloatingTypes(arccos, ArccosBackward_LoopBody,
                                              a, b);
Register_FuseBackward_Operation_Err_Int(arccos, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(arccos, a, b);
Register_FuseBackward_Operation_Array(arccos, a, b);
/*==========================================================================================================================================================*/
/*arctan backward fusion*/
#define ArctanBackward_LoopBody(generic_type, type, i, result_ptr, a_idx,      \
                                b_idx, a_ptr, b_ptr)                           \
  generic_type a_val = Promote(type, a_ptr[a_idx]);                            \
  generic_type square_result = a_val * a_val;                                  \
  generic_type add_result = 1 + square_result;                                 \
  generic_type reciprocal_add_result;                                          \
  Div2(1, add_result, reciprocal_add_result, generic_type, generic_type);      \
  generic_type b_val = Promote(type, b_ptr[b_idx]);                            \
  result_ptr[i] = Demote(type, b_val * reciprocal_add_result);

Register_FuseBackward_Operation_FloatingTypes(arctan, ArctanBackward_LoopBody,
                                              a, b);
Register_FuseBackward_Operation_Err_Int(arctan, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(arctan, a, b);
Register_FuseBackward_Operation_Array(arctan, a, b);
/*==========================================================================================================================================================*/
/*sinh backward fusion*/
#define SinhBackward_LoopBody(generic_type, type, i, result_ptr, a_idx, b_idx, \
                              a_ptr, b_ptr)                                    \
  generic_type a_val = Map_Method(type, npy_cosh, a_ptr[a_idx]);               \
  generic_type b_val = Promote(type, b_ptr[b_idx]);                            \
  result_ptr[i] = Demote(type, a_val * b_val);

Register_FuseBackward_Operation_FloatingTypes(sinh, SinhBackward_LoopBody, a,
                                              b);
Register_FuseBackward_Operation_Err_Int(sinh, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(sinh, a, b);
Register_FuseBackward_Operation_Array(sinh, a, b);
/*==========================================================================================================================================================*/
/*cosh backward fusion*/
#define CoshBackward_LoopBody(generic_type, type, i, result_ptr, a_idx, b_idx, \
                              a_ptr, b_ptr)                                    \
  generic_type a_val = Map_Method(type, npy_sinh, a_ptr[a_idx]);               \
  generic_type b_val = Promote(type, b_ptr[b_idx]);                            \
  result_ptr[i] = Demote(type, a_val * b_val);

Register_FuseBackward_Operation_FloatingTypes(cosh, CoshBackward_LoopBody, a,
                                              b);
Register_FuseBackward_Operation_Err_Int(cosh, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(cosh, a, b);
Register_FuseBackward_Operation_Array(cosh, a, b);
/*==========================================================================================================================================================*/
/*tanh backward fusion*/
#define TanhBackward_LoopBody(generic_type, type, i, result_ptr, a_idx, b_idx, \
                              a_ptr, b_ptr)                                    \
  generic_type a_val = Promote(type, a_ptr[a_idx]);                            \
  generic_type b_val = Promote(type, b_ptr[b_idx]);                            \
  result_ptr[i] = Demote(type, (1 - a_val * a_val) * b_val);

Register_FuseBackward_Operation_FloatingTypes(tanh, TanhBackward_LoopBody, a,
                                              b);
Register_FuseBackward_Operation_Err_Int(tanh, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(tanh, a, b);
Register_FuseBackward_Operation_Array(tanh, a, b);
/*==========================================================================================================================================================*/
/*arcsinh backward fusion*/
#define ArcsinhBackward_LoopBody(generic_type, type, i, result_ptr, a_idx,     \
                                 b_idx, a_ptr, b_ptr)                          \
  generic_type a_val = Promote(type, a_ptr[a_idx]);                            \
  generic_type square_result = a_val * a_val;                                  \
  generic_type add_result = 1 + square_result;                                 \
  generic_type sqrt_result = Map_Method(generic_type, npy_sqrt, add_result);   \
  generic_type b_val = Promote(type, b_ptr[b_idx]);                            \
  Div(b_val, sqrt_result, result_ptr[i], generic_type, type);

Register_FuseBackward_Operation_FloatingTypes(arcsinh, ArcsinhBackward_LoopBody,
                                              a, b);
Register_FuseBackward_Operation_Err_Int(arcsinh, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(arcsinh, a, b);
Register_FuseBackward_Operation_Array(arcsinh, a, b);
/*==========================================================================================================================================================*/
/*arccosh backward fusion*/
#define ArccoshBackward_LoopBody(generic_type, type, i, result_ptr, a_idx,     \
                                 b_idx, a_ptr, b_ptr)                          \
  generic_type a_val = Promote(type, a_ptr[a_idx]);                            \
  generic_type square_result = a_val * a_val;                                  \
  generic_type sub_result = square_result - 1;                                 \
  generic_type sqrt_result = Map_Method(generic_type, npy_sqrt, sub_result);   \
  generic_type b_val = Promote(type, b_ptr[b_idx]);                            \
  Div(b_val, sqrt_result, result_ptr[i], generic_type, type);

Register_FuseBackward_Operation_FloatingTypes(arccosh, ArccoshBackward_LoopBody,
                                              a, b);
Register_FuseBackward_Operation_Err_Int(arccosh, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(arccosh, a, b);
Register_FuseBackward_Operation_Array(arccosh, a, b);
/*==========================================================================================================================================================*/
/*arctanh backward fusion*/
#define ArctanhBackward_LoopBody(generic_type, type, i, result_ptr, a_idx,     \
                                 b_idx, a_ptr, b_ptr)                          \
  generic_type a_val = Promote(type, a_ptr[a_idx]);                            \
  generic_type square_result = a_val * a_val;                                  \
  generic_type sub_result = 1 - square_result;                                 \
  generic_type b_val = Promote(type, b_ptr[b_idx]);                            \
  Div(b_val, sub_result, result_ptr[i], generic_type, type);

Register_FuseBackward_Operation_FloatingTypes(arctanh, ArctanhBackward_LoopBody,
                                              a, b);
Register_FuseBackward_Operation_Err_Int(arctanh, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(arctanh, a, b);
Register_FuseBackward_Operation_Array(arctanh, a, b);
/*==========================================================================================================================================================*/
/*exp backward fusion*/
#define ExpBackward_LoopBody(generic_type, type, i, result_ptr, a_idx, b_idx,  \
                             a_ptr, b_ptr)                                     \
  generic_type a_val = Promote(type, a_ptr[a_idx]);                            \
  generic_type exp_result = Map_Method(generic_type, npy_exp, a_val);                  \
  generic_type b_val = Promote(type, b_ptr[b_idx]);                            \
  result_ptr[i] = Demote(type, exp_result * b_val);

Register_FuseBackward_Operation_FloatingTypes(exp, ExpBackward_LoopBody, a, b);
Register_FuseBackward_Operation_Err_Int(exp, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(exp, a, b);
Register_FuseBackward_Operation_Array(exp, a, b);
/*==========================================================================================================================================================*/
/*log backward fusion*/
#define LogBackward_LoopBody(generic_type, type, i, result_ptr, a_idx, b_idx,  \
                             a_ptr, b_ptr)                                     \
  generic_type a_val = Promote(type, a_ptr[a_idx]);                            \
  generic_type b_val = Promote(type, b_ptr[b_idx]);                            \
  Div(a_val, b_val, result_ptr[i], generic_type, type);

Register_FuseBackward_Operation_FloatingTypes(log, LogBackward_LoopBody, a, b);
Register_FuseBackward_Operation_Err_Int(log, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(log, a, b);
Register_FuseBackward_Operation_Array(log, a, b);
/*==========================================================================================================================================================*/
/*log10 backward fusion*/
#define Log10Backward_LoopBody(generic_type, type, i, result_ptr, a_idx,       \
                               b_idx, a_ptr, b_ptr)                            \
  generic_type a_val = Promote(type, a_ptr[a_idx]);                            \
  generic_type b_val = Promote(type, b_ptr[b_idx]);                            \
  generic_type log_result = Map_Method(type, npy_log, 10);                     \
  generic_type mul_result = Demote(type, log_result * b_val);                  \
  Div(a_val, mul_result, result_ptr[i], generic_type, type);

Register_FuseBackward_Operation_FloatingTypes(log10, Log10Backward_LoopBody, a,
                                              b);
Register_FuseBackward_Operation_Err_Int(log10, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(log10, a, b);
Register_FuseBackward_Operation_Array(log10, a, b);
/*==========================================================================================================================================================*/
/*log2 backward fusion*/
/* future plan */
/*==========================================================================================================================================================*/
/*sqrt backward fusion*/
#define SqrtBackward_LoopBody(generic_type, type, i, result_ptr, a_idx, b_idx, \
                              a_ptr, b_ptr)                                    \
  generic_type a_val = Promote(type, a_ptr[a_idx]);                            \
  generic_type b_val = Promote(type, b_ptr[b_idx]);                            \
  generic_type mul_result = a_val * 2;                                         \
  Div(b_val, mul_result, result_ptr[i], generic_type, type);

Register_FuseBackward_Operation_FloatingTypes(sqrt, SqrtBackward_LoopBody, a,
                                              b);
Register_FuseBackward_Operation_Err_Int(sqrt, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(sqrt, a, b);
Register_FuseBackward_Operation_Array(sqrt, a, b);
/*==========================================================================================================================================================*/
/*abs backward fusion*/
#define AbsBackward_LoopBody(generic_type, type, i, result_ptr, a_idx, b_idx,  \
                             a_ptr, b_ptr)                                     \
  generic_type a_val = Promote(type, a_ptr[a_idx]);                            \
  generic_type b_val = Promote(type, b_ptr[b_idx]);                            \
  result_ptr[i] = Demote(type, a_val > 0 ? b_val : -b_val);

Register_FuseBackward_Operation_FloatingTypes(abs, AbsBackward_LoopBody, a, b);
Register_FuseBackward_Operation_IntergerTypes(abs, AbsBackward_LoopBody, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(abs, a, b);
Register_FuseBackward_Operation_Array(abs, a, b);
/*==========================================================================================================================================================*/
/*power backward fusion*/
#define PowerBackward_LoopBody(generic_type, type, i, result_ptr, a_idx,       \
                               power_idx, grad_idx, a_ptr, power_ptr,          \
                               grad_ptr)                                       \
  generic_type a_val = Promote(type, a_ptr[a_idx]);                            \
  generic_type power_val = Promote(type, power_ptr[power_idx]);                \
  generic_type grad_val = Promote(type, grad_ptr[grad_idx]);                   \
  generic_type tmp = Map_Method(generic_type, npy_pow, a_val, power_val - 1);          \
  result_ptr[i] = Demote(type, tmp * power_val * grad_val);

Register_FuseBackward_Operation_FloatingTypes(power, PowerBackward_LoopBody, a,
                                              b, c);
Register_FuseBackward_Operation_Err_Int(power, a, power, grad);
Register_FuseBackward_Operation_Err_UnsupportTypes(power, a, power, grad);
Register_FuseBackward_Operation_Array(power, a, power, grad);
/*==========================================================================================================================================================*/