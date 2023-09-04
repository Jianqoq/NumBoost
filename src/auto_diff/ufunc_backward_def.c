#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "ufunc_backward_def.h"
#include "omp.h"

#define Div(val1, val2, result, type)                                          \
  if (!(val2)) {                                                               \
    if (val1 > 0)                                                              \
      result = Use_Inf(type);                                                  \
    else if (val1 < 0)                                                         \
      result = -Use_Inf(type);                                                 \
    else                                                                       \
      result = Use_Nan(type);                                                  \
    continue;                                                                  \
  } else                                                                       \
    result = Cast_Half_When_Half(type, val1 / (val2));

#define Div2(val1, val2, result, type)                                         \
  if (!(val2)) {                                                               \
    if (val1 > 0)                                                              \
      result = Use_Inf(type);                                                  \
    else if (val1 < 0)                                                         \
      result = -Use_Inf(type);                                                 \
    else                                                                       \
      result = Use_Nan(type);                                                  \
  } else                                                                       \
    result = Cast_Half_When_Half(type, val1 / (val2));

/*==========================================================================================================================================================*/
/*sin backward fusion*/
#define SinBackward_LoopBody(type, i, result_ptr, stride_a_last,               \
                             stride_b_last, a_ptr, b_ptr)                      \
  Use_Float_When_Half(type) val1 =                                             \
      Use_Method(type, npy_cos, a_ptr[i * stride_a_last]);                     \
  Use_Float_When_Half(type) val2 =                                             \
      Cast_Float_When_Half(type, b_ptr[i * stride_b_last]);                    \
  result_ptr[i] = Cast_Half_When_Half(type, val1 * val2);

#define SinBackward_LoopBody_Sequential(type, i, result_ptr, a_ptr, b_ptr)     \
  Use_Float_When_Half(type) val1 = Use_Method(type, npy_cos, a_ptr[i]);        \
  Use_Float_When_Half(type) val2 = Cast_Float_When_Half(type, b_ptr[i]);       \
  result_ptr[i] = Cast_Half_When_Half(type, val1 * val2);

Register_FuseBackward_Operation_FloatingTypes(sin, SinBackward_LoopBody,
                                              SinBackward_LoopBody_Sequential,
                                              a, b);
Register_FuseBackward_Operation_Err_Int(sin, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(sin, a, b);
Register_FuseBackward_Operation_Array(sin, a, b)
/*==========================================================================================================================================================*/
/*cos backward fusion*/
#define CosBackward_LoopBody(type, i, result_ptr, stride_a_last,               \
                             stride_b_last, a_ptr, b_ptr)                      \
  Use_Float_When_Half(type) val1 =                                             \
      -Use_Method(type, npy_sin, a_ptr[i * stride_a_last]);                    \
  Use_Float_When_Half(type) val2 =                                             \
      Cast_Float_When_Half(type, b_ptr[i * stride_b_last]);                    \
  result_ptr[i] = Cast_Half_When_Half(type, val1 * val2);

#define CosBackward_LoopBody_Sequential(type, i, result_ptr, a_ptr, b_ptr)     \
  Use_Float_When_Half(type) val1 = -Use_Method(type, npy_sin, a_ptr[i]);       \
  Use_Float_When_Half(type) val2 = Cast_Float_When_Half(type, b_ptr[i]);       \
  result_ptr[i] = Cast_Half_When_Half(type, val1 * val2);

    Register_FuseBackward_Operation_FloatingTypes(
        cos, CosBackward_LoopBody, CosBackward_LoopBody_Sequential, a, b);
Register_FuseBackward_Operation_Err_Int(cos, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(cos, a, b);
Register_FuseBackward_Operation_Array(cos, a, b);
/*==========================================================================================================================================================*/
/*tan backward pow(1 / cos(x), 2)*/
#define TanBackward_LoopBody(type, i, result_ptr, stride_a_last,               \
                             stride_b_last, a_ptr, b_ptr)                      \
  Use_Float_When_Half(type) cos_x =                                            \
      Use_Method(type, npy_cos, a_ptr[i * stride_a_last]);                     \
  Use_Float_When_Half(type) grad =                                             \
      Cast_Float_When_Half(type, b_ptr[i * stride_b_last]);                    \
  Use_Float_When_Half(type) val3;                                              \
  Div(1, cos_x, val3, type);                                                   \
  result_ptr[i] = Cast_Half_When_Half(type, grad * val3 * val3);

#define TanBackward_LoopBody_Sequential(type, i, result_ptr, a_ptr, b_ptr)     \
  Use_Float_When_Half(type) cos_x = Use_Method(type, npy_cos, a_ptr[i]);       \
  Use_Float_When_Half(type) val2 = Cast_Float_When_Half(type, b_ptr[i]);       \
  Use_Float_When_Half(type) val3;                                              \
  Div(1, cos_x, val3, type);                                                   \
  result_ptr[i] = Cast_Half_When_Half(type, val2 * val3 * val3);

Register_FuseBackward_Operation_FloatingTypes(tan, TanBackward_LoopBody,
                                              TanBackward_LoopBody_Sequential,
                                              a, b);
Register_FuseBackward_Operation_Err_Int(tan, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(tan, a, b);
Register_FuseBackward_Operation_Array(tan, a, b);
/*==========================================================================================================================================================*/
/*arcsin backward fusion*/
#define ArcsinBackward_LoopBody(type, i, result_ptr, stride_a_last,            \
                                stride_b_last, a_ptr, b_ptr)                   \
  Use_Float_When_Half(type) a_val =                                            \
      Cast_Float_When_Half(type, a_ptr[i * stride_a_last]);                    \
  Use_Float_When_Half(type) square_result = a_val * a_val;                     \
  Use_Float_When_Half(type) sub_result = 1 - square_result;                    \
  Use_Float_When_Half(type) sqrt_result =                                      \
      Use_Method(type, npy_sqrt, sub_result);                                  \
  Use_Float_When_Half(type) reciprocal_sqrt_result;                            \
  Div2(1, sqrt_result, reciprocal_sqrt_result, Use_Float_When_Half(type));     \
  Use_Float_When_Half(type) b_val =                                            \
      Cast_Float_When_Half(type, b_ptr[i * stride_b_last]);                    \
  result_ptr[i] = Cast_Half_When_Half(type, b_val * reciprocal_sqrt_result);

#define ArcsinBackward_LoopBody_Sequential(type, i, result_ptr, a_ptr, b_ptr)  \
  Use_Float_When_Half(type) a_val = Cast_Float_When_Half(type, a_ptr[i]);      \
  Use_Float_When_Half(type) square_result = a_val * a_val;                     \
  Use_Float_When_Half(type) sub_result = 1 - square_result;                    \
  Use_Float_When_Half(type) sqrt_result =                                      \
      Use_Method(type, npy_sqrt, sub_result);                                  \
  Use_Float_When_Half(type) reciprocal_sqrt_result;                            \
  Div2(1, sqrt_result, reciprocal_sqrt_result, Use_Float_When_Half(type));     \
  Use_Float_When_Half(type) b_val = Cast_Float_When_Half(type, b_ptr[i]);      \
  result_ptr[i] = Cast_Half_When_Half(type, b_val * reciprocal_sqrt_result);

Register_FuseBackward_Operation_FloatingTypes(
    arcsin, ArcsinBackward_LoopBody, ArcsinBackward_LoopBody_Sequential, a, b);
Register_FuseBackward_Operation_Err_Int(arcsin, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(arcsin, a, b);
Register_FuseBackward_Operation_Array(arcsin, a, b);
/*==========================================================================================================================================================*/
/*arccos backward fusion*/
#define ArccosBackward_LoopBody(type, i, result_ptr, stride_a_last,            \
                                stride_b_last, a_ptr, b_ptr)                   \
  Use_Float_When_Half(type) a_val =                                            \
      Cast_Float_When_Half(type, a_ptr[i * stride_a_last]);                    \
  Use_Float_When_Half(type) square_result = a_val * a_val;                     \
  Use_Float_When_Half(type) sub_result = 1 - square_result;                    \
  Use_Float_When_Half(type) sqrt_result =                                      \
      Use_Method(type, npy_sqrt, sub_result);                                  \
  Use_Float_When_Half(type) reciprocal_sqrt_result;                            \
  Div2(1, sqrt_result, reciprocal_sqrt_result, Use_Float_When_Half(type));     \
  Use_Float_When_Half(type) b_val =                                            \
      Cast_Float_When_Half(type, b_ptr[i * stride_b_last]);                    \
  result_ptr[i] = -Cast_Half_When_Half(type, b_val * reciprocal_sqrt_result);

#define ArccosBackward_LoopBody_Sequential(type, i, result_ptr, a_ptr, b_ptr)  \
  Use_Float_When_Half(type) a_val = Cast_Float_When_Half(type, a_ptr[i]);      \
  Use_Float_When_Half(type) square_result = a_val * a_val;                     \
  Use_Float_When_Half(type) sub_result = 1 - square_result;                    \
  Use_Float_When_Half(type) sqrt_result =                                      \
      Use_Method(type, npy_sqrt, sub_result);                                  \
  Use_Float_When_Half(type) reciprocal_sqrt_result;                            \
  Div2(1, sqrt_result, reciprocal_sqrt_result, Use_Float_When_Half(type));     \
  Use_Float_When_Half(type) b_val = Cast_Float_When_Half(type, b_ptr[i]);      \
  result_ptr[i] = -Cast_Half_When_Half(type, b_val * reciprocal_sqrt_result);

Register_FuseBackward_Operation_FloatingTypes(
    arccos, ArccosBackward_LoopBody, ArccosBackward_LoopBody_Sequential, a, b);
Register_FuseBackward_Operation_Err_Int(arccos, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(arccos, a, b);
Register_FuseBackward_Operation_Array(arccos, a, b);
/*==========================================================================================================================================================*/
/*arctan backward fusion*/
#define ArctanBackward_LoopBody(type, i, result_ptr, stride_a_last,            \
                                stride_b_last, a_ptr, b_ptr)                   \
  Use_Float_When_Half(type) a_val =                                            \
      Cast_Float_When_Half(type, a_ptr[i * stride_a_last]);                    \
  Use_Float_When_Half(type) square_result = a_val * a_val;                     \
  Use_Float_When_Half(type) add_result = 1 + square_result;                    \
  Use_Float_When_Half(type) reciprocal_add_result;                             \
  Div2(1, add_result, reciprocal_add_result, Use_Float_When_Half(type));       \
  Use_Float_When_Half(type) b_val =                                            \
      Cast_Float_When_Half(type, b_ptr[i * stride_b_last]);                    \
  result_ptr[i] = Cast_Half_When_Half(type, b_val * reciprocal_add_result);

#define ArctanBackward_LoopBody_Sequential(type, i, result_ptr, a_ptr, b_ptr)  \
  Use_Float_When_Half(type) a_val = Cast_Float_When_Half(type, a_ptr[i]);      \
  Use_Float_When_Half(type) square_result = a_val * a_val;                     \
  Use_Float_When_Half(type) add_result = 1 + square_result;                    \
  Use_Float_When_Half(type) reciprocal_add_result;                             \
  Div2(1, add_result, reciprocal_add_result, Use_Float_When_Half(type));       \
  Use_Float_When_Half(type) b_val = Cast_Float_When_Half(type, b_ptr[i]);      \
  result_ptr[i] = Cast_Half_When_Half(type, b_val * reciprocal_add_result);

Register_FuseBackward_Operation_FloatingTypes(
    arctan, ArctanBackward_LoopBody, ArctanBackward_LoopBody_Sequential, a, b);
Register_FuseBackward_Operation_Err_Int(arctan, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(arctan, a, b);
Register_FuseBackward_Operation_Array(arctan, a, b);
/*==========================================================================================================================================================*/
/*sinh backward fusion*/
#define SinhBackward_LoopBody(type, i, result_ptr, stride_a_last,              \
                              stride_b_last, a_ptr, b_ptr)                     \
  Use_Float_When_Half(type) a_val =                                            \
      Use_Method(type, npy_cosh, a_ptr[i * stride_a_last]);                    \
  Use_Float_When_Half(type) b_val =                                            \
      Cast_Float_When_Half(type, b_ptr[i * stride_b_last]);                    \
  result_ptr[i] = Cast_Half_When_Half(type, a_val * b_val);

#define SinhBackward_LoopBody_Sequential(type, i, result_ptr, a_ptr, b_ptr)    \
  Use_Float_When_Half(type) a_val = Use_Method(type, npy_cosh, a_ptr[i]);      \
  Use_Float_When_Half(type) b_val = Cast_Float_When_Half(type, b_ptr[i]);      \
  result_ptr[i] = Cast_Half_When_Half(type, a_val * b_val);

Register_FuseBackward_Operation_FloatingTypes(sinh, SinhBackward_LoopBody,
                                              SinhBackward_LoopBody_Sequential,
                                              a, b);
Register_FuseBackward_Operation_Err_Int(sinh, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(sinh, a, b);
Register_FuseBackward_Operation_Array(sinh, a, b);
/*==========================================================================================================================================================*/
/*cosh backward fusion*/
#define CoshBackward_LoopBody(type, i, result_ptr, stride_a_last,              \
                              stride_b_last, a_ptr, b_ptr)                     \
  Use_Float_When_Half(type) a_val =                                            \
      Use_Method(type, npy_sinh, a_ptr[i * stride_a_last]);                    \
  Use_Float_When_Half(type) b_val =                                            \
      Cast_Float_When_Half(type, b_ptr[i * stride_b_last]);                    \
  result_ptr[i] = Cast_Half_When_Half(type, a_val * b_val);

#define CoshBackward_LoopBody_Sequential(type, i, result_ptr, a_ptr, b_ptr)    \
  Use_Float_When_Half(type) a_val = Use_Method(type, npy_sinh, a_ptr[i]);      \
  Use_Float_When_Half(type) b_val = Cast_Float_When_Half(type, b_ptr[i]);      \
  result_ptr[i] = Cast_Half_When_Half(type, a_val * b_val);

Register_FuseBackward_Operation_FloatingTypes(cosh, CoshBackward_LoopBody,
                                              CoshBackward_LoopBody_Sequential,
                                              a, b);
Register_FuseBackward_Operation_Err_Int(cosh, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(cosh, a, b);
Register_FuseBackward_Operation_Array(cosh, a, b);
/*==========================================================================================================================================================*/
/*tanh backward fusion*/
#define TanhBackward_LoopBody(type, i, result_ptr, stride_a_last,              \
                              stride_b_last, a_ptr, b_ptr)                     \
  Use_Float_When_Half(type) a_val =                                            \
      Cast_Float_When_Half(type, a_ptr[i * stride_a_last]);                    \
  Use_Float_When_Half(type) b_val =                                            \
      Cast_Float_When_Half(type, b_ptr[i * stride_b_last]);                    \
  result_ptr[i] = Cast_Half_When_Half(type, (1 - a_val * a_val) * b_val);

#define TanhBackward_LoopBody_Sequential(type, i, result_ptr, a_ptr, b_ptr)    \
  Use_Float_When_Half(type) a_val = Cast_Float_When_Half(type, a_ptr[i]);      \
  Use_Float_When_Half(type) b_val = Cast_Float_When_Half(type, b_ptr[i]);      \
  result_ptr[i] = Cast_Half_When_Half(type, (1 - a_val * a_val) * b_val);

Register_FuseBackward_Operation_FloatingTypes(tanh, TanhBackward_LoopBody,
                                              TanhBackward_LoopBody_Sequential,
                                              a, b);
Register_FuseBackward_Operation_Err_Int(tanh, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(tanh, a, b);
Register_FuseBackward_Operation_Array(tanh, a, b);
/*==========================================================================================================================================================*/
/*arcsinh backward fusion*/
#define ArcsinhBackward_LoopBody(type, i, result_ptr, stride_a_last,           \
                                 stride_b_last, a_ptr, b_ptr)                  \
  Use_Float_When_Half(type) a_val =                                            \
      Cast_Float_When_Half(type, a_ptr[i * stride_a_last]);                    \
  Use_Float_When_Half(type) square_result = a_val * a_val;                     \
  Use_Float_When_Half(type) add_result = 1 + square_result;                    \
  Use_Float_When_Half(type) sqrt_result =                                      \
      Use_Method(type, npy_sqrt, add_result);                                  \
  Use_Float_When_Half(type) b_val =                                            \
      Cast_Float_When_Half(type, b_ptr[i * stride_b_last]);                    \
  Div(b_val, sqrt_result, result_ptr[i], npy_float);

#define ArcsinhBackward_LoopBody_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
  Use_Float_When_Half(type) a_val = Cast_Float_When_Half(type, a_ptr[i]);      \
  Use_Float_When_Half(type) square_result = a_val * a_val;                     \
  Use_Float_When_Half(type) add_result = 1 + square_result;                    \
  Use_Float_When_Half(type) sqrt_result =                                      \
      Use_Method(type, npy_sqrt, add_result);                                  \
  Use_Float_When_Half(type) b_val = Cast_Float_When_Half(type, b_ptr[i]);      \
  Div(b_val, sqrt_result, result_ptr[i], npy_float);

Register_FuseBackward_Operation_FloatingTypes(
    arcsinh, ArcsinhBackward_LoopBody, ArcsinhBackward_LoopBody_Sequential, a,
    b);
Register_FuseBackward_Operation_Err_Int(arcsinh, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(arcsinh, a, b);
Register_FuseBackward_Operation_Array(arcsinh, a, b);
/*==========================================================================================================================================================*/
/*arccosh backward fusion*/
#define ArccoshBackward_LoopBody(type, i, result_ptr, stride_a_last,           \
                                 stride_b_last, a_ptr, b_ptr)                  \
  Use_Float_When_Half(type) a_val =                                            \
      Cast_Float_When_Half(type, a_ptr[i * stride_a_last]);                    \
  Use_Float_When_Half(type) square_result = a_val * a_val;                     \
  Use_Float_When_Half(type) sub_result = square_result - 1;                    \
  Use_Float_When_Half(type) sqrt_result =                                      \
      Use_Method(type, npy_sqrt, sub_result);                                  \
  Use_Float_When_Half(type) b_val =                                            \
      Cast_Float_When_Half(type, b_ptr[i * stride_b_last]);                    \
  Div(b_val, sqrt_result, result_ptr[i], npy_float);

#define ArccoshBackward_LoopBody_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
  Use_Float_When_Half(type) a_val = Cast_Float_When_Half(type, a_ptr[i]);      \
  Use_Float_When_Half(type) square_result = a_val * a_val;                     \
  Use_Float_When_Half(type) sub_result = square_result - 1;                    \
  Use_Float_When_Half(type) sqrt_result =                                      \
      Use_Method(type, npy_sqrt, sub_result);                                  \
  Use_Float_When_Half(type) b_val = Cast_Float_When_Half(type, b_ptr[i]);      \
  Div(b_val, sqrt_result, result_ptr[i], npy_float);

Register_FuseBackward_Operation_FloatingTypes(
    arccosh, ArccoshBackward_LoopBody, ArccoshBackward_LoopBody_Sequential, a,
    b);
Register_FuseBackward_Operation_Err_Int(arccosh, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(arccosh, a, b);
Register_FuseBackward_Operation_Array(arccosh, a, b);
/*==========================================================================================================================================================*/
/*arctanh backward fusion*/
#define ArctanhBackward_LoopBody(type, i, result_ptr, stride_a_last,           \
                                 stride_b_last, a_ptr, b_ptr)                  \
  Use_Float_When_Half(type) a_val =                                            \
      Cast_Float_When_Half(type, a_ptr[i * stride_a_last]);                    \
  Use_Float_When_Half(type) square_result = a_val * a_val;                     \
  Use_Float_When_Half(type) sub_result = 1 - square_result;                    \
  Use_Float_When_Half(type) b_val =                                            \
      Cast_Float_When_Half(type, b_ptr[i * stride_b_last]);                    \
  Div(b_val, sub_result, result_ptr[i], npy_float);

#define ArctanhBackward_LoopBody_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
  Use_Float_When_Half(type) a_val = Cast_Float_When_Half(type, a_ptr[i]);      \
  Use_Float_When_Half(type) square_result = a_val * a_val;                     \
  Use_Float_When_Half(type) sub_result = 1 - square_result;                    \
  Use_Float_When_Half(type) b_val = Cast_Float_When_Half(type, b_ptr[i]);      \
  Div(b_val, sub_result, result_ptr[i], npy_float);

Register_FuseBackward_Operation_FloatingTypes(
    arctanh, ArctanhBackward_LoopBody, ArctanhBackward_LoopBody_Sequential, a,
    b);
Register_FuseBackward_Operation_Err_Int(arctanh, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(arctanh, a, b);
Register_FuseBackward_Operation_Array(arctanh, a, b);
/*==========================================================================================================================================================*/
/*exp backward fusion*/
#define ExpBackward_LoopBody(type, i, result_ptr, stride_a_last,               \
                             stride_b_last, a_ptr, b_ptr)                      \
  Use_Float_When_Half(type) a_val =                                            \
      Cast_Float_When_Half(type, a_ptr[i * stride_a_last]);                    \
  Use_Float_When_Half(type) exp_result = Use_Method(type, npy_exp, a_val);     \
  Use_Float_When_Half(type) b_val =                                            \
      Cast_Float_When_Half(type, b_ptr[i * stride_b_last]);                    \
  result_ptr[i] = Cast_Half_When_Half(type, exp_result * b_val);

#define ExpBackward_LoopBody_Sequential(type, i, result_ptr, a_ptr, b_ptr)     \
  Use_Float_When_Half(type) a_val = Cast_Float_When_Half(type, a_ptr[i]);      \
  Use_Float_When_Half(type) exp_result = Use_Method(type, npy_exp, a_val);     \
  Use_Float_When_Half(type) b_val = Cast_Float_When_Half(type, b_ptr[i]);      \
  result_ptr[i] = Cast_Half_When_Half(type, exp_result * b_val);

Register_FuseBackward_Operation_FloatingTypes(exp, ExpBackward_LoopBody,
                                              ExpBackward_LoopBody_Sequential,
                                              a, b);
Register_FuseBackward_Operation_Err_Int(exp, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(exp, a, b);
Register_FuseBackward_Operation_Array(exp, a, b);
/*==========================================================================================================================================================*/
/*log backward fusion*/
#define LogBackward_LoopBody(type, i, result_ptr, stride_a_last,               \
                             stride_b_last, a_ptr, b_ptr)                      \
  Use_Float_When_Half(type) a_val =                                            \
      Cast_Float_When_Half(type, a_ptr[i * stride_a_last]);                    \
  Use_Float_When_Half(type) b_val =                                            \
      Cast_Float_When_Half(type, b_ptr[i * stride_b_last]);                    \
  Div(a_val, b_val, result_ptr[i], npy_float);

#define LogBackward_LoopBody_Sequential(type, i, result_ptr, a_ptr, b_ptr)     \
  Use_Float_When_Half(type) a_val = Cast_Float_When_Half(type, a_ptr[i]);      \
  Use_Float_When_Half(type) b_val = Cast_Float_When_Half(type, b_ptr[i]);      \
  Div(a_val, b_val, result_ptr[i], npy_float);

Register_FuseBackward_Operation_FloatingTypes(log, LogBackward_LoopBody,
                                              LogBackward_LoopBody_Sequential,
                                              a, b);
Register_FuseBackward_Operation_Err_Int(log, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(log, a, b);
Register_FuseBackward_Operation_Array(log, a, b);
/*==========================================================================================================================================================*/
/*log10 backward fusion*/
#define Log10Backward_LoopBody(type, i, result_ptr, stride_a_last,             \
                               stride_b_last, a_ptr, b_ptr)                    \
  Use_Float_When_Half(type) a_val =                                            \
      Cast_Float_When_Half(type, a_ptr[i * stride_a_last]);                    \
  Use_Float_When_Half(type) b_val =                                            \
      Cast_Float_When_Half(type, b_ptr[i * stride_b_last]);                    \
  Use_Float_When_Half(type) log_result = Use_Method(type, npy_log, 10);        \
  Use_Float_When_Half(type) mul_result =                                       \
      Cast_Half_When_Half(type, log_result * b_val);                           \
  Div(a_val, mul_result, result_ptr[i], npy_float);

#define Log10Backward_LoopBody_Sequential(type, i, result_ptr, a_ptr, b_ptr)   \
  Use_Float_When_Half(type) a_val = Cast_Float_When_Half(type, a_ptr[i]);      \
  Use_Float_When_Half(type) b_val = Cast_Float_When_Half(type, b_ptr[i]);      \
  Use_Float_When_Half(type) log_result = Use_Method(type, npy_log, 10);        \
  Use_Float_When_Half(type) mul_result =                                       \
      Cast_Half_When_Half(type, log_result * b_val);                           \
  Div(a_val, mul_result, result_ptr[i], npy_float);

Register_FuseBackward_Operation_FloatingTypes(log10, Log10Backward_LoopBody,
                                              Log10Backward_LoopBody_Sequential,
                                              a, b);
Register_FuseBackward_Operation_Err_Int(log10, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(log10, a, b);
Register_FuseBackward_Operation_Array(log10, a, b);
/*==========================================================================================================================================================*/
/*log2 backward fusion*/
/* future plan */
/*==========================================================================================================================================================*/
/*sqrt backward fusion*/
#define SqrtBackward_LoopBody(type, i, result_ptr, stride_a_last,              \
                              stride_b_last, a_ptr, b_ptr)                     \
  Use_Float_When_Half(type) a_val =                                            \
      Cast_Float_When_Half(type, a_ptr[i * stride_a_last]);                    \
  Use_Float_When_Half(type) b_val =                                            \
      Cast_Float_When_Half(type, b_ptr[i * stride_b_last]);                    \
  Use_Float_When_Half(type) mul_result = a_val * 2;                            \
  Div(b_val, mul_result, result_ptr[i], npy_float);

#define SqrtBackward_LoopBody_Sequential(type, i, result_ptr, a_ptr, b_ptr)    \
  Use_Float_When_Half(type) a_val = Cast_Float_When_Half(type, a_ptr[i]);      \
  Use_Float_When_Half(type) b_val = Cast_Float_When_Half(type, b_ptr[i]);      \
  Use_Float_When_Half(type) mul_result = a_val * 2;                            \
  Div(b_val, mul_result, result_ptr[i], npy_float);

Register_FuseBackward_Operation_FloatingTypes(sqrt, SqrtBackward_LoopBody,
                                              SqrtBackward_LoopBody_Sequential,
                                              a, b);
Register_FuseBackward_Operation_Err_Int(sqrt, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(sqrt, a, b);
Register_FuseBackward_Operation_Array(sqrt, a, b);
/*==========================================================================================================================================================*/
/*abs backward fusion*/
#define AbsBackward_LoopBody(type, i, result_ptr, stride_a_last,               \
                             stride_b_last, a_ptr, b_ptr)                      \
  Use_Float_When_Half(type) a_val =                                            \
      Cast_Float_When_Half(type, a_ptr[i * stride_a_last]);                    \
  Use_Float_When_Half(type) b_val =                                            \
      Cast_Float_When_Half(type, b_ptr[i * stride_b_last]);                    \
  result_ptr[i] = Cast_Half_When_Half(type, a_val > 0 ? b_val : -b_val);

#define AbsBackward_LoopBody_Sequential(type, i, result_ptr, a_ptr, b_ptr)     \
  Use_Float_When_Half(type) a_val = Cast_Float_When_Half(type, a_ptr[i]);      \
  Use_Float_When_Half(type) b_val = Cast_Float_When_Half(type, b_ptr[i]);      \
  result_ptr[i] = Cast_Half_When_Half(type, a_val > 0 ? b_val : -b_val);

Register_FuseBackward_Operation_FloatingTypes(abs, AbsBackward_LoopBody,
                                              AbsBackward_LoopBody_Sequential,
                                              a, b);
Register_FuseBackward_Operation_IntergerTypes(abs, AbsBackward_LoopBody,
                                              AbsBackward_LoopBody_Sequential,
                                              a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(abs, a, b);
Register_FuseBackward_Operation_Array(abs, a, b);
/*==========================================================================================================================================================*/
/*power backward fusion*/
#define PowerBackward_LoopBody(type, i, result_ptr, stride_a_last,             \
                               stride_power_last, stride_grad_last, a_ptr,     \
                               power_ptr, grad_ptr)                            \
  Use_Float_When_Half(type) a_val =                                            \
      Cast_Float_When_Half(type, a_ptr[i * stride_a_last]);                    \
  Use_Float_When_Half(type) power_val =                                        \
      Cast_Float_When_Half(type, power_ptr[i * stride_power_last]);            \
  Use_Float_When_Half(type) grad_val =                                         \
      Cast_Float_When_Half(type, grad_ptr[i * stride_grad_last]);              \
  Use_Float_When_Half(type) tmp =                                              \
      Use_Method(type, npy_pow, a_val, power_val - 1);                         \
  result_ptr[i] = tmp * power_val * grad_val;

#define PowerBackward_LoopBody_Sequential(type, i, result_ptr, a_ptr,          \
                                          power_ptr, grad_ptr)                 \
  Use_Float_When_Half(type) a_val = Cast_Float_When_Half(type, a_ptr[i]);      \
  Use_Float_When_Half(type) power_val =                                        \
      Cast_Float_When_Half(type, power_ptr[i]);                                \
  Use_Float_When_Half(type) grad_val =                                         \
      Cast_Float_When_Half(type, grad_ptr[i]);                                 \
  Use_Float_When_Half(type) tmp =                                              \
      Use_Method(type, npy_pow, a_val, power_val - 1);                         \
  result_ptr[i] = tmp * power_val * grad_val;

Register_FuseBackward_Operation_FloatingTypes(power, PowerBackward_LoopBody,
                                              PowerBackward_LoopBody_Sequential,
                                              a, b, c);
Register_FuseBackward_Operation_Err_Int(power, a, power, grad);
Register_FuseBackward_Operation_Err_UnsupportTypes(power, a, power, grad);
Register_FuseBackward_Operation_Array(power, a, power, grad);
/*==========================================================================================================================================================*/