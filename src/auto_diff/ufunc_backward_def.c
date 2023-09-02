#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "ufunc_backward_def.h"

#define Is(x) Concat_(Is_, x)
#define Is_npy_half Place_Holder, 1
#define Should_Cast_To_Half(x) Second(Is(x), 0)
#define Cast_Half_If_Is_Half(x) Concat_(Half_, x)
#define Half_1 float_cast_half
#define Half_0
#define Cast(x) Cast_Half_If_Is_Half(Should_Cast_To_Half(x))

#define Div(val1, val2, result, inf, nan_, type) \
    if (!val2)                                   \
    {                                            \
        if (val1 > 0)                            \
            result = inf;                        \
        else if (val1 < 0)                       \
            result = -inf;                       \
        else                                     \
            result = nan_;                       \
        continue;                                \
    }                                            \
    else                                         \
        result = Cast(type)(val1 / val2);

#define Div2(val1, val2, result, inf, nan_, type) \
    if (!val2)                                    \
    {                                             \
        if (val1 > 0)                             \
            result = inf;                         \
        else if (val1 < 0)                        \
            result = -inf;                        \
        else                                      \
            result = nan_;                        \
    }                                             \
    else                                          \
        result = Cast(type)(val1 / val2);
/*============================================================================= sin backward fusion ===================================================================*/
#define SinBackward_LoopBody_Float(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_float val1 = npy_cosf(a_ptr[i * stride_a_last]);                                            \
    npy_float val2 = b_ptr[i * stride_b_last];                                                      \
    result_ptr[i] = val1 * val2;

#define SinBackward_LoopBody_Double(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_double val1 = npy_cos(a_ptr[i * stride_a_last]);                                             \
    npy_double val2 = b_ptr[i * stride_b_last];                                                      \
    result_ptr[i] = val1 * val2;

#define SinBackward_LoopBody_LongDouble(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    type val1 = npy_cosl(a_ptr[i * stride_a_last]);                                                      \
    type val2 = b_ptr[i * stride_b_last];                                                                \
    result_ptr[i] = val1 * val2;

#define SinBackward_LoopBody_Half(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_float val1 = npy_cos(half_cast_float(*((a_ptr + i * stride_a_last))));                     \
    npy_float val2 = half_cast_float(*((b_ptr + i * stride_b_last)));                              \
    *((result_ptr + i)) = float_cast_half(val1 * val2);

#define SinBackward_LoopBody_Float_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_float val1 = npy_cosf(a_ptr[i]);                                         \
    npy_float val2 = b_ptr[i];                                                   \
    result_ptr[i] = val1 * val2;

#define SinBackward_LoopBody_Double_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_double val1 = npy_cos(a_ptr[i]);                                          \
    npy_double val2 = b_ptr[i];                                                   \
    result_ptr[i] = val1 * val2;

#define SinBackward_LoopBody_LongDouble_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    type val1 = npy_cosl(a_ptr[i]);                                                   \
    type val2 = b_ptr[i];                                                             \
    result_ptr[i] = val1 * val2;

#define SinBackward_LoopBody_Half_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_float val1 = npy_cos(half_cast_float(a_ptr[i]));                        \
    npy_float val2 = half_cast_float(b_ptr[i]);                                 \
    result_ptr[i] = float_cast_half(val1 * val2);

Register_FuseBackward_Operation(sin, float, NPY_FLOAT, SinBackward_LoopBody_Float, SinBackward_LoopBody_Float_Sequential, a, b);
Register_FuseBackward_Operation(sin, double, NPY_DOUBLE, SinBackward_LoopBody_Double, SinBackward_LoopBody_Double_Sequential, a, b);
Register_FuseBackward_Operation(sin, longdouble, NPY_LONGDOUBLE, SinBackward_LoopBody_LongDouble, SinBackward_LoopBody_LongDouble_Sequential, a, b);
Register_FuseBackward_Operation(sin, half, NPY_HALF, SinBackward_LoopBody_Half, SinBackward_LoopBody_Half_Sequential, a, b);
Register_FuseBackward_Operation_Err_Int(sin, a, b);
Register_FuseBackward_Operation_Err_Extra(sin, a, b);
Register_FuseBackward_Operation_Array(sin, a, b)
/*============================================================================= cos backward fusion ===================================================================*/

#define CosBackward_LoopBody_Float(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_float val1 = -npy_sinf(a_ptr[i * stride_a_last]);                                           \
    npy_float val2 = b_ptr[i * stride_b_last];                                                      \
    result_ptr[i] = val1 * val2;

#define CosBackward_LoopBody_Double(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_double val1 = -npy_sin(a_ptr[i * stride_a_last]);                                            \
    npy_double val2 = b_ptr[i * stride_b_last];                                                      \
    result_ptr[i] = val1 * val2;

#define CosBackward_LoopBody_LongDouble(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_longdouble val1 = -npy_sin(a_ptr[i * stride_a_last]);                                            \
    npy_longdouble val2 = b_ptr[i * stride_b_last];                                                      \
    result_ptr[i] = val1 * val2;

#define CosBackward_LoopBody_Half(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_float val1 = -npy_sinf(half_cast_float(*((a_ptr + i * stride_a_last))));                   \
    npy_float val2 = half_cast_float(*((b_ptr + i * stride_b_last)));                              \
    result_ptr[i] = float_cast_half(val1 * val2);

#define CosBackward_LoopBody_Float_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_float val1 = -npy_sinf(a_ptr[i]);                                        \
    npy_float val2 = b_ptr[i];                                                   \
    result_ptr[i] = val1 * val2;

#define CosBackward_LoopBody_Double_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_double val1 = -npy_sin(a_ptr[i]);                                         \
    npy_double val2 = b_ptr[i];                                                   \
    result_ptr[i] = val1 * val2;

#define CosBackward_LoopBody_LongDouble_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_longdouble val1 = -npy_sinl(a_ptr[i]);                                        \
    npy_longdouble val2 = b_ptr[i];                                                   \
    result_ptr[i] = val1 * val2;

#define CosBackward_LoopBody_Half_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_float val1 = -npy_sinf(half_cast_float(a_ptr[i]));                      \
    npy_float val2 = half_cast_float(b_ptr[i]);                                 \
    result_ptr[i] = float_cast_half(val1 * val2);

    Register_FuseBackward_Operation(cos, float, NPY_FLOAT, CosBackward_LoopBody_Float, CosBackward_LoopBody_Float_Sequential, a, b);
Register_FuseBackward_Operation(cos, double, NPY_DOUBLE, CosBackward_LoopBody_Double, CosBackward_LoopBody_Double_Sequential, a, b);
Register_FuseBackward_Operation(cos, longdouble, NPY_LONGDOUBLE, CosBackward_LoopBody_LongDouble, CosBackward_LoopBody_LongDouble_Sequential, a, b);
Register_FuseBackward_Operation(cos, half, NPY_HALF, CosBackward_LoopBody_Half, CosBackward_LoopBody_Half_Sequential, a, b);
Register_FuseBackward_Operation_Err_Int(cos, a, b);
Register_FuseBackward_Operation_Err_Extra(cos, a, b);
Register_FuseBackward_Operation_Array(cos, a, b);
/*============================================================================= tan backward fusion ===================================================================*/
#define TanBackward_LoopBody_Float(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_float val1 = npy_cosf(a_ptr[i * stride_a_last]);                                            \
    npy_float val2 = b_ptr[i * stride_b_last];                                                      \
    Div(val2, val1, result_ptr[i], NPY_INFINITYF, NPY_NANF, type);

#define TanBackward_LoopBody_Double(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_double val1 = npy_cos(a_ptr[i * stride_a_last]);                                             \
    npy_double val2 = b_ptr[i * stride_b_last];                                                      \
    Div(val2, val1, result_ptr[i], NPY_INFINITY, NPY_NAN, type);

#define TanBackward_LoopBody_LongDouble(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_longdouble val1 = npy_cosl(a_ptr[i * stride_a_last]);                                            \
    npy_longdouble val2 = b_ptr[i * stride_b_last];                                                      \
    Div(val2, val1, result_ptr[i], NPY_INFINITYL, NPY_NANL, type);

#define TanBackward_LoopBody_Half(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_float val1 = npy_cosf(half_cast_float(a_ptr[i * stride_a_last]));                          \
    npy_float val2 = half_cast_float(b_ptr[i * stride_b_last]);                                    \
    Div(val2, (val1 * val1), result_ptr[i], 0x7C00, 0x7E00, type);

#define TanBackward_LoopBody_Float_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_float val1 = npy_cosf(a_ptr[i]);                                         \
    npy_float val2 = b_ptr[i];                                                   \
    Div(val2, (val1 * val1), result_ptr[i], NPY_INFINITYF, NPY_NANF, type);

#define TanBackward_LoopBody_Double_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_double val1 = npy_cos(a_ptr[i]);                                          \
    npy_double val2 = b_ptr[i];                                                   \
    Div(val2, (val1 * val1), result_ptr[i], NPY_INFINITY, NPY_NAN, type);

#define TanBackward_LoopBody_LongDouble_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_longdouble val1 = npy_cosl(a_ptr[i]);                                         \
    npy_longdouble val2 = b_ptr[i];                                                   \
    Div(val2, (val1 * val1), result_ptr[i], NPY_INFINITYL, NPY_NANL, type);

#define TanBackward_LoopBody_Half_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_float val1 = npy_cosf(half_cast_float(a_ptr[i]));                       \
    npy_float val2 = half_cast_float(b_ptr[i]);                                 \
    Div(val2, (val1 * val1), result_ptr[i], 0x7C00, 0x7E00, type);

Register_FuseBackward_Operation(tan, float, NPY_FLOAT, TanBackward_LoopBody_Float, TanBackward_LoopBody_Float_Sequential, a, b);
Register_FuseBackward_Operation(tan, double, NPY_DOUBLE, TanBackward_LoopBody_Double, TanBackward_LoopBody_Double_Sequential, a, b);
Register_FuseBackward_Operation(tan, longdouble, NPY_LONGDOUBLE, TanBackward_LoopBody_LongDouble, TanBackward_LoopBody_LongDouble_Sequential, a, b);
Register_FuseBackward_Operation(tan, half, NPY_HALF, TanBackward_LoopBody_Half, TanBackward_LoopBody_Half_Sequential, a, b);
Register_FuseBackward_Operation_Err_Int(tan, a, b);
Register_FuseBackward_Operation_Err_Extra(tan, a, b);
Register_FuseBackward_Operation_Array(tan, a, b);
/*============================================================================= arcsin backward fusion ===================================================================*/
#define ArcsinBackward_LoopBody_Float(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_float a_val = a_ptr[i * stride_a_last];                                                        \
    npy_float square_result = a_val * a_val;                                                           \
    npy_float sub_result = 1.0f - square_result;                                                       \
    npy_float sqrt_result = npy_sqrtf(sub_result);                                                     \
    npy_float reciprocal_sqrt_result;                                                                  \
    Div2(1.0f, sqrt_result, reciprocal_sqrt_result, NPY_INFINITYF, NPY_NANF, npy_float);                \
    npy_float b_val = b_ptr[i * stride_b_last];                                                        \
    result_ptr[i] = b_val * reciprocal_sqrt_result;

#define ArcsinBackward_LoopBody_Double(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_double a_val = a_ptr[i * stride_a_last];                                                        \
    npy_double square_result = a_val * a_val;                                                           \
    npy_double sub_result = 1.0 - square_result;                                                        \
    npy_double sqrt_result = npy_sqrt(sub_result);                                                      \
    npy_double reciprocal_sqrt_result;                                                                  \
    Div2(1.0, sqrt_result, reciprocal_sqrt_result, NPY_INFINITY, NPY_NAN, npy_double);                   \
    npy_double b_val = b_ptr[i * stride_b_last];                                                        \
    result_ptr[i] = b_val * reciprocal_sqrt_result;

#define ArcsinBackward_LoopBody_LongDouble(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_longdouble a_val = a_ptr[i * stride_a_last];                                                        \
    npy_longdouble square_result = a_val * a_val;                                                           \
    npy_longdouble sub_result = 1.0L - square_result;                                                       \
    npy_longdouble sqrt_result = npy_sqrtl(sub_result);                                                     \
    npy_longdouble reciprocal_sqrt_result;                                                                  \
    Div2(1.0L, sqrt_result, reciprocal_sqrt_result, NPY_INFINITYL, NPY_NANL, npy_longdouble);                \
    npy_longdouble b_val = b_ptr[i * stride_b_last];                                                        \
    result_ptr[i] = b_val * reciprocal_sqrt_result;

#define ArcsinBackward_LoopBody_Half(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_float a_val = half_cast_float(a_ptr[i * stride_a_last]);                                      \
    npy_float square_result = a_val * a_val;                                                          \
    npy_float sub_result = 1.0f - square_result;                                                      \
    npy_float sqrt_result = npy_sqrtf(sub_result);                                                    \
    npy_float reciprocal_sqrt_result;                                                                 \
    Div2(1.0f, sqrt_result, reciprocal_sqrt_result, NPY_INFINITYF, NPY_NANF, npy_float);               \
    npy_float b_val = half_cast_float(b_ptr[i * stride_b_last]);                                      \
    result_ptr[i] = float_cast_half(b_val * reciprocal_sqrt_result);

#define ArcsinBackward_LoopBody_Float_Sequential(type, i, result_ptr, a_ptr, b_ptr)     \
    npy_float a_val = a_ptr[i];                                                         \
    npy_float square_result = a_val * a_val;                                            \
    npy_float sub_result = 1.0f - square_result;                                        \
    npy_float sqrt_result = npy_sqrtf(sub_result);                                      \
    npy_float reciprocal_sqrt_result;                                                   \
    Div2(1.0f, sqrt_result, reciprocal_sqrt_result, NPY_INFINITYF, NPY_NANF, npy_float); \
    npy_float b_val = b_ptr[i];                                                         \
    result_ptr[i] = b_val * reciprocal_sqrt_result;

#define ArcsinBackward_LoopBody_Double_Sequential(type, i, result_ptr, a_ptr, b_ptr)  \
    npy_double a_val = a_ptr[i];                                                      \
    npy_double square_result = a_val * a_val;                                         \
    npy_double sub_result = 1.0 - square_result;                                      \
    npy_double sqrt_result = npy_sqrt(sub_result);                                    \
    npy_double reciprocal_sqrt_result;                                                \
    Div2(1.0, sqrt_result, reciprocal_sqrt_result, NPY_INFINITY, NPY_NAN, npy_double); \
    npy_double b_val = b_ptr[i];                                                      \
    result_ptr[i] = b_val * reciprocal_sqrt_result;

#define ArcsinBackward_LoopBody_LongDouble_Sequential(type, i, result_ptr, a_ptr, b_ptr)     \
    npy_longdouble a_val = a_ptr[i];                                                         \
    npy_longdouble square_result = a_val * a_val;                                            \
    npy_longdouble sub_result = 1.0L - square_result;                                        \
    npy_longdouble sqrt_result = npy_sqrtl(sub_result);                                      \
    npy_longdouble reciprocal_sqrt_result;                                                   \
    Div2(1.0L, sqrt_result, reciprocal_sqrt_result, NPY_INFINITYL, NPY_NANL, npy_longdouble); \
    npy_longdouble b_val = b_ptr[i];                                                         \
    result_ptr[i] = b_val * reciprocal_sqrt_result;

#define ArcsinBackward_LoopBody_Half_Sequential(type, i, result_ptr, a_ptr, b_ptr)      \
    npy_float a_val = half_cast_float(a_ptr[i]);                                        \
    npy_float square_result = a_val * a_val;                                            \
    npy_float sub_result = 1.0f - square_result;                                        \
    npy_float sqrt_result = npy_sqrtf(sub_result);                                      \
    npy_float reciprocal_sqrt_result;                                                   \
    Div2(1.0f, sqrt_result, reciprocal_sqrt_result, NPY_INFINITYF, NPY_NANF, npy_float); \
    npy_float b_val = half_cast_float(b_ptr[i]);                                        \
    result_ptr[i] = float_cast_half(b_val * reciprocal_sqrt_result);

Register_FuseBackward_Operation(arcsin, float, NPY_FLOAT, ArcsinBackward_LoopBody_Float, ArcsinBackward_LoopBody_Float_Sequential, a, b);
Register_FuseBackward_Operation(arcsin, double, NPY_DOUBLE, ArcsinBackward_LoopBody_Double, ArcsinBackward_LoopBody_Double_Sequential, a, b);
Register_FuseBackward_Operation(arcsin, longdouble, NPY_LONGDOUBLE, ArcsinBackward_LoopBody_LongDouble, ArcsinBackward_LoopBody_LongDouble_Sequential, a, b);
Register_FuseBackward_Operation(arcsin, half, NPY_HALF, ArcsinBackward_LoopBody_Half, ArcsinBackward_LoopBody_Half_Sequential, a, b);
Register_FuseBackward_Operation_Err_Int(arcsin, a, b);
Register_FuseBackward_Operation_Err_Extra(arcsin, a, b);
Register_FuseBackward_Operation_Array(arcsin, a, b);
/*============================================================================= arccos backward fusion ===================================================================*/
#define ArccosBackward_LoopBody_Float(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_float a_val = a_ptr[i * stride_a_last];                                                        \
    npy_float square_result = a_val * a_val;                                                           \
    npy_float sub_result = 1.0f - square_result;                                                       \
    npy_float sqrt_result = npy_sqrtf(sub_result);                                                     \
    npy_float reciprocal_sqrt_result;                                                                  \
    Div2(1.0f, sqrt_result, reciprocal_sqrt_result, NPY_INFINITYF, NPY_NANF, npy_float);                \
    npy_float b_val = b_ptr[i * stride_b_last];                                                        \
    result_ptr[i] = -b_val * reciprocal_sqrt_result;

#define ArccosBackward_LoopBody_Double(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_double a_val = a_ptr[i * stride_a_last];                                                        \
    npy_double square_result = a_val * a_val;                                                           \
    npy_double sub_result = 1.0 - square_result;                                                        \
    npy_double sqrt_result = npy_sqrt(sub_result);                                                      \
    npy_double reciprocal_sqrt_result;                                                                  \
    Div2(1.0, sqrt_result, reciprocal_sqrt_result, NPY_INFINITY, NPY_NAN, npy_double);                   \
    npy_double b_val = b_ptr[i * stride_b_last];                                                        \
    result_ptr[i] = -b_val * reciprocal_sqrt_result;

#define ArccosBackward_LoopBody_LongDouble(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_longdouble a_val = a_ptr[i * stride_a_last];                                                        \
    npy_longdouble square_result = a_val * a_val;                                                           \
    npy_longdouble sub_result = 1.0L - square_result;                                                       \
    npy_longdouble sqrt_result = npy_sqrtl(sub_result);                                                     \
    npy_longdouble reciprocal_sqrt_result;                                                                  \
    Div2(1.0L, sqrt_result, reciprocal_sqrt_result, NPY_INFINITYL, NPY_NANL, npy_longdouble);                \
    npy_longdouble b_val = b_ptr[i * stride_b_last];                                                        \
    result_ptr[i] = -b_val * reciprocal_sqrt_result;

#define ArccosBackward_LoopBody_Half(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_float a_val = half_cast_float(a_ptr[i * stride_a_last]);                                      \
    npy_float square_result = a_val * a_val;                                                          \
    npy_float sub_result = 1.0f - square_result;                                                      \
    npy_float sqrt_result = npy_sqrtf(sub_result);                                                    \
    npy_float reciprocal_sqrt_result;                                                                 \
    Div2(1.0f, sqrt_result, reciprocal_sqrt_result, NPY_INFINITYF, NPY_NANF, npy_float);               \
    npy_float b_val = half_cast_float(b_ptr[i * stride_b_last]);                                      \
    result_ptr[i] = -float_cast_half(b_val * reciprocal_sqrt_result);

#define ArccosBackward_LoopBody_Float_Sequential(type, i, result_ptr, a_ptr, b_ptr)     \
    npy_float a_val = a_ptr[i];                                                         \
    npy_float square_result = a_val * a_val;                                            \
    npy_float sub_result = 1.0f - square_result;                                        \
    npy_float sqrt_result = npy_sqrtf(sub_result);                                      \
    npy_float reciprocal_sqrt_result;                                                   \
    Div2(1.0f, sqrt_result, reciprocal_sqrt_result, NPY_INFINITYF, NPY_NANF, npy_float); \
    npy_float b_val = b_ptr[i];                                                         \
    result_ptr[i] = -b_val * reciprocal_sqrt_result;

#define ArccosBackward_LoopBody_Double_Sequential(type, i, result_ptr, a_ptr, b_ptr)  \
    npy_double a_val = a_ptr[i];                                                      \
    npy_double square_result = a_val * a_val;                                         \
    npy_double sub_result = 1.0 - square_result;                                      \
    npy_double sqrt_result = npy_sqrt(sub_result);                                    \
    npy_double reciprocal_sqrt_result;                                                \
    Div2(1.0, sqrt_result, reciprocal_sqrt_result, NPY_INFINITY, NPY_NAN, npy_double); \
    npy_double b_val = b_ptr[i];                                                      \
    result_ptr[i] = -b_val * reciprocal_sqrt_result;

#define ArccosBackward_LoopBody_LongDouble_Sequential(type, i, result_ptr, a_ptr, b_ptr)     \
    npy_longdouble a_val = a_ptr[i];                                                         \
    npy_longdouble square_result = a_val * a_val;                                            \
    npy_longdouble sub_result = 1.0L - square_result;                                        \
    npy_longdouble sqrt_result = npy_sqrtl(sub_result);                                      \
    npy_longdouble reciprocal_sqrt_result;                                                   \
    Div2(1.0L, sqrt_result, reciprocal_sqrt_result, NPY_INFINITYL, NPY_NANL, npy_longdouble); \
    npy_longdouble b_val = b_ptr[i];                                                         \
    result_ptr[i] = -b_val * reciprocal_sqrt_result;

#define ArccosBackward_LoopBody_Half_Sequential(type, i, result_ptr, a_ptr, b_ptr)      \
    npy_float a_val = half_cast_float(a_ptr[i]);                                        \
    npy_float square_result = a_val * a_val;                                            \
    npy_float sub_result = 1.0f - square_result;                                        \
    npy_float sqrt_result = npy_sqrtf(sub_result);                                      \
    npy_float reciprocal_sqrt_result;                                                   \
    Div2(1.0f, sqrt_result, reciprocal_sqrt_result, NPY_INFINITYF, NPY_NANF, npy_float); \
    npy_float b_val = half_cast_float(b_ptr[i]);                                        \
    result_ptr[i] = -float_cast_half(b_val * reciprocal_sqrt_result);

Register_FuseBackward_Operation(arccos, float, NPY_FLOAT, ArccosBackward_LoopBody_Float, ArccosBackward_LoopBody_Float_Sequential, a, b);
Register_FuseBackward_Operation(arccos, double, NPY_DOUBLE, ArccosBackward_LoopBody_Double, ArccosBackward_LoopBody_Double_Sequential, a, b);
Register_FuseBackward_Operation(arccos, longdouble, NPY_LONGDOUBLE, ArccosBackward_LoopBody_LongDouble, ArccosBackward_LoopBody_LongDouble_Sequential, a, b);
Register_FuseBackward_Operation(arccos, half, NPY_HALF, ArccosBackward_LoopBody_Half, ArccosBackward_LoopBody_Half_Sequential, a, b);
Register_FuseBackward_Operation_Err_Int(arccos, a, b);
Register_FuseBackward_Operation_Err_Extra(arccos, a, b);
Register_FuseBackward_Operation_Array(arccos, a, b);
/*============================================================================= arctan backward fusion ===================================================================*/
#define ArctanBackward_LoopBody_Float(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_float a_val = a_ptr[i * stride_a_last];                                                        \
    npy_float square_result = a_val * a_val;                                                           \
    npy_float add_result = 1.0f + square_result;                                                       \
    npy_float reciprocal_add_result;                                                                   \
    Div2(1.0f, add_result, reciprocal_add_result, NPY_INFINITYF, NPY_NANF, npy_float);                  \
    npy_float b_val = b_ptr[i * stride_b_last];                                                        \
    result_ptr[i] = b_val * reciprocal_add_result;

#define ArctanBackward_LoopBody_Double(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_double a_val = a_ptr[i * stride_a_last];                                                        \
    npy_double square_result = a_val * a_val;                                                           \
    npy_double add_result = 1.0 + square_result;                                                        \
    npy_double reciprocal_add_result;                                                                   \
    Div2(1.0, add_result, reciprocal_add_result, NPY_INFINITY, NPY_NAN, npy_double);                     \
    npy_double b_val = b_ptr[i * stride_b_last];                                                        \
    result_ptr[i] = b_val * reciprocal_add_result;

#define ArctanBackward_LoopBody_LongDouble(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_longdouble a_val = a_ptr[i * stride_a_last];                                                        \
    npy_longdouble square_result = a_val * a_val;                                                           \
    npy_longdouble add_result = 1.0L + square_result;                                                       \
    npy_longdouble reciprocal_add_result;                                                                   \
    Div2(1.0L, add_result, reciprocal_add_result, NPY_INFINITYL, NPY_NANL, npy_longdouble);                  \
    npy_longdouble b_val = b_ptr[i * stride_b_last];                                                        \
    result_ptr[i] = b_val * reciprocal_add_result;

#define ArctanBackward_LoopBody_Half(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_float a_val = half_cast_float(a_ptr[i * stride_a_last]);                                      \
    npy_float square_result = a_val * a_val;                                                          \
    npy_float add_result = 1.0f + square_result;                                                      \
    npy_float reciprocal_add_result;                                                                  \
    Div2(1.0f, add_result, reciprocal_add_result, NPY_INFINITYF, NPY_NANF, npy_float);                 \
    npy_float b_val = half_cast_float(b_ptr[i * stride_b_last]);                                      \
    result_ptr[i] = float_cast_half(b_val * reciprocal_add_result);

#define ArctanBackward_LoopBody_Float_Sequential(type, i, result_ptr, a_ptr, b_ptr)   \
    npy_float a_val = a_ptr[i];                                                       \
    npy_float square_result = a_val * a_val;                                          \
    npy_float add_result = 1.0f + square_result;                                      \
    npy_float reciprocal_add_result;                                                  \
    Div2(1.0f, add_result, reciprocal_add_result, NPY_INFINITYF, NPY_NANF, npy_float); \
    npy_float b_val = b_ptr[i];                                                       \
    result_ptr[i] = b_val * reciprocal_add_result;

#define ArctanBackward_LoopBody_Double_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_double a_val = a_ptr[i];                                                     \
    npy_double square_result = a_val * a_val;                                        \
    npy_double add_result = 1.0 + square_result;                                     \
    npy_double reciprocal_add_result;                                                \
    Div2(1.0, add_result, reciprocal_add_result, NPY_INFINITY, NPY_NAN, npy_double);  \
    npy_double b_val = b_ptr[i];                                                     \
    result_ptr[i] = b_val * reciprocal_add_result;

#define ArctanBackward_LoopBody_LongDouble_Sequential(type, i, result_ptr, a_ptr, b_ptr)   \
    npy_longdouble a_val = a_ptr[i];                                                       \
    npy_longdouble square_result = a_val * a_val;                                          \
    npy_longdouble add_result = 1.0L + square_result;                                      \
    npy_longdouble reciprocal_add_result;                                                  \
    Div2(1.0L, add_result, reciprocal_add_result, NPY_INFINITYL, NPY_NANL, npy_longdouble); \
    npy_longdouble b_val = b_ptr[i];                                                       \
    result_ptr[i] = b_val * reciprocal_add_result;

#define ArctanBackward_LoopBody_Half_Sequential(type, i, result_ptr, a_ptr, b_ptr)    \
    npy_float a_val = half_cast_float(a_ptr[i]);                                      \
    npy_float square_result = a_val * a_val;                                          \
    npy_float add_result = 1.0f + square_result;                                      \
    npy_float reciprocal_add_result;                                                  \
    Div2(1.0f, add_result, reciprocal_add_result, NPY_INFINITYF, NPY_NANF, npy_float); \
    npy_float b_val = half_cast_float(b_ptr[i]);                                      \
    result_ptr[i] = float_cast_half(b_val * reciprocal_add_result);

Register_FuseBackward_Operation(arctan, float, NPY_FLOAT, ArctanBackward_LoopBody_Float, ArctanBackward_LoopBody_Float_Sequential, a, b);
Register_FuseBackward_Operation(arctan, double, NPY_DOUBLE, ArctanBackward_LoopBody_Double, ArctanBackward_LoopBody_Double_Sequential, a, b);
Register_FuseBackward_Operation(arctan, longdouble, NPY_LONGDOUBLE, ArctanBackward_LoopBody_LongDouble, ArctanBackward_LoopBody_LongDouble_Sequential, a, b);
Register_FuseBackward_Operation(arctan, half, NPY_HALF, ArctanBackward_LoopBody_Half, ArctanBackward_LoopBody_Half_Sequential, a, b);
Register_FuseBackward_Operation_Err_Int(arctan, a, b);
Register_FuseBackward_Operation_Err_Extra(arctan, a, b);
Register_FuseBackward_Operation_Array(arctan, a, b);
/*============================================================================= sinh backward fusion ===================================================================*/
#define SinhBackward_LoopBody_Float(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_float val1 = npy_coshf(a_ptr[i * stride_a_last]);                                            \
    npy_float val2 = b_ptr[i * stride_b_last];                                                       \
    result_ptr[i] = val1 * val2;

#define SinhBackward_LoopBody_Double(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_double val1 = npy_cosh(a_ptr[i * stride_a_last]);                                             \
    npy_double val2 = b_ptr[i * stride_b_last];                                                       \
    result_ptr[i] = val1 * val2;

#define SinhBackward_LoopBody_LongDouble(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_longdouble val1 = npy_coshl(a_ptr[i * stride_a_last]);                                            \
    npy_longdouble val2 = b_ptr[i * stride_b_last];                                                       \
    result_ptr[i] = val1 * val2;

#define SinhBackward_LoopBody_Half(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_float val1 = npy_coshf(half_cast_float(a_ptr[i * stride_a_last]));                          \
    npy_float val2 = half_cast_float(b_ptr[i * stride_b_last]);                                     \
    result_ptr[i] = float_cast_half(val1 * val2);

#define SinhBackward_LoopBody_Float_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_float val1 = npy_coshf(a_ptr[i]);                                         \
    npy_float val2 = b_ptr[i];                                                    \
    result_ptr[i] = val1 * val2;

#define SinhBackward_LoopBody_Double_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_double val1 = npy_cosh(a_ptr[i]);                                          \
    npy_double val2 = b_ptr[i];                                                    \
    result_ptr[i] = val1 * val2;

#define SinhBackward_LoopBody_LongDouble_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_longdouble val1 = npy_coshl(a_ptr[i]);                                         \
    npy_longdouble val2 = b_ptr[i];                                                    \
    result_ptr[i] = val1 * val2;

#define SinhBackward_LoopBody_Half_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_float val1 = npy_coshf(half_cast_float(a_ptr[i]));                       \
    npy_float val2 = half_cast_float(b_ptr[i]);                                  \
    result_ptr[i] = float_cast_half(val1 * val2);

Register_FuseBackward_Operation(sinh, float, NPY_FLOAT, SinhBackward_LoopBody_Float, SinhBackward_LoopBody_Float_Sequential, a, b);
Register_FuseBackward_Operation(sinh, double, NPY_DOUBLE, SinhBackward_LoopBody_Double, SinhBackward_LoopBody_Double_Sequential, a, b);
Register_FuseBackward_Operation(sinh, longdouble, NPY_LONGDOUBLE, SinhBackward_LoopBody_LongDouble, SinhBackward_LoopBody_LongDouble_Sequential, a, b);
Register_FuseBackward_Operation(sinh, half, NPY_HALF, SinhBackward_LoopBody_Half, SinhBackward_LoopBody_Half_Sequential, a, b);
Register_FuseBackward_Operation_Err_Int(sinh, a, b);
Register_FuseBackward_Operation_Err_Extra(sinh, a, b);
Register_FuseBackward_Operation_Array(sinh, a, b);
/*============================================================================= cosh backward fusion ===================================================================*/
#define CoshBackward_LoopBody_Float(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_float val1 = npy_sinhf(a_ptr[i * stride_a_last]);                                            \
    npy_float val2 = b_ptr[i * stride_b_last];                                                       \
    result_ptr[i] = val1 * val2;

#define CoshBackward_LoopBody_Double(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_double val1 = npy_sinh(a_ptr[i * stride_a_last]);                                             \
    npy_double val2 = b_ptr[i * stride_b_last];                                                       \
    result_ptr[i] = val1 * val2;

#define CoshBackward_LoopBody_LongDouble(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_longdouble val1 = npy_sinhl(a_ptr[i * stride_a_last]);                                            \
    npy_longdouble val2 = b_ptr[i * stride_b_last];                                                       \
    result_ptr[i] = val1 * val2;

#define CoshBackward_LoopBody_Half(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_float val1 = npy_sinhf(half_cast_float(a_ptr[i * stride_a_last]));                          \
    npy_float val2 = half_cast_float(b_ptr[i * stride_b_last]);                                     \
    result_ptr[i] = float_cast_half(val1 * val2);

#define CoshBackward_LoopBody_Float_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_float val1 = npy_sinhf(a_ptr[i]);                                         \
    npy_float val2 = b_ptr[i];                                                    \
    result_ptr[i] = val1 * val2;

#define CoshBackward_LoopBody_Double_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_double val1 = npy_sinh(a_ptr[i]);                                          \
    npy_double val2 = b_ptr[i];                                                    \
    result_ptr[i] = val1 * val2;

#define CoshBackward_LoopBody_LongDouble_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_longdouble val1 = npy_sinhl(a_ptr[i]);                                         \
    npy_longdouble val2 = b_ptr[i];                                                    \
    result_ptr[i] = val1 * val2;

#define CoshBackward_LoopBody_Half_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_float val1 = npy_sinhf(half_cast_float(a_ptr[i]));                       \
    npy_float val2 = half_cast_float(b_ptr[i]);                                  \
    result_ptr[i] = float_cast_half(val1 * val2);

Register_FuseBackward_Operation(cosh, float, NPY_FLOAT, CoshBackward_LoopBody_Float, CoshBackward_LoopBody_Float_Sequential, a, b);
Register_FuseBackward_Operation(cosh, double, NPY_DOUBLE, CoshBackward_LoopBody_Double, CoshBackward_LoopBody_Double_Sequential, a, b);
Register_FuseBackward_Operation(cosh, longdouble, NPY_LONGDOUBLE, CoshBackward_LoopBody_LongDouble, CoshBackward_LoopBody_LongDouble_Sequential, a, b);
Register_FuseBackward_Operation(cosh, half, NPY_HALF, CoshBackward_LoopBody_Half, CoshBackward_LoopBody_Half_Sequential, a, b);
Register_FuseBackward_Operation_Err_Int(cosh, a, b);
Register_FuseBackward_Operation_Err_Extra(cosh, a, b);
Register_FuseBackward_Operation_Array(cosh, a, b);
/*============================================================================= tanh backward fusion ===================================================================*/
#define TanhBackward_LoopBody_Float(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_float val1 = 1.0f - (a_ptr[i * stride_a_last] * a_ptr[i * stride_a_last]);                   \
    npy_float val2 = b_ptr[i * stride_b_last];                                                       \
    result_ptr[i] = val1 * val2;

#define TanhBackward_LoopBody_Double(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_double val1 = 1.0 - (a_ptr[i * stride_a_last] * a_ptr[i * stride_a_last]);                    \
    npy_double val2 = b_ptr[i * stride_b_last];                                                       \
    result_ptr[i] = val1 * val2;

#define TanhBackward_LoopBody_LongDouble(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_longdouble val1 = 1.0L - (a_ptr[i * stride_a_last] * a_ptr[i * stride_a_last]);                   \
    npy_longdouble val2 = b_ptr[i * stride_b_last];                                                       \
    result_ptr[i] = val1 * val2;

#define TanhBackward_LoopBody_Half(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr)                  \
    npy_float val1 = 1.0f - (half_cast_float(a_ptr[i * stride_a_last]) * half_cast_float(a_ptr[i * stride_a_last])); \
    npy_float val2 = half_cast_float(b_ptr[i * stride_b_last]);                                                      \
    result_ptr[i] = float_cast_half(val1 * val2);

#define TanhBackward_LoopBody_Float_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_float val1 = 1.0f - (a_ptr[i] * a_ptr[i]);                                \
    npy_float val2 = b_ptr[i];                                                    \
    result_ptr[i] = val1 * val2;

#define TanhBackward_LoopBody_Double_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_double val1 = 1.0 - (a_ptr[i] * a_ptr[i]);                                 \
    npy_double val2 = b_ptr[i];                                                    \
    result_ptr[i] = val1 * val2;

#define TanhBackward_LoopBody_LongDouble_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_longdouble val1 = 1.0L - (a_ptr[i] * a_ptr[i]);                                \
    npy_longdouble val2 = b_ptr[i];                                                    \
    result_ptr[i] = val1 * val2;

#define TanhBackward_LoopBody_Half_Sequential(type, i, result_ptr, a_ptr, b_ptr)     \
    npy_float val1 = 1.0f - (half_cast_float(a_ptr[i]) * half_cast_float(a_ptr[i])); \
    npy_float val2 = half_cast_float(b_ptr[i]);                                      \
    result_ptr[i] = float_cast_half(val1 * val2);

Register_FuseBackward_Operation(tanh, float, NPY_FLOAT, TanhBackward_LoopBody_Float, TanhBackward_LoopBody_Float_Sequential, a, b);
Register_FuseBackward_Operation(tanh, double, NPY_DOUBLE, TanhBackward_LoopBody_Double, TanhBackward_LoopBody_Double_Sequential, a, b);
Register_FuseBackward_Operation(tanh, longdouble, NPY_LONGDOUBLE, TanhBackward_LoopBody_LongDouble, TanhBackward_LoopBody_LongDouble_Sequential, a, b);
Register_FuseBackward_Operation(tanh, half, NPY_HALF, TanhBackward_LoopBody_Half, TanhBackward_LoopBody_Half_Sequential, a, b);
Register_FuseBackward_Operation_Err_Int(tanh, a, b);
Register_FuseBackward_Operation_Err_Extra(tanh, a, b);
Register_FuseBackward_Operation_Array(tanh, a, b);
/*============================================================================= arcsinh backward fusion ===================================================================*/
#define ArcsinhBackward_LoopBody_Float(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_float val1 = 1.0f + (a_ptr[i * stride_a_last] * a_ptr[i * stride_a_last]);                      \
    npy_float val2 = npy_powf(val1, 0.5f);                                                              \
    npy_float val3 = b_ptr[i * stride_b_last];                                                          \
    Div(val3, val2, result_ptr[i], NPY_INFINITYF, NPY_NANF, npy_float);

#define ArcsinhBackward_LoopBody_Double(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_double val1 = 1.0 + (a_ptr[i * stride_a_last] * a_ptr[i * stride_a_last]);                       \
    npy_double val2 = npy_pow(val1, 0.5);                                                                \
    npy_double val3 = b_ptr[i * stride_b_last];                                                          \
    Div(val3, val2, result_ptr[i], NPY_INFINITY, NPY_NAN, npy_double);

#define ArcsinhBackward_LoopBody_LongDouble(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_longdouble val1 = 1.0L + (a_ptr[i * stride_a_last] * a_ptr[i * stride_a_last]);                      \
    npy_longdouble val2 = npy_powl(val1, 0.5L);                                                              \
    npy_longdouble val3 = b_ptr[i * stride_b_last];                                                          \
    Div(val3, val2, result_ptr[i], NPY_INFINITYL, NPY_NANL, npy_longdouble);

#define ArcsinhBackward_LoopBody_Half(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr)               \
    npy_float val1 = 1.0f + (half_cast_float(a_ptr[i * stride_a_last]) * half_cast_float(a_ptr[i * stride_a_last])); \
    npy_float val2 = npy_powf(val1, 0.5f);                                                                           \
    npy_float val3 = half_cast_float(b_ptr[i * stride_b_last]);                                                      \
    Div(val3, val2, result_ptr[i], NPY_INFINITYF, NPY_NANF, npy_half);

#define ArcsinhBackward_LoopBody_Float_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_float val1 = 1.0f + (a_ptr[i] * a_ptr[i]);                                   \
    npy_float val2 = npy_powf(val1, 0.5f);                                           \
    npy_float val3 = b_ptr[i];                                                       \
    Div(val3, val2, result_ptr[i], NPY_INFINITYF, NPY_NANF, npy_float);

#define ArcsinhBackward_LoopBody_Double_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_double val1 = 1.0 + (a_ptr[i] * a_ptr[i]);                                    \
    npy_double val2 = npy_pow(val1, 0.5);                                             \
    npy_double val3 = b_ptr[i];                                                       \
    Div(val3, val2, result_ptr[i], NPY_INFINITY, NPY_NAN, npy_double);

#define ArcsinhBackward_LoopBody_LongDouble_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_longdouble val1 = 1.0L + (a_ptr[i] * a_ptr[i]);                                   \
    npy_longdouble val2 = npy_powl(val1, 0.5L);                                           \
    npy_longdouble val3 = b_ptr[i];                                                       \
    Div(val3, val2, result_ptr[i], NPY_INFINITYL, NPY_NANL, npy_longdouble);

#define ArcsinhBackward_LoopBody_Half_Sequential(type, i, result_ptr, a_ptr, b_ptr)  \
    npy_float val1 = 1.0f + (half_cast_float(a_ptr[i]) * half_cast_float(a_ptr[i])); \
    npy_float val2 = npy_powf(val1, 0.5f);                                           \
    npy_float val3 = half_cast_float(b_ptr[i]);                                      \
    Div(val3, val2, result_ptr[i], NPY_INFINITYF, NPY_NANF, npy_half);

Register_FuseBackward_Operation(arcsinh, float, NPY_FLOAT, ArcsinhBackward_LoopBody_Float, ArcsinhBackward_LoopBody_Float_Sequential, a, b);
Register_FuseBackward_Operation(arcsinh, double, NPY_DOUBLE, ArcsinhBackward_LoopBody_Double, ArcsinhBackward_LoopBody_Double_Sequential, a, b);
Register_FuseBackward_Operation(arcsinh, longdouble, NPY_LONGDOUBLE, ArcsinhBackward_LoopBody_LongDouble, ArcsinhBackward_LoopBody_LongDouble_Sequential, a, b);
Register_FuseBackward_Operation(arcsinh, half, NPY_HALF, ArcsinhBackward_LoopBody_Half, ArcsinhBackward_LoopBody_Half_Sequential, a, b);
Register_FuseBackward_Operation_Err_Int(arcsinh, a, b);
Register_FuseBackward_Operation_Err_Extra(arcsinh, a, b);
Register_FuseBackward_Operation_Array(arcsinh, a, b);
/*============================================================================= arccosh backward fusion ===================================================================*/
#define ArccoshBackward_LoopBody_Float(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_float val1 = a_ptr[i * stride_a_last] * a_ptr[i * stride_a_last];                               \
    npy_float val2 = val1 - 1.0f;                                                                       \
    npy_float val3 = npy_sqrtf(val2);                                                                   \
    npy_float val4 = b_ptr[i * stride_b_last];                                                          \
    Div(val4, val3, result_ptr[i], NPY_INFINITYF, NPY_NANF, npy_float);

#define ArccoshBackward_LoopBody_Double(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_double val1 = a_ptr[i * stride_a_last] * a_ptr[i * stride_a_last];                               \
    npy_double val2 = val1 - 1.0;                                                                        \
    npy_double val3 = npy_sqrt(val2);                                                                    \
    npy_double val4 = b_ptr[i * stride_b_last];                                                          \
    Div(val4, val3, result_ptr[i], NPY_INFINITY, NPY_NAN, npy_double);

#define ArccoshBackward_LoopBody_LongDouble(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_longdouble val1 = a_ptr[i * stride_a_last] * a_ptr[i * stride_a_last];                               \
    npy_longdouble val2 = val1 - 1.0L;                                                                       \
    npy_longdouble val3 = npy_sqrtl(val2);                                                                   \
    npy_longdouble val4 = b_ptr[i * stride_b_last];                                                          \
    Div(val4, val3, result_ptr[i], NPY_INFINITYL, NPY_NANL, npy_longdouble);

#define ArccoshBackward_LoopBody_Half(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr)      \
    npy_float val1 = half_cast_float(a_ptr[i * stride_a_last]) * half_cast_float(a_ptr[i * stride_a_last]); \
    npy_float val2 = val1 - 1.0f;                                                                           \
    npy_float val3 = npy_sqrtf(val2);                                                                       \
    npy_float val4 = half_cast_float(b_ptr[i * stride_b_last]);                                             \
    Div(val4, val3, result_ptr[i], NPY_INFINITYF, NPY_NANF, npy_half);

#define ArccoshBackward_LoopBody_Float_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_float val1 = a_ptr[i] * a_ptr[i];                                            \
    npy_float val2 = val1 - 1.0f;                                                    \
    npy_float val3 = npy_sqrtf(val2);                                                \
    npy_float val4 = b_ptr[i];                                                       \
    Div(val4, val3, result_ptr[i], NPY_INFINITYF, NPY_NANF, npy_float);

#define ArccoshBackward_LoopBody_Double_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_double val1 = a_ptr[i] * a_ptr[i];                                            \
    npy_double val2 = val1 - 1.0;                                                     \
    npy_double val3 = npy_sqrt(val2);                                                 \
    npy_double val4 = b_ptr[i];                                                       \
    Div(val4, val3, result_ptr[i], NPY_INFINITY, NPY_NAN, npy_double);

#define ArccoshBackward_LoopBody_LongDouble_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_longdouble val1 = a_ptr[i] * a_ptr[i];                                            \
    npy_longdouble val2 = val1 - 1.0L;                                                    \
    npy_longdouble val3 = npy_sqrtl(val2);                                                \
    npy_longdouble val4 = b_ptr[i];                                                       \
    Div(val4, val3, result_ptr[i], NPY_INFINITYL, NPY_NANL, npy_longdouble);

#define ArccoshBackward_LoopBody_Half_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_float val1 = half_cast_float(a_ptr[i]) * half_cast_float(a_ptr[i]);         \
    npy_float val2 = val1 - 1.0f;                                                   \
    npy_float val3 = npy_sqrtf(val2);                                               \
    npy_float val4 = half_cast_float(b_ptr[i]);                                     \
    Div(val4, val3, result_ptr[i], NPY_INFINITYF, NPY_NANF, npy_half);

Register_FuseBackward_Operation(arccosh, float, NPY_FLOAT, ArccoshBackward_LoopBody_Float, ArccoshBackward_LoopBody_Float_Sequential, a, b);
Register_FuseBackward_Operation(arccosh, double, NPY_DOUBLE, ArccoshBackward_LoopBody_Double, ArccoshBackward_LoopBody_Double_Sequential, a, b);
Register_FuseBackward_Operation(arccosh, longdouble, NPY_LONGDOUBLE, ArccoshBackward_LoopBody_LongDouble, ArccoshBackward_LoopBody_LongDouble_Sequential, a, b);
Register_FuseBackward_Operation(arccosh, half, NPY_HALF, ArccoshBackward_LoopBody_Half, ArccoshBackward_LoopBody_Half_Sequential, a, b);
Register_FuseBackward_Operation_Err_Int(arccosh, a, b);
Register_FuseBackward_Operation_Err_Extra(arccosh, a, b);
Register_FuseBackward_Operation_Array(arccosh, a, b);
/*============================================================================= arctanh backward fusion ===================================================================*/
#define ArctanhBackward_LoopBody_Float(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_float val1 = 1.0f - (a_ptr[i * stride_a_last] * a_ptr[i * stride_a_last]);                      \
    npy_float val2 = b_ptr[i * stride_b_last];                                                          \
    Div(val2, val1, result_ptr[i], NPY_INFINITYF, NPY_NANF, npy_float);

#define ArctanhBackward_LoopBody_Double(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_double val1 = 1.0 - (a_ptr[i * stride_a_last] * a_ptr[i * stride_a_last]);                       \
    npy_double val2 = b_ptr[i * stride_b_last];                                                          \
    Div(val2, val1, result_ptr[i], NPY_INFINITY, NPY_NAN, npy_double);

#define ArctanhBackward_LoopBody_LongDouble(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_longdouble val1 = 1.0L - (a_ptr[i * stride_a_last] * a_ptr[i * stride_a_last]);                      \
    npy_longdouble val2 = b_ptr[i * stride_b_last];                                                          \
    Div(val2, val1, result_ptr[i], NPY_INFINITYL, NPY_NANL, npy_longdouble);

#define ArctanhBackward_LoopBody_Half(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr)               \
    npy_float val1 = 1.0f - (half_cast_float(a_ptr[i * stride_a_last]) * half_cast_float(a_ptr[i * stride_a_last])); \
    npy_float val2 = half_cast_float(b_ptr[i * stride_b_last]);                                                      \
    Div(val2, val1, result_ptr[i], NPY_INFINITYF, NPY_NANF, npy_half);

#define ArctanhBackward_LoopBody_Float_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_float val1 = 1.0f - (a_ptr[i] * a_ptr[i]);                                   \
    npy_float val2 = b_ptr[i];                                                       \
    Div(val2, val1, result_ptr[i], NPY_INFINITYF, NPY_NANF, npy_float);

#define ArctanhBackward_LoopBody_Double_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_double val1 = 1.0 - (a_ptr[i] * a_ptr[i]);                                    \
    npy_double val2 = b_ptr[i];                                                       \
    Div(val2, val1, result_ptr[i], NPY_INFINITY, NPY_NAN, npy_double);

#define ArctanhBackward_LoopBody_LongDouble_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_longdouble val1 = 1.0L - (a_ptr[i] * a_ptr[i]);                                   \
    npy_longdouble val2 = b_ptr[i];                                                       \
    Div(val2, val1, result_ptr[i], NPY_INFINITYL, NPY_NANL, npy_longdouble);

#define ArctanhBackward_LoopBody_Half_Sequential(type, i, result_ptr, a_ptr, b_ptr)  \
    npy_float val1 = 1.0f - (half_cast_float(a_ptr[i]) * half_cast_float(a_ptr[i])); \
    npy_float val2 = half_cast_float(b_ptr[i]);                                      \
    Div(val2, val1, result_ptr[i], NPY_INFINITYF, NPY_NANF, npy_half);

Register_FuseBackward_Operation(arctanh, float, NPY_FLOAT, ArctanhBackward_LoopBody_Float, ArctanhBackward_LoopBody_Float_Sequential, a, b);
Register_FuseBackward_Operation(arctanh, double, NPY_DOUBLE, ArctanhBackward_LoopBody_Double, ArctanhBackward_LoopBody_Double_Sequential, a, b);
Register_FuseBackward_Operation(arctanh, longdouble, NPY_LONGDOUBLE, ArctanhBackward_LoopBody_LongDouble, ArctanhBackward_LoopBody_LongDouble_Sequential, a, b);
Register_FuseBackward_Operation(arctanh, half, NPY_HALF, ArctanhBackward_LoopBody_Half, ArctanhBackward_LoopBody_Half_Sequential, a, b);
Register_FuseBackward_Operation_Err_Int(arctanh, a, b);
Register_FuseBackward_Operation_Err_Extra(arctanh, a, b);
Register_FuseBackward_Operation_Array(arctanh, a, b);
/*============================================================================= exp backward fusion ===================================================================*/
#define ExpBackward_LoopBody_Float(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_float val1 = npy_expf(a_ptr[i * stride_a_last]);                                            \
    npy_float val2 = b_ptr[i * stride_b_last];                                                      \
    result_ptr[i] = val1 * val2;

#define ExpBackward_LoopBody_Double(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_double val1 = npy_exp(a_ptr[i * stride_a_last]);                                             \
    npy_double val2 = b_ptr[i * stride_b_last];                                                      \
    result_ptr[i] = val1 * val2;

#define ExpBackward_LoopBody_LongDouble(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_longdouble val1 = npy_expl(a_ptr[i * stride_a_last]);                                            \
    npy_longdouble val2 = b_ptr[i * stride_b_last];                                                      \
    result_ptr[i] = val1 * val2;

#define ExpBackward_LoopBody_Half(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_float val1 = npy_expf(half_cast_float(a_ptr[i * stride_a_last]));                          \
    npy_float val2 = half_cast_float(b_ptr[i * stride_b_last]);                                    \
    result_ptr[i] = float_cast_half(val1 * val2);

#define ExpBackward_LoopBody_Float_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_float val1 = npy_expf(a_ptr[i]);                                         \
    npy_float val2 = b_ptr[i];                                                   \
    result_ptr[i] = val1 * val2;

#define ExpBackward_LoopBody_Double_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_double val1 = npy_exp(a_ptr[i]);                                          \
    npy_double val2 = b_ptr[i];                                                   \
    result_ptr[i] = val1 * val2;

#define ExpBackward_LoopBody_LongDouble_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_longdouble val1 = npy_expl(a_ptr[i]);                                         \
    npy_longdouble val2 = b_ptr[i];                                                   \
    result_ptr[i] = val1 * val2;

#define ExpBackward_LoopBody_Half_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_float val1 = npy_expf(half_cast_float(a_ptr[i]));                       \
    npy_float val2 = half_cast_float(b_ptr[i]);                                 \
    result_ptr[i] = float_cast_half(val1 * val2);

Register_FuseBackward_Operation(exp, float, NPY_FLOAT, ExpBackward_LoopBody_Float, ExpBackward_LoopBody_Float_Sequential, a, b);
Register_FuseBackward_Operation(exp, double, NPY_DOUBLE, ExpBackward_LoopBody_Double, ExpBackward_LoopBody_Double_Sequential, a, b);
Register_FuseBackward_Operation(exp, longdouble, NPY_LONGDOUBLE, ExpBackward_LoopBody_LongDouble, ExpBackward_LoopBody_LongDouble_Sequential, a, b);
Register_FuseBackward_Operation(exp, half, NPY_HALF, ExpBackward_LoopBody_Half, ExpBackward_LoopBody_Half_Sequential, a, b);
Register_FuseBackward_Operation_Err_Int(exp, a, b);
Register_FuseBackward_Operation_Err_Extra(exp, a, b);
Register_FuseBackward_Operation_Array(exp, a, b);
/*============================================================================= log backward fusion ===================================================================*/
#define LogBackward_LoopBody_Float(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_float val1 = a_ptr[i * stride_a_last];                                                      \
    npy_float val2 = b_ptr[i * stride_b_last];                                                      \
    Div(val1, val2, result_ptr[i], NPY_INFINITYF, NPY_NANF, npy_float);

#define LogBackward_LoopBody_Double(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_double val1 = a_ptr[i * stride_a_last];                                                      \
    npy_double val2 = b_ptr[i * stride_b_last];                                                      \
    Div(val1, val2, result_ptr[i], NPY_INFINITY, NPY_NAN, npy_double);

#define LogBackward_LoopBody_LongDouble(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_longdouble val1 = a_ptr[i * stride_a_last];                                                      \
    npy_longdouble val2 = b_ptr[i * stride_b_last];                                                      \
    Div(val1, val2, result_ptr[i], NPY_INFINITYL, NPY_NANL, npy_longdouble);

#define LogBackward_LoopBody_Half(type, i, result_ptr, stride_a_last, stride_b_last, a_ptr, b_ptr) \
    npy_float val1 = half_cast_float(a_ptr[i * stride_a_last]);                                    \
    npy_float val2 = half_cast_float(b_ptr[i * stride_b_last]);                                    \
    Div(val1, val2, result_ptr[i], NPY_INFINITYF, NPY_NANF, npy_half);

#define LogBackward_LoopBody_Float_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_float val1 = a_ptr[i];                                                   \
    npy_float val2 = b_ptr[i];                                                   \
    Div(val1, val2, result_ptr[i], NPY_INFINITYF, NPY_NANF, npy_float);

#define LogBackward_LoopBody_Double_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_double val1 = a_ptr[i];                                                   \
    npy_double val2 = b_ptr[i];                                                   \
    Div(val1, val2, result_ptr[i], NPY_INFINITY, NPY_NAN, npy_double);

#define LogBackward_LoopBody_LongDouble_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_longdouble val1 = a_ptr[i];                                                   \
    npy_longdouble val2 = b_ptr[i];                                                   \
    Div(val1, val2, result_ptr[i], NPY_INFINITYL, NPY_NANL, npy_longdouble);

#define LogBackward_LoopBody_Half_Sequential(type, i, result_ptr, a_ptr, b_ptr) \
    npy_float val1 = half_cast_float(a_ptr[i]);                                 \
    npy_float val2 = half_cast_float(b_ptr[i]);                                 \
    Div(val1, val2, result_ptr[i], NPY_INFINITYF, NPY_NANF, npy_half);

Register_FuseBackward_Operation(log, float, NPY_FLOAT, LogBackward_LoopBody_Float, LogBackward_LoopBody_Float_Sequential, a, b);
Register_FuseBackward_Operation(log, double, NPY_DOUBLE, LogBackward_LoopBody_Double, LogBackward_LoopBody_Double_Sequential, a, b);
Register_FuseBackward_Operation(log, longdouble, NPY_LONGDOUBLE, LogBackward_LoopBody_LongDouble, LogBackward_LoopBody_LongDouble_Sequential, a, b);
Register_FuseBackward_Operation(log, half, NPY_HALF, LogBackward_LoopBody_Half, LogBackward_LoopBody_Half_Sequential, a, b);
Register_FuseBackward_Operation_Err_Int(log, a, b);
Register_FuseBackward_Operation_Err_Extra(log, a, b);
Register_FuseBackward_Operation_Array(log, a, b);
/*============================================================================= log10 backward fusion ===================================================================*/