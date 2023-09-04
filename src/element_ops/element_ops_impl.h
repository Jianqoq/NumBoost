#include "../numboost_api.h"

#define Abs_LoopBody(type, i, result_ptr, stride_a_last, a_ptr)                \
  Generic(type) a_val = Cast_Float_When_Half(type, a_ptr[i * stride_a_last]);  \
  result_ptr[i] =                                                              \
      Cast_Half_When_Half(Generic(type), a_val > 0 ? a_val : -a_val);

#define Abs_LoopBody_Sequential(type, i, result_ptr, a_ptr)                    \
  Generic(type) a_val = Cast_Float_When_Half(type, a_ptr[i]);                  \
  result_ptr[i] =                                                              \
      Cast_Half_When_Half(Generic(type), a_val > 0 ? a_val : -a_val);

#define Sin_LoopBody(type, i, result_ptr, stride_a_last, a_ptr)                \
  Generic(type) a_val = Cast_Float_When_Half(type, a_ptr[i * stride_a_last]);  \
  Generic(type) result = Map_Method(Generic(type), npy_sin, a_val);            \
  result_ptr[i] = Cast_Half_When_Half(Generic(type), result);

#define Sin_LoopBody_Sequential(type, i, result_ptr, a_ptr)                    \
  Generic(type) a_val = Cast_Float_When_Half(type, a_ptr[i]);                  \
  Generic(type) result = Map_Method(Generic(type), npy_sin, a_val);            \
  result_ptr[i] = Cast_Half_When_Half(Generic(type), result);

#define Cos_LoopBody(type, i, result_ptr, stride_a_last, a_ptr)                \
  Generic(type) a_val = Cast_Float_When_Half(type, a_ptr[i * stride_a_last]);  \
  Generic(type) result = Map_Method(Generic(type), npy_cos, a_val);            \
  result_ptr[i] = Cast_Half_When_Half(Generic(type), result);

#define Cos_LoopBody_Sequential(type, i, result_ptr, a_ptr)                    \
  Generic(type) a_val = Cast_Float_When_Half(type, a_ptr[i]);                  \
  Generic(type) result = Map_Method(Generic(type), npy_cos, a_val);            \
  result_ptr[i] = Cast_Half_When_Half(Generic(type), result);

#define Tan_LoopBody(type, i, result_ptr, stride_a_last, a_ptr)                \
  Generic(type) a_val = Cast_Float_When_Half(type, a_ptr[i * stride_a_last]);  \
  Generic(type) result = Map_Method(Generic(type), npy_tan, a_val);            \
  result_ptr[i] = Cast_Half_When_Half(Generic(type), result);

#define Tan_LoopBody_Sequential(type, i, result_ptr, a_ptr)                    \
  Generic(type) a_val = Cast_Float_When_Half(type, a_ptr[i]);                  \
  Generic(type) result = Map_Method(Generic(type), npy_tan, a_val);            \
  result_ptr[i] = Cast_Half_When_Half(Generic(type), result);

#define Asin_LoopBody(type, i, result_ptr, stride_a_last, a_ptr)               \
  Generic(type) a_val = Cast_Float_When_Half(type, a_ptr[i * stride_a_last]);  \
  Generic(type) result = Map_Method(Generic(type), npy_asin, a_val);           \
  result_ptr[i] = Cast_Half_When_Half(Generic(type), result);

#define Asin_LoopBody_Sequential(type, i, result_ptr, a_ptr)                   \
  Generic(type) a_val = Cast_Float_When_Half(type, a_ptr[i]);                  \
  Generic(type) result = Map_Method(Generic(type), npy_asin, a_val);           \
  result_ptr[i] = Cast_Half_When_Half(Generic(type), result);

#define Acos_LoopBody(type, i, result_ptr, stride_a_last, a_ptr)               \
  Generic(type) a_val = Cast_Float_When_Half(type, a_ptr[i * stride_a_last]);  \
  Generic(type) result = Map_Method(Generic(type), npy_acos, a_val);           \
  result_ptr[i] = Cast_Half_When_Half(Generic(type), result);

#define Acos_LoopBody_Sequential(type, i, result_ptr, a_ptr)                   \
  Generic(type) a_val = Cast_Float_When_Half(type, a_ptr[i]);                  \
  Generic(type) result = Map_Method(Generic(type), npy_acos, a_val);           \
  result_ptr[i] = Cast_Half_When_Half(Generic(type), result);

#define Atan_LoopBody(type, i, result_ptr, stride_a_last, a_ptr)               \
  Generic(type) a_val = Cast_Float_When_Half(type, a_ptr[i * stride_a_last]);  \
  Generic(type) result = Map_Method(Generic(type), npy_atan, a_val);           \
  result_ptr[i] = Cast_Half_When_Half(Generic(type), result);

#define Atan_LoopBody_Sequential(type, i, result_ptr, a_ptr)                   \
  Generic(type) a_val = Cast_Float_When_Half(type, a_ptr[i]);                  \
  Generic(type) result = Map_Method(Generic(type), npy_atan, a_val);           \
  result_ptr[i] = Cast_Half_When_Half(Generic(type), result);

#define Sinh_LoopBody(type, i, result_ptr, stride_a_last, a_ptr)               \
  Generic(type) a_val = Cast_Float_When_Half(type, a_ptr[i * stride_a_last]);  \
  Generic(type) result = Map_Method(Generic(type), npy_sinh, a_val);           \
  result_ptr[i] = Cast_Half_When_Half(Generic(type), result);

#define Sinh_LoopBody_Sequential(type, i, result_ptr, a_ptr)                   \
  Generic(type) a_val = Cast_Float_When_Half(type, a_ptr[i]);                  \
  Generic(type) result = Map_Method(Generic(type), npy_sinh, a_val);           \
  result_ptr[i] = Cast_Half_When_Half(Generic(type), result);

#define Cosh_LoopBody(type, i, result_ptr, stride_a_last, a_ptr)               \
  Generic(type) a_val = Cast_Float_When_Half(type, a_ptr[i * stride_a_last]);  \
  Generic(type) result = Map_Method(Generic(type), npy_cosh, a_val);           \
  result_ptr[i] = Cast_Half_When_Half(Generic(type), result);

#define Cosh_LoopBody_Sequential(type, i, result_ptr, a_ptr)                   \
  Generic(type) a_val = Cast_Float_When_Half(type, a_ptr[i]);                  \
  Generic(type) result = Map_Method(Generic(type), npy_cosh, a_val);           \
  result_ptr[i] = Cast_Half_When_Half(Generic(type), result);

#define Tanh_LoopBody(type, i, result_ptr, stride_a_last, a_ptr)               \
  Generic(type) a_val = Cast_Float_When_Half(type, a_ptr[i * stride_a_last]);  \
  Generic(type) result = Map_Method(Generic(type), npy_tanh, a_val);           \
  result_ptr[i] = Cast_Half_When_Half(Generic(type), result);

#define Tanh_LoopBody_Sequential(type, i, result_ptr, a_ptr)                   \
  Generic(type) a_val = Cast_Float_When_Half(type, a_ptr[i]);                  \
  Generic(type) result = Map_Method(Generic(type), npy_tanh, a_val);           \
  result_ptr[i] = Cast_Half_When_Half(Generic(type), result);

#define Asinh_LoopBody(type, i, result_ptr, stride_a_last, a_ptr)              \
  Generic(type) a_val = Cast_Float_When_Half(type, a_ptr[i * stride_a_last]);  \
  Generic(type) result = Map_Method(Generic(type), npy_asinh, a_val);          \
  result_ptr[i] = Cast_Half_When_Half(Generic(type), result);

#define Asinh_LoopBody_Sequential(type, i, result_ptr, a_ptr)                  \
  Generic(type) a_val = Cast_Float_When_Half(type, a_ptr[i]);                  \
  Generic(type) result = Map_Method(Generic(type), npy_asinh, a_val);          \
  result_ptr[i] = Cast_Half_When_Half(Generic(type), result);

#define Acosh_LoopBody(type, i, result_ptr, stride_a_last, a_ptr)              \
  Generic(type) a_val = Cast_Float_When_Half(type, a_ptr[i * stride_a_last]);  \
  Generic(type) result = Map_Method(Generic(type), npy_acosh, a_val);          \
  result_ptr[i] = Cast_Half_When_Half(Generic(type), result);

#define Acosh_LoopBody_Sequential(type, i, result_ptr, a_ptr)                  \
  Generic(type) a_val = Cast_Float_When_Half(type, a_ptr[i]);                  \
  Generic(type) result = Map_Method(Generic(type), npy_acosh, a_val);          \
  result_ptr[i] = Cast_Half_When_Half(Generic(type), result);

#define Atanh_LoopBody(type, i, result_ptr, stride_a_last, a_ptr)              \
  Generic(type) a_val = Cast_Float_When_Half(type, a_ptr[i * stride_a_last]);  \
  Generic(type) result = Map_Method(Generic(type), npy_atanh, a_val);          \
  result_ptr[i] = Cast_Half_When_Half(Generic(type), result);

#define Atanh_LoopBody_Sequential(type, i, result_ptr, a_ptr)                  \
  Generic(type) a_val = Cast_Float_When_Half(type, a_ptr[i]);                  \
  Generic(type) result = Map_Method(Generic(type), npy_atanh, a_val);          \
  result_ptr[i] = Cast_Half_When_Half(Generic(type), result);