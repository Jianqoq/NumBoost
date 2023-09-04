#include "../numboost_api.h"

#define Abs_LoopBody(type, i, result_ptr, stride_a_last, a_ptr)                \
  Use_Float_When_Half(type) a_val =                                            \
      Cast_Float_When_Half(type, a_ptr[i * stride_a_last]);                    \
  result_ptr[i] = Cast_Half_When_Half(type, a_val > 0 ? a_val : -a_val);

#define Abs_LoopBody_Sequential(type, i, result_ptr, a_ptr)                    \
  Use_Float_When_Half(type) a_val = Cast_Float_When_Half(type, a_ptr[i]);      \
  result_ptr[i] = Cast_Half_When_Half(type, a_val > 0 ? a_val : -a_val);
