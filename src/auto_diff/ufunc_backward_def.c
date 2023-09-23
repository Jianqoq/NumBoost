#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "ufunc_backward_def.h"
#include "backward_kernels.h"
#include "omp.h"


Register_FuseBackward_Operation_FloatingTypes(sin, SinBackward_LoopBody, a, b);
Register_FuseBackward_Operation_Err_Int(sin, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(sin, a, b);
Register_FuseBackward_Operation_Array(sin, a, b);
Register_Backward_Operation_Method(sin, a, b);

Register_FuseBackward_Operation_FloatingTypes(cos, CosBackward_LoopBody, a, b);
Register_FuseBackward_Operation_Err_Int(cos, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(cos, a, b);
Register_FuseBackward_Operation_Array(cos, a, b);
Register_Backward_Operation_Method(cos, a, b);

Register_FuseBackward_Operation_FloatingTypes(tan, TanBackward_LoopBody, a, b);
Register_FuseBackward_Operation_Err_Int(tan, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(tan, a, b);
Register_FuseBackward_Operation_Array(tan, a, b);
Register_Backward_Operation_Method(tan, a, b);

Register_FuseBackward_Operation_FloatingTypes(arcsin, ArcsinBackward_LoopBody,
                                              a, b);
Register_FuseBackward_Operation_Err_Int(arcsin, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(arcsin, a, b);
Register_FuseBackward_Operation_Array(arcsin, a, b);
Register_Backward_Operation_Method(arcsin, a, b);

Register_FuseBackward_Operation_FloatingTypes(arccos, ArccosBackward_LoopBody,
                                              a, b);
Register_FuseBackward_Operation_Err_Int(arccos, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(arccos, a, b);
Register_FuseBackward_Operation_Array(arccos, a, b);
Register_Backward_Operation_Method(arccos, a, b);

Register_FuseBackward_Operation_FloatingTypes(arctan, ArctanBackward_LoopBody,
                                              a, b);
Register_FuseBackward_Operation_Err_Int(arctan, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(arctan, a, b);
Register_FuseBackward_Operation_Array(arctan, a, b);
Register_Backward_Operation_Method(arctan, a, b);

Register_FuseBackward_Operation_FloatingTypes(sinh, SinhBackward_LoopBody, a,
                                              b);
Register_FuseBackward_Operation_Err_Int(sinh, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(sinh, a, b);
Register_FuseBackward_Operation_Array(sinh, a, b);
Register_Backward_Operation_Method(sinh, a, b);

Register_FuseBackward_Operation_FloatingTypes(cosh, CoshBackward_LoopBody, a,
                                              b);
Register_FuseBackward_Operation_Err_Int(cosh, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(cosh, a, b);
Register_FuseBackward_Operation_Array(cosh, a, b);
Register_Backward_Operation_Method(cosh, a, b);

Register_FuseBackward_Operation_FloatingTypes(tanh, TanhBackward_LoopBody, a,
                                              b);
Register_FuseBackward_Operation_Err_Int(tanh, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(tanh, a, b);
Register_FuseBackward_Operation_Array(tanh, a, b);
Register_Backward_Operation_Method(tanh, a, b);

Register_FuseBackward_Operation_FloatingTypes(arcsinh, ArcsinhBackward_LoopBody,
                                              a, b);
Register_FuseBackward_Operation_Err_Int(arcsinh, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(arcsinh, a, b);
Register_FuseBackward_Operation_Array(arcsinh, a, b);
Register_Backward_Operation_Method(arcsinh, a, b);

Register_FuseBackward_Operation_FloatingTypes(arccosh, ArccoshBackward_LoopBody,
                                              a, b);
Register_FuseBackward_Operation_Err_Int(arccosh, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(arccosh, a, b);
Register_FuseBackward_Operation_Array(arccosh, a, b);
Register_Backward_Operation_Method(arccosh, a, b);

Register_FuseBackward_Operation_FloatingTypes(arctanh, ArctanhBackward_LoopBody,
                                              a, b);
Register_FuseBackward_Operation_Err_Int(arctanh, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(arctanh, a, b);
Register_FuseBackward_Operation_Array(arctanh, a, b);
Register_Backward_Operation_Method(arctanh, a, b);

Register_FuseBackward_Operation_FloatingTypes(exp, ExpBackward_LoopBody, a, b);
Register_FuseBackward_Operation_Err_Int(exp, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(exp, a, b);
Register_FuseBackward_Operation_Array(exp, a, b);
Register_Backward_Operation_Method(exp, a, b);

Register_FuseBackward_Operation_FloatingTypes(log, LogBackward_LoopBody, a, b);
Register_FuseBackward_Operation_Err_Int(log, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(log, a, b);
Register_FuseBackward_Operation_Array(log, a, b);
Register_Backward_Operation_Method(log, a, b);

Register_FuseBackward_Operation_FloatingTypes(log10, Log10Backward_LoopBody, a,
                                              b);
Register_FuseBackward_Operation_Err_Int(log10, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(log10, a, b);
Register_FuseBackward_Operation_Array(log10, a, b);
Register_Backward_Operation_Method(log10, a, b);

Register_FuseBackward_Operation_FloatingTypes(sqrt, SqrtBackward_LoopBody, a,
                                              b);
Register_FuseBackward_Operation_Err_Int(sqrt, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(sqrt, a, b);
Register_FuseBackward_Operation_Array(sqrt, a, b);
Register_Backward_Operation_Method(sqrt, a, b);

Register_FuseBackward_Operation_FloatingTypes(abs, AbsBackward_LoopBody, a, b);
Register_FuseBackward_Operation_IntergerTypes(abs, AbsBackward_LoopBody, a, b);
Register_FuseBackward_Operation_Err_UnsupportTypes(abs, a, b);
Register_FuseBackward_Operation_Array(abs, a, b);
Register_Backward_Operation_Method(abs, a, b);

Register_FuseBackward_Operation_FloatingTypes(power, PowerBackward_LoopBody, a,
                                              b, c);
Register_FuseBackward_Operation_Err_Int(power, a, power, grad);
Register_FuseBackward_Operation_Err_UnsupportTypes(power, a, power, grad);
Register_FuseBackward_Operation_Array(power, a, power, grad);
Register_Backward_Operation_Method(power, a, power, grad);
