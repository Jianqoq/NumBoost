#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "element_ops_def.h"
#include "../numboost_utils.h"
#include "omp.h"

Register_ElementWise_Operations_Floating_Types(abs, Abs_LoopBody);
Register_ElementWise_Operation(abs, bool, bool, NPY_BOOL, Abs_LoopBody);
Register_ElementWise_Operation(abs, byte, byte, NPY_BYTE, Abs_LoopBody);
Register_ElementWise_Operation(abs, ubyte, ubyte, NPY_UBYTE,
                               AbsUnsigned_LoopBody);
Register_ElementWise_Operation(abs, short, short, NPY_SHORT, Abs_LoopBody);
Register_ElementWise_Operation(abs, ushort, ushort, NPY_USHORT,
                               AbsUnsigned_LoopBody);
Register_ElementWise_Operation(abs, int, int, NPY_INT, Abs_LoopBody);
Register_ElementWise_Operation(abs, uint, uint, NPY_UINT, AbsUnsigned_LoopBody);
Register_ElementWise_Operation(abs, long, long, NPY_LONG, Abs_LoopBody);
Register_ElementWise_Operation(abs, ulong, ulong, NPY_ULONG,
                               AbsUnsigned_LoopBody);
Register_ElementWise_Operation(abs, longlong, longlong, NPY_LONGLONG,
                               Abs_LoopBody);
Register_ElementWise_Operation(abs, ulonglong, ulonglong, NPY_ULONGLONG,
                               AbsUnsigned_LoopBody);
Register_ElementWise_Operation_Err_Not_Support_Types(abs);
Register_ElementWise_Operation_Array(abs, );
Register_ElementWise_Operation_Method(abs, ABS);

Register_ElementWise_Operations_Floating_Types(negative, Negative_LoopBody);
Register_ElementWise_Operations_Interger_Types(negative, Negative_LoopBody);
Register_ElementWise_Operation_Err_Not_Support_Types(negative);
Register_ElementWise_Operation_Array(negative, );
Register_ElementWise_Operation_Method(negative, NEGATIVE);

Register_ElementWise_Operations_Floating_Types(sin, Sin_LoopBody);
Register_ElementWise_Operation_Err_Interger_Types(sin);
Register_ElementWise_Operation_Err_Not_Support_Types(sin);
Register_ElementWise_Operation_Array(sin, );
Register_ElementWise_Operation_Method(sin, SIN);

Register_ElementWise_Operations_Floating_Types(cos, Cos_LoopBody);
Register_ElementWise_Operation_Err_Interger_Types(cos);
Register_ElementWise_Operation_Err_Not_Support_Types(cos);
Register_ElementWise_Operation_Array(cos, );
Register_ElementWise_Operation_Method(cos, COS);

Register_ElementWise_Operations_Floating_Types(tan, Tan_LoopBody);
Register_ElementWise_Operation_Err_Interger_Types(tan);
Register_ElementWise_Operation_Err_Not_Support_Types(tan);
Register_ElementWise_Operation_Array(tan, );
Register_ElementWise_Operation_Method(tan, TAN);

Register_ElementWise_Operations_Floating_Types(asin, Asin_LoopBody);
Register_ElementWise_Operation_Err_Interger_Types(asin);
Register_ElementWise_Operation_Err_Not_Support_Types(asin);
Register_ElementWise_Operation_Array(asin, );
Register_ElementWise_Operation_Method(asin, ARCSIN);

Register_ElementWise_Operations_Floating_Types(acos, Acos_LoopBody);
Register_ElementWise_Operation_Err_Interger_Types(acos);
Register_ElementWise_Operation_Err_Not_Support_Types(acos);
Register_ElementWise_Operation_Array(acos, );
Register_ElementWise_Operation_Method(acos, ARCCOS);

Register_ElementWise_Operations_Floating_Types(atan, Atan_LoopBody);
Register_ElementWise_Operation_Err_Interger_Types(atan);
Register_ElementWise_Operation_Err_Not_Support_Types(atan);
Register_ElementWise_Operation_Array(atan, );
Register_ElementWise_Operation_Method(atan, ARCTAN);

Register_ElementWise_Operations_Floating_Types(sinh, Sinh_LoopBody);
Register_ElementWise_Operation_Err_Interger_Types(sinh);
Register_ElementWise_Operation_Err_Not_Support_Types(sinh);
Register_ElementWise_Operation_Array(sinh, );
Register_ElementWise_Operation_Method(sinh, SINH);

Register_ElementWise_Operations_Floating_Types(cosh, Cosh_LoopBody);
Register_ElementWise_Operation_Err_Interger_Types(cosh);
Register_ElementWise_Operation_Err_Not_Support_Types(cosh);
Register_ElementWise_Operation_Array(cosh, );
Register_ElementWise_Operation_Method(cosh, COSH);

Register_ElementWise_Operations_Floating_Types(tanh, Tanh_LoopBody);
Register_ElementWise_Operation_Err_Interger_Types(tanh);
Register_ElementWise_Operation_Err_Not_Support_Types(tanh);
Register_ElementWise_Operation_Array(tanh, );
Register_ElementWise_Operation_Method(tanh, TANH);

Register_ElementWise_Operations_Floating_Types(asinh, Asinh_LoopBody);
Register_ElementWise_Operation_Err_Interger_Types(asinh);
Register_ElementWise_Operation_Err_Not_Support_Types(asinh);
Register_ElementWise_Operation_Array(asinh, );
Register_ElementWise_Operation_Method(asinh, ARCSINH);

Register_ElementWise_Operations_Floating_Types(acosh, Acosh_LoopBody);
Register_ElementWise_Operation_Err_Interger_Types(acosh);
Register_ElementWise_Operation_Err_Not_Support_Types(acosh);
Register_ElementWise_Operation_Array(acosh, );
Register_ElementWise_Operation_Method(acosh, ARCCOSH);

Register_ElementWise_Operations_Floating_Types(atanh, Atanh_LoopBody);
Register_ElementWise_Operation_Err_Interger_Types(atanh);
Register_ElementWise_Operation_Err_Not_Support_Types(atanh);
Register_ElementWise_Operation_Array(atanh, );
Register_ElementWise_Operation_Method(atanh, ARCTANH);

Register_ElementWise_Operations_Floating_Types(sqrt, Sqrt_LoopBody);
Register_ElementWise_Operation_Err_Interger_Types(sqrt);
Register_ElementWise_Operation_Err_Not_Support_Types(sqrt);
Register_ElementWise_Operation_Array(sqrt, );
Register_ElementWise_Operation_Method(sqrt, SQRT);

Register_ElementWise_Operations_Floating_Types(log, Log_LoopBody);
Register_ElementWise_Operation_Err_Interger_Types(log);
Register_ElementWise_Operation_Err_Not_Support_Types(log);
Register_ElementWise_Operation_Array(log, );
Register_ElementWise_Operation_Method(log, LOG);

Register_ElementWise_Operations_Floating_Types(log10, Log10_LoopBody);
Register_ElementWise_Operation_Err_Interger_Types(log10);
Register_ElementWise_Operation_Err_Not_Support_Types(log10);
Register_ElementWise_Operation_Array(log10, );
Register_ElementWise_Operation_Method(log10, LOG10);

Register_ElementWise_Operations_Floating_Types(exp, Exp_LoopBody);
Register_ElementWise_Operation_Err_Interger_Types(exp);
Register_ElementWise_Operation_Err_Not_Support_Types(exp);
Register_ElementWise_Operation_Array(exp, );
Register_ElementWise_Operation_Method(exp, EXP);