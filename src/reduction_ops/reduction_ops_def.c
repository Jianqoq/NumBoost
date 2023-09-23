#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "reduction_ops_def.h"
#include "../numboost_utils.h"
#include "limits.h"
#include "reduction_kernels.h"
#include "immintrin.h"

Register_Reduction_Operations_Floating_Types(sum, 0, Sum_Main, Empty_Pre,
                                             Empty_Post);
Register_Reduction_Operations_Interger_Types(sum, 0, Sum_Main, Empty_Pre,
                                             Empty_Post);
Register_Reduction_Operation_Err_Not_Support_Types(sum);
Register_Reduction_Operation_Array(sum, );
Register_Reduction_Operation_Method(sum, , SUM);

/*min def*/
Register_Reduction_Operation(min, float, NPY_FLOAT, NPY_INFINITYF, Min_Main,
                             Empty_Pre, Empty_Post);
Register_Reduction_Operation(min, double, NPY_DOUBLE, NPY_INFINITY, Min_Main,
                             Empty_Pre, Empty_Post);
Register_Reduction_Operation(min, longdouble, NPY_LONGDOUBLE, NPY_INFINITYL,
                             Min_Main, Empty_Pre, Empty_Post);
Register_Reduction_Operation(min, half, NPY_HALF, NPY_INFINITYF, Min_Main,
                             Empty_Pre, Empty_Post);
Register_Reduction_Operation(min, bool, NPY_BOOL, 1, Min_Main, Empty_Pre,
                             Empty_Post);
Register_Reduction_Operation(min, byte, NPY_BYTE, SCHAR_MAX, Min_Main,
                             Empty_Pre, Empty_Post);
Register_Reduction_Operation(min, ubyte, NPY_UBYTE, UCHAR_MAX, Min_Main,
                             Empty_Pre, Empty_Post);
Register_Reduction_Operation(min, short, NPY_SHORT, SHRT_MAX, Min_Main,
                             Empty_Pre, Empty_Post);
Register_Reduction_Operation(min, ushort, NPY_USHORT, USHRT_MAX, Min_Main,
                             Empty_Pre, Empty_Post);
Register_Reduction_Operation(min, int, NPY_INT, INT_MAX, Min_Main, Empty_Pre,
                             Empty_Post);
Register_Reduction_Operation(min, uint, NPY_UINT, UINT_MAX, Min_Main, Empty_Pre,
                             Empty_Post);
Register_Reduction_Operation(min, long, NPY_LONG, LONG_MAX, Min_Main, Empty_Pre,
                             Empty_Post);
Register_Reduction_Operation(min, ulong, NPY_ULONG, ULONG_MAX, Min_Main,
                             Empty_Pre, Empty_Post);
Register_Reduction_Operation(min, longlong, NPY_LONGLONG, LLONG_MAX, Min_Main,
                             Empty_Pre, Empty_Post);
Register_Reduction_Operation(min, ulonglong, NPY_ULONGLONG, ULLONG_MAX,
                             Min_Main, Empty_Pre, Empty_Post);
Register_Reduction_Operation_Err_Not_Support_Types(min);
Register_Reduction_Operation_Array(min, );
Register_Reduction_Operation_Method(min, , MIN);

/*max def*/
Register_Reduction_Operation(max, float, NPY_FLOAT, -NPY_INFINITYF, Max_Main,
                             Empty_Pre, Empty_Post);
Register_Reduction_Operation(max, double, NPY_DOUBLE, -NPY_INFINITY, Max_Main,
                             Empty_Pre, Empty_Post);
Register_Reduction_Operation(max, longdouble, NPY_LONGDOUBLE, -NPY_INFINITYL,
                             Max_Main, Empty_Pre, Empty_Post);
Register_Reduction_Operation(max, half, NPY_HALF, -NPY_INFINITYF, Max_Main,
                             Empty_Pre, Empty_Post);
Register_Reduction_Operation(max, bool, NPY_BOOL, 0, Max_Main, Empty_Pre,
                             Empty_Post);
Register_Reduction_Operation(max, byte, NPY_BYTE, -SCHAR_MAX, Max_Main,
                             Empty_Pre, Empty_Post);
Register_Reduction_Operation(max, ubyte, NPY_UBYTE, 0, Max_Main, Empty_Pre,
                             Empty_Post);
Register_Reduction_Operation(max, short, NPY_SHORT, -SHRT_MAX, Max_Main,
                             Empty_Pre, Empty_Post);
Register_Reduction_Operation(max, ushort, NPY_USHORT, 0, Max_Main, Empty_Pre,
                             Empty_Post);
Register_Reduction_Operation(max, int, NPY_INT, -INT_MAX, Max_Main, Empty_Pre,
                             Empty_Post);
Register_Reduction_Operation(max, uint, NPY_UINT, 0, Max_Main, Empty_Pre,
                             Empty_Post);
Register_Reduction_Operation(max, long, NPY_LONG, -LONG_MAX, Max_Main,
                             Empty_Pre, Empty_Post);
Register_Reduction_Operation(max, ulong, NPY_ULONG, 0, Max_Main, Empty_Pre,
                             Empty_Post);
Register_Reduction_Operation(max, longlong, NPY_LONGLONG, -LLONG_MAX, Max_Main,
                             Empty_Pre, Empty_Post);
Register_Reduction_Operation(max, ulonglong, NPY_ULONGLONG, 0, Max_Main,
                             Empty_Pre, Empty_Post);
Register_Reduction_Operation_Err_Not_Support_Types(max);
Register_Reduction_Operation_Array(max, );
Register_Reduction_Operation_Method(max, , MAX);

/*argmin def*/
Register_Arg_Reduction_Operation(argmin, float, NPY_FLOAT, NPY_INFINITYF,
                                 ArgMin_Main);
Register_Arg_Reduction_Operation(argmin, double, NPY_DOUBLE, NPY_INFINITY,
                                 ArgMin_Main);
Register_Arg_Reduction_Operation(argmin, longdouble, NPY_LONGDOUBLE,
                                 NPY_INFINITYL, ArgMin_Main);
Register_Arg_Reduction_Operation(argmin, half, NPY_HALF, NPY_INFINITYF,
                                 ArgMin_Main);
Register_Arg_Reduction_Operation(argmin, bool, NPY_BOOL, 1, ArgMin_Main);
Register_Arg_Reduction_Operation(argmin, byte, NPY_BYTE, SCHAR_MAX,
                                 ArgMin_Main);
Register_Arg_Reduction_Operation(argmin, ubyte, NPY_UBYTE, UCHAR_MAX,
                                 ArgMin_Main);
Register_Arg_Reduction_Operation(argmin, short, NPY_SHORT, SHRT_MAX,
                                 ArgMin_Main);
Register_Arg_Reduction_Operation(argmin, ushort, NPY_USHORT, USHRT_MAX,
                                 ArgMin_Main);
Register_Arg_Reduction_Operation(argmin, int, NPY_INT, INT_MAX, ArgMin_Main);
Register_Arg_Reduction_Operation(argmin, uint, NPY_UINT, UINT_MAX, ArgMin_Main);
Register_Arg_Reduction_Operation(argmin, long, NPY_LONG, LONG_MAX, ArgMin_Main);
Register_Arg_Reduction_Operation(argmin, ulong, NPY_ULONG, ULONG_MAX,
                                 ArgMin_Main);
Register_Arg_Reduction_Operation(argmin, longlong, NPY_LONGLONG, LLONG_MAX,
                                 ArgMin_Main);
Register_Arg_Reduction_Operation(argmin, ulonglong, NPY_ULONGLONG, ULLONG_MAX,
                                 ArgMin_Main);
Register_Reduction_Operation_Err_Not_Support_Types(argmin);
Register_Reduction_Operation_Array(argmin, );
Register_Arg_Reduction_Operation_Method(argmin, , ARGMIN);

/*argmax def*/
Register_Arg_Reduction_Operation(argmax, float, NPY_FLOAT, -NPY_INFINITYF,
                                 ArgMax_Main);
Register_Arg_Reduction_Operation(argmax, double, NPY_DOUBLE, -NPY_INFINITY,
                                 ArgMax_Main);
Register_Arg_Reduction_Operation(argmax, longdouble, NPY_LONGDOUBLE,
                                 -NPY_INFINITYL, ArgMax_Main);
Register_Arg_Reduction_Operation(argmax, half, NPY_HALF, -NPY_INFINITYF,
                                 ArgMax_Main);
Register_Arg_Reduction_Operation(argmax, bool, NPY_BOOL, 0, ArgMax_Main);
Register_Arg_Reduction_Operation(argmax, byte, NPY_BYTE, -SCHAR_MAX,
                                 ArgMax_Main);
Register_Arg_Reduction_Operation(argmax, ubyte, NPY_UBYTE, 0, ArgMax_Main);
Register_Arg_Reduction_Operation(argmax, short, NPY_SHORT, -SHRT_MAX,
                                 ArgMax_Main);
Register_Arg_Reduction_Operation(argmax, ushort, NPY_USHORT, 0, ArgMax_Main);
Register_Arg_Reduction_Operation(argmax, int, NPY_INT, -INT_MAX, ArgMax_Main);
Register_Arg_Reduction_Operation(argmax, uint, NPY_UINT, 0, ArgMax_Main);
Register_Arg_Reduction_Operation(argmax, long, NPY_LONG, -LONG_MAX,
                                 ArgMax_Main);
Register_Arg_Reduction_Operation(argmax, ulong, NPY_ULONG, 0, ArgMax_Main);
Register_Arg_Reduction_Operation(argmax, longlong, NPY_LONGLONG, -LLONG_MAX,
                                 ArgMax_Main);
Register_Arg_Reduction_Operation(argmax, ulonglong, NPY_ULONGLONG, 0,
                                 ArgMax_Main);
Register_Reduction_Operation_Err_Not_Support_Types(argmax);
Register_Reduction_Operation_Array(argmax, );
Register_Arg_Reduction_Operation_Method(argmax, , ARGMAX);

/*mean def*/
Register_Mean_Reduction_Operation(mean, float, NPY_FLOAT, 0);
Register_Mean_Reduction_Operation(mean, double, NPY_DOUBLE, 0);
Register_Mean_Reduction_Operation(mean, longdouble, NPY_LONGDOUBLE, 0);
Register_Mean_Reduction_Operation(mean, half, NPY_HALF, 0);
Register_Reduction_Operation_Err_Interger_Types(mean);
Register_Reduction_Operation_Err_Not_Support_Types(mean);
Register_Reduction_Operation_Array(mean, );
Register_Reduction_Operation_Method(mean, , MEAN);