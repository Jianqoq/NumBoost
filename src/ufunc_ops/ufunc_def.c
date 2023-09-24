#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "ufunc_def.h"
#include "ufunc_kernels.h"

Register_UFunc_Operation_FloatingTypes(where, Where_LoopBody, a, b, c);
Register_UFunc_Operation_IntergerTypes(where, Where_LoopBody, a, b, c);
Register_UFunc_Operation_Err_UnsupportTypes(where, a, b, c);
Register_UFunc_Operation_Array(where, a, b, c);
Register_UFunc_Operation_Method(where, a, b, c);