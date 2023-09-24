#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "ufunc_def.h"
#include "ufunc_kernels.h"

Register_UFunc_Operation_FloatingTypes(where, Where_LoopBody, mask, x, y);
Register_UFunc_Operation_IntergerTypes(where, Where_LoopBody, mask, x, y);
Register_UFunc_Operation_Err_UnsupportTypes(where, mask, x, y);
Register_UFunc_Operation_Array(where, mask, x, y);
Register_UFunc_Operation_Method(where, mask, x, y);