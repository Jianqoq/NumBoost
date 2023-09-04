#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "element_ops_def.h"
#include "../op.h"
#include "omp.h"

Register_ElementWise_Operations_Floating_Types(abs, Abs_LoopBody, Abs_LoopBody_Sequential);
Register_ElementWise_Operations_Interger_Types(abs, Abs_LoopBody, Abs_LoopBody_Sequential);
Register_ElementWise_Operation_Err_Not_Support_Types(abs);
Register_ElementWise_Operation_Array(abs, );
Register_ElementWise_Operation_Method(abs, ABS);
