#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "binary_op_def.h"
#include "../numboost_api.h"
#include "../numboost_math.h"
#include "../op.h"

Register_Binary_Operations_Floating_Types(add, Add_LoopBody);
Register_Binary_Operations_Interger_Types(add, Add_LoopBody);
Register_Binary_Operation_Err_Not_Support_Types(add);
Register_Binary_Operation_Array(add, );
Register_Binary_Operation_Method(add, ADD);

Register_Binary_Operations_Floating_Types(sub, Sub_LoopBody);
Register_Binary_Operations_Interger_Types(sub, Sub_LoopBody);
Register_Binary_Operation_Err_Not_Support_Types(sub);
Register_Binary_Operation_Array(sub, );
Register_Binary_Operation_Method(sub, SUB);

Register_Binary_Operations_Floating_Types(mul, Mul_LoopBody);
Register_Binary_Operations_Interger_Types(mul, Mul_LoopBody);
Register_Binary_Operation_Err_Not_Support_Types(mul);
Register_Binary_Operation_Array(mul, );
Register_Binary_Operation_Method(mul, MUL);

Register_Binary_Operations_Floating_Types(div, Div_LoopBody);
Register_Binary_Operations_Err_Interger_Types(div);
Register_Binary_Operation_Err_Not_Support_Types(div);
Register_Binary_Operation_Array(div, );
Register_Binary_Operation_Method(div, DIV);

Register_Binary_Operations_Err_Floating_Types(lshift);
Register_Binary_Operations_Interger_Types(lshift, LShift_LoopBody);
Register_Binary_Operation_Err_Not_Support_Types(lshift);
Register_Binary_Operation_Array(lshift, );
Register_Binary_Operation_Method(lshift, LSHIFT);

Register_Binary_Operations_Err_Floating_Types(rshift);
Register_Binary_Operations_Interger_Types(rshift, RShift_LoopBody);
Register_Binary_Operation_Err_Not_Support_Types(rshift);
Register_Binary_Operation_Array(rshift, );
Register_Binary_Operation_Method(rshift, RSHIFT);

Register_Binary_Operations_Floating_Types(mod, Mod_LoopBody);
Register_Binary_Operations_Interger_Types(mod, Mod_LoopBody);
Register_Binary_Operation_Err_Not_Support_Types(mod);
Register_Binary_Operation_Array(mod, );
Register_Binary_Operation_Method(mod, MOD);

Register_Binary_Operations_Floating_Types(fdiv, FloorDiv_LoopBody);
Register_Binary_Operations_Interger_Types(fdiv, FloorDiv_LoopBody);
Register_Binary_Operation_Err_Not_Support_Types(fdiv);
Register_Binary_Operation_Array(fdiv, );
Register_Binary_Operation_Method(fdiv, FLOOR_DIV);

Register_Binary_Operations_Floating_Types(pow, Pow_LoopBody);
Register_Binary_Operations_Err_Interger_Types(pow);
Register_Binary_Operation_Err_Not_Support_Types(pow);
Register_Binary_Operation_Array(pow, );
Register_Binary_Operation_Method(pow, POW);