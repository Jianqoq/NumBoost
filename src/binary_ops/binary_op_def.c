#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "binary_op_def.h"
#include "../numboost_api.h"
#include "../numboost_math.h"
#include "../numboost_utils.h"

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

Register_Binary_Operations_Err_Floating_Types(bitwise_and);
Register_Binary_Operations_Interger_Types(bitwise_and, And_LoopBody);
Register_Binary_Operation_Err_Not_Support_Types(bitwise_and);
Register_Binary_Operation_Array(bitwise_and, );
Register_Binary_Operation_Method(bitwise_and, BITWISE_AND);

Register_Binary_Operations_Err_Floating_Types(bitwise_xor);
Register_Binary_Operations_Interger_Types(bitwise_xor, Xor_LoopBody);
Register_Binary_Operation_Err_Not_Support_Types(bitwise_xor);
Register_Binary_Operation_Array(bitwise_xor, );
Register_Binary_Operation_Method(bitwise_xor, BITWISE_XOR);

Register_Binary_Operations_Err_Floating_Types(bitwise_or);
Register_Binary_Operations_Interger_Types(bitwise_or, Or_LoopBody);
Register_Binary_Operation_Err_Not_Support_Types(bitwise_or);
Register_Binary_Operation_Array(bitwise_or, );
Register_Binary_Operation_Method(bitwise_or, BITWISE_OR);

Register_Binary_Operations_Floating_Types_MultiOut(divmod, DivMod_LoopBody,
                                                   result1, result2);
Register_Binary_Operations_Interger_Types_MultiOut(divmod, DivMod_LoopBody,
                                                   result1, result2);
Register_Binary_Operation_Err_Not_Support_Types_MultiOut(divmod);
Register_Binary_Operation_Array_MultiOut(divmod, );
Register_Binary_Operation_Method_MultiOut(divmod, DIVMOD);

Register_Compare_Operations_Floating_Types(eq, EQ_LoopBody);
Register_Compare_Operations_Interger_Types(eq, EQ_LoopBody);
Register_Binary_Operation_Err_Not_Support_Types(eq);
Register_Binary_Operation_Array(eq, );
Register_Compare_Operation_Method(eq, EQ);

Register_Compare_Operations_Floating_Types(lt, LT_LoopBody);
Register_Compare_Operations_Interger_Types(lt, LT_LoopBody);
Register_Binary_Operation_Err_Not_Support_Types(lt);
Register_Binary_Operation_Array(lt, );
Register_Compare_Operation_Method(lt, LT);

Register_Compare_Operations_Floating_Types(le, LE_LoopBody);
Register_Compare_Operations_Interger_Types(le, LE_LoopBody);
Register_Binary_Operation_Err_Not_Support_Types(le);
Register_Binary_Operation_Array(le, );
Register_Compare_Operation_Method(le, LE);

Register_Compare_Operations_Floating_Types(gt, GT_LoopBody);
Register_Compare_Operations_Interger_Types(gt, GT_LoopBody);
Register_Binary_Operation_Err_Not_Support_Types(gt);
Register_Binary_Operation_Array(gt, );
Register_Compare_Operation_Method(gt, GT);

Register_Compare_Operations_Floating_Types(ge, GE_LoopBody);
Register_Compare_Operations_Interger_Types(ge, GE_LoopBody);
Register_Binary_Operation_Err_Not_Support_Types(ge);
Register_Binary_Operation_Array(ge, );
Register_Compare_Operation_Method(ge, GE);

Register_Compare_Operations_Floating_Types(neq, NEQ_LoopBody);
Register_Compare_Operations_Interger_Types(neq, NEQ_LoopBody);
Register_Binary_Operation_Err_Not_Support_Types(neq);
Register_Binary_Operation_Array(neq, );
Register_Compare_Operation_Method(neq, GT);