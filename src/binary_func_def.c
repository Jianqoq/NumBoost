#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "binary_func_def.h"

/*====================================================== Add ===============================================================================*/
Register_Int_Binary_Operations(add_, , Binary_Operation, nb_add, , Binary_Loop);
Register_Binary_Operation(add_, float, Binary_Operation, nb_add, NPY_FLOAT, Binary_Loop);
Register_Binary_Operation(add_, double, Binary_Operation, nb_add, NPY_DOUBLE, Binary_Loop);
Register_Binary_Operation(add_, longdouble, Binary_Operation, nb_add, NPY_LONGDOUBLE, Binary_Loop);
NotImplement_Err(add_, , PyArrayObject, PyArrayObject);
Register_Binary_Operation(add_, half, Binary_Operation, nb_add_half, NPY_HALF, Binary_Loop);

Register_Int_Binary_Operations(add_, _A_Scalar, Binary_Operation_A_Scalar, nb_add, , Binary_Loop_a_Scalar);
Register_Binary_Operation_A_Scalar(add_, float, Binary_Operation_A_Scalar, nb_add, NPY_FLOAT, Binary_Loop_a_Scalar);
Register_Binary_Operation_A_Scalar(add_, double, Binary_Operation_A_Scalar, nb_add, NPY_DOUBLE, Binary_Loop_a_Scalar);
Register_Binary_Operation_A_Scalar(add_, longdouble, Binary_Operation_A_Scalar, nb_add, NPY_LONGDOUBLE, Binary_Loop_a_Scalar);
NotImplement_Err(add_, _a_scalar, Python_Number, PyArrayObject);
Register_Binary_Operation_A_Scalar(add_, half, Binary_Operation_A_Scalar, nb_add_half, NPY_HALF, Binary_Loop_a_Scalar);

Register_Int_Binary_Operations(add_, _B_Scalar, Binary_Operation_B_Scalar, nb_add, , Binary_Loop_b_Scalar);
Register_Binary_Operation_B_Scalar(add_, float, Binary_Operation_B_Scalar, nb_add, NPY_FLOAT, Binary_Loop_b_Scalar);
Register_Binary_Operation_B_Scalar(add_, double, Binary_Operation_B_Scalar, nb_add, NPY_DOUBLE, Binary_Loop_b_Scalar);
Register_Binary_Operation_B_Scalar(add_, longdouble, Binary_Operation_B_Scalar, nb_add, NPY_LONGDOUBLE, Binary_Loop_b_Scalar);
NotImplement_Err(add_, _b_scalar, PyArrayObject, Python_Number);
Register_Binary_Operation_B_Scalar(add_, half, Binary_Operation_B_Scalar, nb_add_half, NPY_HALF, Binary_Loop_b_Scalar);

/*====================================================== Sub ===============================================================================*/
Register_Int_Binary_Operations(sub_, , Binary_Operation, nb_subtract, , Binary_Loop);
Register_Binary_Operation(sub_, float, Binary_Operation, nb_subtract, NPY_FLOAT, Binary_Loop);
Register_Binary_Operation(sub_, double, Binary_Operation, nb_subtract, NPY_DOUBLE, Binary_Loop);
Register_Binary_Operation(sub_, longdouble, Binary_Operation, nb_subtract, NPY_LONGDOUBLE, Binary_Loop);
NotImplement_Err(sub_, , PyArrayObject, PyArrayObject);
Register_Binary_Operation(sub_, half, Binary_Operation, nb_subtract_half, NPY_HALF, Binary_Loop);

Register_Int_Binary_Operations(sub_, _A_Scalar, Binary_Operation_A_Scalar, nb_subtract, , Binary_Loop_a_Scalar);
Register_Binary_Operation_A_Scalar(sub_, float, Binary_Operation_A_Scalar, nb_subtract, NPY_FLOAT, Binary_Loop_a_Scalar);
Register_Binary_Operation_A_Scalar(sub_, double, Binary_Operation_A_Scalar, nb_subtract, NPY_DOUBLE, Binary_Loop_a_Scalar);
Register_Binary_Operation_A_Scalar(sub_, longdouble, Binary_Operation_A_Scalar, nb_subtract, NPY_LONGDOUBLE, Binary_Loop_a_Scalar);
NotImplement_Err(sub_, _a_scalar, Python_Number, PyArrayObject);
Register_Binary_Operation_A_Scalar(sub_, half, Binary_Operation_A_Scalar, nb_subtract_half, NPY_HALF, Binary_Loop_a_Scalar);

Register_Int_Binary_Operations(sub_, _B_Scalar, Binary_Operation_B_Scalar, nb_subtract, , Binary_Loop_b_Scalar);
Register_Binary_Operation_B_Scalar(sub_, float, Binary_Operation_B_Scalar, nb_subtract, NPY_FLOAT, Binary_Loop_b_Scalar);
Register_Binary_Operation_B_Scalar(sub_, double, Binary_Operation_B_Scalar, nb_subtract, NPY_DOUBLE, Binary_Loop_b_Scalar);
Register_Binary_Operation_B_Scalar(sub_, longdouble, Binary_Operation_B_Scalar, nb_subtract, NPY_LONGDOUBLE, Binary_Loop_b_Scalar);
NotImplement_Err(sub_, _b_scalar, PyArrayObject, Python_Number);
Register_Binary_Operation_B_Scalar(sub_, half, Binary_Operation_B_Scalar, nb_subtract_half, NPY_HALF, Binary_Loop_b_Scalar);

/*====================================================== Mul ===============================================================================*/
Register_Int_Binary_Operations(mul_, , Binary_Operation, nb_multiply, , Binary_Loop);
Register_Binary_Operation(mul_, float, Binary_Operation, nb_multiply, NPY_FLOAT, Binary_Loop);
Register_Binary_Operation(mul_, double, Binary_Operation, nb_multiply, NPY_DOUBLE, Binary_Loop);
Register_Binary_Operation(mul_, longdouble, Binary_Operation, nb_multiply, NPY_LONGDOUBLE, Binary_Loop);
NotImplement_Err(mul_, , PyArrayObject, PyArrayObject);
Register_Binary_Operation(mul_, half, Binary_Operation, nb_multiply_half, NPY_HALF, Binary_Loop);

Register_Int_Binary_Operations(mul_, _A_Scalar, Binary_Operation_A_Scalar, nb_multiply, , Binary_Loop_a_Scalar);
Register_Binary_Operation_A_Scalar(mul_, float, Binary_Operation_A_Scalar, nb_multiply, NPY_FLOAT, Binary_Loop_a_Scalar);
Register_Binary_Operation_A_Scalar(mul_, double, Binary_Operation_A_Scalar, nb_multiply, NPY_DOUBLE, Binary_Loop_a_Scalar);
Register_Binary_Operation_A_Scalar(mul_, longdouble, Binary_Operation_A_Scalar, nb_multiply, NPY_LONGDOUBLE, Binary_Loop_a_Scalar);
NotImplement_Err(mul_, _a_scalar, Python_Number, PyArrayObject);
Register_Binary_Operation_A_Scalar(mul_, half, Binary_Operation_A_Scalar, nb_multiply_half, NPY_HALF, Binary_Loop_a_Scalar);

Register_Int_Binary_Operations(mul_, _B_Scalar, Binary_Operation_B_Scalar, nb_multiply, , Binary_Loop_b_Scalar);
Register_Binary_Operation_B_Scalar(mul_, float, Binary_Operation_B_Scalar, nb_multiply, NPY_FLOAT, Binary_Loop_b_Scalar);
Register_Binary_Operation_B_Scalar(mul_, double, Binary_Operation_B_Scalar, nb_multiply, NPY_DOUBLE, Binary_Loop_b_Scalar);
Register_Binary_Operation_B_Scalar(mul_, longdouble, Binary_Operation_B_Scalar, nb_multiply, NPY_LONGDOUBLE, Binary_Loop_b_Scalar);
NotImplement_Err(mul_, _b_scalar, PyArrayObject, Python_Number);
Register_Binary_Operation_B_Scalar(mul_, half, Binary_Operation_B_Scalar, nb_multiply_half, NPY_HALF, Binary_Loop_b_Scalar);

/*====================================================== Div ===============================================================================*/
Register_Int_Binary_OperationsErr(div_, , PyArrayObject, PyArrayObject);
Register_Binary_Operation(div_, float, Float_Div_Binary_Operation, nb_divide, div_result_type_pick(NPY_FLOAT), _);
Register_Binary_Operation(div_, double, Double_Div_Binary_Operation, nb_divide, div_result_type_pick(NPY_DOUBLE), _);
Register_Binary_Operation(div_, longdouble, LongDouble_Div_Binary_Operation, nb_divide, div_result_type_pick(NPY_LONGDOUBLE), _);
NotImplement_Err(div_, , PyArrayObject, PyArrayObject);
Register_Binary_Operation(div_, half, Half_Div_Binary_Operation, nb_divide, div_result_type_pick(NPY_HALF), _);

Register_Int_Binary_OperationsErr(div_, _a_scalar, PyArrayObject, PyArrayObject);
Register_Binary_Operation_A_Scalar(div_, float, Float_Div_Binary_Operation_A_Scalar, nb_divide, div_result_type_pick(NPY_FLOAT), _a_scalar);
Register_Binary_Operation_A_Scalar(div_, double, Double_Div_Binary_Operation_A_Scalar, nb_divide, div_result_type_pick(NPY_DOUBLE), _a_scalar);
Register_Binary_Operation_A_Scalar(div_, longdouble, LongDouble_Div_Binary_Operation_A_Scalar, nb_divide, div_result_type_pick(NPY_LONGDOUBLE), _a_scalar);
NotImplement_Err(div_, _a_scalar, Python_Number, PyArrayObject);
Register_Binary_Operation_A_Scalar(div_, half, Half_Div_Binary_Operation_A_Scalar, nb_divide, div_result_type_pick(NPY_HALF), _a_scalar);

Register_Int_Binary_OperationsErr(div_, _b_scalar, PyArrayObject, PyArrayObject);
Register_Binary_Operation_B_Scalar(div_, float, Float_Div_Binary_Operation_B_Scalar, nb_divide, div_result_type_pick(NPY_FLOAT), _b_scalar);
Register_Binary_Operation_B_Scalar(div_, double, Double_Div_Binary_Operation_B_Scalar, nb_divide, div_result_type_pick(NPY_DOUBLE), _b_scalar);
Register_Binary_Operation_B_Scalar(div_, longdouble, LongDouble_Div_Binary_Operation_B_Scalar, nb_divide, div_result_type_pick(NPY_LONGDOUBLE), _b_scalar);
NotImplement_Err(div_, _b_scalar, PyArrayObject, Python_Number);
Register_Binary_Operation_B_Scalar(div_, half, Half_Div_Binary_Operation_B_Scalar, nb_divide, div_result_type_pick(NPY_HALF), _b_scalar);

/*====================================================== Pow ===============================================================================*/
Register_Int_Binary_OperationsErr(pow_, , PyArrayObject, PyArrayObject);
Register_Binary_Operation(pow_, float, Binary_Operation, nb_power_float, NPY_FLOAT, Binary_Loop);
Register_Binary_Operation(pow_, double, Binary_Operation, nb_power_double, NPY_DOUBLE, Binary_Loop);
Register_Binary_Operation(pow_, longdouble, Binary_Operation, nb_power_long_double, NPY_LONGDOUBLE, Binary_Loop);
NotImplement_Err(pow_, , PyArrayObject, PyArrayObject);
Register_Binary_Operation(pow_, half, Binary_Operation, nb_power_half, NPY_HALF, Binary_Loop);

Register_Int_Binary_OperationsErr(pow_, _a_scalar, Python_Number, PyArrayObject);
Register_Binary_Operation_A_Scalar(pow_, float, Binary_Operation_A_Scalar, nb_power_float, NPY_FLOAT, Binary_Loop_a_Scalar);
Register_Binary_Operation_A_Scalar(pow_, double, Binary_Operation_A_Scalar, nb_power_double, NPY_DOUBLE, Binary_Loop_a_Scalar);
Register_Binary_Operation_A_Scalar(pow_, longdouble, Binary_Operation_A_Scalar, nb_power_long_double, NPY_LONGDOUBLE, Binary_Loop_a_Scalar);
NotImplement_Err(pow_, _a_scalar, Python_Number, PyArrayObject);
Register_Binary_Operation_A_Scalar(pow_, half, Binary_Operation_A_Scalar, nb_power_half, NPY_HALF, Binary_Loop_a_Scalar);

Register_Int_Binary_OperationsErr(pow_, _b_scalar, PyArrayObject, Python_Number);
Register_Binary_Operation_B_Scalar(pow_, float, Binary_Operation_B_Scalar, nb_power_float, NPY_FLOAT, Binary_Loop_b_Scalar);
Register_Binary_Operation_B_Scalar(pow_, double, Binary_Operation_B_Scalar, nb_power_double, NPY_DOUBLE, Binary_Loop_b_Scalar);
Register_Binary_Operation_B_Scalar(pow_, longdouble, Binary_Operation_B_Scalar, nb_power_long_double, NPY_LONGDOUBLE, Binary_Loop_b_Scalar);
NotImplement_Err(pow_, _b_scalar, PyArrayObject, Python_Number);
Register_Binary_Operation_B_Scalar(pow_, half, Binary_Operation_B_Scalar, nb_power_half, NPY_HALF, Binary_Loop_b_Scalar);

/*====================================================== Mod ===============================================================================*/
Register_Int_Binary_Operations(mod_, , Binary_Operation, nb_mod_int, , Mod_Binary_Loop);
Register_Binary_Operation(mod_, float, Binary_Operation, nb_mod_float, NPY_FLOAT, Mod_Binary_Loop);
Register_Binary_Operation(mod_, double, Binary_Operation, nb_mod_double, NPY_DOUBLE, Mod_Binary_Loop);
Register_Binary_Operation(mod_, longdouble, Binary_Operation, nb_mod_long_double, NPY_LONGDOUBLE, Mod_Binary_Loop);
NotImplement_Err(mod_, , PyArrayObject, PyArrayObject);
Register_Binary_Operation(mod_, half, Binary_Operation, nb_mod_half, NPY_HALF, Modh_Binary_Loop);

Register_Int_Binary_Operations(mod_, _A_Scalar, Binary_Operation_A_Scalar, nb_mod_int, , Mod_Binary_Loop_a_Scalar);
Register_Binary_Operation_A_Scalar(mod_, float, Binary_Operation_A_Scalar, nb_mod_float, NPY_FLOAT, Mod_Binary_Loop_a_Scalar);
Register_Binary_Operation_A_Scalar(mod_, double, Binary_Operation_A_Scalar, nb_mod_double, NPY_DOUBLE, Mod_Binary_Loop_a_Scalar);
Register_Binary_Operation_A_Scalar(mod_, longdouble, Binary_Operation_A_Scalar, nb_mod_long_double, NPY_LONGDOUBLE, Mod_Binary_Loop_a_Scalar);
NotImplement_Err(mod_, _a_scalar, Python_Number, PyArrayObject);
Register_Binary_Operation_A_Scalar(mod_, half, Binary_Operation_A_Scalar, nb_mod_half, NPY_HALF, Modh_Binary_Loop_a_Scalar);

Register_Int_Binary_Operations(mod_, _B_Scalar, Binary_Operation_B_Scalar, nb_mod_int, , Mod_Binary_Loop_b_Scalar);
Register_Binary_Operation_B_Scalar(mod_, float, Binary_Operation_B_Scalar, nb_mod_float, NPY_FLOAT, Mod_Binary_Loop_b_Scalar);
Register_Binary_Operation_B_Scalar(mod_, double, Binary_Operation_B_Scalar, nb_mod_double, NPY_DOUBLE, Mod_Binary_Loop_b_Scalar);
Register_Binary_Operation_B_Scalar(mod_, longdouble, Binary_Operation_B_Scalar, nb_mod_long_double, NPY_LONGDOUBLE, Mod_Binary_Loop_b_Scalar);
NotImplement_Err(mod_, _b_scalar, PyArrayObject, Python_Number);
Register_Binary_Operation_B_Scalar(mod_, half, Binary_Operation_B_Scalar, nb_mod_half, NPY_HALF, Modh_Binary_Loop_b_Scalar);

/*====================================================== Floor Div ===============================================================================*/
Register_Int_Binary_Operations(floor_div_, , Binary_Operation, nb_floor_divide_int, , Binary_Loop);
Register_Binary_Operation(floor_div_, float, Binary_Operation, nb_floor_divide_float, NPY_FLOAT, Binary_Loop);
Register_Binary_Operation(floor_div_, double, Binary_Operation, nb_floor_divide_double, NPY_DOUBLE, Binary_Loop);
Register_Binary_Operation(floor_div_, longdouble, Binary_Operation, nb_floor_divide_long_double, NPY_LONGDOUBLE, Binary_Loop);
NotImplement_Err(floor_div_, , PyArrayObject, PyArrayObject);
Register_Binary_Operation(floor_div_, half, Binary_Operation, nb_floor_divide_half, NPY_HALF, Binary_Loop);

Register_Int_Binary_Operations(floor_div_, _A_Scalar, Binary_Operation_A_Scalar, nb_floor_divide_int, , Binary_Loop_a_Scalar);
Register_Binary_Operation_A_Scalar(floor_div_, float, Binary_Operation_A_Scalar, nb_floor_divide_float, NPY_FLOAT, Binary_Loop_a_Scalar);
Register_Binary_Operation_A_Scalar(floor_div_, double, Binary_Operation_A_Scalar, nb_floor_divide_double, NPY_DOUBLE, Binary_Loop_a_Scalar);
Register_Binary_Operation_A_Scalar(floor_div_, longdouble, Binary_Operation_A_Scalar, nb_floor_divide_long_double, NPY_LONGDOUBLE, Binary_Loop_a_Scalar);
NotImplement_Err(floor_div_, _a_scalar, Python_Number, PyArrayObject);
Register_Binary_Operation_A_Scalar(floor_div_, half, Binary_Operation_A_Scalar, nb_floor_divide_half, NPY_HALF, Binary_Loop_a_Scalar);

Register_Int_Binary_Operations(floor_div_, _B_Scalar, Binary_Operation_B_Scalar, nb_floor_divide_int, , Binary_Loop_b_Scalar);
Register_Binary_Operation_B_Scalar(floor_div_, float, Binary_Operation_B_Scalar, nb_floor_divide_float, NPY_FLOAT, Binary_Loop_b_Scalar);
Register_Binary_Operation_B_Scalar(floor_div_, double, Binary_Operation_B_Scalar, nb_floor_divide_double, NPY_DOUBLE, Binary_Loop_b_Scalar);
Register_Binary_Operation_B_Scalar(floor_div_, longdouble, Binary_Operation_B_Scalar, nb_floor_divide_long_double, NPY_LONGDOUBLE, Binary_Loop_b_Scalar);
NotImplement_Err(floor_div_, _b_scalar, PyArrayObject, Python_Number);
Register_Binary_Operation_B_Scalar(floor_div_, half, Binary_Operation_B_Scalar, nb_floor_divide_half, NPY_HALF, Binary_Loop_b_Scalar);

/*====================================================== Left Shift ===============================================================================*/
Register_Int_Binary_Operations(left_shift_, , Binary_Operation, nb_lshift, , Binary_Loop);
Register_Binary_Operation_Err(left_shift_, float, , PyArrayObject, PyArrayObject);
Register_Binary_Operation_Err(left_shift_, double, , PyArrayObject, PyArrayObject);
Register_Binary_Operation_Err(left_shift_, longdouble, , PyArrayObject, PyArrayObject);
NotImplement_Err(left_shift_, , PyArrayObject, PyArrayObject);
Register_Binary_Operation_Err(left_shift_, half, , PyArrayObject, PyArrayObject);

Register_Int_Binary_Operations(left_shift_, _A_Scalar, Binary_Operation_A_Scalar, nb_lshift, , Binary_Loop_a_Scalar);
Register_Binary_Operation_Err(left_shift_, float, _a_scalar, Python_Number, PyArrayObject);
Register_Binary_Operation_Err(left_shift_, double, _a_scalar, Python_Number, PyArrayObject);
Register_Binary_Operation_Err(left_shift_, longdouble, _a_scalar, Python_Number, PyArrayObject);
NotImplement_Err(left_shift_, _a_scalar, Python_Number, PyArrayObject);
Register_Binary_Operation_A_Scalar(left_shift_, half, Binary_Operation_A_Scalar, nb_lshift, NPY_HALF, Binary_Loop_a_Scalar);

Register_Int_Binary_Operations(left_shift_, _B_Scalar, Binary_Operation_B_Scalar, nb_lshift, , Binary_Loop_b_Scalar);
Register_Binary_Operation_Err(left_shift_, float, _b_scalar, PyArrayObject, Python_Number);
Register_Binary_Operation_Err(left_shift_, double, _b_scalar, PyArrayObject, Python_Number);
Register_Binary_Operation_Err(left_shift_, longdouble, _b_scalar, PyArrayObject, Python_Number);
NotImplement_Err(left_shift_, _b_scalar, PyArrayObject, Python_Number);
Register_Binary_Operation_B_Scalar(left_shift_, half, Binary_Operation_B_Scalar, nb_lshift, NPY_HALF, Binary_Loop_b_Scalar);

/*====================================================== Right Shift ===============================================================================*/
Register_Int_Binary_Operations(right_shift_, , Binary_Operation, nb_rshift, , Binary_Loop);
Register_Binary_Operation_Err(right_shift_, float, , PyArrayObject, PyArrayObject);
Register_Binary_Operation_Err(right_shift_, double, , PyArrayObject, PyArrayObject);
Register_Binary_Operation_Err(right_shift_, longdouble, , PyArrayObject, PyArrayObject);
NotImplement_Err(right_shift_, , PyArrayObject, PyArrayObject);
Register_Binary_Operation_Err(right_shift_, half, , PyArrayObject, PyArrayObject);

Register_Int_Binary_Operations(right_shift_, _A_Scalar, Binary_Operation_A_Scalar, nb_rshift, , Binary_Loop_a_Scalar);
Register_Binary_Operation_Err(right_shift_, float, _a_scalar, Python_Number, PyArrayObject);
Register_Binary_Operation_Err(right_shift_, double, _a_scalar, Python_Number, PyArrayObject);
Register_Binary_Operation_Err(right_shift_, longdouble, _a_scalar, Python_Number, PyArrayObject);
NotImplement_Err(right_shift_, _a_scalar, Python_Number, PyArrayObject);
Register_Binary_Operation_A_Scalar(right_shift_, half, Binary_Operation_A_Scalar, nb_rshift, NPY_HALF, Binary_Loop_a_Scalar);

Register_Int_Binary_Operations(right_shift_, _B_Scalar, Binary_Operation_B_Scalar, nb_rshift, , Binary_Loop_b_Scalar);
Register_Binary_Operation_Err(right_shift_, float, _b_scalar, PyArrayObject, Python_Number);
Register_Binary_Operation_Err(right_shift_, double, _b_scalar, PyArrayObject, Python_Number);
Register_Binary_Operation_Err(right_shift_, longdouble, _b_scalar, PyArrayObject, Python_Number);
NotImplement_Err(right_shift_, _b_scalar, PyArrayObject, Python_Number);
Register_Binary_Operation_B_Scalar(right_shift_, half, Binary_Operation_B_Scalar, nb_rshift, NPY_HALF, Binary_Loop_b_Scalar);

/*==================================================================================================================================================*/
Register_Binary_Operation_Array(add, , PyArrayObject, PyArrayObject);
Register_Binary_Operation_Array(add, _a_scalar, Python_Number, PyArrayObject);
Register_Binary_Operation_Array(add, _b_scalar, PyArrayObject, Python_Number);

Register_Binary_Operation_Array(sub, , PyArrayObject, PyArrayObject);
Register_Binary_Operation_Array(sub, _a_scalar, Python_Number, PyArrayObject);
Register_Binary_Operation_Array(sub, _b_scalar, PyArrayObject, Python_Number);

Register_Binary_Operation_Array(mul, , PyArrayObject, PyArrayObject);
Register_Binary_Operation_Array(mul, _a_scalar, Python_Number, PyArrayObject);
Register_Binary_Operation_Array(mul, _b_scalar, PyArrayObject, Python_Number);

Register_Binary_Operation_Array(div, , PyArrayObject, PyArrayObject);
Register_Binary_Operation_Array(div, _a_scalar, Python_Number, PyArrayObject);
Register_Binary_Operation_Array(div, _b_scalar, PyArrayObject, Python_Number);

Register_Binary_Operation_Array(mod, , PyArrayObject, PyArrayObject);
Register_Binary_Operation_Array(mod, _a_scalar, Python_Number, PyArrayObject);
Register_Binary_Operation_Array(mod, _b_scalar, PyArrayObject, Python_Number);

Register_Binary_Operation_Array(pow, , PyArrayObject, PyArrayObject);
Register_Binary_Operation_Array(pow, _a_scalar, Python_Number, PyArrayObject);
Register_Binary_Operation_Array(pow, _b_scalar, PyArrayObject, Python_Number);

Register_Binary_Operation_Array(left_shift, , PyArrayObject, PyArrayObject);
Register_Binary_Operation_Array(left_shift, _a_scalar, Python_Number, PyArrayObject);
Register_Binary_Operation_Array(left_shift, _b_scalar, PyArrayObject, Python_Number);

Register_Binary_Operation_Array(right_shift, , PyArrayObject, PyArrayObject);
Register_Binary_Operation_Array(right_shift, _a_scalar, Python_Number, PyArrayObject);
Register_Binary_Operation_Array(right_shift, _b_scalar, PyArrayObject, Python_Number);

Register_Binary_Operation_Array(floor_div, , PyArrayObject, PyArrayObject);
Register_Binary_Operation_Array(floor_div, _a_scalar, Python_Number, PyArrayObject);
Register_Binary_Operation_Array(floor_div, _b_scalar, PyArrayObject, Python_Number);

PyArrayObject *(**operations[])(PyArrayObject *, PyArrayObject *) = {add_operations, sub_operations, mul_operations, div_operations,
                                                                     mod_operations, pow_operations, left_shift_operations,
                                                                     right_shift_operations, floor_div_operations};

PyArrayObject *(**operations_a_scalar[])(Python_Number *, PyArrayObject *) = {add_operations_a_scalar, sub_operations_a_scalar,
                                                                              mul_operations_a_scalar, div_operations_a_scalar,
                                                                              mod_operations_a_scalar, pow_operations_a_scalar,
                                                                              left_shift_operations_a_scalar, right_shift_operations_a_scalar,
                                                                              floor_div_operations_a_scalar};

PyArrayObject *(**operations_b_scalar[])(PyArrayObject *, Python_Number *) = {add_operations_b_scalar, sub_operations_b_scalar,
                                                                              mul_operations_b_scalar, div_operations_b_scalar,
                                                                              mod_operations_b_scalar, pow_operations_b_scalar,
                                                                              left_shift_operations_b_scalar, right_shift_operations_b_scalar,
                                                                              floor_div_operations_b_scalar};