#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "broadcast_func_def.h"

Register_Broadcast_Operation(bool, add_, nb_add, Standard_Inner_Loop);
Register_Broadcast_Operation(bool, sub_, nb_subtract, Standard_Inner_Loop);
Register_Broadcast_Operation(bool, mul_, nb_multiply, Standard_Inner_Loop);
Register_Broadcast_Operation(bool, div_, nb_divide, Standard_Inner_Loop);
Register_Broadcast_Operation(bool, mod_, nb_mod_int, Standard_Inner_Loop);
Register_Broadcast_Operation(bool, lshift_, nb_lshift, Shift_Inner_Loop);
Register_Broadcast_Operation(bool, rshift_, nb_rshift, Shift_Inner_Loop);
Register_Broadcast_Err(bool, pow_);

Register_Broadcast_Operation(byte, add_, nb_add, Standard_Inner_Loop);
Register_Broadcast_Operation(byte, sub_, nb_subtract, Standard_Inner_Loop);
Register_Broadcast_Operation(byte, mul_, nb_multiply, Standard_Inner_Loop);
Register_Broadcast_Operation(byte, div_, nb_divide, Standard_Inner_Loop);
Register_Broadcast_Operation(byte, mod_, nb_mod_int, Standard_Inner_Loop);
Register_Broadcast_Operation(byte, lshift_, nb_lshift, Shift_Inner_Loop);
Register_Broadcast_Operation(byte, rshift_, nb_rshift, Shift_Inner_Loop);
Register_Broadcast_Err(byte, pow_);

Register_Broadcast_Operation(ubyte, add_, nb_add, Standard_Inner_Loop);
Register_Broadcast_Operation(ubyte, sub_, nb_subtract, Standard_Inner_Loop);
Register_Broadcast_Operation(ubyte, mul_, nb_multiply, Standard_Inner_Loop);
Register_Broadcast_Operation(ubyte, div_, nb_divide, Standard_Inner_Loop);
Register_Broadcast_Operation(ubyte, mod_, nb_mod_int, Standard_Inner_Loop);
Register_Broadcast_Operation(ubyte, lshift_, nb_lshift, Shift_Inner_Loop);
Register_Broadcast_Operation(ubyte, rshift_, nb_rshift, Shift_Inner_Loop);
Register_Broadcast_Err(ubyte, pow_);

Register_Broadcast_Operation(short, add_, nb_add, Standard_Inner_Loop);
Register_Broadcast_Operation(short, sub_, nb_subtract, Standard_Inner_Loop);
Register_Broadcast_Operation(short, mul_, nb_multiply, Standard_Inner_Loop);
Register_Broadcast_Operation(short, div_, nb_divide, Standard_Inner_Loop);
Register_Broadcast_Operation(short, mod_, nb_mod_int, Standard_Inner_Loop);
Register_Broadcast_Operation(short, lshift_, nb_lshift, Shift_Inner_Loop);
Register_Broadcast_Operation(short, rshift_, nb_rshift, Shift_Inner_Loop);
Register_Broadcast_Err(short, pow_);

Register_Broadcast_Operation(ushort, add_, nb_add, Standard_Inner_Loop);
Register_Broadcast_Operation(ushort, sub_, nb_subtract, Standard_Inner_Loop);
Register_Broadcast_Operation(ushort, mul_, nb_multiply, Standard_Inner_Loop);
Register_Broadcast_Operation(ushort, div_, nb_divide, Standard_Inner_Loop);
Register_Broadcast_Operation(ushort, mod_, nb_mod_int, Standard_Inner_Loop);
Register_Broadcast_Operation(ushort, lshift_, nb_lshift, Shift_Inner_Loop);
Register_Broadcast_Operation(ushort, rshift_, nb_rshift, Shift_Inner_Loop);
Register_Broadcast_Err(ushort, pow_);

Register_Broadcast_Operation(int, add_, nb_add, Standard_Inner_Loop);
Register_Broadcast_Operation(int, sub_, nb_subtract, Standard_Inner_Loop);
Register_Broadcast_Operation(int, mul_, nb_multiply, Standard_Inner_Loop);
Register_Broadcast_Operation(int, div_, nb_divide, Standard_Inner_Loop);
Register_Broadcast_Operation(int, mod_, nb_mod_int, Standard_Inner_Loop);
Register_Broadcast_Operation(int, lshift_, nb_lshift, Shift_Inner_Loop);
Register_Broadcast_Operation(int, rshift_, nb_rshift, Shift_Inner_Loop);
Register_Broadcast_Err(int, pow_);

Register_Broadcast_Operation(uint, add_, nb_add, Standard_Inner_Loop);
Register_Broadcast_Operation(uint, sub_, nb_subtract, Standard_Inner_Loop);
Register_Broadcast_Operation(uint, mul_, nb_multiply, Standard_Inner_Loop);
Register_Broadcast_Operation(uint, div_, nb_divide, Standard_Inner_Loop);
Register_Broadcast_Operation(uint, mod_, nb_mod_int, Standard_Inner_Loop);
Register_Broadcast_Operation(uint, lshift_, nb_lshift, Shift_Inner_Loop);
Register_Broadcast_Operation(uint, rshift_, nb_rshift, Shift_Inner_Loop);
Register_Broadcast_Err(uint, pow_);

Register_Broadcast_Operation(long, add_, nb_add, Standard_Inner_Loop);
Register_Broadcast_Operation(long, sub_, nb_subtract, Standard_Inner_Loop);
Register_Broadcast_Operation(long, mul_, nb_multiply, Standard_Inner_Loop);
Register_Broadcast_Operation(long, div_, nb_divide, Standard_Inner_Loop);
Register_Broadcast_Operation(long, mod_, nb_mod_int, Standard_Inner_Loop);
Register_Broadcast_Operation(long, lshift_, nb_lshift, Shift_Inner_Loop);
Register_Broadcast_Operation(long, rshift_, nb_rshift, Shift_Inner_Loop);
Register_Broadcast_Err(long, pow_);

Register_Broadcast_Operation(ulong, add_, nb_add, Standard_Inner_Loop);
Register_Broadcast_Operation(ulong, sub_, nb_subtract, Standard_Inner_Loop);
Register_Broadcast_Operation(ulong, mul_, nb_multiply, Standard_Inner_Loop);
Register_Broadcast_Operation(ulong, div_, nb_divide, Standard_Inner_Loop);
Register_Broadcast_Operation(ulong, mod_, nb_mod_int, Standard_Inner_Loop);
Register_Broadcast_Operation(ulong, lshift_, nb_lshift, Shift_Inner_Loop);
Register_Broadcast_Operation(ulong, rshift_, nb_rshift, Shift_Inner_Loop);
Register_Broadcast_Err(ulong, pow_);

Register_Broadcast_Operation(longlong, add_, nb_add, Standard_Inner_Loop);
Register_Broadcast_Operation(longlong, sub_, nb_subtract, Standard_Inner_Loop);
Register_Broadcast_Operation(longlong, mul_, nb_multiply, Standard_Inner_Loop);
Register_Broadcast_Operation(longlong, div_, nb_divide, Standard_Inner_Loop);
Register_Broadcast_Operation(longlong, mod_, nb_mod_int, Standard_Inner_Loop);
Register_Broadcast_Operation(longlong, lshift_, nb_lshift, Shift_Inner_Loop);
Register_Broadcast_Operation(longlong, rshift_, nb_rshift, Shift_Inner_Loop);
Register_Broadcast_Err(longlong, pow_);

Register_Broadcast_Operation(ulonglong, add_, nb_add, Standard_Inner_Loop);
Register_Broadcast_Operation(ulonglong, sub_, nb_subtract, Standard_Inner_Loop);
Register_Broadcast_Operation(ulonglong, mul_, nb_multiply, Standard_Inner_Loop);
Register_Broadcast_Operation(ulonglong, div_, nb_divide, Standard_Inner_Loop);
Register_Broadcast_Operation(ulonglong, mod_, nb_mod_int, Standard_Inner_Loop);
Register_Broadcast_Operation(ulonglong, lshift_, nb_lshift, Shift_Inner_Loop);
Register_Broadcast_Operation(ulonglong, rshift_, nb_rshift, Shift_Inner_Loop);
Register_Broadcast_Err(ulonglong, pow_);

Register_Broadcast_Operation(float, add_, nb_add, Standard_Inner_Loop);
Register_Broadcast_Operation(float, sub_, nb_subtract, Standard_Inner_Loop);
Register_Broadcast_Operation(float, mul_, nb_multiply, Standard_Inner_Loop);
Register_Broadcast_Operation(float, div_, nb_divide, Standard_Inner_Loop);
Register_Broadcast_Err(float, mod_);
Register_Broadcast_Err(float, lshift_);
Register_Broadcast_Err(float, rshift_);
Register_Broadcast_Operation(float, pow_, nb_power_float, Powf_Inner_Loop);

Register_Broadcast_Operation(double, add_, nb_add, Standard_Inner_Loop);
Register_Broadcast_Operation(double, sub_, nb_subtract, Standard_Inner_Loop);
Register_Broadcast_Operation(double, mul_, nb_multiply, Standard_Inner_Loop);
Register_Broadcast_Operation(double, div_, nb_divide, Standard_Inner_Loop);
Register_Broadcast_Err(double, mod_);
Register_Broadcast_Err(double, lshift_);
Register_Broadcast_Err(double, rshift_);
Register_Broadcast_Operation(double, pow_, nb_power_double, Pow_Inner_Loop);

Register_Broadcast_Operation(longdouble, add_, nb_add, Standard_Inner_Loop);
Register_Broadcast_Operation(longdouble, sub_, nb_subtract, Standard_Inner_Loop);
Register_Broadcast_Operation(longdouble, mul_, nb_multiply, Standard_Inner_Loop);
Register_Broadcast_Operation(longdouble, div_, nb_divide, Standard_Inner_Loop);
Register_Broadcast_Err(longdouble, mod_);
Register_Broadcast_Err(longdouble, lshift_);
Register_Broadcast_Err(longdouble, rshift_);
Register_Broadcast_Operation(longdouble, pow_, nb_power_long_double, Powl_Inner_Loop);

Register_All_Err(cfloat);
Register_All_Err(cdouble);
Register_All_Err(clongdouble);
Register_All_Err(object);
Register_All_Err(string);
Register_All_Err(unicode);
Register_All_Err(void);
Register_All_Err(datetime);
Register_All_Err(timedelta);

Register_Broadcast_Operation(half, add_, nb_add_half, Half_Inner_Loop);
Register_Broadcast_Operation(half, sub_, nb_subtract_half, Half_Inner_Loop);
Register_Broadcast_Operation(half, mul_, nb_multiply_half, Half_Inner_Loop);
Register_Broadcast_Operation(half, div_, nb_divide_half, Half_Inner_Loop);
Register_Broadcast_Operation(half, mod_, nb_mod_half, Half_Inner_Loop);
Register_Broadcast_Err(half, lshift_);
Register_Broadcast_Err(half, rshift_);
Register_Broadcast_Operation(half, pow_, nb_power_half, Half_Inner_Loop);

Register_Broadcast_Operation_Array(add_);
Register_Broadcast_Operation_Array(sub_);
Register_Broadcast_Operation_Array(mul_);
Register_Broadcast_Operation_Array(div_);
Register_Broadcast_Operation_Array(mod_);
Register_Broadcast_Operation_Array(pow_);
Register_Broadcast_Operation_Array(lshift_);
Register_Broadcast_Operation_Array(rshift_);

PyArrayObject *(**broadcast_operations[])(PyArrayObject *, PyArrayObject *, int, int) = {broadcast_add_, broadcast_sub_, broadcast_mul_, broadcast_div_,
                                                                                         broadcast_mod_, broadcast_pow_, broadcast_lshift_, broadcast_rshift_};