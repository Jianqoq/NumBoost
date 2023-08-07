#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "type_convertor.h"

/*to uint8*/
void Any_to_Float(PyArrayObject **array, PyArrayObject **result, int type)
{
    switch (type)
    {
    NO_CONVERT_CASE(NPY_BOOL)
    CAST_ARRAY_CASE(NPY_BYTE, npy_byte, npy_float, NPY_FLOAT)
    CAST_ARRAY_CASE(NPY_UBYTE, npy_ubyte, npy_float, NPY_FLOAT)
    CAST_ARRAY_CASE(NPY_SHORT, npy_short, npy_float, NPY_FLOAT)
    CAST_ARRAY_CASE(NPY_USHORT, npy_ushort, npy_float, NPY_FLOAT)
    CAST_ARRAY_CASE(NPY_INT, npy_int, npy_float, NPY_FLOAT)
    CAST_ARRAY_CASE(NPY_UINT, npy_uint, npy_float, NPY_FLOAT)
    CAST_ARRAY_CASE(NPY_LONG, npy_long, npy_float, NPY_FLOAT)
    CAST_ARRAY_CASE(NPY_ULONG, npy_ulong, npy_float, NPY_FLOAT)
    CAST_ARRAY_CASE(NPY_LONGLONG, npy_longlong, npy_float, NPY_FLOAT)
    CAST_ARRAY_CASE(NPY_ULONGLONG, npy_ulonglong, npy_float, NPY_FLOAT)
    NO_CONVERT_CASE(NPY_FLOAT)
    CAST_ARRAY_CASE(NPY_DOUBLE, npy_double, npy_float, NPY_FLOAT)
    CAST_ARRAY_CASE(NPY_LONGDOUBLE, npy_longdouble, npy_float, NPY_FLOAT)
    F_CAST_ARRAY_CASE(NPY_HALF, npy_half, npy_float, NPY_FLOAT, half_cast_float)
    }
}

void Any_to_Double(PyArrayObject **array, PyArrayObject **result, int type)
{
    switch (type)
    {
    NO_CONVERT_CASE(NPY_BOOL)
    CAST_ARRAY_CASE(NPY_BYTE, npy_byte, npy_double, NPY_DOUBLE)
    CAST_ARRAY_CASE(NPY_UBYTE, npy_ubyte, npy_double, NPY_DOUBLE)
    CAST_ARRAY_CASE(NPY_SHORT, npy_short, npy_double, NPY_DOUBLE)
    CAST_ARRAY_CASE(NPY_USHORT, npy_ushort, npy_double, NPY_DOUBLE)
    CAST_ARRAY_CASE(NPY_INT, npy_int, npy_double, NPY_DOUBLE)
    CAST_ARRAY_CASE(NPY_UINT, npy_uint, npy_double, NPY_DOUBLE)
    CAST_ARRAY_CASE(NPY_LONG, npy_long, npy_double, NPY_DOUBLE)
    CAST_ARRAY_CASE(NPY_ULONG, npy_ulong, npy_double, NPY_DOUBLE)
    CAST_ARRAY_CASE(NPY_LONGLONG, npy_longlong, npy_double, NPY_DOUBLE)
    CAST_ARRAY_CASE(NPY_ULONGLONG, npy_ulonglong, npy_double, NPY_DOUBLE)
    CAST_ARRAY_CASE(NPY_FLOAT, npy_float, npy_double, NPY_DOUBLE)
    NO_CONVERT_CASE(NPY_DOUBLE)
    CAST_ARRAY_CASE(NPY_LONGDOUBLE, npy_longdouble, npy_double, NPY_DOUBLE)
    F_CAST_ARRAY_CASE(NPY_HALF, npy_half, npy_double, NPY_DOUBLE, half_cast_double)
    }
}

void Any_to_Half(PyArrayObject **array, PyArrayObject **result, int type)
{
    switch (type)
    {
    F_CAST_ARRAY_CASE(NPY_BOOL, npy_bool, npy_half, NPY_HALF, byte_cast_half)
    F_CAST_ARRAY_CASE(NPY_BYTE, npy_byte, npy_half, NPY_HALF, byte_cast_half)
    F_CAST_ARRAY_CASE(NPY_UBYTE, npy_ubyte, npy_half, NPY_HALF, ubyte_cast_half)
    F_CAST_ARRAY_CASE(NPY_SHORT, npy_short, npy_half, NPY_HALF, short_cast_half)
    F_CAST_ARRAY_CASE(NPY_USHORT, npy_ushort, npy_half, NPY_HALF, ushort_cast_half)
    F_CAST_ARRAY_CASE(NPY_INT, npy_int, npy_half, NPY_HALF, int_cast_half)
    F_CAST_ARRAY_CASE(NPY_UINT, npy_uint, npy_half, NPY_HALF, uint_cast_half)
    F_CAST_ARRAY_CASE(NPY_LONG, npy_long, npy_half, NPY_HALF, long_cast_half)
    F_CAST_ARRAY_CASE(NPY_ULONG, npy_ulong, npy_half, NPY_HALF, ulong_cast_half)
    F_CAST_ARRAY_CASE(NPY_LONGLONG, npy_longlong, npy_half, NPY_HALF, longlong_cast_half)
    F_CAST_ARRAY_CASE(NPY_ULONGLONG, npy_ulonglong, npy_half, NPY_HALF, ulonglong_cast_half)
    F_CAST_ARRAY_CASE(NPY_FLOAT, npy_float, npy_half, NPY_HALF, float_cast_half)
    F_CAST_ARRAY_CASE(NPY_DOUBLE, npy_double, npy_half, NPY_HALF, double_cast_half)
    F_CAST_ARRAY_CASE(NPY_LONGDOUBLE, npy_longdouble, npy_half, NPY_HALF, double_cast_half)
    NO_CONVERT_CASE(NPY_HALF)
    }
}

void Any_to_LongLong(PyArrayObject **array, PyArrayObject **result, int type)
{
    switch (type)
    {
    NO_CONVERT_CASE(NPY_BOOL)
    CAST_ARRAY_CASE(NPY_BYTE, npy_byte, npy_longlong, NPY_LONGLONG)
    CAST_ARRAY_CASE(NPY_UBYTE, npy_ubyte, npy_longlong, NPY_LONGLONG)
    CAST_ARRAY_CASE(NPY_SHORT, npy_short, npy_longlong, NPY_LONGLONG)
    CAST_ARRAY_CASE(NPY_USHORT, npy_ushort, npy_longlong, NPY_LONGLONG)
    CAST_ARRAY_CASE(NPY_INT, npy_int, npy_longlong, NPY_LONGLONG)
    CAST_ARRAY_CASE(NPY_UINT, npy_uint, npy_longlong, NPY_LONGLONG)
    CAST_ARRAY_CASE(NPY_LONG, npy_long, npy_longlong, NPY_LONGLONG)
    CAST_ARRAY_CASE(NPY_ULONG, npy_ulong, npy_longlong, NPY_LONGLONG)
    CAST_ARRAY_CASE(NPY_LONGLONG, npy_longlong, npy_longlong, NPY_LONGLONG)
    CAST_ARRAY_CASE(NPY_ULONGLONG, npy_ulonglong, npy_longlong, NPY_LONGLONG)
    CAST_ARRAY_CASE(NPY_FLOAT, npy_float, npy_longlong, NPY_LONGLONG)
    CAST_ARRAY_CASE(NPY_DOUBLE, npy_double, npy_longlong, NPY_LONGLONG)
    CAST_ARRAY_CASE(NPY_LONGDOUBLE, npy_longdouble, npy_longlong, NPY_LONGLONG)
    F_CAST_ARRAY_CASE(NPY_HALF, npy_half, npy_longlong, NPY_LONGLONG, half_cast_longlong)
    }
}

void Any_to_Long(PyArrayObject **array, PyArrayObject **result, int type)
{
    switch (type)
    {
    case NPY_BOOL:
        break;
    CAST_ARRAY_CASE(NPY_BYTE, npy_byte, npy_long, NPY_LONG)
    CAST_ARRAY_CASE(NPY_UBYTE, npy_ubyte, npy_long, NPY_LONG)
    CAST_ARRAY_CASE(NPY_SHORT, npy_short, npy_long, NPY_LONG)
    CAST_ARRAY_CASE(NPY_USHORT, npy_ushort, npy_long, NPY_LONG)
    CAST_ARRAY_CASE(NPY_INT, npy_int, npy_long, NPY_LONG)
    CAST_ARRAY_CASE(NPY_UINT, npy_uint, npy_long, NPY_LONG)
    NO_CONVERT_CASE(NPY_LONG)
    CAST_ARRAY_CASE(NPY_ULONG, npy_ulong, npy_long, NPY_LONG)
    CAST_ARRAY_CASE(NPY_LONGLONG, npy_longlong, npy_long, NPY_LONG)
    CAST_ARRAY_CASE(NPY_ULONGLONG, npy_ulonglong, npy_long, NPY_LONG)
    CAST_ARRAY_CASE(NPY_FLOAT, npy_float, npy_long, NPY_LONG)
    CAST_ARRAY_CASE(NPY_DOUBLE, npy_double, npy_long, NPY_LONG)
    CAST_ARRAY_CASE(NPY_LONGDOUBLE, npy_longdouble, npy_long, NPY_LONG)
    F_CAST_ARRAY_CASE(NPY_HALF, npy_half, npy_long, NPY_LONG, half_cast_long)
    }
}

void Any_to_Int(PyArrayObject **array, PyArrayObject **result, int type)
{
    switch (type)
    {
    NO_CONVERT_CASE(NPY_BOOL)
    CAST_ARRAY_CASE(NPY_BYTE, npy_byte, npy_int, NPY_INT)
    CAST_ARRAY_CASE(NPY_UBYTE, npy_ubyte, npy_int, NPY_INT)
    CAST_ARRAY_CASE(NPY_SHORT, npy_short, npy_int, NPY_INT)
    CAST_ARRAY_CASE(NPY_USHORT, npy_ushort, npy_int, NPY_INT)
    NO_CONVERT_CASE(NPY_INT)
    CAST_ARRAY_CASE(NPY_UINT, npy_uint, npy_int, NPY_INT)
    CAST_ARRAY_CASE(NPY_LONG, npy_long, npy_int, NPY_INT)
    CAST_ARRAY_CASE(NPY_ULONG, npy_ulong, npy_int, NPY_INT)
    CAST_ARRAY_CASE(NPY_LONGLONG, npy_longlong, npy_int, NPY_INT)
    CAST_ARRAY_CASE(NPY_ULONGLONG, npy_ulonglong, npy_int, NPY_INT)
    CAST_ARRAY_CASE(NPY_FLOAT, npy_float, npy_int, NPY_INT)
    CAST_ARRAY_CASE(NPY_DOUBLE, npy_double, npy_int, NPY_INT)
    CAST_ARRAY_CASE(NPY_LONGDOUBLE, npy_longdouble, npy_int, NPY_INT)
    F_CAST_ARRAY_CASE(NPY_HALF, npy_half, npy_int, NPY_INT, half_cast_short)
    }
}

void Any_to_Short(PyArrayObject **array, PyArrayObject **result, int type)
{
    switch (type)
    {
    NO_CONVERT_CASE(NPY_BOOL)
    CAST_ARRAY_CASE(NPY_BYTE, npy_byte, npy_short, NPY_SHORT)
    CAST_ARRAY_CASE(NPY_UBYTE, npy_ubyte, npy_short, NPY_SHORT)
    NO_CONVERT_CASE(NPY_SHORT)
    CAST_ARRAY_CASE(NPY_USHORT, npy_ushort, npy_short, NPY_SHORT)
    CAST_ARRAY_CASE(NPY_INT, npy_int, npy_short, NPY_SHORT)
    CAST_ARRAY_CASE(NPY_UINT, npy_uint, npy_short, NPY_SHORT)
    CAST_ARRAY_CASE(NPY_LONG, npy_long, npy_short, NPY_SHORT)
    CAST_ARRAY_CASE(NPY_ULONG, npy_ulong, npy_short, NPY_SHORT)
    CAST_ARRAY_CASE(NPY_LONGLONG, npy_longlong, npy_short, NPY_SHORT)
    CAST_ARRAY_CASE(NPY_ULONGLONG, npy_ulonglong, npy_short, NPY_SHORT)
    CAST_ARRAY_CASE(NPY_FLOAT, npy_float, npy_short, NPY_SHORT)
    CAST_ARRAY_CASE(NPY_DOUBLE, npy_double, npy_short, NPY_SHORT)
    CAST_ARRAY_CASE(NPY_LONGDOUBLE, npy_longdouble, npy_short, NPY_SHORT)
    F_CAST_ARRAY_CASE(NPY_HALF, npy_half, npy_short, NPY_SHORT, half_cast_short)
    }
}

void Any_to_Bool(PyArrayObject **array, PyArrayObject **result, int type)
{
    switch (type)
    {
        NO_CONVERT_CASE(NPY_BOOL)
        CAST_ARRAY_CASE(NPY_BYTE, npy_byte, npy_bool, NPY_BOOL)
        CAST_ARRAY_CASE(NPY_UBYTE, npy_ubyte, npy_bool, NPY_BOOL)
        CAST_ARRAY_CASE(NPY_SHORT, npy_short, npy_bool, NPY_BOOL)
        CAST_ARRAY_CASE(NPY_USHORT, npy_ushort, npy_bool, NPY_BOOL)
        CAST_ARRAY_CASE(NPY_INT, npy_int, npy_bool, NPY_BOOL)
        CAST_ARRAY_CASE(NPY_UINT, npy_uint, npy_bool, NPY_BOOL)
        CAST_ARRAY_CASE(NPY_LONG, npy_long, npy_bool, NPY_BOOL)
        CAST_ARRAY_CASE(NPY_ULONG, npy_ulong, npy_bool, NPY_BOOL)
        CAST_ARRAY_CASE(NPY_LONGLONG, npy_longlong, npy_bool, NPY_BOOL)
        CAST_ARRAY_CASE(NPY_ULONGLONG, npy_ulonglong, npy_bool, NPY_BOOL)
        CAST_ARRAY_CASE(NPY_FLOAT, npy_float, npy_bool, NPY_BOOL)
        CAST_ARRAY_CASE(NPY_DOUBLE, npy_double, npy_bool, NPY_BOOL)
        CAST_ARRAY_CASE(NPY_LONGDOUBLE, npy_longdouble, npy_bool, NPY_BOOL)
        F_CAST_ARRAY_CASE(NPY_HALF, npy_half, npy_bool, NPY_BOOL, half_cast_short)
    }
}

inline void As_Type(PyArrayObject **a, PyArrayObject **result, int self_type, int target_type)
{
    switch (target_type)
    {
    As_Type_Cases(Bool, NPY_BOOL)
    As_Type_Cases(Bool, NPY_BYTE)
    As_Type_Cases(Bool, NPY_UBYTE)
    As_Type_Cases(Short, NPY_SHORT)
    As_Type_Cases(Bool, NPY_USHORT)
    As_Type_Cases(Int, NPY_INT)
    As_Type_Cases(Bool, NPY_UINT)
    As_Type_Cases(Long, NPY_LONG)
    As_Type_Cases(Long, NPY_ULONG)
    As_Type_Cases(LongLong, NPY_LONGLONG)
    As_Type_Cases(LongLong, NPY_ULONGLONG)
    As_Type_Cases(Float, NPY_FLOAT)
    As_Type_Cases(Double, NPY_DOUBLE)
    As_Type_Cases(Half, NPY_HALF)
    default:
        *a = NULL;
        break;
    }
}

void as_type(PyArrayObject **a, PyArrayObject **result, int target_type)
{
    int a_dtype = PyArray_TYPE(*a);
    As_Type(a, result, a_dtype, target_type);
}