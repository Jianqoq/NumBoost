#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "type_convertor.h"

/*to uint8*/
void Any_to_Float(PyArrayObject **array, PyArrayObject **result, int type)
{
    switch (type)
    {
        CAST_ARRAY_CASE(NPY_BOOL, npy_bool, npy_float, NPY_FLOAT)
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
        CAST_ARRAY_CASE(NPY_BOOL, npy_bool, npy_double, NPY_DOUBLE)
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
        F_CAST_ARRAY_CASE(NPY_BOOL, npy_bool, npy_half, NPY_HALF, bool_cast_half)
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
        CAST_ARRAY_CASE(NPY_BOOL, npy_bool, npy_longlong, NPY_LONGLONG)
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
        CAST_ARRAY_CASE(NPY_BOOL, npy_bool, npy_long, NPY_LONG)
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
        CAST_ARRAY_CASE(NPY_BOOL, npy_bool, npy_int, NPY_INT)
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
        F_CAST_ARRAY_CASE(NPY_HALF, npy_half, npy_int, NPY_INT, half_cast_int)
    }
}

void Any_to_Byte_(PyArrayObject **array, PyArrayObject **result, int type)
{
    switch (type)
    {
        CAST_ARRAY_CASE(NPY_BOOL, npy_bool, npy_byte, NPY_BYTE)
        NO_CONVERT_CASE(NPY_BYTE)
        CAST_ARRAY_CASE(NPY_UBYTE, npy_ubyte, npy_byte, NPY_BYTE)
        CAST_ARRAY_CASE(NPY_SHORT, npy_short, npy_byte, NPY_BYTE)
        CAST_ARRAY_CASE(NPY_USHORT, npy_ushort, npy_byte, NPY_BYTE)
        CAST_ARRAY_CASE(NPY_INT, npy_int, npy_byte, NPY_BYTE)
        CAST_ARRAY_CASE(NPY_UINT, npy_uint, npy_byte, NPY_BYTE)
        CAST_ARRAY_CASE(NPY_LONG, npy_long, npy_byte, NPY_BYTE)
        CAST_ARRAY_CASE(NPY_ULONG, npy_ulong, npy_byte, NPY_BYTE)
        CAST_ARRAY_CASE(NPY_LONGLONG, npy_longlong, npy_byte, NPY_BYTE)
        CAST_ARRAY_CASE(NPY_ULONGLONG, npy_ulonglong, npy_byte, NPY_BYTE)
        CAST_ARRAY_CASE(NPY_FLOAT, npy_float, npy_byte, NPY_BYTE)
        CAST_ARRAY_CASE(NPY_DOUBLE, npy_double, npy_byte, NPY_BYTE)
        CAST_ARRAY_CASE(NPY_LONGDOUBLE, npy_longdouble, npy_byte, NPY_BYTE)
        F_CAST_ARRAY_CASE(NPY_HALF, npy_half, npy_byte, NPY_BYTE, half_cast_byte)
    }
}

void Any_to_UByte(PyArrayObject **array, PyArrayObject **result, int type)
{
    switch (type)
    {
        CAST_ARRAY_CASE(NPY_BOOL, npy_bool, npy_ubyte, NPY_UBYTE)
        CAST_ARRAY_CASE(NPY_BYTE, npy_byte, npy_ubyte, NPY_UBYTE)
        NO_CONVERT_CASE(NPY_UBYTE)
        CAST_ARRAY_CASE(NPY_SHORT, npy_short, npy_ubyte, NPY_UBYTE)
        CAST_ARRAY_CASE(NPY_USHORT, npy_ushort, npy_ubyte, NPY_UBYTE)
        CAST_ARRAY_CASE(NPY_INT, npy_int, npy_ubyte, NPY_UBYTE)
        CAST_ARRAY_CASE(NPY_UINT, npy_uint, npy_ubyte, NPY_UBYTE)
        CAST_ARRAY_CASE(NPY_LONG, npy_long, npy_ubyte, NPY_UBYTE)
        CAST_ARRAY_CASE(NPY_ULONG, npy_ulong, npy_ubyte, NPY_UBYTE)
        CAST_ARRAY_CASE(NPY_LONGLONG, npy_longlong, npy_ubyte, NPY_UBYTE)
        CAST_ARRAY_CASE(NPY_ULONGLONG, npy_ulonglong, npy_ubyte, NPY_UBYTE)
        CAST_ARRAY_CASE(NPY_FLOAT, npy_float, npy_ubyte, NPY_UBYTE)
        CAST_ARRAY_CASE(NPY_DOUBLE, npy_double, npy_ubyte, NPY_UBYTE)
        CAST_ARRAY_CASE(NPY_LONGDOUBLE, npy_longdouble, npy_ubyte, NPY_UBYTE)
        F_CAST_ARRAY_CASE(NPY_HALF, npy_half, npy_ubyte, NPY_UBYTE, half_cast_ubyte)
    }
}

void Any_to_Uint(PyArrayObject **array, PyArrayObject **result, int type)
{
    switch (type)
    {
        CAST_ARRAY_CASE(NPY_BOOL, npy_bool, npy_uint, NPY_UINT)
        CAST_ARRAY_CASE(NPY_BYTE, npy_byte, npy_uint, NPY_UINT)
        CAST_ARRAY_CASE(NPY_UBYTE, npy_ubyte, npy_uint, NPY_UINT)
        CAST_ARRAY_CASE(NPY_SHORT, npy_short, npy_uint, NPY_UINT)
        CAST_ARRAY_CASE(NPY_USHORT, npy_ushort, npy_uint, NPY_UINT)
        CAST_ARRAY_CASE(NPY_INT, npy_int, npy_uint, NPY_UINT)
        NO_CONVERT_CASE(NPY_UINT)
        CAST_ARRAY_CASE(NPY_LONG, npy_long, npy_uint, NPY_UINT)
        CAST_ARRAY_CASE(NPY_ULONG, npy_ulong, npy_uint, NPY_UINT)
        CAST_ARRAY_CASE(NPY_LONGLONG, npy_longlong, npy_uint, NPY_UINT)
        CAST_ARRAY_CASE(NPY_ULONGLONG, npy_ulonglong, npy_uint, NPY_UINT)
        CAST_ARRAY_CASE(NPY_FLOAT, npy_float, npy_uint, NPY_UINT)
        CAST_ARRAY_CASE(NPY_DOUBLE, npy_double, npy_uint, NPY_UINT)
        CAST_ARRAY_CASE(NPY_LONGDOUBLE, npy_longdouble, npy_uint, NPY_UINT)
        F_CAST_ARRAY_CASE(NPY_HALF, npy_half, npy_uint, NPY_UINT, half_cast_uint)
    }
}

void Any_to_Ulong(PyArrayObject **array, PyArrayObject **result, int type)
{
    switch (type)
    {
        CAST_ARRAY_CASE(NPY_BOOL, npy_bool, npy_ulong, NPY_ULONG)
        CAST_ARRAY_CASE(NPY_BYTE, npy_byte, npy_ulong, NPY_ULONG)
        CAST_ARRAY_CASE(NPY_UBYTE, npy_ubyte, npy_ulong, NPY_ULONG)
        CAST_ARRAY_CASE(NPY_SHORT, npy_short, npy_ulong, NPY_ULONG)
        CAST_ARRAY_CASE(NPY_USHORT, npy_ushort, npy_ulong, NPY_ULONG)
        CAST_ARRAY_CASE(NPY_INT, npy_int, npy_ulong, NPY_ULONG)
        CAST_ARRAY_CASE(NPY_UINT, npy_uint, npy_ulong, NPY_ULONG)
        CAST_ARRAY_CASE(NPY_LONG, npy_long, npy_ulong, NPY_ULONG)
        NO_CONVERT_CASE(NPY_ULONG)
        CAST_ARRAY_CASE(NPY_LONGLONG, npy_longlong, npy_ulong, NPY_ULONG)
        CAST_ARRAY_CASE(NPY_ULONGLONG, npy_ulonglong, npy_ulong, NPY_ULONG)
        CAST_ARRAY_CASE(NPY_FLOAT, npy_float, npy_ulong, NPY_ULONG)
        CAST_ARRAY_CASE(NPY_DOUBLE, npy_double, npy_ulong, NPY_ULONG)
        CAST_ARRAY_CASE(NPY_LONGDOUBLE, npy_longdouble, npy_ulong, NPY_ULONG)
        F_CAST_ARRAY_CASE(NPY_HALF, npy_half, npy_ulong, NPY_ULONG, half_cast_ulong)
    }
}

void Any_to_ULongLong(PyArrayObject **array, PyArrayObject **result, int type)
{
    switch (type)
    {
        CAST_ARRAY_CASE(NPY_BOOL, npy_bool, npy_ulonglong, NPY_ULONGLONG)
        CAST_ARRAY_CASE(NPY_BYTE, npy_byte, npy_ulonglong, NPY_ULONGLONG)
        CAST_ARRAY_CASE(NPY_UBYTE, npy_ubyte, npy_ulonglong, NPY_ULONGLONG)
        CAST_ARRAY_CASE(NPY_SHORT, npy_short, npy_ulonglong, NPY_ULONGLONG)
        CAST_ARRAY_CASE(NPY_USHORT, npy_ushort, npy_ulonglong, NPY_ULONGLONG)
        CAST_ARRAY_CASE(NPY_INT, npy_int, npy_ulonglong, NPY_ULONGLONG)
        CAST_ARRAY_CASE(NPY_UINT, npy_uint, npy_ulonglong, NPY_ULONGLONG)
        CAST_ARRAY_CASE(NPY_LONG, npy_long, npy_ulonglong, NPY_ULONGLONG)
        CAST_ARRAY_CASE(NPY_ULONG, npy_ulong, npy_ulonglong, NPY_ULONGLONG)
        CAST_ARRAY_CASE(NPY_LONGLONG, npy_longlong, npy_ulonglong, NPY_ULONGLONG)
        NO_CONVERT_CASE(NPY_ULONGLONG)
        CAST_ARRAY_CASE(NPY_FLOAT, npy_float, npy_ulonglong, NPY_ULONGLONG)
        CAST_ARRAY_CASE(NPY_DOUBLE, npy_double, npy_ulonglong, NPY_ULONGLONG)
        CAST_ARRAY_CASE(NPY_LONGDOUBLE, npy_longdouble, npy_ulonglong, NPY_ULONGLONG)
        F_CAST_ARRAY_CASE(NPY_HALF, npy_half, npy_ulonglong, NPY_ULONGLONG, half_cast_ulonglong)
    }
}

void Any_to_Short(PyArrayObject **array, PyArrayObject **result, int type)
{
    switch (type)
    {
        CAST_ARRAY_CASE(NPY_BOOL, npy_bool, npy_short, NPY_SHORT)
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

void Any_to_UShort(PyArrayObject **array, PyArrayObject **result, int type)
{
    switch (type)
    {
        CAST_ARRAY_CASE(NPY_BOOL, npy_bool, npy_ushort, NPY_USHORT)
        CAST_ARRAY_CASE(NPY_BYTE, npy_byte, npy_ushort, NPY_USHORT)
        CAST_ARRAY_CASE(NPY_UBYTE, npy_ubyte, npy_ushort, NPY_USHORT)
        CAST_ARRAY_CASE(NPY_SHORT, npy_short, npy_ushort, NPY_USHORT)
        NO_CONVERT_CASE(NPY_USHORT)
        CAST_ARRAY_CASE(NPY_INT, npy_int, npy_ushort, NPY_USHORT)
        CAST_ARRAY_CASE(NPY_UINT, npy_uint, npy_ushort, NPY_USHORT)
        CAST_ARRAY_CASE(NPY_LONG, npy_long, npy_ushort, NPY_USHORT)
        CAST_ARRAY_CASE(NPY_ULONG, npy_ulong, npy_ushort, NPY_USHORT)
        CAST_ARRAY_CASE(NPY_LONGLONG, npy_longlong, npy_ushort, NPY_USHORT)
        CAST_ARRAY_CASE(NPY_ULONGLONG, npy_ulonglong, npy_ushort, NPY_USHORT)
        CAST_ARRAY_CASE(NPY_FLOAT, npy_float, npy_ushort, NPY_USHORT)
        CAST_ARRAY_CASE(NPY_DOUBLE, npy_double, npy_ushort, NPY_USHORT)
        CAST_ARRAY_CASE(NPY_LONGDOUBLE, npy_longdouble, npy_ushort, NPY_USHORT)
        F_CAST_ARRAY_CASE(NPY_HALF, npy_half, npy_ushort, NPY_USHORT, half_cast_ushort)
    }
}

void Any_to_Bool(PyArrayObject **array, PyArrayObject **result, int type)
{
    switch (type)
    {
        NO_CONVERT_CASE(NPY_BOOL)
        CAST_ARRAY_CASE_BOOL(NPY_BYTE, npy_byte)
        CAST_ARRAY_CASE_BOOL(NPY_UBYTE, npy_ubyte)
        CAST_ARRAY_CASE_BOOL(NPY_SHORT, npy_short)
        CAST_ARRAY_CASE_BOOL(NPY_USHORT, npy_ushort)
        CAST_ARRAY_CASE_BOOL(NPY_INT, npy_int)
        CAST_ARRAY_CASE_BOOL(NPY_UINT, npy_uint)
        CAST_ARRAY_CASE_BOOL(NPY_LONG, npy_long)
        CAST_ARRAY_CASE_BOOL(NPY_ULONG, npy_ulong)
        CAST_ARRAY_CASE_BOOL(NPY_LONGLONG, npy_longlong)
        CAST_ARRAY_CASE_BOOL(NPY_ULONGLONG, npy_ulonglong)
        CAST_ARRAY_CASE_BOOL(NPY_FLOAT, npy_float)
        CAST_ARRAY_CASE_BOOL(NPY_DOUBLE, npy_double)
        CAST_ARRAY_CASE_BOOL(NPY_LONGDOUBLE, npy_longdouble)
        F_CAST_ARRAY_CASE(NPY_HALF, npy_half, npy_bool, NPY_BOOL, half_cast_short)
    }
}

inline void As_Type(PyArrayObject **a, PyArrayObject **result, int self_type, int target_type)
{
    switch (target_type)
    {
        As_Type_Cases(Bool, NPY_BOOL);
        As_Type_Cases(Byte_, NPY_BYTE);
        As_Type_Cases(UByte, NPY_UBYTE);
        As_Type_Cases(Short, NPY_SHORT);
        As_Type_Cases(UShort, NPY_USHORT);
        As_Type_Cases(Int, NPY_INT);
        As_Type_Cases(Uint, NPY_UINT);
        As_Type_Cases(Long, NPY_LONG);
        As_Type_Cases(Ulong, NPY_ULONG); // haven't implemented yet
        As_Type_Cases(LongLong, NPY_LONGLONG);
        As_Type_Cases(ULongLong, NPY_ULONGLONG);
        As_Type_Cases(Float, NPY_FLOAT);
        As_Type_Cases(Double, NPY_DOUBLE);
        As_Type_Cases(Half, NPY_HALF);
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

int div_result_type_pick(int npy_enum)
{
    switch (npy_enum)
    {
    case NPY_BOOL:
    case NPY_BYTE:
    case NPY_UBYTE:
    case NPY_SHORT:
    case NPY_USHORT:
        return NPY_HALF;
    case NPY_INT:
    case NPY_UINT:
    case NPY_LONG:
    case NPY_ULONG:
        return NPY_FLOAT;
    case NPY_LONGLONG:
    case NPY_ULONGLONG:
        return NPY_DOUBLE;
    }
}