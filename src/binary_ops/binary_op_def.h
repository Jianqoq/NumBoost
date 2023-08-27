#ifndef BINARY_FUNC_DEF_H
#define BINARY_FUNC_DEF_H
#include "numpy/arrayobject.h"
#include "binary_op_impl.h"
#define Register_Binary_Operation_Array(op, sufix, a_type, b_type)                                                                \
    PyArrayObject* (*op##_operations##sufix[])(a_type *, b_type *) = {                                                         \
        Binary_##op##_bool##sufix, Binary_##op##_byte##sufix, Binary_##op##_ubyte##sufix, Binary_##op##_short##sufix,             \
        Binary_##op##_ushort##sufix, Binary_##op##_int##sufix, Binary_##op##_uint##sufix, Binary_##op##_long##sufix,              \
        Binary_##op##_ulong##sufix, Binary_##op##_longlong##sufix, Binary_##op##_ulonglong##sufix, Binary_##op##_float##sufix,    \
        Binary_##op##_double##sufix, Binary_##op##_longdouble##sufix, Binary_##op##_cfloat##sufix, Binary_##op##_cdouble##sufix,  \
        Binary_##op##_clongdouble##sufix, Binary_##op##_object##sufix, Binary_##op##_string##sufix, Binary_##op##_unicode##sufix, \
        Binary_##op##_void##sufix, Binary_##op##_datetime##sufix, Binary_##op##_timedelta##sufix, Binary_##op##_half##sufix};

extern PyArrayObject* (**operations[])(PyArrayObject *, PyArrayObject *);

extern PyArrayObject* (**operations_a_scalar[])(Python_Number *, PyArrayObject *);

extern PyArrayObject* (**operations_b_scalar[])(PyArrayObject *, Python_Number *);

#endif