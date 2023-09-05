#ifndef BINARY_FUNC_DEF_H
#define BINARY_FUNC_DEF_H
#include "binary_op_impl.h"
#include "numpy/arrayobject.h"

#define NotImplement_Err(name, sufix, a_type, b_type)                          \
  Register_Binary_Operation_Err(name, cfloat, sufix, a_type, b_type);          \
  Register_Binary_Operation_Err(name, cdouble, sufix, a_type, b_type);         \
  Register_Binary_Operation_Err(name, clongdouble, sufix, a_type, b_type);     \
  Register_Binary_Operation_Err(name, object, sufix, a_type, b_type);          \
  Register_Binary_Operation_Err(name, string, sufix, a_type, b_type);          \
  Register_Binary_Operation_Err(name, unicode, sufix, a_type, b_type);         \
  Register_Binary_Operation_Err(name, void, sufix, a_type, b_type);            \
  Register_Binary_Operation_Err(name, datetime, sufix, a_type, b_type);        \
  Register_Binary_Operation_Err(name, timedelta, sufix, a_type, b_type);

#define Register_Int_Binary_OperationsErr(name, sufix, a_type, b_type)         \
  Register_Binary_Operation_Err(name, bool, sufix, a_type, b_type);            \
  Register_Binary_Operation_Err(name, byte, sufix, a_type, b_type);            \
  Register_Binary_Operation_Err(name, ubyte, sufix, a_type, b_type);           \
  Register_Binary_Operation_Err(name, short, sufix, a_type, b_type);           \
  Register_Binary_Operation_Err(name, ushort, sufix, a_type, b_type);          \
  Register_Binary_Operation_Err(name, int, sufix, a_type, b_type);             \
  Register_Binary_Operation_Err(name, uint, sufix, a_type, b_type);            \
  Register_Binary_Operation_Err(name, long, sufix, a_type, b_type);            \
  Register_Binary_Operation_Err(name, ulong, sufix, a_type, b_type);           \
  Register_Binary_Operation_Err(name, longlong, sufix, a_type, b_type);        \
  Register_Binary_Operation_Err(name, ulonglong, sufix, a_type, b_type);

#define Register_Binary_Operation_Array(op, sufix, a_type, b_type)             \
  PyArrayObject *(*op##_operations##sufix[])(a_type *, b_type *) = {           \
      Binary_##op##_bool##sufix,        Binary_##op##_byte##sufix,             \
      Binary_##op##_ubyte##sufix,       Binary_##op##_short##sufix,            \
      Binary_##op##_ushort##sufix,      Binary_##op##_int##sufix,              \
      Binary_##op##_uint##sufix,        Binary_##op##_long##sufix,             \
      Binary_##op##_ulong##sufix,       Binary_##op##_longlong##sufix,         \
      Binary_##op##_ulonglong##sufix,   Binary_##op##_float##sufix,            \
      Binary_##op##_double##sufix,      Binary_##op##_longdouble##sufix,       \
      Binary_##op##_cfloat##sufix,      Binary_##op##_cdouble##sufix,          \
      Binary_##op##_clongdouble##sufix, Binary_##op##_object##sufix,           \
      Binary_##op##_string##sufix,      Binary_##op##_unicode##sufix,          \
      Binary_##op##_void##sufix,        Binary_##op##_datetime##sufix,         \
      Binary_##op##_timedelta##sufix,   Binary_##op##_half##sufix};

#define Register_Binary_Operation_Array_New(name, sufix)                       \
  PyObject *(*name##_operations##sufix##_new[])(PyObject *, PyObject *) = {    \
      binary_##name##_bool##sufix,        binary_##name##_byte##sufix,         \
      binary_##name##_ubyte##sufix,       binary_##name##_short##sufix,        \
      binary_##name##_ushort##sufix,      binary_##name##_int##sufix,          \
      binary_##name##_uint##sufix,        binary_##name##_long##sufix,         \
      binary_##name##_ulong##sufix,       binary_##name##_longlong##sufix,     \
      binary_##name##_ulonglong##sufix,   binary_##name##_float##sufix,        \
      binary_##name##_double##sufix,      binary_##name##_longdouble##sufix,   \
      binary_##name##_cfloat##sufix,      binary_##name##_cdouble##sufix,      \
      binary_##name##_clongdouble##sufix, binary_##name##_object##sufix,       \
      binary_##name##_string##sufix,      binary_##name##_unicode##sufix,      \
      binary_##name##_void##sufix,        binary_##name##_datetime##sufix,     \
      binary_##name##_timedelta##sufix,   binary_##name##_half##sufix};

#define Register_Binary_Operations_Floating_Types(name, loop_body)             \
  Register_Binary_Operation_New(name, float, NPY_FLOAT, loop_body);            \
  Register_Binary_Operation_New(name, double, NPY_DOUBLE, loop_body);          \
  Register_Binary_Operation_New(name, longdouble, NPY_LONGDOUBLE, loop_body);  \
  Register_Binary_Operation_New(name, half, NPY_HALF, loop_body);

#define Register_Binary_Operations_Interger_Types(name, op_enum, loop_body,    \
                                                  sequential_loop_body)        \
  Register_Binary_Operation_New(name, bool, NPY_BOOL, loop_body);              \
  Register_Binary_Operation_New(name, byte, NPY_BYTE, loop_body);              \
  Register_Binary_Operation_New(name, ubyte, NPY_UBYTE, loop_body);            \
  Register_Binary_Operation_New(name, short, NPY_SHORT, loop_body);            \
  Register_Binary_Operation_New(name, ushort, NPY_USHORT, loop_body);          \
  Register_Binary_Operation_New(name, int, NPY_INT, loop_body);                \
  Register_Binary_Operation_New(name, uint, NPY_UINT, loop_body);              \
  Register_Binary_Operation_New(name, long, NPY_LONG, loop_body);              \
  Register_Binary_Operation_New(name, ulong, NPY_ULONG, loop_body);            \
  Register_Binary_Operation_New(name, longlong, NPY_LONGLONG, loop_body);      \
  Register_Binary_Operation_New(name, ulonglong, NPY_ULONGLONG, loop_body);

#define Register_Binary_Operation_Err_Interger_Types(name)                     \
  Register_Binary_Operation_New_Err(name, bool);                               \
  Register_Binary_Operation_New_Err(name, byte);                               \
  Register_Binary_Operation_New_Err(name, ubyte);                              \
  Register_Binary_Operation_New_Err(name, short);                              \
  Register_Binary_Operation_New_Err(name, ushort);                             \
  Register_Binary_Operation_New_Err(name, int);                                \
  Register_Binary_Operation_New_Err(name, uint);                               \
  Register_Binary_Operation_New_Err(name, long);                               \
  Register_Binary_Operation_New_Err(name, ulong);                              \
  Register_Binary_Operation_New_Err(name, longlong);                           \
  Register_Binary_Operation_New_Err(name, ulonglong);

#define Register_Binary_Operation_Err_Not_Support_Types(name)                  \
  Register_Binary_Operation_New_Err(name, cfloat);                             \
  Register_Binary_Operation_New_Err(name, cdouble);                            \
  Register_Binary_Operation_New_Err(name, clongdouble);                        \
  Register_Binary_Operation_New_Err(name, object);                             \
  Register_Binary_Operation_New_Err(name, string);                             \
  Register_Binary_Operation_New_Err(name, unicode);                            \
  Register_Binary_Operation_New_Err(name, void);                               \
  Register_Binary_Operation_New_Err(name, datetime);                           \
  Register_Binary_Operation_New_Err(name, timedelta);

#define Register_Binary_Operation_Method(name, op_enum)                        \
  PyObject *numboost_##name(PyObject *a, PyObject *b) {                        \
    int a_type = any_to_type_enum(a);                                          \
    int b_type = any_to_type_enum(b);                                          \
    int result_type = binary_result_type(op_enum, a_type, type_2_size[a_type], \
                                         b_type, type_2_size[b_type]);         \
    if (result_type == -1) {                                                   \
      PyErr_SetString(PyExc_TypeError,                                         \
                      Str(name not supported for type));                       \
      return NULL;                                                             \
    }                                                                          \
    assert(result_type <= NPY_HALF);                                           \
    PyObject *result = name##_operations_new[result_type](a, b);               \
    return result;                                                             \
  }

extern PyArrayObject *(**operations[])(PyArrayObject *, PyArrayObject *);

extern PyArrayObject *(**operations_a_scalar[])(Python_Number *,
                                                PyArrayObject *);

extern PyArrayObject *(**operations_b_scalar[])(PyArrayObject *,
                                                Python_Number *);

PyObject *numboost_pow(PyObject *a, PyObject *b);

#endif