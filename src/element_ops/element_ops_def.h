#ifndef _ELEMENT_OPS_DEF_H
#define _ELEMENT_OPS_DEF_H
#include "element_ops_impl.h"

#define Register_ElementWise_Operation_Array(name, sufix)                      \
  PyObject *(*name##_operations##sufix[])(PyObject *) = {                      \
      elementwise_##name##_bool##sufix,                                        \
      elementwise_##name##_byte##sufix,                                        \
      elementwise_##name##_ubyte##sufix,                                       \
      elementwise_##name##_short##sufix,                                       \
      elementwise_##name##_ushort##sufix,                                      \
      elementwise_##name##_int##sufix,                                         \
      elementwise_##name##_uint##sufix,                                        \
      elementwise_##name##_long##sufix,                                        \
      elementwise_##name##_ulong##sufix,                                       \
      elementwise_##name##_longlong##sufix,                                    \
      elementwise_##name##_ulonglong##sufix,                                   \
      elementwise_##name##_float##sufix,                                       \
      elementwise_##name##_double##sufix,                                      \
      elementwise_##name##_longdouble##sufix,                                  \
      elementwise_##name##_cfloat##sufix,                                      \
      elementwise_##name##_cdouble##sufix,                                     \
      elementwise_##name##_clongdouble##sufix,                                 \
      elementwise_##name##_object##sufix,                                      \
      elementwise_##name##_string##sufix,                                      \
      elementwise_##name##_unicode##sufix,                                     \
      elementwise_##name##_void##sufix,                                        \
      elementwise_##name##_datetime##sufix,                                    \
      elementwise_##name##_timedelta##sufix,                                   \
      elementwise_##name##_half##sufix};

#define Register_ElementWise_Operation_Err(name, type)                         \
  PyObject *elementwise_##name##_##type(PyObject *a) {                         \
    PyErr_SetString(PyExc_TypeError, Str(name not supported for type));        \
    return NULL;                                                               \
  }

#define Register_ElementWise_Operation(                                        \
    name, type, result_type, inner_loop_body_universal, inner_loop_body_seq)   \
  PyObject *elementwise_##name##_##type(PyObject *a) {                         \
    Perform_Universal_Operation(npy_##type, result_type,                       \
                                inner_loop_body_universal,                     \
                                inner_loop_body_seq, a);                       \
  }

#define Register_ElementWise_Operations_Floating_Types(                        \
    name, universal_loop_body, sequential_loop_body)                           \
  Register_ElementWise_Operation(name, float, NPY_FLOAT, universal_loop_body,  \
                                 sequential_loop_body);                        \
  Register_ElementWise_Operation(name, double, NPY_DOUBLE,                     \
                                 universal_loop_body, sequential_loop_body);   \
  Register_ElementWise_Operation(name, longdouble, NPY_LONGDOUBLE,             \
                                 universal_loop_body, sequential_loop_body);   \
  Register_ElementWise_Operation(name, half, NPY_HALF, universal_loop_body,    \
                                 sequential_loop_body);

#define Register_ElementWise_Operations_Interger_Types(                        \
    name, universal_loop_body, sequential_loop_body)                           \
  Register_ElementWise_Operation(name, bool, NPY_BOOL, universal_loop_body,    \
                                 sequential_loop_body);                        \
  Register_ElementWise_Operation(name, byte, NPY_BYTE, universal_loop_body,    \
                                 sequential_loop_body);                        \
  Register_ElementWise_Operation(name, ubyte, NPY_UBYTE, universal_loop_body,  \
                                 sequential_loop_body);                        \
  Register_ElementWise_Operation(name, short, NPY_SHORT, universal_loop_body,  \
                                 sequential_loop_body);                        \
  Register_ElementWise_Operation(name, ushort, NPY_USHORT,                     \
                                 universal_loop_body, sequential_loop_body);   \
  Register_ElementWise_Operation(name, int, NPY_INT, universal_loop_body,      \
                                 sequential_loop_body);                        \
  Register_ElementWise_Operation(name, uint, NPY_UINT, universal_loop_body,    \
                                 sequential_loop_body);                        \
  Register_ElementWise_Operation(name, long, NPY_LONG, universal_loop_body,    \
                                 sequential_loop_body);                        \
  Register_ElementWise_Operation(name, ulong, NPY_ULONG, universal_loop_body,  \
                                 sequential_loop_body);                        \
  Register_ElementWise_Operation(name, longlong, NPY_LONGLONG,                 \
                                 universal_loop_body, sequential_loop_body);   \
  Register_ElementWise_Operation(name, ulonglong, NPY_ULONGLONG,               \
                                 universal_loop_body, sequential_loop_body);

#define Register_ElementWise_Operation_Err_Interger_Types(name)                \
  Register_ElementWise_Operation_Err(name, bool);                              \
  Register_ElementWise_Operation_Err(name, byte);                              \
  Register_ElementWise_Operation_Err(name, ubyte);                             \
  Register_ElementWise_Operation_Err(name, short);                             \
  Register_ElementWise_Operation_Err(name, ushort);                            \
  Register_ElementWise_Operation_Err(name, int);                               \
  Register_ElementWise_Operation_Err(name, uint);                              \
  Register_ElementWise_Operation_Err(name, long);                              \
  Register_ElementWise_Operation_Err(name, ulong);                             \
  Register_ElementWise_Operation_Err(name, longlong);                          \
  Register_ElementWise_Operation_Err(name, ulonglong);

#define Register_ElementWise_Operation_Err_Not_Support_Types(name)             \
  Register_ElementWise_Operation_Err(name, cfloat);                            \
  Register_ElementWise_Operation_Err(name, cdouble);                           \
  Register_ElementWise_Operation_Err(name, clongdouble);                       \
  Register_ElementWise_Operation_Err(name, object);                            \
  Register_ElementWise_Operation_Err(name, string);                            \
  Register_ElementWise_Operation_Err(name, unicode);                           \
  Register_ElementWise_Operation_Err(name, void);                              \
  Register_ElementWise_Operation_Err(name, datetime);                          \
  Register_ElementWise_Operation_Err(name, timedelta);

#define Register_ElementWise_Operation_Method(name, op_enum)                   \
  PyObject *numboost_##name(PyObject *a) {                                     \
    int input_type = any_to_type_enum(a);                                      \
    int result_type = elementwise_result_type(op_enum, input_type);            \
    if (result_type == -1) {                                                   \
      PyErr_SetString(PyExc_TypeError,                                         \
                      Str(name not supported for type));                       \
      return NULL;                                                             \
    }                                                                          \
    assert(result_type <= NPY_HALF);                                           \
    printf("result_type: %d\n", result_type);                                  \
    PyObject *result = name##_operations[result_type](a);                      \
    return result;                                                             \
  }

PyObject *numboost_abs(PyObject *a);
PyObject *numboost_sin(PyObject *a);
PyObject *numboost_cos(PyObject *a);
PyObject *numboost_tan(PyObject *a);
PyObject *numboost_asin(PyObject *a);
PyObject *numboost_acos(PyObject *a);
PyObject *numboost_atan(PyObject *a);
PyObject *numboost_sinh(PyObject *a);
PyObject *numboost_cosh(PyObject *a);
PyObject *numboost_tanh(PyObject *a);
PyObject *numboost_asinh(PyObject *a);
PyObject *numboost_acosh(PyObject *a);
PyObject *numboost_atanh(PyObject *a);

#endif