#ifndef BINARY_FUNC_DEF_H
#define BINARY_FUNC_DEF_H
#include "binary_op_kernels.h"
#include "numpy/arrayobject.h"

#define Register_Binary_Operation_Array(name, sufix)                           \
  PyObject *(*name##_operations##sufix[])(PyObject *, PyObject *, PyObject **, \
                                          int) = {                             \
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
  Register_Binary_Operation(name, float, NPY_FLOAT, loop_body);                \
  Register_Binary_Operation(name, double, NPY_DOUBLE, loop_body);              \
  Register_Binary_Operation(name, longdouble, NPY_LONGDOUBLE, loop_body);      \
  Register_Binary_Operation(name, half, NPY_HALF, loop_body);

#define Register_Binary_Operations_Interger_Types(name, loop_body)             \
  Register_Binary_Operation(name, bool, NPY_BOOL, loop_body);                  \
  Register_Binary_Operation(name, byte, NPY_BYTE, loop_body);                  \
  Register_Binary_Operation(name, ubyte, NPY_UBYTE, loop_body);                \
  Register_Binary_Operation(name, short, NPY_SHORT, loop_body);                \
  Register_Binary_Operation(name, ushort, NPY_USHORT, loop_body);              \
  Register_Binary_Operation(name, int, NPY_INT, loop_body);                    \
  Register_Binary_Operation(name, uint, NPY_UINT, loop_body);                  \
  Register_Binary_Operation(name, long, NPY_LONG, loop_body);                  \
  Register_Binary_Operation(name, ulong, NPY_ULONG, loop_body);                \
  Register_Binary_Operation(name, longlong, NPY_LONGLONG, loop_body);          \
  Register_Binary_Operation(name, ulonglong, NPY_ULONGLONG, loop_body);

#define Register_Binary_Operations_Err_Floating_Types(name)                    \
  Register_Binary_Operation_Err(name, float);                                  \
  Register_Binary_Operation_Err(name, double);                                 \
  Register_Binary_Operation_Err(name, longdouble);                             \
  Register_Binary_Operation_Err(name, half);

#define Register_Binary_Operations_Err_Interger_Types(name)                    \
  Register_Binary_Operation_Err(name, bool);                                   \
  Register_Binary_Operation_Err(name, byte);                                   \
  Register_Binary_Operation_Err(name, ubyte);                                  \
  Register_Binary_Operation_Err(name, short);                                  \
  Register_Binary_Operation_Err(name, ushort);                                 \
  Register_Binary_Operation_Err(name, int);                                    \
  Register_Binary_Operation_Err(name, uint);                                   \
  Register_Binary_Operation_Err(name, long);                                   \
  Register_Binary_Operation_Err(name, ulong);                                  \
  Register_Binary_Operation_Err(name, longlong);                               \
  Register_Binary_Operation_Err(name, ulonglong);

#define Register_Binary_Operation_Err_Not_Support_Types(name)                  \
  Register_Binary_Operation_Err(name, cfloat);                                 \
  Register_Binary_Operation_Err(name, cdouble);                                \
  Register_Binary_Operation_Err(name, clongdouble);                            \
  Register_Binary_Operation_Err(name, object);                                 \
  Register_Binary_Operation_Err(name, string);                                 \
  Register_Binary_Operation_Err(name, unicode);                                \
  Register_Binary_Operation_Err(name, void);                                   \
  Register_Binary_Operation_Err(name, datetime);                               \
  Register_Binary_Operation_Err(name, timedelta);

#define Register_Binary_Operation_Method(name, op_enum)                        \
  PyObject *numboost_##name(PyObject *a, PyObject *b, PyObject **outs_arr) {   \
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
    PyObject *result = name##_operations[result_type](a, b, outs_arr, 1);      \
    if (result == NULL) {                                                      \
      return NULL;                                                             \
    }                                                                          \
    return result;                                                             \
  }

#define Register_Binary_Operation(name, type, result_type,                     \
                                  inner_loop_body_universal)                   \
  PyObject *binary_##name##_##type(PyObject *a, PyObject *b,                   \
                                   PyObject **out_arr, int out_arr_len) {      \
    PyArrayObject **return_arr =                                               \
        (PyArrayObject **)malloc(sizeof(PyArrayObject *) * 1);                 \
    Perform_Universal_Operation(npy_##type, return_arr, result_type,           \
                                inner_loop_body_universal, out_arr,            \
                                out_arr_len, (result), a, b);                  \
    if (return_arr == NULL) {                                                  \
      return NULL;                                                             \
    } else {                                                                   \
      PyObject *ret = (PyObject *)return_arr[0];                               \
      free(return_arr);                                                        \
      return ret;                                                              \
    }                                                                          \
  }

#define Register_Binary_Operation_Err(name, type)                              \
  PyObject *binary_##name##_##type(PyObject *a, PyObject *b,                   \
                                   PyObject **outs_arr, int outs_arr_len) {    \
    PyErr_SetString(PyExc_TypeError, Str(name not supported for type));        \
    return NULL;                                                               \
  }

PyObject *numboost_add(PyObject *a, PyObject *b, PyObject **out);
PyObject *numboost_pow(PyObject *a, PyObject *b, PyObject **out);
PyObject *numboost_sub(PyObject *a, PyObject *b, PyObject **out);
PyObject *numboost_mul(PyObject *a, PyObject *b, PyObject **out);
PyObject *numboost_div(PyObject *a, PyObject *b, PyObject **out);
PyObject *numboost_lshift(PyObject *a, PyObject *b, PyObject **out);
PyObject *numboost_rshift(PyObject *a, PyObject *b, PyObject **out);
PyObject *numboost_mod(PyObject *a, PyObject *b, PyObject **out);
PyObject *numboost_fdiv(PyObject *a, PyObject *b, PyObject **out);
PyObject *numboost_bitwise_and(PyObject *a, PyObject *b, PyObject **out);
PyObject *numboost_bitwise_xor(PyObject *a, PyObject *b, PyObject **out);
PyObject *numboost_bitwise_or(PyObject *a, PyObject *b, PyObject **out);
#endif