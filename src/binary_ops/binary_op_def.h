#ifndef BINARY_FUNC_DEF_H
#define BINARY_FUNC_DEF_H
#include "binary_op_kernels.h"
#include "numpy/arrayobject.h"

#define Register_Binary_Operation_Array(name, sufix)                           \
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

#define Register_Binary_Operation_Array_New(name, sufix)                       \
  PyObject *(*name##_operations##sufix[])(PyObject *, PyObject *, PyObject **, \
                                          int) = {                             \
      binary_##name##_bool##sufix##_new,                                       \
      binary_##name##_byte##sufix##_new,                                       \
      binary_##name##_ubyte##sufix##_new,                                      \
      binary_##name##_short##sufix##_new,                                      \
      binary_##name##_ushort##sufix##_new,                                     \
      binary_##name##_int##sufix##_new,                                        \
      binary_##name##_uint##sufix##_new,                                       \
      binary_##name##_long##sufix##_new,                                       \
      binary_##name##_ulong##sufix##_new,                                      \
      binary_##name##_longlong##sufix##_new,                                   \
      binary_##name##_ulonglong##sufix##_new,                                  \
      binary_##name##_float##sufix##_new,                                      \
      binary_##name##_double##sufix##_new,                                     \
      binary_##name##_longdouble##sufix##_new,                                 \
      binary_##name##_cfloat##sufix##_new,                                     \
      binary_##name##_cdouble##sufix##_new,                                    \
      binary_##name##_clongdouble##sufix##_new,                                \
      binary_##name##_object##sufix##_new,                                     \
      binary_##name##_string##sufix##_new,                                     \
      binary_##name##_unicode##sufix##_new,                                    \
      binary_##name##_void##sufix##_new,                                       \
      binary_##name##_datetime##sufix##_new,                                   \
      binary_##name##_timedelta##sufix##_new,                                  \
      binary_##name##_half##sufix##_new};

#define Register_Binary_Operations_Floating_Types_New(name, loop_body)         \
  Register_Binary_Operation_New(name, float, NPY_FLOAT, loop_body);            \
  Register_Binary_Operation_New(name, double, NPY_DOUBLE, loop_body);          \
  Register_Binary_Operation_New(name, longdouble, NPY_LONGDOUBLE, loop_body);  \
  Register_Binary_Operation_New(name, half, NPY_HALF, loop_body);

#define Register_Binary_Operations_Interger_Types_New(name, loop_body)         \
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

#define Register_Binary_Operations_Err_Floating_Types_New(name)                \
  Register_Binary_Operation_Err_New(name, float);                              \
  Register_Binary_Operation_Err_New(name, double);                             \
  Register_Binary_Operation_Err_New(name, longdouble);                         \
  Register_Binary_Operation_Err_New(name, half);

#define Register_Binary_Operations_Err_Interger_Types_New(name)                \
  Register_Binary_Operation_Err_New(name, bool);                               \
  Register_Binary_Operation_Err_New(name, byte);                               \
  Register_Binary_Operation_Err_New(name, ubyte);                              \
  Register_Binary_Operation_Err_New(name, short);                              \
  Register_Binary_Operation_Err_New(name, ushort);                             \
  Register_Binary_Operation_Err_New(name, int);                                \
  Register_Binary_Operation_Err_New(name, uint);                               \
  Register_Binary_Operation_Err_New(name, long);                               \
  Register_Binary_Operation_Err_New(name, ulong);                              \
  Register_Binary_Operation_Err_New(name, longlong);                           \
  Register_Binary_Operation_Err_New(name, ulonglong);

#define Register_Binary_Operation_Err_Not_Support_Types_New(name)              \
  Register_Binary_Operation_Err_New(name, cfloat);                             \
  Register_Binary_Operation_Err_New(name, cdouble);                            \
  Register_Binary_Operation_Err_New(name, clongdouble);                        \
  Register_Binary_Operation_Err_New(name, object);                             \
  Register_Binary_Operation_Err_New(name, string);                             \
  Register_Binary_Operation_Err_New(name, unicode);                            \
  Register_Binary_Operation_Err_New(name, void);                               \
  Register_Binary_Operation_Err_New(name, datetime);                           \
  Register_Binary_Operation_Err_New(name, timedelta);

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
    if (result == NULL) {                                                      \
      return NULL;                                                             \
    }                                                                          \
    return result;                                                             \
  }

#define Register_Binary_Operation_Method_New(name, op_enum)                    \
  PyObject *numboost_##name##_new(PyObject *a, PyObject *b,                    \
                                  PyObject **outs_arr) {                       \
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
  PyObject *binary_##name##_##type(PyObject *a, PyObject *b) {                 \
    PyArrayObject **return_arr =                                               \
        (PyArrayObject **)malloc(sizeof(PyArrayObject *) * 1);                 \
    Perform_Universal_Operation(npy_##type, return_arr, result_type,           \
                                inner_loop_body_universal, (result), a, b);    \
    if (return_arr == NULL) {                                                  \
      return NULL;                                                             \
    } else {                                                                   \
      PyObject *ret = (PyObject *)return_arr[0];                               \
      free(return_arr);                                                        \
      return ret;                                                              \
    }                                                                          \
  }

#define Register_Binary_Operation_New(name, type, result_type,                 \
                                      inner_loop_body_universal)               \
  PyObject *binary_##name##_##type##_new(                                      \
      PyObject *a, PyObject *b, PyObject **out_arr, int out_arr_len) {         \
    PyArrayObject **return_arr =                                               \
        (PyArrayObject **)malloc(sizeof(PyArrayObject *) * 1);                 \
    Perform_Universal_Operation_Inplace(npy_##type, return_arr, result_type,   \
                                        inner_loop_body_universal, out_arr,    \
                                        out_arr_len, (result), a, b);          \
    if (return_arr == NULL) {                                                  \
      return NULL;                                                             \
    } else {                                                                   \
      PyObject *ret = (PyObject *)return_arr[0];                               \
      free(return_arr);                                                        \
      return ret;                                                              \
    }                                                                          \
  }

#define Register_Binary_Operation_Err(name, type)                              \
  PyObject *binary_##name##_##type(PyObject *a, PyObject *b) {                 \
    PyErr_SetString(PyExc_TypeError, Str(name not supported for type));        \
    return NULL;                                                               \
  }

#define Register_Binary_Operation_Err_New(name, type)                          \
  PyObject *binary_##name##_##type##_new(                                      \
      PyObject *a, PyObject *b, PyObject **outs_arr, int outs_arr_len) {       \
    PyErr_SetString(PyExc_TypeError, Str(name not supported for type));        \
    return NULL;                                                               \
  }

PyObject *numboost_pow(PyObject *a, PyObject *b);
PyObject *numboost_add(PyObject *a, PyObject *b);
PyObject *numboost_sub(PyObject *a, PyObject *b);
PyObject *numboost_mul(PyObject *a, PyObject *b);
PyObject *numboost_div(PyObject *a, PyObject *b);
PyObject *numboost_lshift(PyObject *a, PyObject *b);
PyObject *numboost_rshift(PyObject *a, PyObject *b);
PyObject *numboost_mod(PyObject *a, PyObject *b);
PyObject *numboost_fdiv(PyObject *a, PyObject *b);

PyObject *numboost_add_new(PyObject *a, PyObject *b, PyObject **out);
PyObject *numboost_pow_new(PyObject *a, PyObject *b, PyObject **out);
PyObject *numboost_add_new(PyObject *a, PyObject *b, PyObject **out);
PyObject *numboost_sub_new(PyObject *a, PyObject *b, PyObject **out);
PyObject *numboost_mul_new(PyObject *a, PyObject *b, PyObject **out);
PyObject *numboost_div_new(PyObject *a, PyObject *b, PyObject **out);
PyObject *numboost_lshift_new(PyObject *a, PyObject *b, PyObject **out);
PyObject *numboost_rshift_new(PyObject *a, PyObject *b, PyObject **out);
PyObject *numboost_mod_new(PyObject *a, PyObject *b, PyObject **out);
PyObject *numboost_fdiv_new(PyObject *a, PyObject *b, PyObject **out);
#endif