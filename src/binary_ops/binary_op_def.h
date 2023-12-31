#ifndef BINARY_FUNC_DEF_H
#define BINARY_FUNC_DEF_H
#include "binary_op_kernels.h"
#include "numpy/arrayobject.h"

/*=== Array Def ===*/
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

#define Register_Binary_Operation_Array_MultiOut(name, sufix)                  \
  PyObject **(*name##_operations##sufix[])(PyObject *, PyObject *,             \
                                           PyObject **, int) = {               \
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

/*=== Floating Def ===*/
#define Register_Binary_Operations_Floating_Types(name, loop_body)             \
  Register_Binary_Operation(name, float, float, NPY_FLOAT, loop_body);         \
  Register_Binary_Operation(name, double, double, NPY_DOUBLE, loop_body);      \
  Register_Binary_Operation(name, longdouble, longdouble, NPY_LONGDOUBLE,      \
                            loop_body);                                        \
  Register_Binary_Operation(name, half, half, NPY_HALF, loop_body);

#define Register_Compare_Operations_Floating_Types(name, loop_body)            \
  Register_Compare_Operation(name, float, bool, NPY_FLOAT, NPY_BOOL,            \
                            loop_body);                                        \
  Register_Compare_Operation(name, double, bool, NPY_DOUBLE, NPY_BOOL,          \
                            loop_body);                                        \
  Register_Compare_Operation(name, longdouble, bool, NPY_LONGDOUBLE, NPY_BOOL,  \
                            loop_body);                                        \
  Register_Compare_Operation(name, half, bool, NPY_HALF, NPY_BOOL, loop_body);

#define Register_Binary_Operations_Floating_Types_MultiOut(name, loop_body,    \
                                                           ...)                \
  Register_Binary_Operation_MultiOut(name, float, float, NPY_FLOAT, loop_body, \
                                     __VA_ARGS__);                             \
  Register_Binary_Operation_MultiOut(name, double, double, NPY_DOUBLE,         \
                                     loop_body, __VA_ARGS__);                  \
  Register_Binary_Operation_MultiOut(name, longdouble, longdouble,             \
                                     NPY_LONGDOUBLE, loop_body, __VA_ARGS__);  \
  Register_Binary_Operation_MultiOut(name, half, half, NPY_HALF, loop_body,    \
                                     __VA_ARGS__);

/*=== Interger Def ===*/
#define Register_Binary_Operations_Interger_Types(name, loop_body)             \
  Register_Binary_Operation(name, bool, bool, NPY_BOOL, loop_body);            \
  Register_Binary_Operation(name, byte, byte, NPY_BYTE, loop_body);            \
  Register_Binary_Operation(name, ubyte, ubyte, NPY_UBYTE, loop_body);         \
  Register_Binary_Operation(name, short, short, NPY_SHORT, loop_body);         \
  Register_Binary_Operation(name, ushort, ushort, NPY_USHORT, loop_body);      \
  Register_Binary_Operation(name, int, int, NPY_INT, loop_body);               \
  Register_Binary_Operation(name, uint, uint, NPY_UINT, loop_body);            \
  Register_Binary_Operation(name, long, long, NPY_LONG, loop_body);            \
  Register_Binary_Operation(name, ulong, ulong, NPY_ULONG, loop_body);         \
  Register_Binary_Operation(name, longlong, longlong, NPY_LONGLONG,            \
                            loop_body);                                        \
  Register_Binary_Operation(name, ulonglong, ulonglong, NPY_ULONGLONG,         \
                            loop_body);

#define Register_Compare_Operations_Interger_Types(name, loop_body)            \
  Register_Compare_Operation(name, bool, bool, NPY_BOOL, NPY_BOOL, loop_body); \
  Register_Compare_Operation(name, byte, bool, NPY_BYTE, NPY_BOOL, loop_body); \
  Register_Compare_Operation(name, ubyte, bool, NPY_UBYTE, NPY_BOOL,           \
                             loop_body);                                       \
  Register_Compare_Operation(name, short, bool, NPY_SHORT, NPY_BOOL,           \
                             loop_body);                                       \
  Register_Compare_Operation(name, ushort, bool, NPY_USHORT, NPY_BOOL,         \
                             loop_body);                                       \
  Register_Compare_Operation(name, int, bool, NPY_INT, NPY_BOOL, loop_body);   \
  Register_Compare_Operation(name, uint, bool, NPY_UINT, NPY_BOOL, loop_body); \
  Register_Compare_Operation(name, long, bool, NPY_LONG, NPY_BOOL, loop_body); \
  Register_Compare_Operation(name, ulong, bool, NPY_ULONG, NPY_BOOL,           \
                             loop_body);                                       \
  Register_Compare_Operation(name, longlong, bool, NPY_LONGLONG, NPY_BOOL,     \
                             loop_body);                                       \
  Register_Compare_Operation(name, ulonglong, bool, NPY_ULONGLONG, NPY_BOOL,   \
                             loop_body);

#define Register_Binary_Operations_Interger_Types_MultiOut(name, loop_body,    \
                                                           ...)                \
  Register_Binary_Operation_MultiOut(name, bool, bool, NPY_BOOL, loop_body,    \
                                     __VA_ARGS__);                             \
  Register_Binary_Operation_MultiOut(name, byte, byte, NPY_BYTE, loop_body,    \
                                     __VA_ARGS__);                             \
  Register_Binary_Operation_MultiOut(name, ubyte, ubyte, NPY_UBYTE, loop_body, \
                                     __VA_ARGS__);                             \
  Register_Binary_Operation_MultiOut(name, short, short, NPY_SHORT, loop_body, \
                                     __VA_ARGS__);                             \
  Register_Binary_Operation_MultiOut(name, ushort, ushort, NPY_USHORT,         \
                                     loop_body, __VA_ARGS__);                  \
  Register_Binary_Operation_MultiOut(name, int, int, NPY_INT, loop_body,       \
                                     __VA_ARGS__);                             \
  Register_Binary_Operation_MultiOut(name, uint, uint, NPY_UINT, loop_body,    \
                                     __VA_ARGS__);                             \
  Register_Binary_Operation_MultiOut(name, long, long, NPY_LONG, loop_body,    \
                                     __VA_ARGS__);                             \
  Register_Binary_Operation_MultiOut(name, ulong, ulong, NPY_ULONG, loop_body, \
                                     __VA_ARGS__);                             \
  Register_Binary_Operation_MultiOut(name, longlong, longlong, NPY_LONGLONG,   \
                                     loop_body, __VA_ARGS__);                  \
  Register_Binary_Operation_MultiOut(name, ulonglong, ulonglong,               \
                                     NPY_ULONGLONG, loop_body, __VA_ARGS__);

/*=== Floating Err Def ===*/
#define Register_Binary_Operations_Err_Floating_Types(name)                    \
  Register_Binary_Operation_Err(name, float);                                  \
  Register_Binary_Operation_Err(name, double);                                 \
  Register_Binary_Operation_Err(name, longdouble);                             \
  Register_Binary_Operation_Err(name, half);

#define Register_Binary_Operations_Err_Floating_Types_MultiOut(name)           \
  Register_Binary_Operation_Err_MultiOut(name, float);                         \
  Register_Binary_Operation_Err_MultiOut(name, double);                        \
  Register_Binary_Operation_Err_MultiOut(name, longdouble);                    \
  Register_Binary_Operation_Err_MultiOut(name, half);

/*=== Interger Err Def ===*/
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

#define Register_Binary_Operations_Err_Interger_Types_MultiOut(name)           \
  Register_Binary_Operation_Err_MultiOut(name, bool);                          \
  Register_Binary_Operation_Err_MultiOut(name, byte);                          \
  Register_Binary_Operation_Err_MultiOut(name, ubyte);                         \
  Register_Binary_Operation_Err_MultiOut(name, short);                         \
  Register_Binary_Operation_Err_MultiOut(name, ushort);                        \
  Register_Binary_Operation_Err_MultiOut(name, int);                           \
  Register_Binary_Operation_Err_MultiOut(name, uint);                          \
  Register_Binary_Operation_Err_MultiOut(name, long);                          \
  Register_Binary_Operation_Err_MultiOut(name, ulong);                         \
  Register_Binary_Operation_Err_MultiOut(name, longlong);                      \
  Register_Binary_Operation_Err_MultiOut(name, ulonglong);

/*=== Not Support Type Def ===*/
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

#define Register_Binary_Operation_Err_Not_Support_Types_MultiOut(name)         \
  Register_Binary_Operation_Err_MultiOut(name, cfloat);                        \
  Register_Binary_Operation_Err_MultiOut(name, cdouble);                       \
  Register_Binary_Operation_Err_MultiOut(name, clongdouble);                   \
  Register_Binary_Operation_Err_MultiOut(name, object);                        \
  Register_Binary_Operation_Err_MultiOut(name, string);                        \
  Register_Binary_Operation_Err_MultiOut(name, unicode);                       \
  Register_Binary_Operation_Err_MultiOut(name, void);                          \
  Register_Binary_Operation_Err_MultiOut(name, datetime);                      \
  Register_Binary_Operation_Err_MultiOut(name, timedelta);

/*=== Register Method Def ===*/
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

#define Register_Compare_Operation_Method(name, op_enum)                       \
  PyObject *numboost_##name(PyObject *a, PyObject *b, PyObject **outs_arr) {   \
    int a_type = any_to_type_enum(a);                                          \
    int b_type = any_to_type_enum(b);                                          \
    int type = type_2_size[a_type] > type_2_size[b_type] ? a_type : b_type;    \
    PyObject *result = name##_operations[type](a, b, outs_arr, 1);             \
    if (result == NULL) {                                                      \
      return NULL;                                                             \
    }                                                                          \
    return result;                                                             \
  }

#define Register_Binary_Operation_Method_MultiOut(name, op_enum)               \
  PyObject **numboost_##name(PyObject *a, PyObject *b, PyObject **outs,        \
                             int outs_len) {                                   \
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
    PyObject **result = name##_operations[result_type](a, b, outs, outs_len);  \
    if (result == NULL) {                                                      \
      return NULL;                                                             \
    }                                                                          \
    return result;                                                             \
  }

/*=== Array Element Function Def ===*/
#define Register_Binary_Operation(name, in_type, out_type, result_type,        \
                                  inner_loop_body_universal)                   \
  PyObject *binary_##name##_##in_type(PyObject *a, PyObject *b,                \
                                      PyObject **out_arr, int out_arr_len) {   \
    PyArrayObject **return_arr =                                               \
        (PyArrayObject **)malloc(sizeof(PyArrayObject *) * 1);                 \
    Perform_Universal_Operation(                                               \
        npy_##in_type, npy_##out_type, result_type, result_type, return_arr,   \
        inner_loop_body_universal, out_arr, out_arr_len, (result), a, b);      \
    if (return_arr == NULL) {                                                  \
      return NULL;                                                             \
    } else {                                                                   \
      PyObject *ret = (PyObject *)return_arr[0];                               \
      free(return_arr);                                                        \
      return ret;                                                              \
    }                                                                          \
  }

#define Register_Compare_Operation(name, in_type, out_type, in_type_enum,      \
                                   result_type, inner_loop_body_universal)     \
  PyObject *binary_##name##_##in_type(PyObject *a, PyObject *b,                \
                                      PyObject **out_arr, int out_arr_len) {   \
    PyArrayObject **return_arr =                                               \
        (PyArrayObject **)malloc(sizeof(PyArrayObject *) * 1);                 \
    Perform_Universal_Operation(                                               \
        npy_##in_type, npy_##out_type, in_type_enum, result_type, return_arr,  \
        inner_loop_body_universal, out_arr, out_arr_len, (result), a, b);      \
    if (return_arr == NULL) {                                                  \
      return NULL;                                                             \
    } else {                                                                   \
      PyObject *ret = (PyObject *)return_arr[0];                               \
      free(return_arr);                                                        \
      return ret;                                                              \
    }                                                                          \
  }

#define Register_Binary_Operation_MultiOut(                                    \
    name, in_type, out_type, result_type, inner_loop_body_universal, ...)      \
  PyObject **binary_##name##_##in_type(PyObject *a, PyObject *b,               \
                                       PyObject **out_arr, int out_arr_len) {  \
    PyArrayObject **return_arr = (PyArrayObject **)malloc(                     \
        sizeof(PyArrayObject *) * (Args_Num(__VA_ARGS__)));                    \
    Perform_Universal_Operation(                                               \
        npy_##in_type, npy_##out_type, result_type, result_type, return_arr,   \
        inner_loop_body_universal, out_arr, out_arr_len, (__VA_ARGS__), a, b); \
    if (return_arr == NULL) {                                                  \
      return NULL;                                                             \
    } else {                                                                   \
      for (int i = 0; i < (Args_Num(__VA_ARGS__)); i++) {                      \
        if (return_arr[i] == NULL) {                                           \
          for (int j = 0; j < (Args_Num(__VA_ARGS__)); j++) {                  \
            if (return_arr[j] != NULL) {                                       \
              Py_DECREF(return_arr[j]);                                        \
            }                                                                  \
          }                                                                    \
          free(return_arr);                                                    \
          return NULL;                                                         \
        }                                                                      \
      }                                                                        \
      return (PyObject **)return_arr;                                          \
    }                                                                          \
  }

#define Register_Binary_Operation_Err(name, type)                              \
  PyObject *binary_##name##_##type(PyObject *a, PyObject *b,                   \
                                   PyObject **outs_arr, int outs_arr_len) {    \
    PyErr_SetString(PyExc_TypeError, Str(name not supported for type));        \
    return NULL;                                                               \
  }

#define Register_Binary_Operation_Err_MultiOut(name, type)                     \
  PyObject **binary_##name##_##type(PyObject *a, PyObject *b,                  \
                                    PyObject **outs_arr, int outs_arr_len) {   \
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
PyObject *numboost_eq(PyObject *a, PyObject *b, PyObject **out);
PyObject *numboost_neq(PyObject *a, PyObject *b, PyObject **out);
PyObject *numboost_lt(PyObject *a, PyObject *b, PyObject **out);
PyObject *numboost_gt(PyObject *a, PyObject *b, PyObject **out);
PyObject *numboost_le(PyObject *a, PyObject *b, PyObject **out);
PyObject *numboost_ge(PyObject *a, PyObject *b, PyObject **out);
PyObject **numboost_divmod(PyObject *a, PyObject *b, PyObject **outs,
                           int outs_len);
PyObject *numboost_add_test(PyObject *a, PyObject *b, PyObject **outs_arr);
#endif