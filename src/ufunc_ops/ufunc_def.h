#ifndef UFUNC_DEF_H
#define UFUNC_DEF_H

#include "../numboost_api.h"
#include "../numboost_utils.h"

#define Register_UFunc_Operation(name, in_type, out_type, result_type,         \
                                 inner_loop_body_universal, ...)               \
  PyObject **name##_ufunc_##in_type(                                           \
      Replicate0_With_Comma(Parameter_type, __VA_ARGS__), PyObject **out,      \
      int out_arr_len) {                                                       \
    PyArrayObject **result =                                                   \
        (PyArrayObject **)malloc(sizeof(PyArrayObject *));                     \
    Perform_Universal_Operation(                                               \
        npy_##in_type, npy_##out_type, result_type, result_type, result,       \
        inner_loop_body_universal, out, out_arr_len, (result), __VA_ARGS__);   \
    if (result == NULL) {                                                      \
      return NULL;                                                             \
    } else {                                                                   \
      return (PyObject **)result;                                              \
    }                                                                          \
  }

#define Register_UFunc_Operation_Err(name, type, ...)                          \
  PyObject **name##_ufunc_##type(                                              \
      Replicate0_With_Comma(Parameter_type, __VA_ARGS__), PyObject **out,      \
      int out_arr_len) {                                                       \
        PyErr_SetString(PyExc_TypeError, Str(Not support for type.));          \
    return NULL;                                                               \
  }

#define Register_UFunc_Operation_FloatingTypes(name, universal_loop_body, ...) \
  Register_UFunc_Operation(name, float, float, NPY_FLOAT, universal_loop_body, \
                           __VA_ARGS__);                                       \
  Register_UFunc_Operation(name, double, double, NPY_DOUBLE,                   \
                           universal_loop_body, __VA_ARGS__);                  \
  Register_UFunc_Operation(name, longdouble, longdouble, NPY_LONGDOUBLE,       \
                           universal_loop_body, __VA_ARGS__);                  \
  Register_UFunc_Operation(name, half, half, NPY_HALF, universal_loop_body,    \
                           __VA_ARGS__);

#define Register_UFunc_Operation_IntergerTypes(name, universal_loop_body, ...) \
  Register_UFunc_Operation(name, bool, bool, NPY_FLOAT, universal_loop_body,   \
                           __VA_ARGS__);                                       \
  Register_UFunc_Operation(name, byte, byte, NPY_BYTE, universal_loop_body,    \
                           __VA_ARGS__);                                       \
  Register_UFunc_Operation(name, ubyte, ubyte, NPY_UBYTE, universal_loop_body, \
                           __VA_ARGS__);                                       \
  Register_UFunc_Operation(name, short, short, NPY_SHORT, universal_loop_body, \
                           __VA_ARGS__);                                       \
  Register_UFunc_Operation(name, ushort, ushort, NPY_USHORT,                   \
                           universal_loop_body, __VA_ARGS__);                  \
  Register_UFunc_Operation(name, int, int, NPY_INT, universal_loop_body,       \
                           __VA_ARGS__);                                       \
  Register_UFunc_Operation(name, uint, uint, NPY_UINT, universal_loop_body,    \
                           __VA_ARGS__);                                       \
  Register_UFunc_Operation(name, long, long, NPY_LONG, universal_loop_body,    \
                           __VA_ARGS__);                                       \
  Register_UFunc_Operation(name, ulong, ulong, NPY_ULONG, universal_loop_body, \
                           __VA_ARGS__);                                       \
  Register_UFunc_Operation(name, longlong, longlong, NPY_LONGLONG,             \
                           universal_loop_body, __VA_ARGS__);                  \
  Register_UFunc_Operation(name, ulonglong, ulonglong, NPY_ULONGLONG,          \
                           universal_loop_body, __VA_ARGS__);

#define Register_UFunc_Operation_Err_Int(name, ...)                            \
  Register_UFunc_Operation_Err(name, bool, __VA_ARGS__);                       \
  Register_UFunc_Operation_Err(name, byte, __VA_ARGS__);                       \
  Register_UFunc_Operation_Err(name, ubyte, __VA_ARGS__);                      \
  Register_UFunc_Operation_Err(name, short, __VA_ARGS__);                      \
  Register_UFunc_Operation_Err(name, ushort, __VA_ARGS__);                     \
  Register_UFunc_Operation_Err(name, int, __VA_ARGS__);                        \
  Register_UFunc_Operation_Err(name, uint, __VA_ARGS__);                       \
  Register_UFunc_Operation_Err(name, long, __VA_ARGS__);                       \
  Register_UFunc_Operation_Err(name, ulong, __VA_ARGS__);                      \
  Register_UFunc_Operation_Err(name, longlong, __VA_ARGS__);                   \
  Register_UFunc_Operation_Err(name, ulonglong, __VA_ARGS__);

#define Register_UFunc_Operation_Err_UnsupportTypes(name, ...)                 \
  Register_UFunc_Operation_Err(name, cfloat, __VA_ARGS__);                     \
  Register_UFunc_Operation_Err(name, cdouble, __VA_ARGS__);                    \
  Register_UFunc_Operation_Err(name, clongdouble, __VA_ARGS__);                \
  Register_UFunc_Operation_Err(name, object, __VA_ARGS__);                     \
  Register_UFunc_Operation_Err(name, string, __VA_ARGS__);                     \
  Register_UFunc_Operation_Err(name, unicode, __VA_ARGS__);                    \
  Register_UFunc_Operation_Err(name, void, __VA_ARGS__);                       \
  Register_UFunc_Operation_Err(name, datetime, __VA_ARGS__);                   \
  Register_UFunc_Operation_Err(name, timedelta, __VA_ARGS__);

#define Register_UFunc_Operation_Array(name, ...)                              \
  PyObject **(*name##_ufunc_[])(                                               \
      Replicate0_With_Comma(Parameter_type_, __VA_ARGS__), PyObject **,        \
      int) = {                                                                 \
      name##_ufunc_bool,     name##_ufunc_byte,        name##_ufunc_ubyte,     \
      name##_ufunc_short,    name##_ufunc_ushort,      name##_ufunc_int,       \
      name##_ufunc_uint,     name##_ufunc_long,        name##_ufunc_ulong,     \
      name##_ufunc_longlong, name##_ufunc_ulonglong,   name##_ufunc_float,     \
      name##_ufunc_double,   name##_ufunc_longdouble,  name##_ufunc_cfloat,    \
      name##_ufunc_cdouble,  name##_ufunc_clongdouble, name##_ufunc_object,    \
      name##_ufunc_string,   name##_ufunc_unicode,     name##_ufunc_void,      \
      name##_ufunc_datetime, name##_ufunc_timedelta,   name##_ufunc_half};

#define Register_UFunc_Operation_Method(name, ...)                             \
  PyObject **numboost_##name(                                                  \
      Replicate0_With_Comma(Parameter_type, __VA_ARGS__), PyObject **out,      \
      int out_arr_len, int result_type) {                                      \
    assert(result_type <= NPY_HALF);                                           \
    PyObject **result =                                                        \
        name##_ufunc_[result_type](__VA_ARGS__, out, out_arr_len);             \
    return result;                                                             \
  }

PyObject **numboost_where(PyObject *mask, PyObject *x, PyObject *y,
                          PyObject **out, int out_arr_len, int result_type);

#endif // UFUNC_DEF_H