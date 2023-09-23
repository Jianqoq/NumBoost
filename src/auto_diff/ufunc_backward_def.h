
#ifndef _NUMBOOST_AUTO_DIFF_UFUNC_BACKWARD_DEF_H_
#define _NUMBOOST_AUTO_DIFF_UFUNC_BACKWARD_DEF_H_
#include "../numboost_api.h"
#include "../numboost_utils.h"

#define Register_FuseBackward_Operation_Set(name, inner_loop_body, ...)        \
  Register_FuseBackward_Operation(name, bool, NPY_BOOL, inner_loop_body,       \
                                  __VA_ARGS__);                                \
  Register_FuseBackward_Operation(name, byte, NPY_BYTE, inner_loop_body,       \
                                  __VA_ARGS__);                                \
  Register_FuseBackward_Operation(name, ubyte, NPY_UBYTE, inner_loop_body,     \
                                  __VA_ARGS__);                                \
  Register_FuseBackward_Operation(name, short, NPY_SHORT, inner_loop_body,     \
                                  __VA_ARGS__);                                \
  Register_FuseBackward_Operation(name, ushort, NPY_USHORT, inner_loop_body,   \
                                  __VA_ARGS__);                                \
  Register_FuseBackward_Operation(name, int, NPY_INT, inner_loop_body,         \
                                  __VA_ARGS__);                                \
  Register_FuseBackward_Operation(name, uint, NPY_UINT, inner_loop_body,       \
                                  __VA_ARGS__);                                \
  Register_FuseBackward_Operation(name, long, NPY_LONG, inner_loop_body,       \
                                  __VA_ARGS__);                                \
  Register_FuseBackward_Operation(name, ulong, NPY_ULONG, inner_loop_body,     \
                                  __VA_ARGS__);                                \
  Register_FuseBackward_Operation(name, longlong, NPY_LONGLONG,                \
                                  inner_loop_body, __VA_ARGS__);               \
  Register_FuseBackward_Operation(name, ulonglong, NPY_ULONGLONG,              \
                                  inner_loop_body, __VA_ARGS__);               \
  Register_FuseBackward_Operation(name, float, NPY_FLOAT, inner_loop_body,     \
                                  __VA_ARGS__);                                \
  Register_FuseBackward_Operation(name, double, NPY_DOUBLE, inner_loop_body,   \
                                  __VA_ARGS__);                                \
  Register_FuseBackward_Operation(name, longdouble, NPY_LONGDOUBLE,            \
                                  inner_loop_body, __VA_ARGS__);               \
  Register_FuseBackward_Operation(name, half, NPY_HALF, inner_loop_body,       \
                                  __VA_ARGS__);

/*! @function
@param  name  backward_fn_name [ sin, cos, etc.]
@param  type  C-type [ long, int, float, double, etc.]
@param  inner_loop_body     the macro the user defined
@param  ...                the tensors that need to be fused
 */
#define Register_FuseBackward_Operation(name, in_type, out_type, result_type,  \
                                        inner_loop_body_universal, ...)        \
  PyObject **name##_backward_##in_type(                                        \
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

#define Register_FuseBackward_Operation_Err(name, type, ...)                   \
  PyObject **name##_backward_##type(                                           \
      Replicate0_With_Comma(Parameter_type, __VA_ARGS__), PyObject **out,      \
      int out_arr_len) {                                                       \
        PyErr_SetString(PyExc_TypeError, Str(Not support for type.));          \
    return NULL;                                                               \
  }

#define Register_FuseBackward_Operation_FloatingTypes(                         \
    name, universal_loop_body, ...)                                            \
  Register_FuseBackward_Operation(name, float, float, NPY_FLOAT,               \
                                  universal_loop_body, __VA_ARGS__);           \
  Register_FuseBackward_Operation(name, double, double, NPY_DOUBLE,            \
                                  universal_loop_body, __VA_ARGS__);           \
  Register_FuseBackward_Operation(name, longdouble, longdouble,                \
                                  NPY_LONGDOUBLE, universal_loop_body,         \
                                  __VA_ARGS__);                                \
  Register_FuseBackward_Operation(name, half, half, NPY_HALF,                  \
                                  universal_loop_body, __VA_ARGS__);

#define Register_FuseBackward_Operation_IntergerTypes(                         \
    name, universal_loop_body, ...)                                            \
  Register_FuseBackward_Operation(name, bool, bool, NPY_FLOAT,                 \
                                  universal_loop_body, __VA_ARGS__);           \
  Register_FuseBackward_Operation(name, byte, byte, NPY_BYTE,                  \
                                  universal_loop_body, __VA_ARGS__);           \
  Register_FuseBackward_Operation(name, ubyte, ubyte, NPY_UBYTE,               \
                                  universal_loop_body, __VA_ARGS__);           \
  Register_FuseBackward_Operation(name, short, short, NPY_SHORT,               \
                                  universal_loop_body, __VA_ARGS__);           \
  Register_FuseBackward_Operation(name, ushort, ushort, NPY_USHORT,            \
                                  universal_loop_body, __VA_ARGS__);           \
  Register_FuseBackward_Operation(name, int, int, NPY_INT,                     \
                                  universal_loop_body, __VA_ARGS__);           \
  Register_FuseBackward_Operation(name, uint, uint, NPY_UINT,                  \
                                  universal_loop_body, __VA_ARGS__);           \
  Register_FuseBackward_Operation(name, long, long, NPY_LONG,                  \
                                  universal_loop_body, __VA_ARGS__);           \
  Register_FuseBackward_Operation(name, ulong, ulong, NPY_ULONG,               \
                                  universal_loop_body, __VA_ARGS__);           \
  Register_FuseBackward_Operation(name, longlong, longlong, NPY_LONGLONG,      \
                                  universal_loop_body, __VA_ARGS__);           \
  Register_FuseBackward_Operation(name, ulonglong, ulonglong, NPY_ULONGLONG,   \
                                  universal_loop_body, __VA_ARGS__);

#define Register_FuseBackward_Operation_Err_Int(name, ...)                     \
  Register_FuseBackward_Operation_Err(name, bool, __VA_ARGS__);                \
  Register_FuseBackward_Operation_Err(name, byte, __VA_ARGS__);                \
  Register_FuseBackward_Operation_Err(name, ubyte, __VA_ARGS__);               \
  Register_FuseBackward_Operation_Err(name, short, __VA_ARGS__);               \
  Register_FuseBackward_Operation_Err(name, ushort, __VA_ARGS__);              \
  Register_FuseBackward_Operation_Err(name, int, __VA_ARGS__);                 \
  Register_FuseBackward_Operation_Err(name, uint, __VA_ARGS__);                \
  Register_FuseBackward_Operation_Err(name, long, __VA_ARGS__);                \
  Register_FuseBackward_Operation_Err(name, ulong, __VA_ARGS__);               \
  Register_FuseBackward_Operation_Err(name, longlong, __VA_ARGS__);            \
  Register_FuseBackward_Operation_Err(name, ulonglong, __VA_ARGS__);

#define Register_FuseBackward_Operation_Err_Int(name, ...)                     \
  Register_FuseBackward_Operation_Err(name, bool, __VA_ARGS__);                \
  Register_FuseBackward_Operation_Err(name, byte, __VA_ARGS__);                \
  Register_FuseBackward_Operation_Err(name, ubyte, __VA_ARGS__);               \
  Register_FuseBackward_Operation_Err(name, short, __VA_ARGS__);               \
  Register_FuseBackward_Operation_Err(name, ushort, __VA_ARGS__);              \
  Register_FuseBackward_Operation_Err(name, int, __VA_ARGS__);                 \
  Register_FuseBackward_Operation_Err(name, uint, __VA_ARGS__);                \
  Register_FuseBackward_Operation_Err(name, long, __VA_ARGS__);                \
  Register_FuseBackward_Operation_Err(name, ulong, __VA_ARGS__);               \
  Register_FuseBackward_Operation_Err(name, longlong, __VA_ARGS__);            \
  Register_FuseBackward_Operation_Err(name, ulonglong, __VA_ARGS__);

#define Register_FuseBackward_Operation_Err_UnsupportTypes(name, ...)          \
  Register_FuseBackward_Operation_Err(name, cfloat, __VA_ARGS__);              \
  Register_FuseBackward_Operation_Err(name, cdouble, __VA_ARGS__);             \
  Register_FuseBackward_Operation_Err(name, clongdouble, __VA_ARGS__);         \
  Register_FuseBackward_Operation_Err(name, object, __VA_ARGS__);              \
  Register_FuseBackward_Operation_Err(name, string, __VA_ARGS__);              \
  Register_FuseBackward_Operation_Err(name, unicode, __VA_ARGS__);             \
  Register_FuseBackward_Operation_Err(name, void, __VA_ARGS__);                \
  Register_FuseBackward_Operation_Err(name, datetime, __VA_ARGS__);            \
  Register_FuseBackward_Operation_Err(name, timedelta, __VA_ARGS__);

#define Register_FuseBackward_Operation_Array(name, ...)                       \
  PyObject **(*name##_backward_fn_[])(                                         \
      Replicate0_With_Comma(Parameter_type_, __VA_ARGS__), PyObject **,        \
      int) = {name##_backward_bool,        name##_backward_byte,               \
              name##_backward_ubyte,       name##_backward_short,              \
              name##_backward_ushort,      name##_backward_int,                \
              name##_backward_uint,        name##_backward_long,               \
              name##_backward_ulong,       name##_backward_longlong,           \
              name##_backward_ulonglong,   name##_backward_float,              \
              name##_backward_double,      name##_backward_longdouble,         \
              name##_backward_cfloat,      name##_backward_cdouble,            \
              name##_backward_clongdouble, name##_backward_object,             \
              name##_backward_string,      name##_backward_unicode,            \
              name##_backward_void,        name##_backward_datetime,           \
              name##_backward_timedelta,   name##_backward_half};

#define Register_Backward_Operation_Method(name, ...)                          \
  PyObject **numboost_##name##_backward(                                       \
      Replicate0_With_Comma(Parameter_type, __VA_ARGS__), PyObject **out,      \
      int out_arr_len, int result_type) {                                      \
    assert(result_type <= NPY_HALF);                                           \
    PyObject **result =                                                        \
        name##_backward_fn_[result_type](__VA_ARGS__, out, out_arr_len);       \
    return result;                                                             \
  }

PyObject **numboost_sin_backward(PyObject *a, PyObject *b, PyObject **out,
                                 int out_arr_len, int result_type);
PyObject **numboost_cos_backward(PyObject *a, PyObject *b, PyObject **out,
                                 int out_arr_len, int result_type);
PyObject **numboost_tan_backward(PyObject *a, PyObject *b, PyObject **out,
                                 int out_arr_len, int result_type);
PyObject **numboost_arcsin_backward(PyObject *a, PyObject *b, PyObject **out,
                                    int out_arr_len, int result_type);
PyObject **numboost_arccos_backward(PyObject *a, PyObject *b, PyObject **out,
                                    int out_arr_len, int result_type);
PyObject **numboost_arctan_backward(PyObject *a, PyObject *b, PyObject **out,
                                    int out_arr_len, int result_type);
PyObject **numboost_sinh_backward(PyObject *a, PyObject *b, PyObject **out,
                                  int out_arr_len, int result_type);
PyObject **numboost_cosh_backward(PyObject *a, PyObject *b, PyObject **out,
                                  int out_arr_len, int result_type);
PyObject **numboost_tanh_backward(PyObject *a, PyObject *b, PyObject **out,
                                  int out_arr_len, int result_type);
PyObject **numboost_arcsinh_backward(PyObject *a, PyObject *b, PyObject **out,
                                     int out_arr_len, int result_type);
PyObject **numboost_arccosh_backward(PyObject *a, PyObject *b, PyObject **out,
                                     int out_arr_len, int result_type);
PyObject **numboost_arctanh_backward(PyObject *a, PyObject *b, PyObject **out,
                                     int out_arr_len, int result_type);
PyObject **numboost_exp_backward(PyObject *a, PyObject *b, PyObject **out,
                                 int out_arr_len, int result_type);
PyObject **numboost_log_backward(PyObject *a, PyObject *b, PyObject **out,
                                 int out_arr_len, int result_type);
PyObject **numboost_log10_backward(PyObject *a, PyObject *b, PyObject **out,
                                   int out_arr_len, int result_type);
PyObject **numboost_sqrt_backward(PyObject *a, PyObject *b, PyObject **out,
                                  int out_arr_len, int result_type);
PyObject **numboost_abs_backward(PyObject *a, PyObject *b, PyObject **out,
                                 int out_arr_len, int result_type);
PyObject **numboost_power_backward(PyObject *a, PyObject *power, PyObject *grad,
                                   PyObject **out, int out_arr_len,
                                   int result_type);
#endif // _NUMBOOST_AUTO_DIFF_UFUNC_BACKWARD_DEF_H_