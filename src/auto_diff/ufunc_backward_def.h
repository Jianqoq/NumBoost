
#ifndef _NUMBOOST_AUTO_DIFF_UFUNC_BACKWARD_DEF_H_
#define _NUMBOOST_AUTO_DIFF_UFUNC_BACKWARD_DEF_H_
#include "../numboost_api.h"
#include "../op.h"

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
#define Register_FuseBackward_Operation(name, type, result_type,               \
                                        inner_loop_body_universal, ...)          \
  PyObject *name##_backward_fuse_##type(                                       \
      Replicate0_With_Comma(Parameter_type, __VA_ARGS__)) {                    \
    Perform_Universal_Operation(npy_##type, result_type,                       \
                                inner_loop_body_universal, __VA_ARGS__);       \
  }

#define Register_FuseBackward_Operation_Err(name, type, ...)                   \
  PyObject *name##_backward_fuse_##type(                                       \
      Replicate0_With_Comma(Parameter_type, __VA_ARGS__)) {                    \
        PyErr_SetString(PyExc_TypeError, Str(Not support for type.));          \
    return NULL;                                                               \
  }

#define Register_FuseBackward_Operation_FloatingTypes(                         \
    name, universal_loop_body, ...)                                            \
  Register_FuseBackward_Operation(name, float, NPY_FLOAT, universal_loop_body, \
                                  __VA_ARGS__);                                \
  Register_FuseBackward_Operation(name, double, NPY_DOUBLE,                    \
                                  universal_loop_body, __VA_ARGS__);           \
  Register_FuseBackward_Operation(name, longdouble, NPY_LONGDOUBLE,            \
                                  universal_loop_body, __VA_ARGS__);           \
  Register_FuseBackward_Operation(name, half, NPY_HALF, universal_loop_body,   \
                                  __VA_ARGS__);

#define Register_FuseBackward_Operation_IntergerTypes(                         \
    name, universal_loop_body, ...)                                            \
  Register_FuseBackward_Operation(name, bool, NPY_FLOAT, universal_loop_body,  \
                                  __VA_ARGS__);                                \
  Register_FuseBackward_Operation(name, byte, NPY_BYTE, universal_loop_body,   \
                                  __VA_ARGS__);                                \
  Register_FuseBackward_Operation(name, ubyte, NPY_UBYTE, universal_loop_body, \
                                  __VA_ARGS__);                                \
  Register_FuseBackward_Operation(name, short, NPY_SHORT, universal_loop_body, \
                                  __VA_ARGS__);                                \
  Register_FuseBackward_Operation(name, ushort, NPY_USHORT,                    \
                                  universal_loop_body, __VA_ARGS__);           \
  Register_FuseBackward_Operation(name, int, NPY_INT, universal_loop_body,     \
                                  __VA_ARGS__);                                \
  Register_FuseBackward_Operation(name, uint, NPY_UINT, universal_loop_body,   \
                                  __VA_ARGS__);                                \
  Register_FuseBackward_Operation(name, long, NPY_LONG, universal_loop_body,   \
                                  __VA_ARGS__);                                \
  Register_FuseBackward_Operation(name, ulong, NPY_ULONG, universal_loop_body, \
                                  __VA_ARGS__);                                \
  Register_FuseBackward_Operation(name, longlong, NPY_LONGLONG,                \
                                  universal_loop_body, __VA_ARGS__);           \
  Register_FuseBackward_Operation(name, ulonglong, NPY_ULONGLONG,              \
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
  PyObject *(*name##_backward_fusefn[])(                                       \
      Replicate0_With_Comma(Parameter_type_, __VA_ARGS__)) = {                 \
      name##_backward_fuse_bool,        name##_backward_fuse_byte,             \
      name##_backward_fuse_ubyte,       name##_backward_fuse_short,            \
      name##_backward_fuse_ushort,      name##_backward_fuse_int,              \
      name##_backward_fuse_uint,        name##_backward_fuse_long,             \
      name##_backward_fuse_ulong,       name##_backward_fuse_longlong,         \
      name##_backward_fuse_ulonglong,   name##_backward_fuse_float,            \
      name##_backward_fuse_double,      name##_backward_fuse_longdouble,       \
      name##_backward_fuse_cfloat,      name##_backward_fuse_cdouble,          \
      name##_backward_fuse_clongdouble, name##_backward_fuse_object,           \
      name##_backward_fuse_string,      name##_backward_fuse_unicode,          \
      name##_backward_fuse_void,        name##_backward_fuse_datetime,         \
      name##_backward_fuse_timedelta,   name##_backward_fuse_half};

extern PyObject *(*sin_backward_fusefn[])(PyObject *a, PyObject *b);
extern PyObject *(*cos_backward_fusefn[])(PyObject *a, PyObject *b);
extern PyObject *(*tan_backward_fusefn[])(PyObject *a, PyObject *b);
extern PyObject *(*arcsin_backward_fusefn[])(PyObject *a, PyObject *b);
extern PyObject *(*arccos_backward_fusefn[])(PyObject *a, PyObject *b);
extern PyObject *(*arctan_backward_fusefn[])(PyObject *a, PyObject *b);
extern PyObject *(*sinh_backward_fusefn[])(PyObject *a, PyObject *b);
extern PyObject *(*cosh_backward_fusefn[])(PyObject *a, PyObject *b);
extern PyObject *(*tanh_backward_fusefn[])(PyObject *a, PyObject *b);
extern PyObject *(*arcsinh_backward_fusefn[])(PyObject *a, PyObject *b);
extern PyObject *(*arccosh_backward_fusefn[])(PyObject *a, PyObject *b);
extern PyObject *(*arctanh_backward_fusefn[])(PyObject *a, PyObject *b);
extern PyObject *(*exp_backward_fusefn[])(PyObject *a, PyObject *b);
extern PyObject *(*log_backward_fusefn[])(PyObject *a, PyObject *b);
extern PyObject *(*log10_backward_fusefn[])(PyObject *a, PyObject *b);
extern PyObject *(*sqrt_backward_fusefn[])(PyObject *a, PyObject *b);
extern PyObject *(*abs_backward_fusefn[])(PyObject *a, PyObject *b);
extern PyObject *(*power_backward_fusefn[])(PyObject *a, PyObject *b,
                                            PyObject *c);
#endif // _NUMBOOST_AUTO_DIFF_UFUNC_BACKWARD_DEF_H_