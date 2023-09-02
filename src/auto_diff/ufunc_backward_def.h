
#ifndef _NUMBOOST_AUTO_DIFF_UFUNC_BACKWARD_DEF_H_
#define _NUMBOOST_AUTO_DIFF_UFUNC_BACKWARD_DEF_H_
#include "../numboost_api.h"
#include "../op.h"

#define Register_FuseBackward_Operation_Set(name, inner_loop_body, ...)                              \
    Register_FuseBackward_Operation(name, bool, NPY_BOOL, inner_loop_body, __VA_ARGS__);             \
    Register_FuseBackward_Operation(name, byte, NPY_BYTE, inner_loop_body, __VA_ARGS__);             \
    Register_FuseBackward_Operation(name, ubyte, NPY_UBYTE, inner_loop_body, __VA_ARGS__);           \
    Register_FuseBackward_Operation(name, short, NPY_SHORT, inner_loop_body, __VA_ARGS__);           \
    Register_FuseBackward_Operation(name, ushort, NPY_USHORT, inner_loop_body, __VA_ARGS__);         \
    Register_FuseBackward_Operation(name, int, NPY_INT, inner_loop_body, __VA_ARGS__);               \
    Register_FuseBackward_Operation(name, uint, NPY_UINT, inner_loop_body, __VA_ARGS__);             \
    Register_FuseBackward_Operation(name, long, NPY_LONG, inner_loop_body, __VA_ARGS__);             \
    Register_FuseBackward_Operation(name, ulong, NPY_ULONG, inner_loop_body, __VA_ARGS__);           \
    Register_FuseBackward_Operation(name, longlong, NPY_LONGLONG, inner_loop_body, __VA_ARGS__);     \
    Register_FuseBackward_Operation(name, ulonglong, NPY_ULONGLONG, inner_loop_body, __VA_ARGS__);   \
    Register_FuseBackward_Operation(name, float, NPY_FLOAT, inner_loop_body, __VA_ARGS__);           \
    Register_FuseBackward_Operation(name, double, NPY_DOUBLE, inner_loop_body, __VA_ARGS__);         \
    Register_FuseBackward_Operation(name, longdouble, NPY_LONGDOUBLE, inner_loop_body, __VA_ARGS__); \
    Register_FuseBackward_Operation(name, half, NPY_HALF, inner_loop_body, __VA_ARGS__);

/*! @function
@param  name  backward_fn_name [ sin, cos, etc.]
@param  type  C-type [ long, int, float, double, etc.]
@param  inner_loop_body     the macro the user defined
@param  ...                the tensors that need to be fused
 */
#define Register_FuseBackward_Operation(name, type, result_type, inner_loop_body_universal, inner_loop_body_seq, ...) \
    PyArrayObject *name##_backward_fuse_##type(Replicate0_With_Comma(Parameter_type, __VA_ARGS__))                    \
    {                                                                                                                 \
        PyArrayObject *result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(_First_(__VA_ARGS__)),                    \
                                                               PyArray_DIMS(_First_(__VA_ARGS__)), result_type, 0);   \
        Replicate0_No_Comma(Handlers, __VA_ARGS__);                                                                   \
        Replicate2(Correct_Type, result_type, type, __VA_ARGS__);                                                     \
        if (Replicate_Or(Contiguous_OrNot, __VA_ARGS__))                                                              \
            BinaryOperation_Universal(npy_##type, result, inner_loop_body_universal, __VA_ARGS__);                    \
        else                                                                                                          \
            BinaryOperation_Sequential(npy_##type, result, inner_loop_body_seq, __VA_ARGS__);                         \
        Replicate0_No_Comma(Free_Array, __VA_ARGS__);                                                                 \
        return result;                                                                                                \
    }

#define Register_FuseBackward_Operation_Err(name, type, ...)                                       \
    PyArrayObject *name##_backward_fuse_##type(Replicate0_With_Comma(Parameter_type, __VA_ARGS__)) \
    {                                                                                              \
        PyErr_SetString(PyExc_TypeError, Str(Not support for type.));                              \
        return NULL;                                                                               \
    }

#define Register_FuseBackward_Operation_Err_Int(name, ...)            \
    Register_FuseBackward_Operation_Err(name, bool, __VA_ARGS__);     \
    Register_FuseBackward_Operation_Err(name, byte, __VA_ARGS__);     \
    Register_FuseBackward_Operation_Err(name, ubyte, __VA_ARGS__);    \
    Register_FuseBackward_Operation_Err(name, short, __VA_ARGS__);    \
    Register_FuseBackward_Operation_Err(name, ushort, __VA_ARGS__);   \
    Register_FuseBackward_Operation_Err(name, int, __VA_ARGS__);      \
    Register_FuseBackward_Operation_Err(name, uint, __VA_ARGS__);     \
    Register_FuseBackward_Operation_Err(name, long, __VA_ARGS__);     \
    Register_FuseBackward_Operation_Err(name, ulong, __VA_ARGS__);    \
    Register_FuseBackward_Operation_Err(name, longlong, __VA_ARGS__); \
    Register_FuseBackward_Operation_Err(name, ulonglong, __VA_ARGS__);

#define Register_FuseBackward_Operation_Err_Extra(name, ...)             \
    Register_FuseBackward_Operation_Err(name, cfloat, __VA_ARGS__);      \
    Register_FuseBackward_Operation_Err(name, cdouble, __VA_ARGS__);     \
    Register_FuseBackward_Operation_Err(name, clongdouble, __VA_ARGS__); \
    Register_FuseBackward_Operation_Err(name, object, __VA_ARGS__);      \
    Register_FuseBackward_Operation_Err(name, string, __VA_ARGS__);      \
    Register_FuseBackward_Operation_Err(name, unicode, __VA_ARGS__);     \
    Register_FuseBackward_Operation_Err(name, void, __VA_ARGS__);        \
    Register_FuseBackward_Operation_Err(name, datetime, __VA_ARGS__);    \
    Register_FuseBackward_Operation_Err(name, timedelta, __VA_ARGS__);

#define Register_FuseBackward_Operation_Array(name, ...)                                                                                                                                                                         \
    PyArrayObject *(*name##_backward_fusefn[])(Replicate0_With_Comma(Parameter_type_, __VA_ARGS__)) = {name##_backward_fuse_bool, name##_backward_fuse_byte, name##_backward_fuse_ubyte, name##_backward_fuse_short,             \
                                                                                                       name##_backward_fuse_ushort, name##_backward_fuse_int, name##_backward_fuse_uint, name##_backward_fuse_long,              \
                                                                                                       name##_backward_fuse_ulong, name##_backward_fuse_longlong, name##_backward_fuse_ulonglong,                                \
                                                                                                       name##_backward_fuse_float, name##_backward_fuse_double, name##_backward_fuse_longdouble, name##_backward_fuse_cfloat,    \
                                                                                                       name##_backward_fuse_cdouble, name##_backward_fuse_clongdouble, name##_backward_fuse_object, name##_backward_fuse_string, \
                                                                                                       name##_backward_fuse_unicode, name##_backward_fuse_void, name##_backward_fuse_datetime, name##_backward_fuse_timedelta,   \
                                                                                                       name##_backward_fuse_half};

extern PyArrayObject *(*sin_backward_fusefn[])(PyArrayObject *a, PyArrayObject *b);
extern PyArrayObject *(*cos_backward_fusefn[])(PyArrayObject *a, PyArrayObject *b);
extern PyArrayObject *(*tan_backward_fusefn[])(PyArrayObject *a, PyArrayObject *b);
extern PyArrayObject *(*arcsin_backward_fusefn[])(PyArrayObject *a, PyArrayObject *b);
extern PyArrayObject *(*arccos_backward_fusefn[])(PyArrayObject *a, PyArrayObject *b);
extern PyArrayObject *(*arctan_backward_fusefn[])(PyArrayObject *a, PyArrayObject *b);
extern PyArrayObject *(*sinh_backward_fusefn[])(PyArrayObject *a, PyArrayObject *b);
extern PyArrayObject *(*cosh_backward_fusefn[])(PyArrayObject *a, PyArrayObject *b);
extern PyArrayObject *(*tanh_backward_fusefn[])(PyArrayObject *a, PyArrayObject *b);
extern PyArrayObject *(*arcsinh_backward_fusefn[])(PyArrayObject *a, PyArrayObject *b);
extern PyArrayObject *(*arccosh_backward_fusefn[])(PyArrayObject *a, PyArrayObject *b);
extern PyArrayObject *(*arctanh_backward_fusefn[])(PyArrayObject *a, PyArrayObject *b);
extern PyArrayObject *(*exp_backward_fusefn[])(PyArrayObject *a, PyArrayObject *b);
extern PyArrayObject *(*log_backward_fusefn[])(PyArrayObject *a, PyArrayObject *b);
#endif // _NUMBOOST_AUTO_DIFF_UFUNC_BACKWARD_DEF_H_