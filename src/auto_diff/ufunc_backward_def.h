
#ifndef _NUMBOOST_AUTO_DIFF_UFUNC_BACKWARD_DEF_H_
#define _NUMBOOST_AUTO_DIFF_UFUNC_BACKWARD_DEF_H_
#include "../numboost_api.h"
#include "../op.h"

#define Register_FuseBackward_Operation_Set(name, inner_loop_body, ...)                                  \
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
#define Register_FuseBackward_Operation(name, type, result_type, inner_loop_body, ...)                       \
    PyArrayObject *name##_backward_fuse_##type(Replicate0_No_Comma(Parameter_type, __VA_ARGS__) int *op_set) \
    {                                                                                                        \
        PyArrayObject *result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a),                              \
                                                               PyArray_DIMS(a), result_type, 0);             \
        BinaryOperation_Universal(npy_##type, result, inner_loop_body, __VA_ARGS__);                         \
        return result;                                                                                       \
    }

extern PyArrayObject *(*sin_backward_arr[])(PyArrayObject *, PyArrayObject *, int *);

#endif // _NUMBOOST_AUTO_DIFF_UFUNC_BACKWARD_DEF_H_