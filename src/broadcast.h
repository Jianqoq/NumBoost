#ifndef SHAPE_H
#define SHAPE_H
#include "shape.h"
#endif

#ifndef SIMD_H
#define SIMD_H
#include <immintrin.h>
#endif

typedef void (*BroadcastFunction)(char *a_data_ptr, char *b_data_ptr,
                                  char *result_data_ptr, npy_intp inner_loop_size,
                                  npy_intp *shape, npy_intp *strides_a,
                                  npy_intp *strides_b, int ndim,
                                  int axis, npy_intp left_prod);

inline npy_intp dot_prod(npy_intp *strides, npy_intp *indice, int ndim)
{
    npy_intp index_ = 0;
    for (int i = 0; i < ndim; i++)
        index_ += strides[i] * indice[i];
    return index_;
}

BroadcastFunction Broadcast_OperationPicker(int npy_type, int operation);

void _BroadCast(PyArrayObject *a, PyArrayObject *b, PyObject **array_result, int npy_type, int enum_op);

void BroadCast_Core(int ndim, npy_intp *shape_a, npy_intp *shape_b, PyArrayObject *bigger, PyArrayObject *to_broadcast, PyArrayObject *result,
                    npy_intp *strides_a, npy_intp *strides_b, npy_intp *shape, int npy_type, int enum_op);

#define BroadCast(a, b, result, op, type) _BroadCast(a, b, result, type, op);
