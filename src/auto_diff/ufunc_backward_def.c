#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "ufunc_backward_def.h"

#define SinBackward_LoopBody_Float(type, inner_loop_size, result_ptr, stride_a_last, \
                                   stride_b_last, a_ptr, b_ptr)                      \
    for (npy_intp i = 0; i < inner_loop_size; ++i)                                   \
    {                                                                                \
        type val1 = npy_cosf(*((a_ptr + i * stride_a_last)));                        \
        type val2 = *((b_ptr + i * stride_b_last));                                  \
        *((result_ptr + i * stride_a_last)) = val1 * val2;                           \
    }

#define SinBackward_LoopBody(type, inner_loop_size, result_ptr, stride_a_last, \
                             stride_b_last, a_ptr, b_ptr)                      \
    for (npy_intp i = 0; i < inner_loop_size; ++i)                             \
    {                                                                          \
        printf("i: %lld\n", i);                                                \
        type val1 = npy_cos(*((a_ptr + i * stride_a_last)));                   \
        type val2 = *((b_ptr + i * stride_b_last));                            \
        *((result_ptr + i * stride_a_last)) = val1 * val2;                     \
    }

#define SinBackward_LoopBody_Half(type, inner_loop_size, result_ptr, stride_a_last, \
                                  stride_b_last, a_ptr, b_ptr)                      \
    for (npy_intp i = 0; i < inner_loop_size; ++i)                                  \
    {                                                                               \
        type val1 = npy_cos(*((a_ptr + i * stride_a_last)));                        \
        type val2 = *((b_ptr + i * stride_b_last));                                 \
        *((result_ptr + i * stride_a_last)) = val1 * val2;                          \
    }

Register_FuseBackward_Operation(sin, float, NPY_FLOAT, SinBackward_LoopBody_Float, a, b);
Register_FuseBackward_Operation(sin, half, NPY_HALF, SinBackward_LoopBody_Half, a, b);

PyArrayObject *sin_backward_fuse_double(PyArrayObject *a, PyArrayObject *b, int *op_set)
{
    PyArrayObject *result = (PyArrayObject *)(*(PyObject * (*)(int, npy_intp const *, PyArray_Descr *, int)) PyArray_API[184])((((PyArrayObject_fields *)(a))->nd), (((PyArrayObject_fields *)(a))->dimensions), (*(PyArray_Descr * (*)(int)) PyArray_API[45])(NPY_DOUBLE), 0);
    npy_double *a_data_ptr_saved = (npy_double *)((void *)((PyArrayObject_fields *)(a))->data);
    npy_double *b_data_ptr_saved = (npy_double *)((void *)((PyArrayObject_fields *)(b))->data);
    npy_intp _size = (*(npy_intp(*)(npy_intp const *, int))PyArray_API[158])((((PyArrayObject_fields *)(result))->dimensions), (((PyArrayObject_fields *)(result))->nd));
    npy_double *result_data_ptr_ = (npy_double *)((void *)((PyArrayObject_fields *)(result))->data);
    npy_intp i;
    _Pragma("omp parallel for schedule(static)") for (i = 0; i < _size; ++i)
    {
        npy_double val1 = npy_cos(a_data_ptr_saved[i]);
        npy_double val2 = b_data_ptr_saved[i];
        result_data_ptr_[i] = val1 * val2;
    }
    return result;
}
PyArrayObject *(*sin_backward_arr[])(PyArrayObject *, PyArrayObject *, int *) = {sin_backward_fuse_float, sin_backward_fuse_double, sin_backward_fuse_half};