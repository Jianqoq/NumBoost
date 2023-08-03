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

BroadcastFunction OperationPicker(int npy_type, int operation);

#define BroadCastLoop_Vec(type, strides_a, strides_b, shape,                   \
                          simd_type, simd_method, simd_op_method,              \
                          simd_store_method, a_ptr, b_ptr, result_ptr,         \
                          vec_size, vec_loop_size, remain_size, ndim, op,      \
                          simd_type_cast)                                      \
    {                                                                          \
        int max_dim = ndim - 1;                                                \
        int outer_start = max_dim - axis;                                      \
        npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);                \
        npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);            \
        npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);            \
        for (int i = 0; i < ndim; i++)                                         \
        {                                                                      \
            shape[i]--;                                                        \
            shape_copy[i] = 0;                                                 \
            indice_a_cache[i] = strides_a[i] * shape[i];                       \
            indice_b_cache[i] = strides_b[i] * shape[i];                       \
        }                                                                      \
        bool done = false;                                                     \
        int32_t *bigger_data_ptr_ = (int32_t *)a_ptr;                          \
        int32_t *to_broadcast_data_ptr_ = (int32_t *)b_ptr;                    \
        npy_intp a_last_stride = vec_size * strides_a[max_dim];                \
        npy_intp b_last_stride = vec_size * strides_b[max_dim];                \
        while (!done)                                                          \
        {                                                                      \
            done = true;                                                       \
            char *bigger_data_ptr_save = a_ptr;                                \
            char *to_broadcast_data_ptr_save = b_ptr;                          \
            for (int i = 0; i < vec_loop_size; i++)                            \
            {                                                                  \
                simd_type val1 = simd_method((const simd_type_cast *)(b_ptr)); \
                simd_type val2 = simd_method((const simd_type_cast *)a_ptr);   \
                simd_type result = simd_op_method(val1, val2);                 \
                simd_store_method((simd_type_cast *)result_ptr, result);       \
                result_ptr += vec_size;                                        \
                a_ptr += a_last_stride;                                        \
                b_ptr += b_last_stride;                                        \
            }                                                                  \
            for (int k = 0; k < remain_size; k++)                              \
            {                                                                  \
                type val1 = *((type *)(b_ptr));                                \
                type val2 = *((type *)(a_ptr));                                \
                *result_ptr = val1 op val2;                                    \
                result_ptr++;                                                  \
                a_ptr += strides_a[max_dim];                                   \
                b_ptr += strides_b[max_dim];                                   \
            }                                                                  \
            a_ptr = bigger_data_ptr_save;                                      \
            b_ptr = to_broadcast_data_ptr_save;                                \
            DEBUG_PRINT("Done loop\n");                                        \
            for (int j = outer_start; j >= 0; j--)                             \
            {                                                                  \
                if (shape_copy[j] < shape[j])                                  \
                {                                                              \
                    shape_copy[j]++;                                           \
                    done = false;                                              \
                    a_ptr += strides_a[j];                                     \
                    b_ptr += strides_b[j];                                     \
                    break;                                                     \
                }                                                              \
                else                                                           \
                {                                                              \
                    shape_copy[j] = 0;                                         \
                    a_ptr -= indice_a_cache[j];                                \
                    b_ptr -= indice_b_cache[j];                                \
                }                                                              \
            }                                                                  \
        }                                                                      \
        free(indice_a_cache);                                                  \
        free(indice_b_cache);                                                  \
        free(shape_copy);                                                      \
    }


void broadcast(PyArrayObject *a, PyArrayObject *b, PyObject *result, int type);

void* Generic_BroadCast(PyArrayObject *a, PyArrayObject *b, PyObject*array_result, int npy_type, int enum_op);