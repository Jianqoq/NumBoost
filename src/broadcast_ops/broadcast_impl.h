#ifndef _BROADCAST_H
#define _BROADCAST_H
#include "../shape.h"
#include <immintrin.h>
#include "../type_convertor/type_convertor.h"
#include "../op.h"
#include "omp.h"
#include "numpy/npy_math.h"
#include "../numboost_api.h"

inline npy_intp dot_prod(npy_intp *strides, npy_intp *indice, int ndim)
{
    npy_intp index_ = 0;
    for (int i = 0; i < ndim; i++)
        index_ += strides[i] * indice[i];
    return index_;
}

PyArrayObject *numboost_broadcast(PyArrayObject *, PyArrayObject *, int);

#define ThrowError(op_enum, msg)                  \
    case op_enum:                                 \
        PyErr_SetString(PyExc_RuntimeError, msg); \
        return NULL;

#define BroadCast(a, b, op) _BroadCast(a, b, op);

#define Standard_Inner_Loop(type, op, inner_loop_size, stride_a, stride_b, a_ptr, b_ptr, result_ptr) \
    {                                                                                                \
        for (int i = 0; i < inner_loop_size; i++)                                                    \
        {                                                                                            \
            type val1 = *(a_ptr + i * stride_a);                                                     \
            type val2 = *(b_ptr + i * stride_b);                                                     \
            *(result_ptr + i) = op(val1, val2);                                                      \
        }                                                                                            \
    }

#define Shift_Inner_Loop(type, op, inner_loop_size, stride_a, stride_b, a_ptr, b_ptr, result_ptr) \
    {                                                                                             \
        for (int i = 0; i < inner_loop_size; i++)                                                 \
        {                                                                                         \
            type val1 = *(a_ptr + i * stride_a);                                                  \
            type val2 = *(b_ptr + i * stride_b);                                                  \
            *(result_ptr + i) = op(val1, val2);                                                   \
        }                                                                                         \
    }

#define Half_Inner_Loop(type, op, inner_loop_size, stride_a, stride_b, a_ptr, b_ptr, result_ptr) \
    {                                                                                            \
        for (int i = 0; i < inner_loop_size; i++)                                                \
        {                                                                                        \
            npy_half val1 = *(a_ptr + i * stride_a);                                             \
            npy_half val2 = *(b_ptr + i * stride_b);                                             \
            *(result_ptr + i) = op(val1, val2);                                                  \
        }                                                                                        \
    }

#define Pow_Inner_Loop(type, op, inner_loop_size, stride_a, stride_b, a_ptr, b_ptr, result_ptr) \
    {                                                                                           \
        for (int i = 0; i < inner_loop_size; i++)                                               \
        {                                                                                       \
            type val1 = *(a_ptr + i * stride_a);                                                \
            type val2 = *(b_ptr + i * stride_b);                                                \
            *(result_ptr + i) = npy_pow((val1), (val2));                                        \
        }                                                                                       \
    }

#define Powf_Inner_Loop(type, op, inner_loop_size, stride_a, stride_b, a_ptr, b_ptr, result_ptr) \
    {                                                                                            \
        for (int i = 0; i < inner_loop_size; i++)                                                \
        {                                                                                        \
            type val1 = *(a_ptr + i * stride_a);                                                 \
            type val2 = *(b_ptr + i * stride_b);                                                 \
            *(result_ptr + i) = npy_powf((val1), (val2));                                        \
        }                                                                                        \
    }

#define Powl_Inner_Loop(type, op, inner_loop_size, stride_a, stride_b, a_ptr, b_ptr, result_ptr) \
    {                                                                                            \
        for (int i = 0; i < inner_loop_size; i++)                                                \
        {                                                                                        \
            type val1 = *(a_ptr + i * stride_a);                                                 \
            type val2 = *(b_ptr + i * stride_b);                                                 \
            *(result_ptr + i) = npy_powl((val1), (val2));                                        \
        }                                                                                        \
    }

#define NestedLoop_Maintainer(last_index, current_shape_process, shape, a_ptr, b_ptr, indice_a_cache, indice_b_cache) \
    for (int j = last_index; j >= 0; j--)                                                                             \
    {                                                                                                                 \
        if (current_shape_process[j] < shape[j])                                                                      \
        {                                                                                                             \
            current_shape_process[j]++;                                                                               \
            a_ptr += strides_a[j];                                                                                    \
            b_ptr += strides_b[j];                                                                                    \
            break;                                                                                                    \
        }                                                                                                             \
        else                                                                                                          \
        {                                                                                                             \
            current_shape_process[j] = 0;                                                                             \
            a_ptr -= indice_a_cache[j];                                                                               \
            b_ptr -= indice_b_cache[j];                                                                               \
        }                                                                                                             \
    }

#define Prepare_For_ResultPtr(a_ptr, b_ptr, result_ptr, type) \
    type *b_data_ptr_saved = (type *)b_ptr;                   \
    type *a_data_ptr_saved = (type *)a_ptr;                   \
    type *result_data_ptr_ = (type *)result_ptr;              \
    type *result_data_ptr_cpy = (type *)result_ptr;

#define Change_Strides_Based_On_Ctype(a_strides, b_strides, type, ndim) \
    for (int i = 0; i < ndim; i++)                                      \
    {                                                                   \
        a_strides[i] /= sizeof(type);                                   \
        b_strides[i] /= sizeof(type);                                   \
    }

#define Prepare_For_Parallel(outer_loop_size, inner_loop_size, a_strides, b_strides, shape, result_ptr1, result_ptr2, result_ptr3) \
    int thread_id = omp_get_thread_num();                                                                                          \
    npy_intp start_index = thread_id * (outer_loop_size / num_threads) + min(thread_id, outer_loop_size % num_threads);            \
    npy_intp end_index = start_index + outer_loop_size / num_threads + (thread_id < outer_loop_size % num_threads ? 1 : 0);        \
    result_ptr1 = result_ptr2;                                                                                                     \
    result_ptr2 += (end_index - start_index) * inner_loop_size;                                                                    \
    npy_intp prd = result_ptr1 - result_ptr3;                                                                                      \
    npy_intp *current_shape_process = (npy_intp *)calloc(ndim, sizeof(npy_intp));                                                  \
    for (int j = max_dim; j >= 0; j--)                                                                                             \
    {                                                                                                                              \
        current_shape_process[j] = prd % (shape[j] + 1);                                                                           \
        prd /= (shape[j] + 1);                                                                                                     \
        a_data_ptr_saved += current_shape_process[j] * a_strides[j];                                                               \
        b_data_ptr_saved += current_shape_process[j] * b_strides[j];                                                               \
    }

#define Fill_Shape_and_Indice(shape_, shape_copy_, indice_a_, indice_b_, strides_a_, strides_b_, ndim) \
    for (int i = 0; i < ndim; i++)                                                                     \
    {                                                                                                  \
        shape_[i]--;                                                                                   \
        shape_copy_[i] = 0;                                                                            \
        indice_a_[i] = strides_a_[i] * shape_[i];                                                      \
        indice_b_[i] = strides_b_[i] * shape_[i];                                                      \
    }

#define n(x) #x

#define Register_Broadcast_Operation(type, suffix, op, inner_loop_body)                                                                               \
    static PyArrayObject *Broadcast_Standard_##type##_##suffix(PyArrayObject *a_, PyArrayObject *b_, int op_enum, int result_type)                    \
    {                                                                                                                                                 \
        int a_ndim = PyArray_NDIM(a_);                                                                                                                \
        int b_ndim = PyArray_NDIM(b_);                                                                                                                \
        npy_intp *a_shape = NULL;                                                                                                                     \
        npy_intp *b_shape = NULL;                                                                                                                     \
        npy_intp *handler = NULL;                                                                                                                     \
        int ndim = 0;                                                                                                                                 \
        if (a_ndim < b_ndim)                                                                                                                          \
        {                                                                                                                                             \
            ndim = b_ndim;                                                                                                                            \
            b_shape = PyArray_SHAPE(b_);                                                                                                              \
            if (!shape_isbroadcastable_to_ex(PyArray_SHAPE(a_), b_shape, a_ndim, b_ndim, &a_shape))                                                   \
            {                                                                                                                                         \
                PyErr_SetString(PyExc_ValueError, "Cannot broadcast shapes");                                                                         \
                return NULL;                                                                                                                          \
            }                                                                                                                                         \
            handler = a_shape;                                                                                                                        \
        }                                                                                                                                             \
        else                                                                                                                                          \
        {                                                                                                                                             \
            ndim = a_ndim;                                                                                                                            \
            a_shape = PyArray_SHAPE(a_);                                                                                                              \
            if (!shape_isbroadcastable_to_ex(PyArray_SHAPE(b_), a_shape, b_ndim, a_ndim, &b_shape))                                                   \
            {                                                                                                                                         \
                PyErr_SetString(PyExc_ValueError, "Cannot broadcast shapes");                                                                         \
                return NULL;                                                                                                                          \
            }                                                                                                                                         \
            handler = b_shape;                                                                                                                        \
        }                                                                                                                                             \
        npy_intp stride_last = PyArray_STRIDE((const PyArrayObject *)a_, a_ndim - 1);                                                                 \
        npy_intp *strides_a = NULL, *strides_b = NULL, *shape = NULL;                                                                                 \
        predict_broadcast_shape(a_shape, b_shape, ndim, &shape);                                                                                      \
        preprocess_strides(a_shape, stride_last, ndim, &strides_a);                                                                                   \
        preprocess_strides(b_shape, stride_last, ndim, &strides_b);                                                                                   \
        PyArrayObject *result = (PyArrayObject *)PyArray_EMPTY(ndim, (npy_intp const *)shape, result_type, 0);                                        \
        if (result == NULL)                                                                                                                           \
        {                                                                                                                                             \
            return NULL;                                                                                                                              \
        }                                                                                                                                             \
        char *result_data_ptr = (char *)PyArray_DATA(result);                                                                                         \
        char *a_data_ptr = (char *)PyArray_DATA((PyArrayObject *)a_);                                                                                 \
        char *b_data_ptr = (char *)PyArray_DATA((PyArrayObject *)b_);                                                                                 \
        npy_intp inner_loop_size = 1;                                                                                                                 \
        int axis = 0;                                                                                                                                 \
        for (int i = ndim - 1; i >= 0; i--)                                                                                                           \
        {                                                                                                                                             \
            if (a_shape[i] == b_shape[i])                                                                                                             \
            {                                                                                                                                         \
                axis++;                                                                                                                               \
                inner_loop_size *= a_shape[i];                                                                                                        \
            }                                                                                                                                         \
            else                                                                                                                                      \
            {                                                                                                                                         \
                if (i == ndim - 1)                                                                                                                    \
                {                                                                                                                                     \
                    inner_loop_size *= shape[ndim - 1];                                                                                               \
                    axis++;                                                                                                                           \
                }                                                                                                                                     \
                break;                                                                                                                                \
            }                                                                                                                                         \
        }                                                                                                                                             \
        npy_intp outer_loop_size = PyArray_SIZE(result) / inner_loop_size;                                                                            \
        int max_dim = ndim - 1;                                                                                                                       \
        int outer_start = max_dim - axis;                                                                                                             \
        npy_intp *shape_copy = (npy_intp *)malloc(sizeof(npy_intp) * ndim);                                                                           \
        npy_intp *indice_a_cache = (npy_intp *)malloc(sizeof(npy_intp) * ndim);                                                                       \
        npy_intp *indice_b_cache = (npy_intp *)malloc(sizeof(npy_intp) * ndim);                                                                       \
        npy_##type *b_data_ptr_saved = (npy_##type *)b_data_ptr;                                                                                      \
        npy_##type *a_data_ptr_saved = (npy_##type *)a_data_ptr;                                                                                      \
        npy_##type *result_data_ptr_ = (npy_##type *)result_data_ptr;                                                                                 \
        npy_##type *result_data_ptr_cpy = (npy_##type *)result_data_ptr;                                                                              \
        for (int i = 0; i < ndim; i++)                                                                                                                \
        {                                                                                                                                             \
            strides_a[i] /= sizeof(npy_##type);                                                                                                       \
            strides_b[i] /= sizeof(npy_##type);                                                                                                       \
        }                                                                                                                                             \
        for (int i = 0; i < ndim; i++)                                                                                                                \
        {                                                                                                                                             \
            shape[i]--;                                                                                                                               \
            shape_copy[i] = 0;                                                                                                                        \
            indice_a_cache[i] = strides_a[i] * shape[i];                                                                                              \
            indice_b_cache[i] = strides_b[i] * shape[i];                                                                                              \
        }                                                                                                                                             \
        npy_intp k = 0;                                                                                                                               \
        npy_intp num_threads = outer_loop_size < omp_get_max_threads() ? outer_loop_size : omp_get_max_threads();                                     \
        npy_##type **result_ptr_ = (npy_##type **)malloc(sizeof(npy_##type *) * num_threads);                                                         \
        npy_intp **current_shape_process_ = (npy_intp **)malloc(sizeof(npy_intp *) * num_threads);                                                    \
        for (int id = 0; id < num_threads; id++)                                                                                                      \
        {                                                                                                                                             \
            npy_intp start_index = id * (outer_loop_size / num_threads) + min(id, outer_loop_size % num_threads);                                     \
            npy_intp end_index = start_index + outer_loop_size / num_threads + (id < outer_loop_size % num_threads);                                  \
            result_ptr_[id] = result_data_ptr_cpy;                                                                                                    \
            result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;                                                                       \
            npy_intp prd = result_ptr_[id] - (npy_##type *)result_data_ptr;                                                                           \
            npy_intp *current_shape_process = (npy_intp *)calloc(ndim, sizeof(npy_intp));                                                             \
            for (int j = max_dim; j >= 0; j--)                                                                                                        \
            {                                                                                                                                         \
                current_shape_process[j] = prd % (shape[j] + 1);                                                                                      \
                prd /= (shape[j] + 1);                                                                                                                \
            }                                                                                                                                         \
            current_shape_process_[id] = current_shape_process;                                                                                       \
        }                                                                                                                                             \
        npy_intp stride_a_last = strides_a[max_dim];                                                                                                  \
        npy_intp stride_b_last = strides_b[max_dim];                                                                                                  \
        _Pragma("omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)")                           \
        {                                                                                                                                             \
            int thread_id = omp_get_thread_num();                                                                                                     \
            result_data_ptr_ = result_ptr_[thread_id];                                                                                                \
            npy_intp *current_process = current_shape_process_[thread_id];                                                                            \
            for (int j = max_dim; j >= 0; j--)                                                                                                        \
            {                                                                                                                                         \
                a_data_ptr_saved += current_process[j] * strides_a[j];                                                                                \
                b_data_ptr_saved += current_process[j] * strides_b[j];                                                                                \
            }                                                                                                                                         \
            _Pragma("omp for schedule(static)") for (k = 0; k < outer_loop_size; k++)                                                                 \
            {                                                                                                                                         \
                inner_loop_body(npy_##type, op, inner_loop_size, stride_a_last, stride_b_last, a_data_ptr_saved, b_data_ptr_saved, result_data_ptr_); \
                result_data_ptr_ += inner_loop_size;                                                                                                  \
                for (int j = outer_start; j >= 0; j--)                                                                                                \
                {                                                                                                                                     \
                    if (current_process[j] < shape[j])                                                                                                \
                    {                                                                                                                                 \
                        current_process[j]++;                                                                                                         \
                        a_data_ptr_saved += strides_a[j];                                                                                             \
                        b_data_ptr_saved += strides_b[j];                                                                                             \
                        break;                                                                                                                        \
                    }                                                                                                                                 \
                    else                                                                                                                              \
                    {                                                                                                                                 \
                        current_process[j] = 0;                                                                                                       \
                        a_data_ptr_saved -= indice_a_cache[j];                                                                                        \
                        b_data_ptr_saved -= indice_b_cache[j];                                                                                        \
                    }                                                                                                                                 \
                }                                                                                                                                     \
            }                                                                                                                                         \
            free(current_process);                                                                                                                    \
        }                                                                                                                                             \
        free(result_ptr_);                                                                                                                            \
        free(current_shape_process_);                                                                                                                 \
        free(indice_a_cache);                                                                                                                         \
        free(indice_b_cache);                                                                                                                         \
        free(shape_copy);                                                                                                                             \
        free(strides_a);                                                                                                                              \
        free(strides_b);                                                                                                                              \
        free(shape);                                                                                                                                  \
        free(handler);                                                                                                                                \
        return result;                                                                                                                                \
    }

#define Register_Broadcast_Operation_All_Err(type)   \
    Register_Broadcast_Operation_Err(type, add_);    \
    Register_Broadcast_Operation_Err(type, sub_);    \
    Register_Broadcast_Operation_Err(type, mul_);    \
    Register_Broadcast_Operation_Err(type, div_);    \
    Register_Broadcast_Operation_Err(type, mod_);    \
    Register_Broadcast_Operation_Err(type, lshift_); \
    Register_Broadcast_Operation_Err(type, rshift_); \
    Register_Broadcast_Operation_Err(type, pow_);

#define Register_Broadcast_Operation_Err(type, suffix)                                             \
    static PyArrayObject *Broadcast_Standard_##type##_##suffix(PyArrayObject *a, PyArrayObject *b, \
                                                               int op_enum, int result_type)       \
    {                                                                                              \
        const char *string[] = {"Operation not supported for", #type, "type"};                     \
        size_t length = strlen(string[0]) + strlen(string[1]) + strlen(string[2]) + 1;             \
        char *string_cat = (char *)malloc(length);                                                 \
        strcpy(string_cat, string[0]);                                                             \
        strcat(string_cat, string[1]);                                                             \
        strcat(string_cat, string[2]);                                                             \
        PyErr_SetString(PyExc_TypeError, string_cat);                                              \
        free(string_cat);                                                                          \
        return NULL;                                                                               \
    }

#endif