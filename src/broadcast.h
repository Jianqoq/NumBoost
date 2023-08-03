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

#define BroadCast_Core(ndim, shape_a, shape_b, bigger, to_broadcast, result, strides_a, strides_b, shape, type, op, \
                       simd_type, simd_load, simd_op, simd_store, vec_size, simd_type_cast, npy_type, enum_op)      \
    {                                                                                                               \
        DEBUG_PRINT("BroadCast_Core\n");                                                                            \
        char *result_data_ptr = (char *)PyArray_DATA(result);                                             \
        DEBUG_PRINT("Result data ptr: %p\n", result_data_ptr);                                                      \
        char *a_data_ptr = (char *)PyArray_DATA((PyArrayObject *)bigger);                                           \
        DEBUG_PRINT("A data ptr: %p\n", a_data_ptr);                                                                \
        char *b_data_ptr = (char *)PyArray_DATA((PyArrayObject *)to_broadcast);                                     \
        DEBUG_PRINT("Shape a: ");                                                                                   \
        for (int i = 0; i < ndim; i++)                                                                              \
            DEBUG_PRINT("%d ", shape_a[i]);                                                                         \
        DEBUG_PRINT("\n");                                                                                          \
        DEBUG_PRINT("Shape b: ");                                                                                   \
        for (int i = 0; i < ndim; i++)                                                                              \
            DEBUG_PRINT("%d ", shape_b[i]);                                                                         \
        DEBUG_PRINT("\n");                                                                                          \
        npy_intp prod = 1;                                                                                          \
        int axis = 0;                                                                                               \
        bool vectorizable = true;                                                                                   \
        for (int i = ndim - 1; i >= 0; i--)                                                                         \
        {                                                                                                           \
            if (shape_a[i] == shape_b[i])                                                                           \
            {                                                                                                       \
                axis++;                                                                                             \
                prod *= shape_a[i];                                                                                 \
            }                                                                                                       \
            else                                                                                                    \
            {                                                                                                       \
                if (i == ndim - 1)                                                                                  \
                {                                                                                                   \
                    prod *= shape[ndim - 1];                                                                        \
                    vectorizable = false;                                                                           \
                    axis++;                                                                                         \
                }                                                                                                   \
                break;                                                                                              \
            }                                                                                                       \
        }                                                                                                           \
        npy_intp left_prod = PyArray_SIZE(result) / prod;                                                           \
        DEBUG_PRINT("Axis: %d\n", axis);                                                                            \
        if (!vectorizable || vec_size == -1)                                                                        \
        {                                                                                                           \
            OperationPicker(npy_type, enum_op)(a_data_ptr, b_data_ptr,                                              \
                                               result_data_ptr, prod,                                               \
                                               shape, strides_a,                                                    \
                                               strides_b, ndim,                                                     \
                                               axis, left_prod);                                                    \
        }                                                                                                           \
        else                                                                                                        \
        {                                                                                                           \
            int vec_loop_size = prod / vec_size;                                                                    \
            int remain_size = prod % vec_size;                                                                      \
            BroadCastLoop_Vec(type, strides_a, strides_b, shape, simd_type, simd_load, simd_op, simd_store,         \
                              a_data_ptr, b_data_ptr,                                                               \
                              result_data_ptr, vec_size, vec_loop_size, remain_size, ndim, op, simd_type_cast);     \
        }                                                                                                           \
    }

void broadcast(PyArrayObject *a, PyArrayObject *b, PyObject *result, int type);

#define _BroadCast(a, b, array_result, op, type, npy_type, simd_type, simd_load, simd_op, simd_store, vec_size,                             \
                   simd_type_cast, enum_op)                                                                                                 \
    {                                                                                                                                       \
        DEBUG_PRINT("BroadCast\n");                                                                                                         \
        int a_ndim = PyArray_NDIM(a);                                                                                                       \
        int b_ndim = PyArray_NDIM(b);                                                                                                       \
        npy_intp *a_shape = PyArray_SHAPE(a);                                                                                               \
        npy_intp *b_shape = PyArray_SHAPE(b);                                                                                               \
        npy_intp *to_broadcast_shape_pad_one = NULL;                                                                                        \
        npy_intp *bigger_shape = NULL;                                                                                                      \
        PyArrayObject *to_broadcast = NULL;                                                                                                 \
        PyArrayObject *bigger = NULL;                                                                                                       \
        int ndim = 0;                                                                                                                       \
        if (a_ndim < b_ndim)                                                                                                                \
        {                                                                                                                                   \
            bigger_shape = b_shape;                                                                                                         \
            to_broadcast = a;                                                                                                               \
            bigger = b;                                                                                                                     \
            ndim = PyArray_NDIM(b);                                                                                                         \
            if (!shape_isbroadcastable_to_ex(a_shape, b_shape, a_ndim, b_ndim, &to_broadcast_shape_pad_one))                                \
            {                                                                                                                               \
                PyErr_SetString(PyExc_ValueError, "Cannot broadcast shapes");                                                               \
                return NULL;                                                                                                                \
            }                                                                                                                               \
        }                                                                                                                                   \
        else                                                                                                                                \
        {                                                                                                                                   \
            bigger_shape = a_shape;                                                                                                         \
            to_broadcast = b;                                                                                                               \
            bigger = a;                                                                                                                     \
            ndim = PyArray_NDIM(a);                                                                                                         \
            if (!shape_isbroadcastable_to_ex(b_shape, a_shape, b_ndim, a_ndim, &to_broadcast_shape_pad_one))                                \
            {                                                                                                                               \
                PyErr_SetString(PyExc_ValueError, "Cannot broadcast shapes");                                                               \
                return NULL;                                                                                                                \
            }                                                                                                                               \
        }                                                                                                                                   \
        DEBUG_PRINT("ndim: %d\n", ndim);                                                                                                    \
        npy_intp stride_last = PyArray_STRIDE((const PyArrayObject *)bigger, ndim - 1);                                                     \
        npy_intp *strides_a = NULL, *strides_b = NULL, *shape = NULL;                                                                       \
        predict_broadcast_shape(to_broadcast_shape_pad_one, bigger_shape, ndim, &shape);                                                    \
        preprocess_strides(bigger_shape, stride_last, ndim, &strides_a);                                                                    \
        preprocess_strides(to_broadcast_shape_pad_one, stride_last, ndim, &strides_b);                                                      \
        PyArrayObject *result = (PyArrayObject *)PyArray_EMPTY(ndim, (npy_intp const *)shape, npy_type, 0);                                 \
        if (result == NULL)                                                                                                                 \
        {                                                                                                                                   \
            return NULL;                                                                                                                    \
        }                                                                                                                                   \
        DEBUG_PRINT("Result shape: ");                                                                                                      \
        for (int i = 0; i < ndim; i++)                                                                                                      \
            DEBUG_PRINT("%d ", shape[i]);                                                                                                   \
        DEBUG_PyObject_Print(result);                                                                                                       \
        BroadCast_Core(ndim, bigger_shape, to_broadcast_shape_pad_one, bigger, to_broadcast, result, strides_a, strides_b, shape, type, op, \
                       simd_type, simd_load, simd_op, simd_store, vec_size, simd_type_cast, npy_type, enum_op);                             \
        array_result = (PyObject *)result;                                                                                                  \
        free(strides_a);                                                                                                                    \
        free(strides_b);                                                                                                                    \
        free(shape);                                                                                                                        \
        free(to_broadcast_shape_pad_one);                                                                                                   \
    }

#ifdef _WIN32
#define Simd_int32_methods(a, b, result, op, c_type, type, simd_op)                                  \
    {                                                                                                \
        switch (simd_op)                                                                             \
        {                                                                                            \
        case ADD:                                                                                    \
            _BroadCast(a, b, result, op, c_type, type, __m256i, _mm256_load_si256, _mm256_add_epi32, \
                       _mm256_store_si256, 8, __m256i, simd_op);                                     \
            break;                                                                                   \
        case SUB:                                                                                    \
            _BroadCast(a, b, result, op, c_type, type, __m256i, _mm256_load_si256, _mm256_sub_epi32, \
                       _mm256_store_si256, 8, __m256i, simd_op);                                     \
            break;                                                                                   \
        case DIV:                                                                                    \
            _BroadCast(a, b, result, op, c_type, type, __m256i, _mm256_load_si256, _mm256_div_epi32, \
                       _mm256_store_si256, 8, __m256i, simd_op);                                     \
            break;                                                                                   \
        case MUL:                                                                                    \
            _BroadCast(a, b, result, op, c_type, type, __m256i, _mm256_load_si256, _mm256_mul_epi32, \
                       _mm256_store_si256, 8, __m256i, simd_op);                                     \
            break;                                                                                   \
        case MOD:                                                                                    \
            _BroadCast(a, b, result, op, c_type, type, __m256i, _mm256_load_si256, _mm256_rem_epi32, \
                       _mm256_store_si256, 8, __m256i, simd_op);                                     \
            break;                                                                                   \
        }                                                                                            \
    }

#define Simd_int64_methods(a, b, result, op, c_type, type, simd_op)                                  \
    {                                                                                                \
        switch (simd_op)                                                                             \
        {                                                                                            \
        case ADD:                                                                                    \
            _BroadCast(a, b, result, op, c_type, type, __m256i, _mm256_load_si256, _mm256_add_epi64, \
                       _mm256_store_si256, 4, __m256i, simd_op);                                     \
            break;                                                                                   \
        case SUB:                                                                                    \
            _BroadCast(a, b, result, op, c_type, type, __m256i, _mm256_load_si256, _mm256_sub_epi64, \
                       _mm256_store_si256, 4, __m256i, simd_op);                                     \
            break;                                                                                   \
        case DIV:                                                                                    \
            _BroadCast(a, b, result, op, c_type, type, __m256i, _mm256_load_si256, _mm256_div_epi64, \
                       _mm256_store_si256, 4, __m256i, simd_op);                                     \
            break;                                                                                   \
        case MUL:                                                                                    \
            _BroadCast(a, b, result, op, c_type, type, __m256i, _mm256_load_si256, _mm256_sub_epi64, \
                       _mm256_store_si256, -1, __m256i, simd_op);                                    \
            break;                                                                                   \
        case MOD:                                                                                    \
            _BroadCast(a, b, result, op, c_type, type, __m256i, _mm256_load_si256, _mm256_sub_epi64, \
                       _mm256_store_si256, -1, __m256i, simd_op);                                    \
            break;                                                                                   \
        }                                                                                            \
    }
#endif

#ifdef __linux__
#define Simd_int32_methods(a, b, result, op, c_type, type, simd_op)                                  \
    {                                                                                                \
        switch (simd_op)                                                                             \
        {                                                                                            \
        case ADD:                                                                                    \
            _BroadCast(a, b, result, op, c_type, type, __m256i, _mm256_load_si256, _mm256_add_epi32, \
                       _mm256_store_si256, 8, __m256i, simd_op);                                     \
            break;                                                                                   \
        case SUB:                                                                                    \
            _BroadCast(a, b, result, op, c_type, type, __m256i, _mm256_load_si256, _mm256_sub_epi32, \
                       _mm256_store_si256, 8, __m256i, simd_op);                                     \
            break;                                                                                   \
        case DIV:                                                                                    \
            _BroadCast(a, b, result, op, c_type, type, __m256i, _mm256_load_si256, _mm256_sub_epi32, \
                       _mm256_store_si256, -1, __m256i, simd_op);                                    \
            break;                                                                                   \
        case MUL:                                                                                    \
            _BroadCast(a, b, result, op, c_type, type, __m256i, _mm256_load_si256, _mm256_mul_epi32, \
                       _mm256_store_si256, 8, __m256i, simd_op);                                     \
            break;                                                                                   \
        case MOD:                                                                                    \
            _BroadCast(a, b, result, op, c_type, type, __m256i, _mm256_load_si256, _mm256_sub_epi32, \
                       _mm256_store_si256, -1, __m256i, simd_op);                                    \
            break;                                                                                   \
        }                                                                                            \
    }

#define Simd_int64_methods(a, b, result, op, c_type, type, simd_op)                                  \
    {                                                                                                \
        switch (simd_op)                                                                             \
        {                                                                                            \
        case ADD:                                                                                    \
            _BroadCast(a, b, result, op, c_type, type, __m256i, _mm256_load_si256, _mm256_add_epi64, \
                       _mm256_store_si256, 4, __m256i, simd_op);                                     \
            break;                                                                                   \
        case SUB:                                                                                    \
            _BroadCast(a, b, result, op, c_type, type, __m256i, _mm256_load_si256, _mm256_sub_epi64, \
                       _mm256_store_si256, 4, __m256i, simd_op);                                     \
            break;                                                                                   \
        case DIV:                                                                                    \
            _BroadCast(a, b, result, op, c_type, type, __m256i, _mm256_load_si256, _mm256_sub_epi64, \
                       _mm256_store_si256, -1, __m256i, simd_op);                                    \
            break;                                                                                   \
        case MUL:                                                                                    \
            _BroadCast(a, b, result, op, c_type, type, __m256i, _mm256_load_si256, _mm256_sub_epi64, \
                       _mm256_store_si256, -1, __m256i, simd_op);                                    \
            break;                                                                                   \
        case MOD:                                                                                    \
            _BroadCast(a, b, result, op, c_type, type, __m256i, _mm256_load_si256, _mm256_sub_epi64, \
                       _mm256_store_si256, -1, __m256i, simd_op);                                    \
            break;                                                                                   \
        }                                                                                            \
    }
#endif

#define Simd_float32_methods(a, b, result, op, c_type, type, simd_op)                         \
    {                                                                                         \
        switch (simd_op)                                                                      \
        {                                                                                     \
        case ADD:                                                                             \
            _BroadCast(a, b, result, op, c_type, type, __m256, _mm256_load_ps, _mm256_add_ps, \
                       _mm256_store_ps, 8, float, simd_op);                                   \
            break;                                                                            \
        case SUB:                                                                             \
            _BroadCast(a, b, result, op, c_type, type, __m256, _mm256_load_ps, _mm256_sub_ps, \
                       _mm256_store_ps, 8, float, simd_op);                                   \
            break;                                                                            \
        case DIV:                                                                             \
            _BroadCast(a, b, result, op, c_type, type, __m256, _mm256_load_ps, _mm256_div_ps, \
                       _mm256_store_ps, 8, float, simd_op);                                   \
            break;                                                                            \
        case MUL:                                                                             \
            _BroadCast(a, b, result, op, c_type, type, __m256, _mm256_load_ps, _mm256_mul_ps, \
                       _mm256_store_ps, 8, float, simd_op);                                   \
            break;                                                                            \
        case MOD:                                                                             \
            _BroadCast(a, b, result, op, c_type, type, __m256, _mm256_load_ps, _mm256_mul_ps, \
                       _mm256_store_ps, -1, float, simd_op);                                  \
            break;                                                                            \
        }                                                                                     \
    }

#define Simd_float64_methods(a, b, result, op, c_type, type, simd_op)                          \
    {                                                                                          \
        switch (simd_op)                                                                       \
        {                                                                                      \
        case ADD:                                                                              \
            _BroadCast(a, b, result, op, c_type, type, __m256d, _mm256_load_pd, _mm256_add_pd, \
                       _mm256_storeu_pd, 4, double);                                           \
            break;                                                                             \
        case SUB:                                                                              \
            _BroadCast(a, b, result, op, c_type, type, __m256d, _mm256_load_pd, _mm256_sub_pd, \
                       _mm256_storeu_pd, 4, double);                                           \
            break;                                                                             \
        case DIV:                                                                              \
            _BroadCast(a, b, result, op, c_type, type, __m256d, _mm256_load_pd, _mm256_div_pd, \
                       _mm256_storeu_pd, 4, double);                                           \
            break;                                                                             \
        case MUL:                                                                              \
            _BroadCast(a, b, result, op, c_type, type, __m256d, _mm256_load_pd, _mm256_mul_pd, \
                       _mm256_storeu_pd, 4, double);                                           \
            break;                                                                             \
        case MOD:                                                                              \
            _BroadCast(a, b, result, op, c_type, type, __m256d, _mm256_load_pd, _mm256_mul_pd, \
                       _mm256_storeu_pd, -1, double);                                          \
            break;                                                                             \
        }                                                                                      \
    }

#define BroadCast(a, b, result, op, simd_op, type)                                                               \
    {                                                                                                            \
        switch (type)                                                                                            \
        {                                                                                                        \
        case NPY_BOOL:                                                                                           \
            _BroadCast(a, b, result, op, bool, type, __m256i, _mm256_load_si256, _mm256_sub_epi64,               \
                       _mm256_store_si256, -1, __m256i, simd_op);                                                \
            break;                                                                                               \
        case NPY_BYTE:                                                                                           \
            break;                                                                                               \
        case NPY_UBYTE:                                                                                          \
            break;                                                                                               \
        case NPY_SHORT:                                                                                          \
            break;                                                                                               \
        case NPY_USHORT:                                                                                         \
            break;                                                                                               \
        case NPY_INT:                                                                                            \
            Simd_int32_methods(a, b, result, op, int32_t, type, simd_op);                                        \
            break;                                                                                               \
        case NPY_UINT:                                                                                           \
            Simd_int32_methods(a, b, result, op, uint8_t, type, simd_op);                                        \
            break;                                                                                               \
        case NPY_LONG:                                                                                           \
            Simd_int64_methods(a, b, result, op, int32_t, type, simd_op) break;                                  \
        case NPY_ULONG:                                                                                          \
            Simd_int64_methods(a, b, result, op, uint32_t, type, simd_op) break;                                 \
        case NPY_LONGLONG:                                                                                       \
            _BroadCast(a, b, result, op, int64_t, type, __m256i, _mm256_load_si256, _mm256_sub_epi64,            \
                       _mm256_store_si256, -1, __m256i, simd_op);                                                \
            break;                                                                                               \
        case NPY_ULONGLONG:                                                                                      \
            _BroadCast(a, b, result, op, unsigned long long, type, __m256i, _mm256_load_si256, _mm256_sub_epi64, \
                       _mm256_store_si256, -1, __m256i, simd_op);                                                \
            break;                                                                                               \
        case NPY_FLOAT:                                                                                          \
            Simd_float32_methods(a, b, result, op, float, type, simd_op) break;                                  \
        case NPY_DOUBLE:                                                                                         \
            Simd_float32_methods(a, b, result, op, double, type, simd_op) break;                                 \
        case NPY_LONGDOUBLE:                                                                                     \
            _BroadCast(a, b, result, op, long double, type, __m256i, _mm256_load_si256, _mm256_sub_epi64,        \
                       _mm256_store_si256, -1, __m256i, simd_op);                                                \
            break;                                                                                               \
        default:                                                                                                 \
            break;                                                                                               \
        }                                                                                                        \
    }