#define PY_SSIZE_T_CLEAN
#include "broadcast.h"
#define NO_IMPORT_ARRAY

void Broad_Cast_(PyArrayObject *a, PyArrayObject *b, PyObject **array_result)
{
    int a_ndim = PyArray_NDIM(a);
    int b_ndim = PyArray_NDIM(b);
    npy_intp *a_shape = PyArray_SHAPE(a);
    npy_intp *b_shape = PyArray_SHAPE(b);
    npy_intp *to_broadcast_shape_pad_one = NULL;
    npy_intp *bigger_shape = NULL;
    PyArrayObject *to_broadcast = NULL;
    PyArrayObject *bigger = NULL;
    int ndim = 0;
    if (a_ndim < b_ndim)
    {
        bigger_shape = b_shape;
        to_broadcast = a;
        bigger = b;
        ndim = PyArray_NDIM(b);
        if (!shape_isbroadcastable_to_ex(a_shape, b_shape, a_ndim, b_ndim, &to_broadcast_shape_pad_one))
        {
            PyErr_SetString(PyExc_ValueError, "Cannot broadcast shapes");
            return;
        }
    }
    else
    {
        bigger_shape = a_shape;
        to_broadcast = b;
        bigger = a;
        ndim = PyArray_NDIM(a);
        if (!shape_isbroadcastable_to_ex(b_shape, a_shape, b_ndim, a_ndim, &to_broadcast_shape_pad_one))
        {
            PyErr_SetString(PyExc_ValueError, "Cannot broadcast shapes");
            return;
        }
    }
    npy_intp stride_last = PyArray_STRIDE((const PyArrayObject *)bigger, ndim - 1);
    npy_intp *strides_a = NULL, *strides_b = NULL, *shape = NULL;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);

    predict_broadcast_shape(to_broadcast_shape_pad_one, bigger_shape, ndim, &shape);
    preprocess_strides(bigger_shape, stride_last, ndim, &strides_a);
    preprocess_strides(to_broadcast_shape_pad_one, stride_last, ndim, &strides_b);
    PyArrayObject *result = (PyArrayObject *)PyArray_EMPTY(ndim, (npy_intp const *)shape, NPY_INT32, 0);
    if (shape_copy == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Cannot allocate memory");
        return;
    }
    int32_t *result_data_ptr = (int32_t *)PyArray_DATA(result);
    char *to_broadcast_data_ptr = (char *)PyArray_DATA(to_broadcast);
    char *bigger_data_ptr = (char *)PyArray_DATA(bigger);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    int max_dim = ndim - 1;
    int axis = 0;
    npy_intp prod = 1;
    for (int i = ndim - 1; i >= 0; i--)
    {
        if (bigger_shape[i] == to_broadcast_shape_pad_one[i])
        {
            axis++;
            prod *= bigger_shape[i];
        }
        else
        {
            if (i == ndim - 1)
            {
                prod *= shape[ndim - 1];
                axis++;
            }
            break;
        }
    }
    bool done = false;
    int outer_start = max_dim - axis;
    while (!done)
    {
        done = true;
        char *bigger_data_ptr_save = bigger_data_ptr;
        char *to_broadcast_data_ptr_save = to_broadcast_data_ptr;
        for (int i = 0; i < prod; i++)
        {
            double val1 = *((double *)(to_broadcast_data_ptr));
            double val2 = *((double *)(bigger_data_ptr));
            *result_data_ptr = val1 + val2;
            result_data_ptr++;
            bigger_data_ptr += strides_a[max_dim];
            to_broadcast_data_ptr += strides_b[max_dim];
        }
        bigger_data_ptr = bigger_data_ptr_save;
        to_broadcast_data_ptr = to_broadcast_data_ptr_save;
        for (int j = outer_start; j >= 0; j--)
        {
            if (shape_copy[j] < shape[j])
            {
                shape_copy[j]++;
                done = false;
                bigger_data_ptr += strides_a[j];
                to_broadcast_data_ptr += strides_b[j];
                break;
            }
            else
            {
                shape_copy[j] = 0;
                bigger_data_ptr -= indice_a_cache[j];
                to_broadcast_data_ptr -= indice_b_cache[j];
            }
        }
    }
    free(indice_a_cache);
    free(indice_b_cache);
    *array_result = (PyObject *)result;
    free(shape_copy);
    free(strides_a);
    free(strides_b);
    free(shape);
    free(to_broadcast_shape_pad_one);
}

void Broad_Cast_vec_(PyArrayObject *a, PyArrayObject *b, PyObject **array_result)
{
    int a_ndim = PyArray_NDIM(a);
    int b_ndim = PyArray_NDIM(b);
    npy_intp *a_shape = PyArray_SHAPE(a);
    npy_intp *b_shape = PyArray_SHAPE(b);
    npy_intp *to_broadcast_shape_pad_one = NULL;
    npy_intp *bigger_shape = NULL;
    PyArrayObject *to_broadcast = NULL;
    PyArrayObject *bigger = NULL;
    int ndim = 0;
    if (a_ndim < b_ndim)
    {
        bigger_shape = b_shape;
        to_broadcast = a;
        bigger = b;
        ndim = PyArray_NDIM(b);
        if (!shape_isbroadcastable_to_ex(a_shape, b_shape, a_ndim, b_ndim, &to_broadcast_shape_pad_one))
        {
            PyErr_SetString(PyExc_ValueError, "Cannot broadcast shapes");
            return;
        }
    }
    else
    {
        bigger_shape = a_shape;
        to_broadcast = b;
        bigger = a;
        ndim = PyArray_NDIM(a);
        if (!shape_isbroadcastable_to_ex(b_shape, a_shape, b_ndim, a_ndim, &to_broadcast_shape_pad_one))
        {
            PyErr_SetString(PyExc_ValueError, "Cannot broadcast shapes");
            return;
        }
    }
    npy_intp stride_last = PyArray_STRIDE((const PyArrayObject *)bigger, ndim - 1);
    npy_intp *strides_a = NULL, *strides_b = NULL, *shape = NULL;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    bool done = false;

    predict_broadcast_shape(to_broadcast_shape_pad_one, bigger_shape, ndim, &shape);
    preprocess_strides(bigger_shape, stride_last, ndim, &strides_a);
    preprocess_strides(to_broadcast_shape_pad_one, stride_last, ndim, &strides_b);
    PyArrayObject *result = (PyArrayObject *)PyArray_EMPTY(ndim, (npy_intp const *)shape, NPY_INT32, 0);
    if (shape_copy == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Cannot allocate memory");
        return;
    }
    int32_t *result_data_ptr = (int32_t *)PyArray_DATA(result);
    char *to_broadcast_data_ptr = (char *)PyArray_DATA(to_broadcast);
    char *bigger_data_ptr = (char *)PyArray_DATA(bigger);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    int max_dim = ndim - 1;
    int vec_size = 8;
    npy_intp total_size = find_innerloop_size(bigger_shape, to_broadcast_shape_pad_one, ndim);
    int vec_loop_size = (total_size / vec_size) * vec_size;
    int remain_size = total_size % vec_size;
    while (!done)
    {
        done = true;
        char *bigger_data_ptr_save = bigger_data_ptr;
        char *to_broadcast_data_ptr_save = to_broadcast_data_ptr;
        DEBUG_PRINT("Vectorized loop, vec_loop_size: %d\n", vec_loop_size);
        for (int i = 0; i < vec_loop_size; i += vec_size)
        {
            __m256i val1 = _mm256_load_si256((__m256i *)(to_broadcast_data_ptr));
            __m256i val2 = _mm256_load_si256((__m256i *)(bigger_data_ptr));
            DEBUG_PRINT("val1: %d %d %d %d %d %d %d %d\n", val1.m256i_i32[0], val1.m256i_i32[1], val1.m256i_i32[2], val1.m256i_i32[3], val1.m256i_i32[4], val1.m256i_i32[5], val1.m256i_i32[6], val1.m256i_i32[7]);
            DEBUG_PRINT("val2: %d %d %d %d %d %d %d %d\n", val2.m256i_i32[0], val2.m256i_i32[1], val2.m256i_i32[2], val2.m256i_i32[3], val2.m256i_i32[4], val2.m256i_i32[5], val2.m256i_i32[6], val2.m256i_i32[7]);
            __m256i result = _mm256_add_epi32(val1, val2);
            DEBUG_PRINT("result: %d %d %d %d %d %d %d %d\n", result.m256i_i32[0], result.m256i_i32[1], result.m256i_i32[2], result.m256i_i32[3], result.m256i_i32[4], result.m256i_i32[5], result.m256i_i32[6], result.m256i_i32[7]);
            _mm256_store_si256((__m256i *)result_data_ptr, result);
            DEBUG_PRINT("result_data_ptr: %d %d %d %d %d %d %d %d\n", result_data_ptr[0], result_data_ptr[1], result_data_ptr[2], result_data_ptr[3], result_data_ptr[4], result_data_ptr[5], result_data_ptr[6], result_data_ptr[7]);
            result_data_ptr += vec_size;
            bigger_data_ptr += vec_size * strides_a[max_dim];
            to_broadcast_data_ptr += vec_size * strides_b[max_dim];
        }
        DEBUG_PRINT("Remain loop, remain_size: %d\n", remain_size);
        for (int k = 0; k < remain_size; k++)
        {
            int32_t val1 = *((int32_t *)(to_broadcast_data_ptr));
            int32_t val2 = *((int32_t *)(bigger_data_ptr));
            *result_data_ptr = val1 + val2;
            result_data_ptr++;
            bigger_data_ptr += strides_a[max_dim];
            to_broadcast_data_ptr += strides_b[max_dim];
        }
        bigger_data_ptr = bigger_data_ptr_save;
        to_broadcast_data_ptr = to_broadcast_data_ptr_save;
        DEBUG_PRINT("Done loop\n");
        for (int j = max_dim - 1; j >= 0; j--)
        {
            DEBUG_PRINT("shape[%d]: %d\n", j, shape[j]);
            DEBUG_PRINT("shape_copy[%d]: %d\n", j, shape_copy[j]);
            if (shape_copy[j] < shape[j])
            {
                shape_copy[j]++;
                done = false;
                bigger_data_ptr += strides_a[j];
                to_broadcast_data_ptr += strides_b[j];
                break;
            }
            else
            {
                shape_copy[j] = 0;
                bigger_data_ptr -= indice_a_cache[j];
                to_broadcast_data_ptr -= indice_b_cache[j];
            }
        }
    }
    DEBUG_PRINT("Done all loop\n");
    free(indice_a_cache);
    free(indice_b_cache);
    *array_result = (PyObject *)result;
    free(shape_copy);
    free(strides_a);
    free(strides_b);
    free(shape);
    free(to_broadcast_shape_pad_one);
}