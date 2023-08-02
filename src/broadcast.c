#define PY_SSIZE_T_CLEAN
#include "broadcast.h"
#include "omp.h"
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
    int32_t *bigger_data_ptr_ = (int32_t *)bigger_data_ptr;
    int32_t *to_broadcast_data_ptr_ = (int32_t *)to_broadcast_data_ptr;
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp total_size = find_innerloop_size(bigger_shape, to_broadcast_shape_pad_one, ndim);
    int vec_loop_size = (total_size / 8) * 8;
    int remain_size = total_size % 8;
    int max_dim = ndim - 1;
    int outer_start = max_dim - 1;

    npy_intp a_last_stride = 8 * strides_a[max_dim];
    npy_intp b_last_stride = 8 * strides_b[max_dim];
    while (!done)
    {
        done = true;
        char *bigger_data_ptr_save = bigger_data_ptr;
        char *to_broadcast_data_ptr_save = to_broadcast_data_ptr;
        for (int i = 0; i < vec_loop_size; i++)
        {
            __m256i val1 = _mm256_load_si256((const __m256i *)to_broadcast_data_ptr);
            __m256i val2 = _mm256_load_si256((const __m256i *)bigger_data_ptr);
            __m256i result = _mm256_add_epi32(val1, val2);
            _mm256_store_si256((__m256i *)result_data_ptr, result);
            result_data_ptr += 8;
            bigger_data_ptr += a_last_stride;
            to_broadcast_data_ptr += b_last_stride;
        }
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

void Broadcast_Standard(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                        npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                        int ndim, int axis)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    bool done = false;
    while (!done)
    {
        done = true;
        for (int i = 0; i < inner_loop_size; i++)
        {
            double val1 = *((double *)(b_data_ptr + i * strides_b[max_dim]));
            double val2 = *((double *)(a_data_ptr + i * strides_a[max_dim]));
            *(result_data_ptr + i) = val1 + val2;
        }
        for (int j = outer_start; j >= 0; j--)
        {
            if (shape_copy[j] < shape[j])
            {
                shape_copy[j]++;
                done = false;
                a_data_ptr += strides_a[j];
                b_data_ptr += strides_b[j];
                break;
            }
            else
            {
                shape_copy[j] = 0;
                a_data_ptr -= indice_a_cache[j];
                b_data_ptr -= indice_b_cache[j];
            }
        }
    }
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

void Broadcast_Standard_fa64(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                             npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                             int ndim, int axis)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    bool done = false;
    while (!done)
    {
        done = true;
        char *bigger_data_ptr_save = a_data_ptr;
        char *to_broadcast_data_ptr_save = b_data_ptr;

        for (int i = 0; i < inner_loop_size; i++)
        {
            npy_float64 val1 = *((npy_float64 *)(b_data_ptr));
            npy_float64 val2 = *((npy_float64 *)(a_data_ptr));
            *result_data_ptr = val1 + val2;
            result_data_ptr++;
            a_data_ptr += strides_a[max_dim];
            b_data_ptr += strides_b[max_dim];
        }
        a_data_ptr = bigger_data_ptr_save;
        b_data_ptr = to_broadcast_data_ptr_save;
        for (int j = outer_start; j >= 0; j--)
        {
            if (shape_copy[j] < shape[j])
            {
                shape_copy[j]++;
                done = false;
                a_data_ptr += strides_a[j];
                b_data_ptr += strides_b[j];
                break;
            }
            else
            {
                shape_copy[j] = 0;
                a_data_ptr -= indice_a_cache[j];
                b_data_ptr -= indice_b_cache[j];
            }
        }
    }
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

void Broadcast_Standard_fa32(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                             npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                             int ndim, int axis)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    bool done = false;
    while (!done)
    {
        done = true;
        char *bigger_data_ptr_save = a_data_ptr;
        char *to_broadcast_data_ptr_save = b_data_ptr;

        for (int i = 0; i < inner_loop_size; i++)
        {
            npy_float32 val1 = *((npy_float32 *)(b_data_ptr));
            npy_float32 val2 = *((npy_float32 *)(a_data_ptr));
            *result_data_ptr = val1 + val2;
            result_data_ptr++;
            a_data_ptr += strides_a[max_dim];
            b_data_ptr += strides_b[max_dim];
        }
        a_data_ptr = bigger_data_ptr_save;
        b_data_ptr = to_broadcast_data_ptr_save;
        for (int j = outer_start; j >= 0; j--)
        {
            if (shape_copy[j] < shape[j])
            {
                shape_copy[j]++;
                done = false;
                a_data_ptr += strides_a[j];
                b_data_ptr += strides_b[j];
                break;
            }
            else
            {
                shape_copy[j] = 0;
                a_data_ptr -= indice_a_cache[j];
                b_data_ptr -= indice_b_cache[j];
            }
        }
    }
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

void Broadcast_Standard_fa16(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                             npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                             int ndim, int axis)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    bool done = false;
    while (!done)
    {
        done = true;
        char *bigger_data_ptr_save = a_data_ptr;
        char *to_broadcast_data_ptr_save = b_data_ptr;

        for (int i = 0; i < inner_loop_size; i++)
        {
            npy_float16 val1 = *((npy_float16 *)(b_data_ptr));
            npy_float16 val2 = *((npy_float16 *)(a_data_ptr));
            *result_data_ptr = val1 + val2;
            result_data_ptr++;
            a_data_ptr += strides_a[max_dim];
            b_data_ptr += strides_b[max_dim];
        }
        a_data_ptr = bigger_data_ptr_save;
        b_data_ptr = to_broadcast_data_ptr_save;
        for (int j = outer_start; j >= 0; j--)
        {
            if (shape_copy[j] < shape[j])
            {
                shape_copy[j]++;
                done = false;
                a_data_ptr += strides_a[j];
                b_data_ptr += strides_b[j];
                break;
            }
            else
            {
                shape_copy[j] = 0;
                a_data_ptr -= indice_a_cache[j];
                b_data_ptr -= indice_b_cache[j];
            }
        }
    }
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

void Broadcast_Standard_ia16(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                             npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                             int ndim, int axis)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    bool done = false;
    while (!done)
    {
        done = true;
        char *bigger_data_ptr_save = a_data_ptr;
        char *to_broadcast_data_ptr_save = b_data_ptr;

        for (int i = 0; i < inner_loop_size; i++)
        {
            npy_int16 val1 = *((npy_int16 *)(b_data_ptr));
            npy_int16 val2 = *((npy_int16 *)(a_data_ptr));
            *result_data_ptr = val1 + val2;
            result_data_ptr++;
            a_data_ptr += strides_a[max_dim];
            b_data_ptr += strides_b[max_dim];
        }
        a_data_ptr = bigger_data_ptr_save;
        b_data_ptr = to_broadcast_data_ptr_save;
        for (int j = outer_start; j >= 0; j--)
        {
            if (shape_copy[j] < shape[j])
            {
                shape_copy[j]++;
                done = false;
                a_data_ptr += strides_a[j];
                b_data_ptr += strides_b[j];
                break;
            }
            else
            {
                shape_copy[j] = 0;
                a_data_ptr -= indice_a_cache[j];
                b_data_ptr -= indice_b_cache[j];
            }
        }
    }
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

void Broadcast_Standard_ia64(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                             npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                             int ndim, int axis)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    bool done = false;
    while (!done)
    {
        done = true;
        char *bigger_data_ptr_save = a_data_ptr;
        char *to_broadcast_data_ptr_save = b_data_ptr;

        for (int i = 0; i < inner_loop_size; i++)
        {
            npy_int64 val1 = *((npy_int64 *)(b_data_ptr));
            npy_int64 val2 = *((npy_int64 *)(a_data_ptr));
            *result_data_ptr = val1 + val2;
            result_data_ptr++;
            a_data_ptr += strides_a[max_dim];
            b_data_ptr += strides_b[max_dim];
        }
        a_data_ptr = bigger_data_ptr_save;
        b_data_ptr = to_broadcast_data_ptr_save;
        for (int j = outer_start; j >= 0; j--)
        {
            if (shape_copy[j] < shape[j])
            {
                shape_copy[j]++;
                done = false;
                a_data_ptr += strides_a[j];
                b_data_ptr += strides_b[j];
                break;
            }
            else
            {
                shape_copy[j] = 0;
                a_data_ptr -= indice_a_cache[j];
                b_data_ptr -= indice_b_cache[j];
            }
        }
    }
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

void Broadcast_Standard_ia32(char *a_data_ptr, char *b_data_ptr, npy_int32 *result_data_ptr, npy_intp inner_loop_size,
                             npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                             int ndim, int axis)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    int vec_loop_size = inner_loop_size / 8;
    int remain_size = inner_loop_size % 8;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_int32 *b_data_ptr_ = (npy_int32 *)b_data_ptr;
    npy_int32 *a_data_ptr_ = (npy_int32 *)a_data_ptr;
    npy_intp i = 0;
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    bool done = false;
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel
    while (!done)
    {
        done = true;
        b_data_ptr_ = (npy_int32 *)b_data_ptr;
        a_data_ptr_ = (npy_int32 *)a_data_ptr;
#pragma omp parallel for
        for (i = 0; i < inner_loop_size; i++)
        {
            int32_t val1 = *((b_data_ptr_ + i));
            int32_t val2 = *((a_data_ptr_));
            *(result_data_ptr + i) = val1 + val2;
        }
        result_data_ptr += inner_loop_size;
        for (int j = outer_start; j >= 0; j--)
        {
            if (shape_copy[j] < shape[j])
            {
                shape_copy[j]++;
                done = false;
                a_data_ptr += strides_a[j];
                b_data_ptr += strides_b[j];
                break;
            }
            else
            {
                shape_copy[j] = 0;
                a_data_ptr -= indice_a_cache[j];
                b_data_ptr -= indice_b_cache[j];
            }
        }
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

void Broadcast_Standard_is32(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                             npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                             int ndim, int axis)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    bool done = false;
    while (!done)
    {
        done = true;
        char *bigger_data_ptr_save = a_data_ptr;
        char *to_broadcast_data_ptr_save = b_data_ptr;

        for (int i = 0; i < inner_loop_size; i++)
        {
            npy_int32 val1 = *((npy_int32 *)(b_data_ptr));
            npy_int32 val2 = *((npy_int32 *)(a_data_ptr));
            *result_data_ptr = val1 - val2;
            result_data_ptr++;
            a_data_ptr += strides_a[max_dim];
            b_data_ptr += strides_b[max_dim];
        }
        a_data_ptr = bigger_data_ptr_save;
        b_data_ptr = to_broadcast_data_ptr_save;
        for (int j = outer_start; j >= 0; j--)
        {
            if (shape_copy[j] < shape[j])
            {
                shape_copy[j]++;
                done = false;
                a_data_ptr += strides_a[j];
                b_data_ptr += strides_b[j];
                break;
            }
            else
            {
                shape_copy[j] = 0;
                a_data_ptr -= indice_a_cache[j];
                b_data_ptr -= indice_b_cache[j];
            }
        }
    }
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

void Broadcast_Standard_is64(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                             npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                             int ndim, int axis)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    bool done = false;
    while (!done)
    {
        done = true;
        char *bigger_data_ptr_save = a_data_ptr;
        char *to_broadcast_data_ptr_save = b_data_ptr;

        for (int i = 0; i < inner_loop_size; i++)
        {
            npy_int64 val1 = *((npy_int64 *)(b_data_ptr));
            npy_int64 val2 = *((npy_int64 *)(a_data_ptr));
            *result_data_ptr = val1 - val2;
            result_data_ptr++;
            a_data_ptr += strides_a[max_dim];
            b_data_ptr += strides_b[max_dim];
        }
        a_data_ptr = bigger_data_ptr_save;
        b_data_ptr = to_broadcast_data_ptr_save;
        for (int j = outer_start; j >= 0; j--)
        {
            if (shape_copy[j] < shape[j])
            {
                shape_copy[j]++;
                done = false;
                a_data_ptr += strides_a[j];
                b_data_ptr += strides_b[j];
                break;
            }
            else
            {
                shape_copy[j] = 0;
                a_data_ptr -= indice_a_cache[j];
                b_data_ptr -= indice_b_cache[j];
            }
        }
    }
}

void Broadcast_Standard_is16(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                             npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                             int ndim, int axis)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    bool done = false;
    while (!done)
    {
        done = true;
        char *bigger_data_ptr_save = a_data_ptr;
        char *to_broadcast_data_ptr_save = b_data_ptr;

        for (int i = 0; i < inner_loop_size; i++)
        {
            npy_int16 val1 = *((npy_int16 *)(b_data_ptr));
            npy_int16 val2 = *((npy_int16 *)(a_data_ptr));
            *result_data_ptr = val1 - val2;
            result_data_ptr++;
            a_data_ptr += strides_a[max_dim];
            b_data_ptr += strides_b[max_dim];
        }
        a_data_ptr = bigger_data_ptr_save;
        b_data_ptr = to_broadcast_data_ptr_save;
        for (int j = outer_start; j >= 0; j--)
        {
            if (shape_copy[j] < shape[j])
            {
                shape_copy[j]++;
                done = false;
                a_data_ptr += strides_a[j];
                b_data_ptr += strides_b[j];
                break;
            }
            else
            {
                shape_copy[j] = 0;
                a_data_ptr -= indice_a_cache[j];
                b_data_ptr -= indice_b_cache[j];
            }
        }
    }
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

void Broadcast_Standard_fs16(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                             npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                             int ndim, int axis)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    bool done = false;
    while (!done)
    {
        done = true;
        char *bigger_data_ptr_save = a_data_ptr;
        char *to_broadcast_data_ptr_save = b_data_ptr;

        for (int i = 0; i < inner_loop_size; i++)
        {
            npy_float16 val1 = *((npy_float16 *)(b_data_ptr));
            npy_float16 val2 = *((npy_float16 *)(a_data_ptr));
            *result_data_ptr = val1 - val2;
            result_data_ptr++;
            a_data_ptr += strides_a[max_dim];
            b_data_ptr += strides_b[max_dim];
        }
        a_data_ptr = bigger_data_ptr_save;
        b_data_ptr = to_broadcast_data_ptr_save;
        for (int j = outer_start; j >= 0; j--)
        {
            if (shape_copy[j] < shape[j])
            {
                shape_copy[j]++;
                done = false;
                a_data_ptr += strides_a[j];
                b_data_ptr += strides_b[j];
                break;
            }
            else
            {
                shape_copy[j] = 0;
                a_data_ptr -= indice_a_cache[j];
                b_data_ptr -= indice_b_cache[j];
            }
        }
    }
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

void Broadcast_OuterPrallel_ia32(char *a_data_ptr, char *b_data_ptr, npy_int32 *result_data_ptr, npy_intp inner_loop_size,
                                 npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                 int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    int vec_loop_size = inner_loop_size / 8;
    int remain_size = inner_loop_size % 8;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_int32 *b_data_ptr_saved = (npy_int32 *)b_data_ptr;
    npy_int32 *a_data_ptr_saved = (npy_int32 *)a_data_ptr;
    npy_int32 *b_data_ptr_ = (npy_int32 *)b_data_ptr;
    npy_int32 *a_data_ptr_ = (npy_int32 *)a_data_ptr;
    npy_int32 *result_data_ptr_saved = result_data_ptr;
    npy_int32 *result_data_ptr_ = result_data_ptr;
    npy_int32 num_elements = 0;
    npy_intp i = 0;
    npy_intp k = 0;
    int cnt = 0;
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
    bool done = false;
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel firstprivate(left_prod, shape_copy1, result_data_ptr_, done, strides_a, strides_b, a_data_ptr_saved, b_data_ptr_saved, k, indice_a_cache, indice_b_cache, inner_loop_size, outer_start, shape)
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        int start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        int end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        npy_intp prd = num_elements;
        result_data_ptr_ = result_data_ptr;
        result_data_ptr += (end_index - start_index) * inner_loop_size;
        num_elements = (result_data_ptr - result_data_ptr_saved);
        for (int j = 1; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * (strides_a[j] / sizeof(npy_int32));
            b_data_ptr_saved += shape_copy1[j] * (strides_b[j] / sizeof(npy_int32));
        }
#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                int32_t val1 = *((b_data_ptr_saved + i * (strides_b[max_dim] / sizeof(npy_int32))));
                int32_t val2 = *((a_data_ptr_saved + i * (strides_a[max_dim] / sizeof(npy_int32))));
                *(result_data_ptr_ + i) = val1 + val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += (strides_a[j] / sizeof(npy_int32));
                    b_data_ptr_saved += (strides_b[j] / sizeof(npy_int32));
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= (indice_a_cache[j] / sizeof(npy_int32));
                    b_data_ptr_saved -= (indice_b_cache[j] / sizeof(npy_int32));
                }
            }
        }
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}