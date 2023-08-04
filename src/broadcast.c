#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define PY_SSIZE_T_CLEAN
#include "broadcast.h"
#include "omp.h"
#include "op.h"
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

void BroadCast_Core(int ndim, npy_intp *shape_a, npy_intp *shape_b, PyArrayObject *bigger, PyArrayObject *to_broadcast, PyArrayObject *result,
                    npy_intp *strides_a, npy_intp *strides_b, npy_intp *shape, int npy_type, int enum_op)
{
    char *result_data_ptr = (char *)PyArray_DATA(result);
    char *a_data_ptr = (char *)PyArray_DATA((PyArrayObject *)bigger);
    char *b_data_ptr = (char *)PyArray_DATA((PyArrayObject *)to_broadcast);
    npy_intp prod = 1;
    int axis = 0;
    bool vectorizable = true;

    for (int i = ndim - 1; i >= 0; i--)
    {
        if (shape_a[i] == shape_b[i])
        {
            axis++;
            prod *= shape_a[i];
        }
        else
        {
            if (i == ndim - 1)
            {
                prod *= shape[ndim - 1];
                vectorizable = false;
                axis++;
            }
            break;
        }
    }
    npy_intp left_prod = PyArray_SIZE(result) / prod;
    Broadcast_OperationPicker(npy_type, enum_op)(a_data_ptr, b_data_ptr,
                                                 result_data_ptr, prod,
                                                 shape, strides_a,
                                                 strides_b, ndim,
                                                 axis, left_prod);
}

void _BroadCast(PyArrayObject *a, PyArrayObject *b, PyObject **array_result, int npy_type, int enum_op)
{
    DEBUG_PRINT("BroadCast\n");
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
            return NULL;
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
            return NULL;
        }
    }
    DEBUG_PRINT("ndim: %d\n", ndim);
    npy_intp stride_last = PyArray_STRIDE((const PyArrayObject *)bigger, ndim - 1);
    npy_intp *strides_a = NULL, *strides_b = NULL, *shape = NULL;
    predict_broadcast_shape(to_broadcast_shape_pad_one, bigger_shape, ndim, &shape);
    preprocess_strides(bigger_shape, stride_last, ndim, &strides_a);
    preprocess_strides(to_broadcast_shape_pad_one, stride_last, ndim, &strides_b);
    PyArrayObject *result = (PyArrayObject *)PyArray_EMPTY(ndim, (npy_intp const *)shape, npy_type, 0);
    if (result == NULL)
    {
        return NULL;
    }
    DEBUG_PRINT("Result shape: ");
    for (int i = 0; i < ndim; i++)
        DEBUG_PRINT("%d ", shape[i]);
    DEBUG_PyObject_Print(result);
    DEBUG_PRINT("\n");
    DEBUG_PRINT("Size: ");
    BroadCast_Core(ndim, bigger_shape, to_broadcast_shape_pad_one, bigger, to_broadcast, result, strides_a, strides_b, shape,
                   npy_type, enum_op);
    *array_result = (PyObject *)result;
    free(strides_a);
    free(strides_b);
    free(shape);
    free(to_broadcast_shape_pad_one);
}

/*============================================================= ADD =================================================================================*/
static void Broadcast_Standard_uint8_add(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                         npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                         int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_uint8 *b_data_ptr_saved = (npy_uint8 *)b_data_ptr;
    npy_uint8 *a_data_ptr_saved = (npy_uint8 *)a_data_ptr;
    npy_uint8 *result_data_ptr_saved = (npy_uint8 *)result_data_ptr;
    npy_uint8 *result_data_ptr_ = (npy_uint8 *)result_data_ptr;
    npy_uint8 *result_data_ptr_cpy = (npy_uint8 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_uint8);
        strides_b[i] /= sizeof(npy_uint8);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_uint8 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_uint8 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 + val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_uint16_add(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                          npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                          int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_uint16 *b_data_ptr_saved = (npy_uint16 *)b_data_ptr;
    npy_uint16 *a_data_ptr_saved = (npy_uint16 *)a_data_ptr;
    npy_uint16 *result_data_ptr_saved = (npy_uint16 *)result_data_ptr;
    npy_uint16 *result_data_ptr_ = (npy_uint16 *)result_data_ptr;
    npy_uint16 *result_data_ptr_cpy = (npy_uint16 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_uint16);
        strides_b[i] /= sizeof(npy_uint16);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_uint16 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_uint16 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 + val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_uint32_add(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                          npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                          int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_uint32 *b_data_ptr_saved = (npy_uint32 *)b_data_ptr;
    npy_uint32 *a_data_ptr_saved = (npy_uint32 *)a_data_ptr;
    npy_uint32 *result_data_ptr_saved = (npy_uint32 *)result_data_ptr;
    npy_uint32 *result_data_ptr_ = (npy_uint32 *)result_data_ptr;
    npy_uint32 *result_data_ptr_cpy = (npy_uint32 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_uint32);
        strides_b[i] /= sizeof(npy_uint32);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_uint32 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_uint32 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 + val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_uint64_add(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                          npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                          int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_uint64 *b_data_ptr_saved = (npy_uint64 *)b_data_ptr;
    npy_uint64 *a_data_ptr_saved = (npy_uint64 *)a_data_ptr;
    npy_uint64 *result_data_ptr_saved = (npy_uint64 *)result_data_ptr;
    npy_uint64 *result_data_ptr_ = (npy_uint64 *)result_data_ptr;
    npy_uint64 *result_data_ptr_cpy = (npy_uint64 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_uint64);
        strides_b[i] /= sizeof(npy_uint64);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_uint64 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_uint64 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 + val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_int8_add(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                        npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                        int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_int8 *b_data_ptr_saved = (npy_int8 *)b_data_ptr;
    npy_int8 *a_data_ptr_saved = (npy_int8 *)a_data_ptr;
    npy_int8 *result_data_ptr_saved = (npy_int8 *)result_data_ptr;
    npy_int8 *result_data_ptr_ = (npy_int8 *)result_data_ptr;
    npy_int8 *result_data_ptr_cpy = (npy_int8 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_int8);
        strides_b[i] /= sizeof(npy_int8);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_int8 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_int8 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 + val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_int32_add(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                         npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                         int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_int32 *b_data_ptr_saved = (npy_int32 *)b_data_ptr;
    npy_int32 *a_data_ptr_saved = (npy_int32 *)a_data_ptr;
    npy_int32 *result_data_ptr_saved = (npy_int32 *)result_data_ptr;
    npy_int32 *result_data_ptr_ = (npy_int32 *)result_data_ptr;
    npy_int32 *result_data_ptr_cpy = (npy_int32 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_int32);
        strides_b[i] /= sizeof(npy_int32);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_int32 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_int32 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 + val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_int64_add(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                         npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                         int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_int64 *b_data_ptr_saved = (npy_int64 *)b_data_ptr;
    npy_int64 *a_data_ptr_saved = (npy_int64 *)a_data_ptr;
    npy_int64 *result_data_ptr_saved = (npy_int64 *)result_data_ptr;
    npy_int64 *result_data_ptr_ = (npy_int64 *)result_data_ptr;
    npy_int64 *result_data_ptr_cpy = (npy_int64 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_int64);
        strides_b[i] /= sizeof(npy_int64);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_int64 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_int64 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 + val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_int16_add(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                         npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                         int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_int16 *b_data_ptr_saved = (npy_int16 *)b_data_ptr;
    npy_int16 *a_data_ptr_saved = (npy_int16 *)a_data_ptr;
    npy_int16 *result_data_ptr_saved = (npy_int16 *)result_data_ptr;
    npy_int16 *result_data_ptr_ = (npy_int16 *)result_data_ptr;
    npy_int16 *result_data_ptr_cpy = (npy_int16 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_int16);
        strides_b[i] /= sizeof(npy_int16);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_int16 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_int16 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 + val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_float16_add(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                           npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                           int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_float16 *b_data_ptr_saved = (npy_float16 *)b_data_ptr;
    npy_float16 *a_data_ptr_saved = (npy_float16 *)a_data_ptr;
    npy_float16 *result_data_ptr_saved = (npy_float16 *)result_data_ptr;
    npy_float16 *result_data_ptr_ = (npy_float16 *)result_data_ptr;
    npy_float16 *result_data_ptr_cpy = (npy_float16 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_float16);
        strides_b[i] /= sizeof(npy_float16);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_float16 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_float16 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 + val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_float32_add(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                           npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                           int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_float32 *b_data_ptr_saved = (npy_float32 *)b_data_ptr;
    npy_float32 *a_data_ptr_saved = (npy_float32 *)a_data_ptr;
    npy_float32 *result_data_ptr_saved = (npy_float32 *)result_data_ptr;
    npy_float32 *result_data_ptr_ = (npy_float32 *)result_data_ptr;
    npy_float32 *result_data_ptr_cpy = (npy_float32 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_float32);
        strides_b[i] /= sizeof(npy_float32);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_float32 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_float32 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 + val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_float64_add(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                           npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                           int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_float64 *b_data_ptr_saved = (npy_float64 *)b_data_ptr;
    npy_float64 *a_data_ptr_saved = (npy_float64 *)a_data_ptr;
    npy_float64 *result_data_ptr_saved = (npy_float64 *)result_data_ptr;
    npy_float64 *result_data_ptr_ = (npy_float64 *)result_data_ptr;
    npy_float64 *result_data_ptr_cpy = (npy_float64 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_float64);
        strides_b[i] /= sizeof(npy_float64);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_float64 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_float64 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 + val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_float128_add(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                            npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                            int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_longdouble *b_data_ptr_saved = (npy_longdouble *)b_data_ptr;
    npy_longdouble *a_data_ptr_saved = (npy_longdouble *)a_data_ptr;
    npy_longdouble *result_data_ptr_saved = (npy_longdouble *)result_data_ptr;
    npy_longdouble *result_data_ptr_ = (npy_longdouble *)result_data_ptr;
    npy_longdouble *result_data_ptr_cpy = (npy_longdouble *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_longdouble);
        strides_b[i] /= sizeof(npy_longdouble);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_longdouble val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_longdouble val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 + val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

/*============================================================= SUB =================================================================================*/
static void Broadcast_Standard_int8_sub(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                        npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                        int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_int8 *b_data_ptr_saved = (npy_int8 *)b_data_ptr;
    npy_int8 *a_data_ptr_saved = (npy_int8 *)a_data_ptr;
    npy_int8 *result_data_ptr_saved = (npy_int8 *)result_data_ptr;
    npy_int8 *result_data_ptr_ = (npy_int8 *)result_data_ptr;
    npy_int8 *result_data_ptr_cpy = (npy_int8 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_int8);
        strides_b[i] /= sizeof(npy_int8);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_int8 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_int8 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 - val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_int32_sub(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                         npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                         int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_int32 *b_data_ptr_saved = (npy_int32 *)b_data_ptr;
    npy_int32 *a_data_ptr_saved = (npy_int32 *)a_data_ptr;
    npy_int32 *result_data_ptr_saved = (npy_int32 *)result_data_ptr;
    npy_int32 *result_data_ptr_ = (npy_int32 *)result_data_ptr;
    npy_int32 *result_data_ptr_cpy = (npy_int32 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_int32);
        strides_b[i] /= sizeof(npy_int32);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_int32 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_int32 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 - val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_int64_sub(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                         npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                         int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_int64 *b_data_ptr_saved = (npy_int64 *)b_data_ptr;
    npy_int64 *a_data_ptr_saved = (npy_int64 *)a_data_ptr;
    npy_int64 *result_data_ptr_saved = (npy_int64 *)result_data_ptr;
    npy_int64 *result_data_ptr_ = (npy_int64 *)result_data_ptr;
    npy_int64 *result_data_ptr_cpy = (npy_int64 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_int64);
        strides_b[i] /= sizeof(npy_int64);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_int64 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_int64 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 - val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_int16_sub(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                         npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                         int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_int16 *b_data_ptr_saved = (npy_int16 *)b_data_ptr;
    npy_int16 *a_data_ptr_saved = (npy_int16 *)a_data_ptr;
    npy_int16 *result_data_ptr_saved = (npy_int16 *)result_data_ptr;
    npy_int16 *result_data_ptr_ = (npy_int16 *)result_data_ptr;
    npy_int16 *result_data_ptr_cpy = (npy_int16 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_int16);
        strides_b[i] /= sizeof(npy_int16);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_int16 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_int16 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 - val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_float16_sub(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                           npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                           int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_float16 *b_data_ptr_saved = (npy_float16 *)b_data_ptr;
    npy_float16 *a_data_ptr_saved = (npy_float16 *)a_data_ptr;
    npy_float16 *result_data_ptr_saved = (npy_float16 *)result_data_ptr;
    npy_float16 *result_data_ptr_ = (npy_float16 *)result_data_ptr;
    npy_float16 *result_data_ptr_cpy = (npy_float16 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_float16);
        strides_b[i] /= sizeof(npy_float16);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_float16 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_float16 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 - val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_float32_sub(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                           npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                           int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_float32 *b_data_ptr_saved = (npy_float32 *)b_data_ptr;
    npy_float32 *a_data_ptr_saved = (npy_float32 *)a_data_ptr;
    npy_float32 *result_data_ptr_saved = (npy_float32 *)result_data_ptr;
    npy_float32 *result_data_ptr_ = (npy_float32 *)result_data_ptr;
    npy_float32 *result_data_ptr_cpy = (npy_float32 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_float32);
        strides_b[i] /= sizeof(npy_float32);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_float32 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_float32 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 - val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_float64_sub(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                           npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                           int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_float64 *b_data_ptr_saved = (npy_float64 *)b_data_ptr;
    npy_float64 *a_data_ptr_saved = (npy_float64 *)a_data_ptr;
    npy_float64 *result_data_ptr_saved = (npy_float64 *)result_data_ptr;
    npy_float64 *result_data_ptr_ = (npy_float64 *)result_data_ptr;
    npy_float64 *result_data_ptr_cpy = (npy_float64 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_float64);
        strides_b[i] /= sizeof(npy_float64);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_float64 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_float64 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 - val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_float128_sub(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                            npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                            int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_longdouble *b_data_ptr_saved = (npy_longdouble *)b_data_ptr;
    npy_longdouble *a_data_ptr_saved = (npy_longdouble *)a_data_ptr;
    npy_longdouble *result_data_ptr_saved = (npy_longdouble *)result_data_ptr;
    npy_longdouble *result_data_ptr_ = (npy_longdouble *)result_data_ptr;
    npy_longdouble *result_data_ptr_cpy = (npy_longdouble *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_longdouble);
        strides_b[i] /= sizeof(npy_longdouble);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_longdouble val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_longdouble val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 - val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

/*============================================================= DIV =================================================================================*/
static void Broadcast_Standard_uint8_div(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                         npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                         int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_uint8 *b_data_ptr_saved = (npy_uint8 *)b_data_ptr;
    npy_uint8 *a_data_ptr_saved = (npy_uint8 *)a_data_ptr;
    npy_uint8 *result_data_ptr_saved = (npy_uint8 *)result_data_ptr;
    npy_uint8 *result_data_ptr_ = (npy_uint8 *)result_data_ptr;
    npy_uint8 *result_data_ptr_cpy = (npy_uint8 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_uint8);
        strides_b[i] /= sizeof(npy_uint8);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_uint8 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_uint8 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 / val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_uint16_div(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                          npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                          int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_uint16 *b_data_ptr_saved = (npy_uint16 *)b_data_ptr;
    npy_uint16 *a_data_ptr_saved = (npy_uint16 *)a_data_ptr;
    npy_uint16 *result_data_ptr_saved = (npy_uint16 *)result_data_ptr;
    npy_uint16 *result_data_ptr_ = (npy_uint16 *)result_data_ptr;
    npy_uint16 *result_data_ptr_cpy = (npy_uint16 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_uint16);
        strides_b[i] /= sizeof(npy_uint16);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_uint16 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_uint16 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 / val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_uint32_div(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                          npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                          int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_uint32 *b_data_ptr_saved = (npy_uint32 *)b_data_ptr;
    npy_uint32 *a_data_ptr_saved = (npy_uint32 *)a_data_ptr;
    npy_uint32 *result_data_ptr_saved = (npy_uint32 *)result_data_ptr;
    npy_uint32 *result_data_ptr_ = (npy_uint32 *)result_data_ptr;
    npy_uint32 *result_data_ptr_cpy = (npy_uint32 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_uint16);
        strides_b[i] /= sizeof(npy_uint16);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_uint32 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_uint32 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 / val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_uint64_div(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                          npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                          int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_uint64 *b_data_ptr_saved = (npy_uint64 *)b_data_ptr;
    npy_uint64 *a_data_ptr_saved = (npy_uint64 *)a_data_ptr;
    npy_uint64 *result_data_ptr_saved = (npy_uint64 *)result_data_ptr;
    npy_uint64 *result_data_ptr_ = (npy_uint64 *)result_data_ptr;
    npy_uint64 *result_data_ptr_cpy = (npy_uint64 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_uint64);
        strides_b[i] /= sizeof(npy_uint64);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_uint64 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_uint64 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 / val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_int8_div(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                        npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                        int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_int8 *b_data_ptr_saved = (npy_int8 *)b_data_ptr;
    npy_int8 *a_data_ptr_saved = (npy_int8 *)a_data_ptr;
    npy_int8 *result_data_ptr_saved = (npy_int8 *)result_data_ptr;
    npy_int8 *result_data_ptr_ = (npy_int8 *)result_data_ptr;
    npy_int8 *result_data_ptr_cpy = (npy_int8 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_int8);
        strides_b[i] /= sizeof(npy_int8);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_int8 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_int8 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 / val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_int32_div(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                         npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                         int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_int32 *b_data_ptr_saved = (npy_int32 *)b_data_ptr;
    npy_int32 *a_data_ptr_saved = (npy_int32 *)a_data_ptr;
    npy_int32 *result_data_ptr_saved = (npy_int32 *)result_data_ptr;
    npy_int32 *result_data_ptr_ = (npy_int32 *)result_data_ptr;
    npy_int32 *result_data_ptr_cpy = (npy_int32 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_int32);
        strides_b[i] /= sizeof(npy_int32);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_int32 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_int32 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 / val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_int64_div(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                         npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                         int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_int64 *b_data_ptr_saved = (npy_int64 *)b_data_ptr;
    npy_int64 *a_data_ptr_saved = (npy_int64 *)a_data_ptr;
    npy_int64 *result_data_ptr_saved = (npy_int64 *)result_data_ptr;
    npy_int64 *result_data_ptr_ = (npy_int64 *)result_data_ptr;
    npy_int64 *result_data_ptr_cpy = (npy_int64 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_int64);
        strides_b[i] /= sizeof(npy_int64);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_int64 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_int64 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 / val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_int16_div(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                         npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                         int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_int16 *b_data_ptr_saved = (npy_int16 *)b_data_ptr;
    npy_int16 *a_data_ptr_saved = (npy_int16 *)a_data_ptr;
    npy_int16 *result_data_ptr_saved = (npy_int16 *)result_data_ptr;
    npy_int16 *result_data_ptr_ = (npy_int16 *)result_data_ptr;
    npy_int16 *result_data_ptr_cpy = (npy_int16 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_int16);
        strides_b[i] /= sizeof(npy_int16);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_int16 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_int16 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 / val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_float16_div(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                           npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                           int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_float16 *b_data_ptr_saved = (npy_float16 *)b_data_ptr;
    npy_float16 *a_data_ptr_saved = (npy_float16 *)a_data_ptr;
    npy_float16 *result_data_ptr_saved = (npy_float16 *)result_data_ptr;
    npy_float16 *result_data_ptr_ = (npy_float16 *)result_data_ptr;
    npy_float16 *result_data_ptr_cpy = (npy_float16 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_float16);
        strides_b[i] /= sizeof(npy_float16);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_float16 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_float16 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 / val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_float32_div(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                           npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                           int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_float32 *b_data_ptr_saved = (npy_float32 *)b_data_ptr;
    npy_float32 *a_data_ptr_saved = (npy_float32 *)a_data_ptr;
    npy_float32 *result_data_ptr_saved = (npy_float32 *)result_data_ptr;
    npy_float32 *result_data_ptr_ = (npy_float32 *)result_data_ptr;
    npy_float32 *result_data_ptr_cpy = (npy_float32 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_float32);
        strides_b[i] /= sizeof(npy_float32);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_float32 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_float32 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 / val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_float64_div(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                           npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                           int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_float64 *b_data_ptr_saved = (npy_float64 *)b_data_ptr;
    npy_float64 *a_data_ptr_saved = (npy_float64 *)a_data_ptr;
    npy_float64 *result_data_ptr_saved = (npy_float64 *)result_data_ptr;
    npy_float64 *result_data_ptr_ = (npy_float64 *)result_data_ptr;
    npy_float64 *result_data_ptr_cpy = (npy_float64 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_float64);
        strides_b[i] /= sizeof(npy_float64);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_float64 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_float64 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 / val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_float128_div(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                            npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                            int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_longdouble *b_data_ptr_saved = (npy_longdouble *)b_data_ptr;
    npy_longdouble *a_data_ptr_saved = (npy_longdouble *)a_data_ptr;
    npy_longdouble *result_data_ptr_saved = (npy_longdouble *)result_data_ptr;
    npy_longdouble *result_data_ptr_ = (npy_longdouble *)result_data_ptr;
    npy_longdouble *result_data_ptr_cpy = (npy_longdouble *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_longdouble);
        strides_b[i] /= sizeof(npy_longdouble);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_longdouble val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_longdouble val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 / val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

/*============================================================= MUL =================================================================================*/
static void Broadcast_Standard_uint8_mul(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                         npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                         int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_uint8 *b_data_ptr_saved = (npy_uint8 *)b_data_ptr;
    npy_uint8 *a_data_ptr_saved = (npy_uint8 *)a_data_ptr;
    npy_uint8 *result_data_ptr_saved = (npy_uint8 *)result_data_ptr;
    npy_uint8 *result_data_ptr_ = (npy_uint8 *)result_data_ptr;
    npy_uint8 *result_data_ptr_cpy = (npy_uint8 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_uint8);
        strides_b[i] /= sizeof(npy_uint8);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_uint8 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_uint8 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 * val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_uint16_mul(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                          npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                          int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_uint16 *b_data_ptr_saved = (npy_uint16 *)b_data_ptr;
    npy_uint16 *a_data_ptr_saved = (npy_uint16 *)a_data_ptr;
    npy_uint16 *result_data_ptr_saved = (npy_uint16 *)result_data_ptr;
    npy_uint16 *result_data_ptr_ = (npy_uint16 *)result_data_ptr;
    npy_uint16 *result_data_ptr_cpy = (npy_uint16 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_uint16);
        strides_b[i] /= sizeof(npy_uint16);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_uint16 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_uint16 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 * val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_uint32_mul(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                          npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                          int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_uint32 *b_data_ptr_saved = (npy_uint32 *)b_data_ptr;
    npy_uint32 *a_data_ptr_saved = (npy_uint32 *)a_data_ptr;
    npy_uint32 *result_data_ptr_saved = (npy_uint32 *)result_data_ptr;
    npy_uint32 *result_data_ptr_ = (npy_uint32 *)result_data_ptr;
    npy_uint32 *result_data_ptr_cpy = (npy_uint32 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_uint32);
        strides_b[i] /= sizeof(npy_uint32);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_uint32 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_uint32 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 * val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_uint64_mul(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                          npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                          int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_uint64 *b_data_ptr_saved = (npy_uint64 *)b_data_ptr;
    npy_uint64 *a_data_ptr_saved = (npy_uint64 *)a_data_ptr;
    npy_uint64 *result_data_ptr_saved = (npy_uint64 *)result_data_ptr;
    npy_uint64 *result_data_ptr_ = (npy_uint64 *)result_data_ptr;
    npy_uint64 *result_data_ptr_cpy = (npy_uint64 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_uint64);
        strides_b[i] /= sizeof(npy_uint64);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_uint64 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_uint64 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 * val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_int8_mul(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                        npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                        int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_int8 *b_data_ptr_saved = (npy_int8 *)b_data_ptr;
    npy_int8 *a_data_ptr_saved = (npy_int8 *)a_data_ptr;
    npy_int8 *result_data_ptr_saved = (npy_int8 *)result_data_ptr;
    npy_int8 *result_data_ptr_ = (npy_int8 *)result_data_ptr;
    npy_int8 *result_data_ptr_cpy = (npy_int8 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_int8);
        strides_b[i] /= sizeof(npy_int8);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_int8 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_int8 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 * val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_int32_mul(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                         npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                         int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_int32 *b_data_ptr_saved = (npy_int32 *)b_data_ptr;
    npy_int32 *a_data_ptr_saved = (npy_int32 *)a_data_ptr;
    npy_int32 *result_data_ptr_saved = (npy_int32 *)result_data_ptr;
    npy_int32 *result_data_ptr_ = (npy_int32 *)result_data_ptr;
    npy_int32 *result_data_ptr_cpy = (npy_int32 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_int32);
        strides_b[i] /= sizeof(npy_int32);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_int32 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_int32 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 * val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_int64_mul(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                         npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                         int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_int64 *b_data_ptr_saved = (npy_int64 *)b_data_ptr;
    npy_int64 *a_data_ptr_saved = (npy_int64 *)a_data_ptr;
    npy_int64 *result_data_ptr_saved = (npy_int64 *)result_data_ptr;
    npy_int64 *result_data_ptr_ = (npy_int64 *)result_data_ptr;
    npy_int64 *result_data_ptr_cpy = (npy_int64 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_int64);
        strides_b[i] /= sizeof(npy_int64);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_int64 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_int64 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 * val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_int16_mul(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                         npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                         int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_int16 *b_data_ptr_saved = (npy_int16 *)b_data_ptr;
    npy_int16 *a_data_ptr_saved = (npy_int16 *)a_data_ptr;
    npy_int16 *result_data_ptr_saved = (npy_int16 *)result_data_ptr;
    npy_int16 *result_data_ptr_ = (npy_int16 *)result_data_ptr;
    npy_int16 *result_data_ptr_cpy = (npy_int16 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_int16);
        strides_b[i] /= sizeof(npy_int16);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_int16 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_int16 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 * val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_float16_mul(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                           npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                           int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_float16 *b_data_ptr_saved = (npy_float16 *)b_data_ptr;
    npy_float16 *a_data_ptr_saved = (npy_float16 *)a_data_ptr;
    npy_float16 *result_data_ptr_saved = (npy_float16 *)result_data_ptr;
    npy_float16 *result_data_ptr_ = (npy_float16 *)result_data_ptr;
    npy_float16 *result_data_ptr_cpy = (npy_float16 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_float16);
        strides_b[i] /= sizeof(npy_float16);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_float16 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_float16 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 * val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_float32_mul(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                           npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                           int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_float32 *b_data_ptr_saved = (npy_float32 *)b_data_ptr;
    npy_float32 *a_data_ptr_saved = (npy_float32 *)a_data_ptr;
    npy_float32 *result_data_ptr_saved = (npy_float32 *)result_data_ptr;
    npy_float32 *result_data_ptr_ = (npy_float32 *)result_data_ptr;
    npy_float32 *result_data_ptr_cpy = (npy_float32 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_float32);
        strides_b[i] /= sizeof(npy_float32);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_float32 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_float32 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 * val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_float64_mul(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                           npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                           int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_float64 *b_data_ptr_saved = (npy_float64 *)b_data_ptr;
    npy_float64 *a_data_ptr_saved = (npy_float64 *)a_data_ptr;
    npy_float64 *result_data_ptr_saved = (npy_float64 *)result_data_ptr;
    npy_float64 *result_data_ptr_ = (npy_float64 *)result_data_ptr;
    npy_float64 *result_data_ptr_cpy = (npy_float64 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_float64);
        strides_b[i] /= sizeof(npy_float64);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_float64 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_float64 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 * val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_float128_mul(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                            npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                            int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_longdouble *b_data_ptr_saved = (npy_longdouble *)b_data_ptr;
    npy_longdouble *a_data_ptr_saved = (npy_longdouble *)a_data_ptr;
    npy_longdouble *result_data_ptr_saved = (npy_longdouble *)result_data_ptr;
    npy_longdouble *result_data_ptr_ = (npy_longdouble *)result_data_ptr;
    npy_longdouble *result_data_ptr_cpy = (npy_longdouble *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_longdouble);
        strides_b[i] /= sizeof(npy_longdouble);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_longdouble val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_longdouble val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 * val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

/*============================================================= MOD =================================================================================*/
static void Broadcast_Standard_uint8_mod(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                         npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                         int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_uint8 *b_data_ptr_saved = (npy_uint8 *)b_data_ptr;
    npy_uint8 *a_data_ptr_saved = (npy_uint8 *)a_data_ptr;
    npy_uint8 *result_data_ptr_saved = (npy_uint8 *)result_data_ptr;
    npy_uint8 *result_data_ptr_ = (npy_uint8 *)result_data_ptr;
    npy_uint8 *result_data_ptr_cpy = (npy_uint8 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_uint8);
        strides_b[i] /= sizeof(npy_uint8);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_uint8 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_uint8 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 % val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_uint16_mod(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                          npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                          int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_uint16 *b_data_ptr_saved = (npy_uint16 *)b_data_ptr;
    npy_uint16 *a_data_ptr_saved = (npy_uint16 *)a_data_ptr;
    npy_uint16 *result_data_ptr_saved = (npy_uint16 *)result_data_ptr;
    npy_uint16 *result_data_ptr_ = (npy_uint16 *)result_data_ptr;
    npy_uint16 *result_data_ptr_cpy = (npy_uint16 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_uint16);
        strides_b[i] /= sizeof(npy_uint16);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_uint16 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_uint16 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 % val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_uint32_mod(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                          npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                          int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_uint32 *b_data_ptr_saved = (npy_uint32 *)b_data_ptr;
    npy_uint32 *a_data_ptr_saved = (npy_uint32 *)a_data_ptr;
    npy_uint32 *result_data_ptr_saved = (npy_uint32 *)result_data_ptr;
    npy_uint32 *result_data_ptr_ = (npy_uint32 *)result_data_ptr;
    npy_uint32 *result_data_ptr_cpy = (npy_uint32 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_uint32);
        strides_b[i] /= sizeof(npy_uint32);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_uint32 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_uint32 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 % val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_uint64_mod(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                          npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                          int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_uint64 *b_data_ptr_saved = (npy_uint64 *)b_data_ptr;
    npy_uint64 *a_data_ptr_saved = (npy_uint64 *)a_data_ptr;
    npy_uint64 *result_data_ptr_saved = (npy_uint64 *)result_data_ptr;
    npy_uint64 *result_data_ptr_ = (npy_uint64 *)result_data_ptr;
    npy_uint64 *result_data_ptr_cpy = (npy_uint64 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_uint64);
        strides_b[i] /= sizeof(npy_uint64);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_uint64 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_uint64 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 % val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_int8_mod(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                        npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                        int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_int8 *b_data_ptr_saved = (npy_int8 *)b_data_ptr;
    npy_int8 *a_data_ptr_saved = (npy_int8 *)a_data_ptr;
    npy_int8 *result_data_ptr_saved = (npy_int8 *)result_data_ptr;
    npy_int8 *result_data_ptr_ = (npy_int8 *)result_data_ptr;
    npy_int8 *result_data_ptr_cpy = (npy_int8 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_int8);
        strides_b[i] /= sizeof(npy_int8);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_int8 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_int8 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 % val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_int32_mod(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                         npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                         int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_int32 *b_data_ptr_saved = (npy_int32 *)b_data_ptr;
    npy_int32 *a_data_ptr_saved = (npy_int32 *)a_data_ptr;
    npy_int32 *result_data_ptr_saved = (npy_int32 *)result_data_ptr;
    npy_int32 *result_data_ptr_ = (npy_int32 *)result_data_ptr;
    npy_int32 *result_data_ptr_cpy = (npy_int32 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_int32);
        strides_b[i] /= sizeof(npy_int32);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_int32 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_int32 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 % val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_int64_mod(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                         npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                         int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_int64 *b_data_ptr_saved = (npy_int64 *)b_data_ptr;
    npy_int64 *a_data_ptr_saved = (npy_int64 *)a_data_ptr;
    npy_int64 *result_data_ptr_saved = (npy_int64 *)result_data_ptr;
    npy_int64 *result_data_ptr_ = (npy_int64 *)result_data_ptr;
    npy_int64 *result_data_ptr_cpy = (npy_int64 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_int64);
        strides_b[i] /= sizeof(npy_int64);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_int64 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_int64 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 % val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_int16_mod(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                         npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                         int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_int16 *b_data_ptr_saved = (npy_int16 *)b_data_ptr;
    npy_int16 *a_data_ptr_saved = (npy_int16 *)a_data_ptr;
    npy_int16 *result_data_ptr_saved = (npy_int16 *)result_data_ptr;
    npy_int16 *result_data_ptr_ = (npy_int16 *)result_data_ptr;
    npy_int16 *result_data_ptr_cpy = (npy_int16 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_int16);
        strides_b[i] /= sizeof(npy_int16);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_int16 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_int16 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 % val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

static void Broadcast_Standard_float16_mod(char *a_data_ptr, char *b_data_ptr, char *result_data_ptr, npy_intp inner_loop_size,
                                           npy_intp *shape, npy_intp *strides_a, npy_intp *strides_b,
                                           int ndim, int axis, npy_intp left_prod)
{
    int max_dim = ndim - 1;
    int outer_start = max_dim - axis;
    npy_intp *shape_copy = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_a_cache = malloc(sizeof(npy_intp) * ndim);
    npy_intp *indice_b_cache = malloc(sizeof(npy_intp) * ndim);
    npy_float16 *b_data_ptr_saved = (npy_float16 *)b_data_ptr;
    npy_float16 *a_data_ptr_saved = (npy_float16 *)a_data_ptr;
    npy_float16 *result_data_ptr_saved = (npy_float16 *)result_data_ptr;
    npy_float16 *result_data_ptr_ = (npy_float16 *)result_data_ptr;
    npy_float16 *result_data_ptr_cpy = (npy_float16 *)result_data_ptr;
    npy_intp k = 0;
    for (int i = 0; i < ndim; i++)
    {
        strides_a[i] /= sizeof(npy_float16);
        strides_b[i] /= sizeof(npy_float16);
    }
    for (int i = 0; i < ndim; i++)
    {
        shape[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape[i];
        indice_b_cache[i] = strides_b[i] * shape[i];
    }
    npy_intp num_threads = left_prod < omp_get_max_threads() ? left_prod : omp_get_max_threads();
    Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)
    {
        int thread_id = omp_get_thread_num();
        npy_intp start_index = thread_id * (left_prod / num_threads) + min(thread_id, left_prod % num_threads);
        npy_intp end_index = start_index + left_prod / num_threads + (thread_id < left_prod % num_threads ? 1 : 0);
        result_data_ptr_ = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_data_ptr_ - result_data_ptr_saved;
        npy_intp *shape_copy1 = calloc(ndim, sizeof(npy_intp));
        for (int j = max_dim; j >= 0; j--)
        {
            shape_copy1[j] = prd % (shape[j] + 1);
            prd /= (shape[j] + 1);
            a_data_ptr_saved += shape_copy1[j] * strides_a[j];
            b_data_ptr_saved += shape_copy1[j] * strides_b[j];
        }

#pragma omp for schedule(static)
        for (k = 0; k < left_prod; k++)
        {
            for (int i = 0; i < inner_loop_size; i++)
            {
                npy_float16 val1 = *((b_data_ptr_saved + i * strides_b[max_dim]));
                npy_float16 val2 = *((a_data_ptr_saved + i * strides_a[max_dim]));
                *(result_data_ptr_ + i) = val1 % val2;
            }
            result_data_ptr_ += inner_loop_size;
            for (int j = outer_start; j >= 0; j--)
            {
                if (shape_copy1[j] < shape[j])
                {
                    shape_copy1[j]++;
                    a_data_ptr_saved += strides_a[j];
                    b_data_ptr_saved += strides_b[j];
                    break;
                }
                else
                {
                    shape_copy1[j] = 0;
                    a_data_ptr_saved -= indice_a_cache[j];
                    b_data_ptr_saved -= indice_b_cache[j];
                }
            }
        }
        free(shape_copy1);
    }
    Py_END_ALLOW_THREADS;
    free(indice_a_cache);
    free(indice_b_cache);
    free(shape_copy);
}

BroadcastFunction Broadcast_OperationPicker(int npy_type, int operation)
{
    switch (npy_type)
    {
    case NPY_BOOL:
        return NULL;
    case NPY_BYTE:
        switch (operation)
        {
        case ADD:
            return Broadcast_Standard_int8_add;
        case SUB:
            return Broadcast_Standard_int8_sub;
        case MUL:
            return Broadcast_Standard_int8_mul;
        case DIV:
            return Broadcast_Standard_int8_div;
        case MOD:
            return Broadcast_Standard_int8_mod;
        }
    case NPY_UBYTE:
        switch (operation)
        {
        case ADD:
            return Broadcast_Standard_uint8_add;
        case SUB:
            return NULL;
        case MUL:
            return Broadcast_Standard_uint8_mul;
        case DIV:
            return Broadcast_Standard_uint8_div;
        case MOD:
            return Broadcast_Standard_uint8_mod;
        }
    case NPY_SHORT:
        switch (operation)
        {
        case ADD:
            return Broadcast_Standard_int16_add;
        case SUB:
            return Broadcast_Standard_int16_sub;
        case MUL:
            return Broadcast_Standard_int16_mul;
        case DIV:
            return Broadcast_Standard_int16_div;
        case MOD:
            return Broadcast_Standard_int16_mod;
        }
    case NPY_USHORT:
        switch (operation)
        {
        case ADD:
            return Broadcast_Standard_uint16_add;
        case SUB:
            return NULL;
        case MUL:
            return Broadcast_Standard_uint16_mul;
        case DIV:
            return Broadcast_Standard_uint16_div;
        case MOD:
            return Broadcast_Standard_uint16_mod;
        }
    case NPY_INT:
        switch (operation)
        {
        case ADD:
            return Broadcast_Standard_int32_add;
        case SUB:
            return Broadcast_Standard_int32_sub;
        case MUL:
            return Broadcast_Standard_int32_mul;
        case DIV:
            return Broadcast_Standard_int32_div;
        case MOD:
            return Broadcast_Standard_int32_mod;
        }
    case NPY_UINT:
        switch (operation)
        {
        case ADD:
            return Broadcast_Standard_uint32_add;
        case SUB:
            return NULL;
        case MUL:
            return Broadcast_Standard_uint32_mul;
        case DIV:
            return Broadcast_Standard_uint32_div;
        case MOD:
            return Broadcast_Standard_uint32_mod;
        }
    case NPY_LONG:
        switch (operation)
        {
        case ADD:
            return Broadcast_Standard_int32_add;
        case SUB:
            return Broadcast_Standard_int32_sub;
        case MUL:
            return Broadcast_Standard_int32_mul;
        case DIV:
            return Broadcast_Standard_int32_div;
        case MOD:
            return Broadcast_Standard_int32_mod;
        }
    case NPY_ULONG:
        switch (operation)
        {
        case ADD:
            return Broadcast_Standard_uint32_add;
        case SUB:
            return NULL;
        case MUL:
            return Broadcast_Standard_uint32_mul;
        case DIV:
            return Broadcast_Standard_uint32_div;
        case MOD:
            return Broadcast_Standard_uint32_mod;
        }
    case NPY_LONGLONG:
        switch (operation)
        {
        case ADD:
            return Broadcast_Standard_int64_add;
        case SUB:
            return Broadcast_Standard_int64_sub;
        case MUL:
            return Broadcast_Standard_int64_mul;
        case DIV:
            return Broadcast_Standard_int64_div;
        case MOD:
            return Broadcast_Standard_int64_mod;
        }
    case NPY_ULONGLONG:
        switch (operation)
        {
        case ADD:
            return Broadcast_Standard_uint64_add;
        case SUB:
            return NULL;
        case MUL:
            return Broadcast_Standard_uint64_mul;
        case DIV:
            return Broadcast_Standard_uint64_div;
        case MOD:
            return Broadcast_Standard_uint64_mod;
        }
    case NPY_FLOAT:
        switch (operation)
        {
        case ADD:
            return Broadcast_Standard_float32_add;
        case SUB:
            return Broadcast_Standard_float32_sub;
        case MUL:
            return Broadcast_Standard_float32_mul;
        case DIV:
            return Broadcast_Standard_float32_div;
        }
    case NPY_DOUBLE:
        switch (operation)
        {
        case ADD:
            return Broadcast_Standard_float64_add;
        case SUB:
            return Broadcast_Standard_float64_sub;
        case MUL:
            return Broadcast_Standard_float64_mul;
        case DIV:
            return Broadcast_Standard_float64_div;
        }
    case NPY_LONGDOUBLE:
        switch (operation)
        {
        case ADD:
            return Broadcast_Standard_float128_add;
        case SUB:
            return Broadcast_Standard_float128_sub;
        case MUL:
            return Broadcast_Standard_float128_mul;
        case DIV:
            return Broadcast_Standard_float128_div;
        }
    }
}