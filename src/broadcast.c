#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#define PY_SSIZE_T_CLEAN
#define min(a, b) ((a) < (b) ? (a) : (b))
#include "broadcast.h"
#include "op.h"
#include "broadcast_func_def.h"

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

PyArrayObject *numboost_broadcast(PyArrayObject *a, PyArrayObject *b, int enum_op)
{
    PyArrayObject *a_handler = NULL;
    PyArrayObject *b_handler = NULL;
    PyArray_Descr *descr_a = ((PyArrayObject_fields *)a)->descr;
    PyArray_Descr *descr_b = ((PyArrayObject_fields *)b)->descr;
    int result_type = binary_result_type(enum_op, ((PyArrayObject_fields *)a)->descr->type_num,
                                         ((PyArrayObject_fields *)a)->descr->elsize, ((PyArrayObject_fields *)b)->descr->type_num,
                                         ((PyArrayObject_fields *)b)->descr->elsize);
    PyArrayObject *a_ = a;
    PyArrayObject *b_ = b;
    if (descr_a->type_num != result_type)
    {
        as_type(&a, &a_, result_type);
        a_handler = a_;
    }
    if (descr_b->type_num != result_type)
    {
        as_type(&b, &b_, result_type);
        b_handler = b_;
    }
    PyArrayObject *result = broadcast_operations[enum_op][result_type](a_, b_, enum_op, result_type);
    if (a_handler)
        Py_DECREF(a_handler);
    if (b_handler)
        Py_DECREF(b_handler);
    return result;
}