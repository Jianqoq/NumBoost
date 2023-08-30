#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#include "../tensor.h"
#include <omp.h>
#include "mkl.h"
#include "../utils.h"
#include "../op.h"
#include "binary_op_impl.h"
#include "binary_op_def.h"
#include "../binary_ops/binary_node_fuse.h"

PyArrayObject *numboost_binary(PyArrayObject *a, PyArrayObject *b, int op_enum)
{
    PyArrayObject *a_ = a;
    PyArrayObject *b_ = b;
    PyArrayObject *a_handler = NULL;
    PyArrayObject *b_handler = NULL;
    PyArray_Descr *descr_a = ((PyArrayObject_fields *)a)->descr;
    PyArray_Descr *descr_b = ((PyArrayObject_fields *)b)->descr;
    int npy_type = binary_result_type(op_enum, ((PyArrayObject_fields *)a)->descr->type_num,
                                      ((PyArrayObject_fields *)a)->descr->elsize, ((PyArrayObject_fields *)b)->descr->type_num,
                                      ((PyArrayObject_fields *)b)->descr->elsize);
    if (npy_type == -1)
    {
        return NULL;
    }
    if (descr_a->type_num != npy_type)
    {
        as_type(&a, &a_, npy_type);
        a_handler = a_;
    }
    if (descr_b->type_num != npy_type)
    {
        as_type(&b, &b_, npy_type);
        b_handler = b_;
    }

    PyArrayObject *result = operations[op_enum][npy_type](a_, b_);

    if (a_handler)
        Py_DECREF(a_handler);
    if (b_handler)
        Py_DECREF(b_handler);
    if (!result)
        return NULL;
    return result;
}

PyArrayObject *numboost_binary_scalar_left(PyObject *a, PyArrayObject *b, int op_enum)
{
    Python_Number *a_ = NULL;
    PyArrayObject *b_ = b;
    PyArrayObject *handler = NULL;
    PyArray_Descr *descr_a = NULL;
    bool is_float = false;
    int npy_type = -1;
    if (PyLong_Check(a))
    {
        descr_a = PyArray_DescrFromType(NPY_LONG);
    }
    else if (PyFloat_Check(a))
    {
        descr_a = PyArray_DescrFromType(NPY_DOUBLE);
        is_float = true;
    }
    else if (PyBool_Check(a))
    {
        descr_a = PyArray_DescrFromType(NPY_BOOL);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Scalar type not supported");
        return NULL;
    }
    PyArray_Descr *descr_b = ((PyArrayObject_fields *)b)->descr;
    npy_type = binary_result_type(op_enum, descr_a->type_num, descr_a->elsize, descr_b->type_num, descr_b->elsize);
    a_ = (Python_Number *)malloc(sizeof(Python_Number));
    Numboost_CheckAlloc(a_);
    a_->type = npy_type;
    Store_Number(a, a_, Long, npy_type, is_float);
    if (descr_b->type_num != npy_type)
    {
        as_type(&b, &b_, npy_type);
        handler = b_;
    }
    PyArrayObject *result = operations_a_scalar[op_enum][npy_type](a_, b_);
    free(a_);
    if (handler)
        Py_DECREF(handler);
    if (!result)
        return NULL;
    return result;
}

PyArrayObject *numboost_binary_scalar_right(PyArrayObject *a, PyObject *b, int op_enum)
{
    PyArrayObject *a_ = a;
    Python_Number *b_ = NULL;
    PyArrayObject *handler = NULL;
    bool is_float = false;
    int npy_type = -1;
    PyArray_Descr *descr_b = NULL;
    if (PyLong_Check(b))
    {
        descr_b = PyArray_DescrFromType(NPY_LONG);
    }
    else if (PyFloat_Check(b))
    {
        descr_b = PyArray_DescrFromType(NPY_DOUBLE);
        is_float = true;
    }
    else if (PyBool_Check(b))
    {
        descr_b = PyArray_DescrFromType(NPY_BOOL);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Scalar type not supported");
        return NULL;
    }
    PyArray_Descr *descr_a = ((PyArrayObject_fields *)a)->descr;
    npy_type = binary_result_type(op_enum, descr_a->type_num, descr_a->elsize, descr_b->type_num, descr_b->elsize);
    b_ = (Python_Number *)malloc(sizeof(Python_Number));
    Numboost_CheckAlloc(b_);
    b_->type = npy_type;
    Store_Number(b, b_, Long, npy_type, is_float);
    if (descr_a->type_num != npy_type)
    {
        as_type(&a, &a_, npy_type);
        handler = a_;
    }
    PyArrayObject *result = operations_b_scalar[op_enum][npy_type](a_, b_);
    free(b_);
    if (handler)
        Py_DECREF(handler);
    if (!result)
        return NULL;
    return result;
}

PyArrayObject *numboost_binary_test(PyArrayObject *a, PyArrayObject *b, int op_enum)
{
    PyArrayObject *a_ = a;
    PyArrayObject *b_ = b;
    PyArrayObject *a_handler = NULL;
    PyArrayObject *b_handler = NULL;
    PyArray_Descr *descr_a = ((PyArrayObject_fields *)a)->descr;
    PyArray_Descr *descr_b = ((PyArrayObject_fields *)b)->descr;
    int npy_type = binary_result_type(op_enum, ((PyArrayObject_fields *)a)->descr->type_num,
                                      ((PyArrayObject_fields *)a)->descr->elsize, ((PyArrayObject_fields *)b)->descr->type_num,
                                      ((PyArrayObject_fields *)b)->descr->elsize);
    if (npy_type == -1)
    {
        return NULL;
    }
    if (descr_a->type_num != npy_type)
    {
        as_type(&a, &a_, npy_type);
        a_handler = a_;
    }
    if (descr_b->type_num != npy_type)
    {
        as_type(&b, &b_, npy_type);
        b_handler = b_;
    }

    PyArrayObject *result = operations[op_enum][npy_type](a_, b_);
    int ndim = PyArray_NDIM(result);
    npy_intp max_dim = ndim - 1;
    npy_intp *__strides_a = (((PyArrayObject_fields *)(a))->strides);
    npy_intp *strides_a = (npy_intp *)malloc(sizeof(npy_intp) * (((PyArrayObject_fields *)(result))->nd));
    npy_intp *indice_a_cache = (npy_intp *)malloc(sizeof(npy_intp) * ndim);
    npy_float *a_data_ptr_saved = (npy_float *)((void *)((PyArrayObject_fields *)(a))->data);
    memcpy(strides_a, __strides_a, sizeof(npy_intp) * (((PyArrayObject_fields *)(result))->nd));
    npy_intp *__strides_b = (((PyArrayObject_fields *)(b))->strides);
    npy_intp *strides_b = (npy_intp *)malloc(sizeof(npy_intp) * (((PyArrayObject_fields *)(result))->nd));
    npy_intp *indice_b_cache = (npy_intp *)malloc(sizeof(npy_intp) * ndim);
    npy_float *b_data_ptr_saved = (npy_float *)((void *)((PyArrayObject_fields *)(b))->data);
    memcpy(strides_b, __strides_b, sizeof(npy_intp) * (((PyArrayObject_fields *)(result))->nd));
    ;
    for (int i = 0; i < ndim; i++)
    {
        Replicate(Normalize_Strides_By_Type, npy_float, a, b);
    }
    Replicate(Retrieve_Last_Stride, max_dim, a, b);

    npy_intp _size = PyArray_SIZE(result);
    npy_intp *shape_cpy = (npy_intp *)malloc(sizeof(npy_intp) * PyArray_NDIM(result));
    npy_intp *__shape = PyArray_SHAPE(result);
    memcpy(shape_cpy, PyArray_SHAPE(result), sizeof(npy_intp) * PyArray_NDIM(result));
    int axis_sep = ndim - 1;
    npy_intp inner_loop_size = PyArray_SHAPE(result)[axis_sep];
    npy_intp outter_loop_size = _size / inner_loop_size;
    npy_intp outer_start = max_dim - axis_sep;
    npy_intp *shape_copy = (npy_intp *)malloc(sizeof(npy_intp) * ndim);
    npy_float *result_data_ptr_saved = (npy_float *)PyArray_DATA(result);
    npy_float *result_data_ptr_ = (npy_float *)PyArray_DATA(result);
    npy_float *result_data_ptr_cpy = (npy_float *)PyArray_DATA(result);

    for (int i = 0; i < ndim; i++)
    {
        shape_cpy[i]--;
        shape_copy[i] = 0;
        Replicate2(Cache_Indice, i, shape_cpy, a, b);
    }
    npy_intp k = 0;
    npy_intp num_threads = outter_loop_size < omp_get_max_threads() ? outter_loop_size : omp_get_max_threads();
    npy_float **result_ptr_ = (npy_float **)malloc(sizeof(npy_float *) * num_threads);
    npy_intp **current_shape_process_ = (npy_intp **)malloc(sizeof(npy_intp *) * num_threads);
    for (npy_intp id = 0; id < num_threads; id++)
    {
        npy_intp start_index = id * (outter_loop_size / num_threads) + min(id, outter_loop_size % num_threads);
        npy_intp end_index = start_index + outter_loop_size / num_threads + (id < outter_loop_size % num_threads);
        result_ptr_[id] = result_data_ptr_cpy;
        result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp prd = result_ptr_[id] - result_data_ptr_saved;
        npy_intp *current_shape_process = (npy_intp *)calloc(ndim, sizeof(npy_intp));
        for (npy_intp j = max_dim; j >= 0; j--)
        {
            current_shape_process[j] = prd % __shape[j];
            prd /= __shape[j];
        }
        current_shape_process_[id] = current_shape_process;
    }
    Omp_Parallel(num_threads, result_data_ptr_, Replicate0(Empty, a, b, c, d, e, f, g))
    {
    }
}