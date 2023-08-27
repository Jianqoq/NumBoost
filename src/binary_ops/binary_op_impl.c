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