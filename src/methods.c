#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "import_methods.h"
#include <omp.h>
#include "mkl_vml_functions.h"
#include <numpy/npy_math.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include "numpy/ndarraytypes.h"
#include "set_Tensor_properties.h"
#include "tensor.h"
#include "operators.h"
extern np_method *NP_METHOD;
extern Array_Shape *ARRAY_SHAPE;
extern Power_Dict *POWER_DICT;
extern Log_Dict *LOG_DICT;

inline bool not_in(long i, long *to_search, long range)
{
    long count = 0;
    if (to_search == NULL)
        return true;
    for (long j = 0; j < range; j++)
    {
        if (i == to_search[j])
        {
            count++;
        }
    }
    if (count == 0)
        return true;
    else
        return false;
}

void store_base(Tensor *key, PyObject *base)
{
    Log_Dict *s = NULL;
    if (LOG_DICT != NULL)
        HASH_FIND_PTR(LOG_DICT, &key, s);
    if (s == NULL)
    {
        s = (Log_Dict *)malloc(sizeof(Power_Dict));
        s->key = key;
        s->base = base;
        Py_INCREF(key);
        HASH_ADD_PTR(LOG_DICT, key, s);
    }
}

PyObject *get_base(Tensor *key)
{
    Log_Dict *s;
    HASH_FIND_PTR(LOG_DICT, &key, s);
    if (s == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Base not found in dict");
        return NULL;
    }
    return s->base;
}

void store_power(Tensor *key, PyObject *power)
{
    Power_Dict *s = NULL;
    if (POWER_DICT != NULL)
        HASH_FIND_PTR(POWER_DICT, &key, s);
    if (s == NULL)
    {
        s = (Power_Dict *)malloc(sizeof(Power_Dict));
        s->key = key;
        s->prev_power = power;
        Py_INCREF(key);
        HASH_ADD_PTR(POWER_DICT, key, s);
    }
}

PyObject *get_power(Tensor *key)
{
    Power_Dict *s;
    HASH_FIND_PTR(POWER_DICT, &key, s);
    if (s == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Power not found in dict");
        return NULL;
    }
    return s->prev_power;
}

void store_array_shape(Tensor *key, npy_intp *shape, int len)
{
    Array_Shape *s = NULL;
    if (ARRAY_SHAPE != NULL)
        HASH_FIND_PTR(ARRAY_SHAPE, &key, s);
    if (s == NULL)
    {
        s = (Array_Shape *)malloc(sizeof(Array_Shape));
        s->key = key;
        s->shape = shape;
        s->len = len;
        Py_INCREF(key);
        HASH_ADD_PTR(ARRAY_SHAPE, key, s);
    }
}

npy_intp *get_array_shape(Tensor *key)
{
    Array_Shape *s;
    HASH_FIND_PTR(ARRAY_SHAPE, &key, s);
    if (s == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Array shape not found in dict");
        return NULL;
    }
    return s->shape;
}

int *get_shape_len(Tensor *key)
{
    Array_Shape *s;
    HASH_FIND_PTR(ARRAY_SHAPE, &key, s);
    if (s == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Array shape not found in dict");
        return NULL;
    }
    return &s->len;
}

inline static Tensor *Generic_function_new_float(void (*vect_func)(const int, const float *, float *),
                                                 float (*func)(float),
                                                 Tensor *self,
                                                 PyArrayObject *array,
                                                 Tensor *out,
                                                 const char *grad_fn)
{
    npy_intp ndims = PyArray_NDIM(array);
    npy_intp *shape = PyArray_SHAPE(array);
    npy_intp size = 1;
    for (npy_intp i = 0; i < ndims; i++)
    {
        size *= shape[i];
    }
    if (out == NULL)
    {
        PyArrayObject *array2 = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT, 0);
        float *data2 = (float *)PyArray_DATA(array2);
        const float *data = (const float *)PyArray_DATA(array);
        vect_func((const int)size, data, data2); // need benchmark to see if needed to release GIL
        return (Tensor *)__new_Tensor(self, (PyObject *)array2, NULL, self->require_grad ? grad_fn : "");
    }
    else
    {
        long long i;

        npy_float *data = (npy_float *)PyArray_DATA((PyArrayObject *)out->data); // ONLY WHEN MEM SIZE IS SAME OR SMALLER
#pragma omp parallel for                                                         // need benchmark to see if needed to release GIL
        for (i = 0; i < size; i++)
        {
            data[i] = func(data[i]);
        }
        out->grad_fn = out->require_grad ? grad_fn : "";
        Py_INCREF(out);
        return out;
    }
}

inline static Tensor *Generic_function_new_double(void (*vect_func)(const int, const double *, double *),
                                                  double (*func)(double),
                                                  Tensor *self,
                                                  PyArrayObject *array,
                                                  Tensor *out,
                                                  const char *grad_fn)
{
    npy_intp ndims = PyArray_NDIM(array);
    npy_intp *shape = PyArray_SHAPE(array);
    npy_intp size = 1;
    for (npy_intp i = 0; i < ndims; i++)
    {
        size *= shape[i];
    }
    if (out == NULL)
    {
        PyArrayObject *array2 = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_DOUBLE, 0);
        double *data2 = (double *)PyArray_DATA(array2);
        const double *data = (const double *)PyArray_DATA(array);
        vect_func((const int)size, data, data2); // need benchmark to see if needed to release GIL
        return (Tensor *)__new_Tensor(self, (PyObject *)array2, NULL, self->require_grad ? grad_fn : "");
    }
    else
    {
        long long i;
        double *data = (double *)PyArray_DATA((PyArrayObject *)out->data); // ONLY WHEN MEM SIZE IS SAME OR SMALLER
#pragma omp parallel for
        for (i = 0; i < size; i++)
        {
            data[i] = func(data[i]);
        }
        out->grad_fn = out->require_grad ? grad_fn : "";
        Py_INCREF(out);
        return out;
    }
}

inline static Tensor *Generic_function(PyObject *func, const char *grad_fn, PyObject *self, PyObject *const *args, size_t nargsf)
{
    PyObject *result;
    Tensor *tensor = NULL;
    if (nargsf == 1)
    {
        tensor = (Tensor *)args[0];
        result = PyObject_CallOneArg(func, tensor->data);
        if (result == NULL)
        {
            return NULL;
        }
    }
    else
    {
        tensor = (Tensor *)args[0];
        Tensor *out = NULL;
        if (args[1] != Py_None)
            out = (Tensor *)args[1];
        if (out == NULL)
        {
            result = PyObject_CallFunctionObjArgs(func, tensor->data, Py_None, NULL);
        }
        else
        {
            result = PyObject_CallFunctionObjArgs(func, tensor->data, out->data, NULL);
        }
        if (result == NULL)
        {
            return NULL;
        }
    }
    if (!tensor->require_grad)
        grad_fn = "";
    Tensor *to_return = (Tensor *)__new_Tensor(tensor, result, NULL, grad_fn);
    return to_return;
}

inline static PyObject *Generic_function_internal(PyObject *func, PyObject *args, PyObject *out)
{
    PyObject *result;
    if (out == NULL)
    {
        result = PyObject_CallOneArg(func, args);
    }
    else
    {
        result = PyObject_CallFunctionObjArgs(func, args, out, NULL);
    }
    if (result == NULL)
    {
        return NULL;
    }
    return result;
}

inline static PyObject *Generic_function_new_float_internal(void (*vect_func)(const int, const float *, float *),
                                                            float (*func)(float),
                                                            PyArrayObject *array,
                                                            PyObject *out)
{
    npy_intp ndims = PyArray_NDIM(array);
    npy_intp *shape = PyArray_SHAPE(array);
    npy_intp size = 1;
    for (npy_intp i = 0; i < ndims; i++)
    {
        size *= shape[i];
    }
    if (out == NULL)
    {
        PyArrayObject *array2 = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT, 0);
        float *data2 = (float *)PyArray_DATA(array2);
        const float *data = (const float *)PyArray_DATA(array);
        vect_func((const int)size, data, data2); // need benchmark to see if needed to release GIL
        return (PyObject *)array2;
    }
    else
    {
        npy_intp i;
        npy_float *data = (npy_float *)PyArray_DATA((PyArrayObject *)out); // ONLY WHEN MEM SIZE IS SAME OR SMALLER
#pragma omp parallel for                                                   // need benchmark to see if needed to release GIL
        for (i = 0; i < size; i++)
        {
            data[i] = func(data[i]);
        }
        Py_INCREF(out);
        return out;
    }
}

inline static PyObject *Generic_function_new_double_internal(void (*vect_func)(const int, const double *, double *),
                                                             double (*func)(double),
                                                             PyArrayObject *array, PyObject *out)
{
    npy_intp ndims = PyArray_NDIM(array);
    npy_intp *shape = PyArray_SHAPE(array);
    npy_intp size = 1;
    for (npy_intp i = 0; i < ndims; i++)
    {
        size *= shape[i];
    }
    if (out == NULL)
    {
        PyArrayObject *array2 = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_DOUBLE, 0);
        double *data2 = (double *)PyArray_DATA(array2);
        const double *data = (const double *)PyArray_DATA(array);
        vect_func((const int)size, data, data2);
        PyObject *ret = (PyObject *)array2;
        return ret;
    }
    else
    {
        long long i;
        npy_float64 *data = (npy_float64 *)PyArray_DATA((PyArrayObject *)out); // ONLY WHEN MEM SIZE IS SAME OR SMALLER
#pragma omp parallel for
        for (i = 0; i < size; i++)
        {
            data[i] = func(data[i]);
        }
        Py_INCREF(out);
        return out;
    }
}

Tensor *reshape(PyObject *self, PyObject *const *args, size_t nargsf, PyObject *kwnames)
{
    size_t nargs = PyVectorcall_NARGS(nargsf);
    PyArrayObject *array;
    npy_intp *pre_shape = NULL;
    npy_intp *pre_shape2 = malloc(sizeof(npy_intp) * NPY_MAXDIMS);
    bool isNULL = kwnames == NULL;
    int order = 0;
    if (nargs < 2 && isNULL)
    {
        PyErr_SetString(PyExc_TypeError, "Expected at least 2 positional arguments");
        return NULL;
    }

    Tensor *tensor = (Tensor *)args[0];
    int length = (int)PyTuple_GET_SIZE(args[1]);
    npy_intp dims[NPY_MAXDIMS] = {0};
    for (uint8_t i = 0; i < length; i++)
    {
        long item = PyLong_AsLong(PyTuple_GET_ITEM(args[1], i));
        dims[i] = item;
    }
    PyArray_Dims shape = {dims, length};
    array = (PyArrayObject *)tensor->data;
    PyObject *result = PyArray_Newshape(array, &shape, order);
    if (result == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Error in reshape");
        return NULL;
    }
    const char *grad_fn = "ReshapeBackward";
    if (!tensor->require_grad)
        grad_fn = "";
    else
    {
        pre_shape = PyArray_SHAPE(array);
    }
    int ndim = (int)PyArray_NDIM(array);
    for (npy_intp i = 0; i < NPY_MAXDIMS; i++)
    {
        if (i < ndim)
            pre_shape2[i] = pre_shape[i];
        else
            pre_shape2[i] = 0;
    }
    Tensor *to_return = (Tensor *)__new_Tensor(tensor, result, NULL, grad_fn);
    if (pre_shape != NULL)
    {
        store_array_shape(to_return, pre_shape2, ndim);
    }
    return to_return;
}

Tensor *tensordot(PyObject *self, PyObject *const *args, size_t nargsf, PyObject *kwnames)
{
    if (nargsf != 3)
    {
        PyErr_SetString(PyExc_TypeError, "Expected 3 positional arguments");
        return NULL;
    }
    Tensor *tensor1 = (Tensor *)args[0];
    Tensor *tensor2 = (Tensor *)args[1];
    npy_intp *a_shape = NULL, *b_shape = NULL;
    long na = 1, nb = 1;
    long axes = 1;
    long *axes_a = NULL, *axes_b = NULL;
    PyObject *axes_a_tuple = NULL;
    PyObject *axes_b_tuple = NULL;
    printf("tensordot\n");
    if (PyIter_Check(args[2]))
    {
        axes_a_tuple = PyTuple_GET_ITEM(args[2], 0);
        axes_b_tuple = PyTuple_GET_ITEM(args[2], 1);
    }
    else
    {
        printf("tensordot2\n");
        axes = abs(PyLong_AsLong(args[2]));
        if (axes == -1 && PyErr_Occurred())
        {
            PyErr_SetString(PyExc_TypeError, "Invalid data type for axes");
            return NULL;
        }
        else
        {
            axes_a = malloc(sizeof(long) * axes);
            axes_b = malloc(sizeof(long) * axes);
            printf("axes: %ld\n", axes);
            if (axes < 0)
            {
                for (long i = 0; i < axes; i--)
                {
                    axes_a[i] = -axes + i;
                }
                for (long i = 0; i < axes; i++)
                {
                    axes_b[i] = i;
                }
            }
            else if (axes > 0)
            {
                for (long i = 0; i < axes; i++)
                {
                    axes_a[i] = -axes + i;
                }
                for (long i = 0; i < -axes; i--)
                {
                    axes_b[i] = i;
                }
            }
            else {
                na = 0;
                nb = 0;
                free(axes_a);
                free(axes_b);
                axes_a = NULL;
                axes_b = NULL;
            }
        }
    }
    printf("tensordot3 %p\n", axes_a);
    PyObject_Print(axes_a_tuple, stdout, 0);
    if (axes_a == NULL && axes_a_tuple!= NULL && PyIter_Check(axes_a_tuple))
    {
        printf("tensordot3.25\n");
        na = (long)PyObject_Length(axes_a_tuple);
        axes_a = malloc(sizeof(long) * na);
        for (Py_ssize_t i = 0; i < na; i++)
        {
            PyObject *item = PyIter_Next(axes_a_tuple);
            axes_a[i] = PyLong_AsLong(item);
            if (axes_a[i] == -1 && PyErr_Occurred())
            {
                PyErr_SetString(PyExc_TypeError, "Invalid data type for axes");
                return NULL;
            }
        }
    }
    else if (axes_a == NULL && axes && axes_a_tuple!= NULL)
    {
        printf("tensordot3.5\n");
        long *axes_a = malloc(sizeof(long) * 1);
        axes_a[0] = PyLong_AsLong(axes_a_tuple);
        if (axes_a[0] == -1 && PyErr_Occurred())
        {
            PyErr_SetString(PyExc_TypeError, "Invalid data type for axes");
            return NULL;
        }
    }
        printf("tensordot4\n");
    if (axes_b == NULL && axes_b_tuple!= NULL && PyIter_Check(axes_b_tuple))
    {
        na = (long)PyObject_Length(axes_b_tuple);
        axes_b = malloc(sizeof(long) * na);
        for (Py_ssize_t i = 0; i < na; i++)
        {
            PyObject *item = PyIter_Next(axes_b_tuple);
            axes_b[i] = PyLong_AsLong(item);
            if (axes_b[i] == -1 && PyErr_Occurred())
            {
                PyErr_SetString(PyExc_TypeError, "Invalid data type for axes");
                return NULL;
            }
        }
    }
    else if (axes_b == NULL && axes && axes_b_tuple!= NULL)
    {
        long *axes_b = malloc(sizeof(long) * 1);
        axes_b[0] = PyLong_AsLong(axes_b_tuple);
        if (axes_b[0] == -1 && PyErr_Occurred())
        {
            PyErr_SetString(PyExc_TypeError, "Invalid data type for axes");
            return NULL;
        }
    }
        printf("tensordot5\n");
    PyObject *a = PyArray_FromAny(tensor1->data, NULL, 0, 0, NPY_ARRAY_DEFAULT, NULL);
    PyObject *b = PyArray_FromAny(tensor2->data, NULL, 0, 0, NPY_ARRAY_DEFAULT, NULL);
    if (a == NULL || b == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "connot convert tensor to numpy array");
        return NULL;
    }
    PyArrayObject *a_ = (PyArrayObject *)a;
    PyArrayObject *b_ = (PyArrayObject *)b;
    a_shape = PyArray_SHAPE(a_);
    b_shape = PyArray_SHAPE(b_);
    int ndim_a = ((PyArrayObject_fields *)a_)->nd;
    int ndim_b = ((PyArrayObject_fields *)b_)->nd;
    bool shape_equal = true;
    printf("na and nb, %d != %d\n", na, nb);
    if (na != nb)
        shape_equal = false;
    else if (axes_a != NULL && axes_b != NULL)
    {
        for (int i = 0; i < na; i++)
        {
            printf("%d %d\n", axes_a[i], axes_b[i]);
            if (a_shape[axes_a[i]] != b_shape[axes_b[i]])
            {
                shape_equal = false;
                break;
            }
            if (axes_a[i] < 0)
                axes_a[i] += ndim_a;
            if (axes_b[i] < 0)
                axes_b[i] += ndim_b;
        }
    }
    if (!shape_equal)
    {
        PyErr_SetString(PyExc_TypeError, "shape-mismatch for sum");
        return NULL;
    }
    long a_len = 0;
    long *__notin = malloc(sizeof(long) * (ndim_a));
    for (int i = 0; i < ndim_a; i++)
    {
        if (not_in(i, axes_a, na))
            __notin[i] = i;
        else
            __notin[i] = -100;
    }
    // get len
    for (int i = 0; i < ndim_a; i++)
        if (__notin[i] != -100)
            a_len++;
    long *notin = malloc(sizeof(long) * (a_len));
    int index = 0;
    for (int i = 0; i < ndim_a; i++)
        if (__notin[i] != -100)
        {
            notin[index] = __notin[i];
            index++;
        }
    printf("at: na + len = %d\n", na + a_len);
    // newaxes_a
    long *newaxes_a = malloc(sizeof(long) * (na + a_len));
    int j = 0;
    for (j = 0; j < a_len; j++)
        newaxes_a[j] = notin[j];
    for (j; j < na + a_len; j++)
        newaxes_a[j] = axes_a[j];
    printf("newaxes_a:");
    for (int i = 0; i < na + a_len; i++)
        printf("%d ", newaxes_a[i]);
    printf("\n");
    npy_intp N2 = 1;
    for (long i = 0; i < na; i++)
    {
        long axis = axes_a[i];
        N2 *= a_shape[axis];
    }
    printf("N2 = %d\n", N2);
    // newshape_a
    npy_intp *newshape_a = malloc(sizeof(npy_intp) * 2);
    npy_intp multiply_reduce = 1;
    for (int i = 0; i < a_len; i++)
        multiply_reduce *= a_shape[notin[i]];
    newshape_a[0] = multiply_reduce;
    newshape_a[1] = N2;
    printf("newshape_a:");
    for (int i = 0; i < 2; i++)
        printf("%d ", newshape_a[i]);
    printf("\n");
    // old_a
    npy_intp *oldshape_a = malloc(sizeof(npy_intp) * a_len);
    for (int i = 0; i < a_len; i++)
        oldshape_a[i] = a_shape[notin[i]];
    free(__notin);
    free(notin);
    free(axes_a);
    PyArray_Dims at_dims = {newshape_a, 2};
    PyArray_Dims at_new_dims = {newaxes_a, a_len};
    printf("at processed\n");
    ////////////////////////////////////////////////////////////////////
    long b_len = 0;
    __notin = malloc(sizeof(long) * (ndim_b));
    for (int i = 0; i < ndim_b; i++)
    {
        if (not_in(i, axes_b, nb))
            __notin[i] = i;
        else
            __notin[i] = -100;
    }
    // get len
    for (int i = 0; i < ndim_b; i++)
        if (__notin[i] != -100)
            b_len++;
    notin = malloc(sizeof(long) * (b_len));
    index = 0;
    for (int i = 0; i < ndim_b; i++)
        if (__notin[i] != -100)
        {
            notin[index] = __notin[i];
            index++;
        }
    // newaxes_b
    npy_intp *newaxes_b = malloc(sizeof(long) * (nb + b_len));
    j = 0;
    for (j; j < b_len; j++)
        newaxes_b[j] = notin[j];
    for (j; j < na + b_len; j++)
        newaxes_b[j] = axes_b[j];
    printf("newaxes_b:");
    for (int i = 0; i < nb + b_len; i++)
        printf("%d ", newaxes_b[i]);
    printf("\n");
    N2 = 1;
    for (long i = 0; i < nb; i++)
    {
        long axis = axes_b[i];
        N2 *= b_shape[axis];
    }
    printf("N2 = %d\n", N2);
    // newshape_b
    npy_intp *newshape_b = malloc(sizeof(npy_intp) * 2);
    multiply_reduce = 1;
    for (int i = 0; i < b_len; i++)
        multiply_reduce *= b_shape[notin[i]];
    newshape_b[0] = N2;
    newshape_b[1] = multiply_reduce;
    printf("newshape_b:");
    for (int i = 0; i < 2; i++)
        printf("%d ", newshape_b[i]);
    printf("\n");
    // old_b
    npy_intp *oldshape_b = malloc(sizeof(npy_intp) * b_len);
    for (int i = 0; i < b_len; i++)
        oldshape_b[i] = b_shape[notin[i]];
    printf("oldshape_b:");
    for (int i = 0; i < b_len; i++)
        printf("%d ", oldshape_b[i]);
    printf("\n");
    PyArray_Dims bt_dims = {newshape_b, 2};
    PyArray_Dims bt_new_dims = {newaxes_b, b_len};
    free(__notin);
    free(notin);
    free(axes_b);
    ///////////////////////////////////////////////////////////////////
    PyObject *at_ = PyArray_Transpose(a_, &at_new_dims);
    PyObject *bt_ = PyArray_Transpose(b_, &bt_new_dims);
        PyObject_Print(at_, stdout, 0);
    PyObject_Print(bt_, stdout, 0);
    printf("\n");
    if (at_ == NULL || bt_ == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "transpose error");
        return NULL;
    }
    for (int i = 0; i < 2; i++)
        printf("%ld ", newshape_a[i]);
    printf("\n");
    for (int i = 0; i < 2; i++)
        printf("%ld ", newshape_b[i]);
    printf("\n");
    printf("at_dims.len = %d\n", at_dims.len);
    printf("bt_dims.len = %d\n", bt_dims.len);
    PyObject *at = PyArray_Newshape((PyArrayObject*)at_, &at_dims, 0);
    PyObject *bt = PyArray_Newshape((PyArrayObject*)bt_, &bt_dims, 0);
    PyObject_Print(at, stdout, 0);
    PyObject_Print(bt, stdout, 0);
    printf("\n");
    if (at == NULL || bt == NULL)
    {
        return NULL;
    }
    PyObject *res = PyArray_MatrixProduct(at, bt);
    PyObject_Print(res, stdout, 0);
    printf("\n");
    if (res == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "matmul error");
        return NULL;
    }
    int total_len = a_len + b_len;
    npy_intp *olds_merge_shape = malloc(sizeof(long) * (total_len));
    j = 0;
    for (j; j < a_len; j++)
        olds_merge_shape[j] = oldshape_a[j];
    for (j; j < total_len; j++)
        olds_merge_shape[j] = oldshape_b[j - a_len];
    PyArray_Dims olds_merge_dims = {olds_merge_shape, total_len};
    printf("olds_merge_dims:");
    for (int i = 0; i < total_len; i++)
        printf("%d ", olds_merge_dims.ptr[i]);
    printf("\n");
    printf("total_len = %d\n", total_len);

    PyObject *result = PyArray_Newshape(res, &olds_merge_dims, 0);
    Tensor *to_return = (Tensor *)__new_Tensor(args[0], result, NULL, "TensordotBackward");
    free(olds_merge_shape);
    Py_DECREF(at_);
    Py_DECREF(bt_);
    Py_DECREF(at);
    Py_DECREF(bt);
    return to_return;
}

PyObject *class_reshape(PyObject *self, PyObject *const *args, size_t nargsf, PyObject *kwnames)
{
    if (nargsf != 1)
    {
        PyErr_SetString(PyExc_TypeError, "Expected 1 positional arguments");
        return NULL;
    }
    Tensor *tensor = (Tensor *)args[0];
    PyObject *result = PyObject_CallOneArg(NP_METHOD->reshape, tensor->data);
    if (result == NULL)
    {
        return NULL;
    }
    return result;
}

PyObject *transpose(PyObject *self, PyObject *const *args, size_t nargsf, PyObject *kwnames)
{
    int nargs = (int)PyVectorcall_NARGS(nargsf);
    if (nargs < 2)
    {
        PyErr_SetString(PyExc_TypeError, "Expected at least 2 positional arguments");
        return NULL;
    }
    Tensor *tensor = (Tensor *)args[0];
    PyArrayObject *array = (PyArrayObject *)tensor->data;
    int length = nargs - 1;
    npy_intp *dims = malloc(sizeof(npy_intp) * length);
    for (uint8_t i = 1; i < length; i++)
    {
        long item = PyLong_AsLong(args[i]);
        dims[i - 1] = item;
    }
    PyArray_Dims shape = {dims, length};
    PyObject *result = PyArray_Transpose(array, &shape);
    free(dims);
    if (result == NULL)
    {
        return NULL;
    }
    return result;
}

Tensor *_sum(PyObject *self, PyObject *const *args, size_t nargsf)
{
    Tensor *tensor = (Tensor *)args[0];
    PyArrayObject *tmp = (PyArrayObject *)tensor->data;
    int axis = NPY_MAXDIMS;
    PyArray_Descr *descr = NULL;
    if (args[2] != Py_None)
    {
        PyArray_DescrConverter(args[2], &descr);
    }
    else
    {
        PyArrayObject_fields *fields = (PyArrayObject_fields *)tmp;
        descr = fields->descr;
    }
    if (descr == NULL)
        return NULL;
    int dtype_enum = descr->type_num;
    int ndims;
    uint8_t i;
    PyObject *result = NULL;
    PyArrayObject *out = NULL;
    if (args[1] != Py_None)
        axis = PyLong_AsLong(args[1]);
    if (args[3] != Py_None)
        out = (PyArrayObject *)args[3];
    if (PyArray_CheckAxis(tmp, &axis, 0) == NULL)
    {
        return NULL;
    };
    if (PyObject_IsTrue(args[4]))
    {
        npy_intp new_shape[NPY_MAXDIMS] = {0};
        if (out != NULL)
            result = PyArray_Sum(tmp, axis, dtype_enum, out);
        else
            result = PyArray_Sum(tmp, axis, dtype_enum, NULL);
        if (result == NULL)
            return NULL;
        PyArrayObject *r = (PyArrayObject *)result;
        npy_intp *shape = PyArray_SHAPE(r);
        ndims = PyArray_NDIM(r);
        for (i = 0; i < axis; i++)
        {
            new_shape[i] = shape[i];
        }
        new_shape[axis] = 1;
        for (i = 0; i < ndims - axis; i++)
        {
            new_shape[i + axis + 1] = shape[axis];
            axis++;
        }
        PyArray_Dims d = {new_shape, ndims + 1};
        result = PyArray_Newshape(r, &d, 0);
    }
    else
    {
        result = PyArray_Sum(tmp, axis, dtype_enum, out);
    }
    if (result == NULL)
    {
        return NULL;
    }
    Tensor *to_return = (Tensor *)__new_Tensor(tensor, result, NULL, "");
    return to_return;
}

Tensor *_max(PyObject *self, PyObject *const *args, size_t nargsf)
{
    Tensor *tensor = (Tensor *)args[0];
    PyArrayObject *tmp = (PyArrayObject *)tensor->data;
    int axis = NPY_MAXDIMS;
    PyArray_Descr *descr = NULL;
    if (args[2] != Py_None)
    {
        PyArray_DescrConverter(args[2], &descr);
    }
    else
    {
        PyArrayObject_fields *fields = (PyArrayObject_fields *)tmp;
        descr = fields->descr;
    }
    if (descr == NULL)
        return NULL;
    int dtype_enum = descr->type_num;
    int ndims;
    uint8_t i;
    PyObject *result = NULL;
    PyArrayObject *out = NULL;
    if (args[1] != Py_None)
        axis = PyLong_AsLong(args[1]);
    if (args[3] != Py_None)
        out = (PyArrayObject *)args[3];
    if (PyArray_CheckAxis(tmp, &axis, 0) == NULL)
    {
        return NULL;
    };
    if (PyObject_IsTrue(args[4]))
    {
        npy_intp new_shape[NPY_MAXDIMS] = {0};
        if (out != NULL)
            result = PyArray_Sum(tmp, axis, dtype_enum, out);
        else
            result = PyArray_Sum(tmp, axis, dtype_enum, NULL);
        if (result == NULL)
            return NULL;
        PyArrayObject *r = (PyArrayObject *)result;
        npy_intp *shape = PyArray_SHAPE(r);
        ndims = PyArray_NDIM(r);
        for (i = 0; i < axis; i++)
        {
            new_shape[i] = shape[i];
        }
        new_shape[axis] = 1;
        for (i = 0; i < ndims - axis; i++)
        {
            new_shape[i + axis + 1] = shape[axis];
            axis++;
        }
        PyArray_Dims d = {new_shape, ndims + 1};
        result = PyArray_Newshape(r, &d, 0);
    }
    else
    {
        result = PyArray_Sum(tmp, axis, dtype_enum, out);
    }
    if (result == NULL)
    {
        return NULL;
    }
    Tensor *to_return = (Tensor *)__new_Tensor(tensor, result, NULL, "");
    return to_return;
}

Tensor *_min(PyObject *self, PyObject *const *args, size_t nargsf)
{
    Tensor *tensor = (Tensor *)args[0];
    PyArrayObject *tmp = (PyArrayObject *)tensor->data;
    int axis = NPY_MAXDIMS;
    int ndims;
    uint8_t i;
    PyObject *result = NULL;
    PyArrayObject *out = NULL;
    if (args[1] != Py_None)
        axis = PyLong_AsLong(args[1]);
    if (args[3] != Py_None)
        out = (PyArrayObject *)args[3];
    if (PyArray_CheckAxis(tmp, &axis, 0) == NULL)
    {
        return NULL;
    };
    if (PyObject_IsTrue(args[2]))
    {
        npy_intp new_shape[NPY_MAXDIMS] = {0};
        if (out != NULL)
            result = PyArray_Min(tmp, axis, out);
        else
            result = PyArray_Min(tmp, axis, NULL);
        if (result == NULL)
            return NULL;
        PyArrayObject *r = (PyArrayObject *)result;
        npy_intp *shape = PyArray_SHAPE(r);
        ndims = PyArray_NDIM(r);
        for (i = 0; i < axis; i++)
        {
            new_shape[i] = shape[i];
        }
        new_shape[axis] = 1;
        for (i = 0; i < ndims - axis; i++)
        {
            new_shape[i + axis + 1] = shape[axis];
            axis++;
        }
        PyArray_Dims d = {new_shape, ndims + 1};
        result = PyArray_Newshape(r, &d, 0);
    }
    else
    {
        result = PyArray_Min(tmp, axis, NULL);
    }
    if (result == NULL)
    {
        return NULL;
    }
    Tensor *to_return = (Tensor *)__new_Tensor(tensor, result, NULL, "");
    return to_return;
}

Tensor *_mean(PyObject *self, PyObject *const *args, size_t nargsf)
{
    Tensor *tensor = (Tensor *)args[0];
    PyArrayObject *tmp = (PyArrayObject *)tensor->data;
    int axis = NPY_MAXDIMS;
    PyArray_Descr *descr = NULL;
    if (args[2] != Py_None)
    {
        PyArray_DescrConverter(args[2], &descr);
    }
    else
    {
        PyArrayObject_fields *fields = (PyArrayObject_fields *)tmp;
        descr = fields->descr;
    }
    if (descr == NULL)
        return NULL;
    int dtype_enum = descr->type_num;
    int ndims;
    uint8_t i;
    PyObject *result = NULL;
    PyArrayObject *out = NULL;
    if (args[1] != Py_None)
        axis = PyLong_AsLong(args[1]);
    if (args[3] != Py_None)
        out = (PyArrayObject *)args[3];
    if (PyArray_CheckAxis(tmp, &axis, 0) == NULL)
    {
        return NULL;
    };
    if (PyObject_IsTrue(args[4]))
    {
        npy_intp new_shape[NPY_MAXDIMS] = {0};
        if (out != NULL)
            result = PyArray_Mean(tmp, axis, dtype_enum, out);
        else
            result = PyArray_Mean(tmp, axis, dtype_enum, NULL);
        if (result == NULL)
            return NULL;
        PyArrayObject *r = (PyArrayObject *)result;
        npy_intp *shape = PyArray_SHAPE(r);
        ndims = PyArray_NDIM(r);
        for (i = 0; i < axis; i++)
        {
            new_shape[i] = shape[i];
        }
        new_shape[axis] = 1;
        for (i = 0; i < ndims - axis; i++)
        {
            new_shape[i + axis + 1] = shape[axis];
            axis++;
        }
        PyArray_Dims d = {new_shape, ndims + 1};
        result = PyArray_Newshape(r, &d, 0);
    }
    else
    {
        result = PyArray_Mean(tmp, axis, dtype_enum, out);
    }
    if (result == NULL)
    {
        return NULL;
    }
    Tensor *to_return = (Tensor *)__new_Tensor(tensor, result, NULL, "");
    return to_return;
}

Tensor *_argmax_wrapper(PyObject *self, PyObject *const *args, size_t nargsf)
{
    Tensor *tensor = (Tensor *)args[0];
    PyArrayObject *tmp = (PyArrayObject *)tensor->data;
    int axis = NPY_MAXDIMS;
    int ndims;
    uint8_t i;
    PyObject *result = NULL;
    PyArrayObject *out = NULL;
    if (args[1] != Py_None)
        axis = PyLong_AsLong(args[1]);
    if (args[3] != Py_None)
        out = (PyArrayObject *)args[3];
    if (PyArray_CheckAxis(tmp, &axis, 0) == NULL)
    {
        return NULL;
    };
    if (PyObject_IsTrue(args[2]))
    {
        npy_intp new_shape[NPY_MAXDIMS] = {0};
        if (out != NULL)
            result = PyArray_ArgMax(tmp, axis, out);
        else
            result = PyArray_ArgMax(tmp, axis, NULL);
        if (result == NULL)
            return NULL;
        PyArrayObject *r = (PyArrayObject *)result;
        npy_intp *shape = PyArray_SHAPE(r);
        ndims = PyArray_NDIM(r);
        for (i = 0; i < axis; i++)
        {
            new_shape[i] = shape[i];
        }
        new_shape[axis] = 1;
        for (i = 0; i < ndims - axis; i++)
        {
            new_shape[i + axis + 1] = shape[axis];
            axis++;
        }
        PyArray_Dims d = {new_shape, ndims + 1};
        result = PyArray_Newshape(r, &d, 0);
    }
    else
    {
        result = PyArray_ArgMax(tmp, axis, NULL);
    }
    if (result == NULL)
    {
        return NULL;
    }
    Tensor *to_return = (Tensor *)__new_Tensor(tensor, result, NULL, "");
    return to_return;
}

Tensor *_argmin_wrapper(PyObject *self, PyObject *const *args, size_t nargsf)
{
    Tensor *tensor = (Tensor *)args[0];
    PyArrayObject *tmp = (PyArrayObject *)tensor->data;
    int axis = NPY_MAXDIMS;
    int ndims;
    uint8_t i;
    PyObject *result = NULL;
    PyArrayObject *out = NULL;
    if (args[1] != Py_None)
        axis = PyLong_AsLong(args[1]);
    if (args[3] != Py_None)
        out = (PyArrayObject *)args[3];
    if (PyArray_CheckAxis(tmp, &axis, 0) == NULL)
    {
        return NULL;
    };
    if (PyObject_IsTrue(args[2]))
    {
        npy_intp new_shape[NPY_MAXDIMS] = {0};
        if (out != NULL)
            result = PyArray_ArgMin(tmp, axis, out);
        else
            result = PyArray_ArgMin(tmp, axis, NULL);
        if (result == NULL)
            return NULL;
        PyArrayObject *r = (PyArrayObject *)result;
        npy_intp *shape = PyArray_SHAPE(r);
        ndims = PyArray_NDIM(r);
        for (i = 0; i < axis; i++)
        {
            new_shape[i] = shape[i];
        }
        new_shape[axis] = 1;
        for (i = 0; i < ndims - axis; i++)
        {
            new_shape[i + axis + 1] = shape[axis];
            axis++;
        }
        PyArray_Dims d = {new_shape, ndims + 1};
        result = PyArray_Newshape(r, &d, 0);
    }
    else
    {
        result = PyArray_ArgMin(tmp, axis, NULL);
    }
    if (result == NULL)
    {
        return NULL;
    }
    Tensor *to_return = (Tensor *)__new_Tensor(tensor, result, NULL, "");
    return to_return;
}

Tensor *_sin(PyObject *self, PyObject *const *args, size_t nargsf)
{
    Tensor *tensor = (Tensor *)args[0];
    Tensor *out = nargsf > 1 && !Py_IsNone(args[1]) ? (Tensor *)args[1] : NULL;
    PyArrayObject *array = (PyArrayObject *)tensor->data;
    int typenum = PyArray_TYPE(array);
    if (typenum == NPY_FLOAT)
    {
        return Generic_function_new_float(vsSin, sinf, tensor, array, out, "SinBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(vdSin, sin, tensor, array, out, "SinBackward");
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Invalid type");
        return NULL;
    }
}

Tensor *_cos(PyObject *self, PyObject *const *args, size_t nargsf)
{
    Tensor *tensor = (Tensor *)args[0];
    Tensor *out = nargsf > 1 && !Py_IsNone(args[1]) ? (Tensor *)args[1] : NULL;
    PyArrayObject *array = (PyArrayObject *)tensor->data;
    int typenum = PyArray_TYPE(array);
    if (typenum == NPY_FLOAT)
    {
        return Generic_function_new_float(vsCos, cosf, tensor, array, out, "CosBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(vdCos, cos, tensor, array, out, "CosBackward");
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Invalid type");
        return NULL;
    }
}

Tensor *_tan(PyObject *self, PyObject *const *args, size_t nargsf)
{
    Tensor *tensor = (Tensor *)args[0];
    PyArrayObject *array = (PyArrayObject *)tensor->data;
    Tensor *out = nargsf > 1 && !Py_IsNone(args[1]) ? (Tensor *)args[1] : NULL;
    int typenum = PyArray_TYPE(array);
    if (typenum == NPY_FLOAT)
    {
        return Generic_function_new_float(vsTan, tanf, tensor, array, out, "TanBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(vdTan, tan, tensor, array, out, "TanBackward");
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Invalid type");
        return NULL;
    }
}

Tensor *_asin(PyObject *self, PyObject *const *args, size_t nargsf)
{
    Tensor *tensor = (Tensor *)args[0];
    PyArrayObject *array = (PyArrayObject *)tensor->data;
    Tensor *out = nargsf > 1 && !Py_IsNone(args[1]) ? (Tensor *)args[1] : NULL;
    int typenum = PyArray_TYPE(array);
    if (typenum == NPY_FLOAT)
    {
        return Generic_function_new_float(vsAsin, asinf, tensor, array, out, "ArcSinBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(vdAsin, asin, tensor, array, out, "ArcSinBackward");
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Invalid type");
        return NULL;
    }
}

Tensor *_acos(PyObject *self, PyObject *const *args, size_t nargsf)
{
    Tensor *tensor = (Tensor *)args[0];
    PyArrayObject *array = (PyArrayObject *)tensor->data;
    Tensor *out = nargsf > 1 && !Py_IsNone(args[1]) ? (Tensor *)args[1] : NULL;
    int typenum = PyArray_TYPE(array);
    if (typenum == NPY_FLOAT)
    {
        return Generic_function_new_float(vsAcos, acosf, tensor, array, out, "ArcCosBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(vdAcos, acos, tensor, array, out, "ArcCosBackward");
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Invalid type");
        return NULL;
    }
}

Tensor *_atan(PyObject *self, PyObject *const *args, size_t nargsf)
{
    Tensor *tensor = (Tensor *)args[0];
    PyArrayObject *array = (PyArrayObject *)tensor->data;
    Tensor *out = nargsf > 1 && !Py_IsNone(args[1]) ? (Tensor *)args[1] : NULL;
    int typenum = PyArray_TYPE(array);
    if (typenum == NPY_FLOAT)
    {
        return Generic_function_new_float(vsAtan, atanf, tensor, array, out, "ArcTanBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(vdAtan, atan, tensor, array, out, "ArcTanBackward");
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Invalid type");
        return NULL;
    }
}

Tensor *_sinh(PyObject *self, PyObject *const *args, size_t nargsf)
{
    Tensor *tensor = (Tensor *)args[0];
    PyArrayObject *array = (PyArrayObject *)tensor->data;
    Tensor *out = nargsf > 1 && !Py_IsNone(args[1]) ? (Tensor *)args[1] : NULL;
    int typenum = PyArray_TYPE(array);
    if (typenum == NPY_FLOAT)
    {
        return Generic_function_new_float(vsSinh, sinhf, tensor, array, out, "SinhBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(vdSinh, sinh, tensor, array, out, "SinhBackward");
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Invalid type");
        return NULL;
    }
}

Tensor *_cosh(PyObject *self, PyObject *const *args, size_t nargsf)
{
    Tensor *tensor = (Tensor *)args[0];
    PyArrayObject *array = (PyArrayObject *)tensor->data;
    Tensor *out = nargsf > 1 && !Py_IsNone(args[1]) ? (Tensor *)args[1] : NULL;
    int typenum = PyArray_TYPE(array);
    if (typenum == NPY_FLOAT)
    {
        return Generic_function_new_float(vsCosh, coshf, tensor, array, out, "CoshBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(vdCosh, cosh, tensor, array, out, "CoshBackward");
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Invalid type");
        return NULL;
    }
}

Tensor *_exp(PyObject *self, PyObject *const *args, size_t nargsf)
{
    Tensor *tensor = (Tensor *)args[0];
    PyArrayObject *array = (PyArrayObject *)tensor->data;
    Tensor *out = nargsf > 1 && !Py_IsNone(args[1]) ? (Tensor *)args[1] : NULL;
    int typenum = PyArray_TYPE(array);
    if (typenum == NPY_FLOAT)
    {
        return Generic_function_new_float(vsExp, expf, tensor, array, out, "ExpBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(vdExp, exp, tensor, array, out, "ExpBackward");
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Invalid type");
        return NULL;
    }
}

Tensor *_log10(PyObject *self, PyObject *const *args, size_t nargsf)
{
    Tensor *tensor = (Tensor *)args[0];
    PyArrayObject *array = (PyArrayObject *)tensor->data;
    Tensor *out = nargsf > 1 && !Py_IsNone(args[1]) ? (Tensor *)args[1] : NULL;
    int typenum = PyArray_TYPE(array);
    if (typenum == NPY_FLOAT)
    {
        return Generic_function_new_float(vsLog10, log10f, tensor, array, out, "Log10Backward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(vdLog10, log10, tensor, array, out, "Log10Backward");
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Invalid type");
        return NULL;
    }
}

Tensor *_log(PyObject *self, PyObject *const *args, size_t nargsf)
{
    Tensor *tensor = (Tensor *)args[0];
    PyArrayObject *array = (PyArrayObject *)tensor->data;
    Tensor *out = nargsf > 1 && !Py_IsNone(args[1]) ? (Tensor *)args[1] : NULL;
    int typenum = PyArray_TYPE(array);
    if (typenum == NPY_FLOAT)
    {
        return Generic_function_new_float(vsLn, logf, tensor, array, out, "LogBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(vdLn, log, tensor, array, out, "LogBackward");
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Invalid type");
        return NULL;
    }
}

Tensor *_tanh(PyObject *self, PyObject *const *args, size_t nargsf)
{
    Tensor *tensor = (Tensor *)args[0];
    PyArrayObject *array = (PyArrayObject *)tensor->data;
    Tensor *out = nargsf > 1 && !Py_IsNone(args[1]) ? (Tensor *)args[1] : NULL;
    int typenum = PyArray_TYPE(array);
    if (typenum == NPY_FLOAT)
    {
        return Generic_function_new_float(vsTanh, tanhf, tensor, array, out, "TanhBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(vdTanh, tanh, tensor, array, out, "TanhBackward");
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Invalid type");
        return NULL;
    }
}

Tensor *_asinh(PyObject *self, PyObject *const *args, size_t nargsf)
{
    Tensor *tensor = (Tensor *)args[0];
    PyArrayObject *array = (PyArrayObject *)tensor->data;
    Tensor *out = nargsf > 1 && !Py_IsNone(args[1]) ? (Tensor *)args[1] : NULL;
    int typenum = PyArray_TYPE(array);
    if (typenum == NPY_FLOAT)
    {
        return Generic_function_new_float(vsAsinh, asinhf, tensor, array, out, "ArcSinhBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(vdAsinh, asinh, tensor, array, out, "ArcSinhBackward");
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Invalid type");
        return NULL;
    }
}

Tensor *_acosh(PyObject *self, PyObject *const *args, size_t nargsf)
{
    Tensor *tensor = (Tensor *)args[0];
    PyArrayObject *array = (PyArrayObject *)tensor->data;
    Tensor *out = nargsf > 1 && !Py_IsNone(args[1]) ? (Tensor *)args[1] : NULL;
    int typenum = PyArray_TYPE(array);
    if (typenum == NPY_FLOAT)
    {
        return Generic_function_new_float(vsAcosh, acoshf, tensor, array, out, "ArcCoshBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(vdAcosh, acosh, tensor, array, out, "ArcCoshBackward");
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Invalid type");
        return NULL;
    }
}

Tensor *_atanh(PyObject *self, PyObject *const *args, size_t nargsf)
{
    Tensor *tensor = (Tensor *)args[0];
    PyArrayObject *array = (PyArrayObject *)tensor->data;
    Tensor *out = nargsf > 1 && !Py_IsNone(args[1]) ? (Tensor *)args[1] : NULL;
    int typenum = PyArray_TYPE(array);
    if (typenum == NPY_FLOAT)
    {
        return Generic_function_new_float(vsTanh, atanhf, tensor, array, out, "ArcTanhBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(vdTanh, atanh, tensor, array, out, "ArcTanhBackward");
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Invalid type");
        return NULL;
    }
}

Tensor *_sqrt(PyObject *self, PyObject *const *args, size_t nargsf)
{
    Tensor *tensor = (Tensor *)args[0];
    PyArrayObject *array = (PyArrayObject *)tensor->data;
    Tensor *out = nargsf > 1 && !Py_IsNone(args[1]) ? (Tensor *)args[1] : NULL;
    int typenum = PyArray_TYPE(array);
    if (typenum == NPY_FLOAT)
    {
        return Generic_function_new_float(vsSqrt, sqrtf, tensor, array, out, "SqrtBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(vdSqrt, sqrt, tensor, array, out, "SqrtBackward");
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Invalid type");
        return NULL;
    }
}

Tensor *_arcsinh(PyObject *self, PyObject *const *args, size_t nargsf)
{
    Tensor *tensor = (Tensor *)args[0];
    PyArrayObject *array = (PyArrayObject *)tensor->data;
    Tensor *out = nargsf > 1 && !Py_IsNone(args[1]) ? (Tensor *)args[1] : NULL;
    int typenum = PyArray_TYPE(array);
    if (typenum == NPY_FLOAT)
    {
        return Generic_function_new_float(vsAsinh, asinhf, tensor, array, out, "ArcSinhBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(vdAsinh, asinh, tensor, array, out, "ArcSinhBackward");
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Invalid type");
        return NULL;
    }
}

Tensor *_arccosh(PyObject *self, PyObject *const *args, size_t nargsf)
{
    Tensor *tensor = (Tensor *)args[0];
    PyArrayObject *array = (PyArrayObject *)tensor->data;
    Tensor *out = nargsf > 1 && !Py_IsNone(args[1]) ? (Tensor *)args[1] : NULL;
    int typenum = PyArray_TYPE(array);
    if (typenum == NPY_FLOAT)
    {
        return Generic_function_new_float(vsAcosh, acoshf, tensor, array, out, "ArcCoshBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(vdAcosh, acosh, tensor, array, out, "ArcCoshBackward");
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Invalid type");
        return NULL;
    }
}

Tensor *_arctanh(PyObject *self, PyObject *const *args, size_t nargsf)
{
    Tensor *tensor = (Tensor *)args[0];
    PyArrayObject *array = (PyArrayObject *)tensor->data;
    Tensor *out = nargsf > 1 && !Py_IsNone(args[1]) ? (Tensor *)args[1] : NULL;
    int typenum = PyArray_TYPE(array);
    if (typenum == NPY_FLOAT)
    {
        return Generic_function_new_float(vsAtanh, atanhf, tensor, array, out, "ArcTanhBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(vdAtanh, atanh, tensor, array, out, "ArcTanhBackward");
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Invalid type");
        return NULL;
    }
}

Tensor *_abs(PyObject *self, PyObject *const *args, size_t nargsf)
{
    return Generic_function(NP_METHOD->abs, "", self, args, nargsf);
}

Tensor *_pow(PyObject *self, PyObject *const *args, size_t nargsf)
{
    PyObject *result = NULL;
    Tensor *tensor = NULL;
    PyObject *power = NULL;
    if (nargsf == 2)
    {
        tensor = (Tensor *)args[0];
        power = args[1];
        if (Py_IS_TYPE(args[1], &Tensor_type))
        {
            Tensor *tmp = (Tensor *)args[1];
            power = tmp->data;
        }
    }
    else if (nargsf > 2)
    {
        tensor = (Tensor *)args[0];
        power = args[1];
        if (Py_IS_TYPE(args[1], &Tensor_type))
        {
            Tensor *tmp = (Tensor *)args[1];
            power = tmp->data;
        }
        Tensor *out = NULL;
        if (args[2] != Py_None)
            out = (Tensor *)args[2];
        if (out == NULL)
        {
            result = PyObject_CallFunctionObjArgs(NP_METHOD->power, tensor->data, power, Py_None, NULL);
        }
        else
        {
            result = PyObject_CallFunctionObjArgs(NP_METHOD->power, tensor->data, power, out->data, NULL);
        }
        if (result == NULL)
        {
            return NULL;
        }
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "pow expected at least 2 arguments, got 0");
        return NULL;
    }
    Tensor *to_return = (Tensor *)__new_Tensor(tensor, result, NULL, "PowBackward");
    if (to_return->require_grad)
        store_power(to_return, power);
    return to_return;
}

static inline PyObject *internal_npy_cal_oneArgs(void (*vect_func1)(const int, const double *, double *),
                                                 void (*vect_func2)(const int, const float *, float *),
                                                 float (*func1)(float), double (*func2)(double),
                                                 PyObject *args,
                                                 PyObject *out)
{
    PyArrayObject *array = (PyArrayObject *)args;
    int typenum = PyArray_TYPE(array);
    if (typenum == NPY_FLOAT)
    {
        return Generic_function_new_float_internal(vect_func2, func1, array, out);
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double_internal(vect_func1, func2, array, out);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Invalid type");
        return NULL;
    }
}

PyObject *_sin_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(vdSin, vsSin, sinf, sin, args, out);
}

PyObject *_cos_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(vdCos, vsCos, cosf, cos, args, out);
}

PyObject *_tan_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(vdTan, vsTan, tanf, tan, args, out);
}

PyObject *_asin_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(vdAsin, vsAsin, asinf, asin, args, out);
}

PyObject *_acos_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(vdAcos, vsAcos, acosf, acos, args, out);
}

PyObject *_atan_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(vdAtan, vsAtan, atanf, atan, args, out);
}

PyObject *_sinh_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(vdSinh, vsSinh, sinhf, sinh, args, out);
}

PyObject *_cosh_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(vdCosh, vsCosh, coshf, cosh, args, out);
}

PyObject *_exp_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(vdExp, vsExp, expf, exp, args, out);
}

PyObject *_log10_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(vdLog10, vsLog10, log10f, log10, args, out);
}

PyObject *_log_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(vdLn, vsLn, logf, log, args, out);
}

PyObject *_tanh_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(vdTanh, vsTanh, tanhf, tanh, args, out);
}

PyObject *_asinh_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(vdAsinh, vsAsinh, asinhf, asinh, args, out);
}

PyObject *_acosh_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(vdAcosh, vsAcosh, acoshf, acosh, args, out);
}

PyObject *_atanh_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(vdAtanh, vsAtanh, atanhf, atanh, args, out);
}

PyObject *_sqrt_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(vdSqrt, vsSqrt, sqrtf, sqrt, args, out);
}

PyObject *_abs_internal(PyObject *args, PyObject *out)
{
    return Generic_function_internal(NP_METHOD->abs, args, out);
}