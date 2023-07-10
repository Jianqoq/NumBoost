#define PY_ARRAY_UNIQUE_SYMBOL core_c
#define NO_IMPORT_ARRAY
#include <omp.h>
#include <numpy/npy_math.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include "numpy/ndarraytypes.h"
#include "set_Tensor_properties.h"
#include "tensor.h"
#include "core.h"
#include "operators.h"
extern np_method *NP_METHOD;
extern Array_Shape *ARRAY_SHAPE;
extern Power_Dict *POWER_DICT;
extern Log_Dict *LOG_DICT;

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

inline static Tensor *Generic_function_new_float(float (*func)(float), Tensor *self, PyArrayObject *array, Tensor *out, const char *grad_fn)
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
        npy_float *data2 = (npy_float *)PyArray_DATA(array2);
        npy_float *data = (npy_float *)PyArray_DATA(array);
        for (npy_intp i = 0; i < size; i++)
        {
            data2[i] = func(data[i]);
        }
        return __new_Tensor(self, (PyObject *)array2, NULL, self->require_grad ? grad_fn : "");
    }
    else
    {
        npy_float *data = (npy_float *)PyArray_DATA((PyArrayObject*)out->data); // ONLY WHEN MEM SIZE IS SAME OR SMALLER
        for (npy_intp i = 0; i < size; i++)
        {
            data[i] = func(data[i]);
        }
        out->grad_fn = out->require_grad ? grad_fn : "";
        Py_INCREF(out);
        return out;
    }
}

inline static Tensor *Generic_function_new_double(double (*func)(double), Tensor *self, PyArrayObject *array, Tensor *out, const char *grad_fn)
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
        double *data = (double *)PyArray_DATA(array);
        for (npy_intp i = 0; i < size; i++)
        {
            data2[i] = func(data[i]);
        }
        return __new_Tensor(self, (PyObject *)array2, NULL, self->require_grad ? grad_fn : "");
    }
    else
    {
        npy_float64 *data = (npy_float64 *)PyArray_DATA((PyArrayObject*)out->data); // ONLY WHEN MEM SIZE IS SAME OR SMALLER
        for (npy_intp i = 0; i < size; i++)
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
    Tensor *to_return = __new_Tensor(tensor, result, NULL, grad_fn);
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

inline static PyObject *Generic_function_new_float_internal(float (*func)(float), PyArrayObject *array, PyObject *out)
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
        npy_float *data2 = (npy_float *)PyArray_DATA(array2);
        npy_float *data = (npy_float *)PyArray_DATA(array);
        for (npy_intp i = 0; i < size; i++)
        {
            data2[i] = func(data[i]);
        }
        return (PyObject*)array2;
    }
    else
    {
        npy_float *data = (npy_float *)PyArray_DATA((PyArrayObject*)out); // ONLY WHEN MEM SIZE IS SAME OR SMALLER
        for (npy_intp i = 0; i < size; i++)
        {
            data[i] = func(data[i]);
        }
        Py_INCREF(out);
        return out;
    }
}

inline static PyObject *Generic_function_new_double_internal(double (*func)(double), PyArrayObject *array, PyObject *out)
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
        double *data = (double *)PyArray_DATA(array);
        for (npy_intp i = 0; i < size; i++)
        {
            data2[i] = func(data[i]);
        }
        PyObject *ret = (PyObject*)array2;
        return ret;
    }
    else
    {
        npy_float64 *data = (npy_float64 *)PyArray_DATA((PyArrayObject*)out); // ONLY WHEN MEM SIZE IS SAME OR SMALLER
        for (npy_intp i = 0; i < size; i++)
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
    Tensor *to_return = __new_Tensor(tensor, result, NULL, grad_fn);
    if (pre_shape != NULL)
    {
        store_array_shape(to_return, pre_shape2, ndim);
    }
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
    Tensor *to_return = __new_Tensor(tensor, result, NULL, "");
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
    Tensor *to_return = __new_Tensor(tensor, result, NULL, "");
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
    Tensor *to_return = __new_Tensor(tensor, result, NULL, "");
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
    Tensor *to_return = __new_Tensor(tensor, result, NULL, "");
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
    Tensor *to_return = __new_Tensor(tensor, result, NULL, "");
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
    Tensor *to_return = __new_Tensor(tensor, result, NULL, "");
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
        return Generic_function_new_float(sinf, tensor, array, out, "SinBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(sin, tensor, array, out, "SinBackward");
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
        return Generic_function_new_float(cosf, tensor, array, out, "CosBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(cos, tensor, array, out, "CosBackward");
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
        return Generic_function_new_float(tanf, tensor, array, out, "TanBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(tan, tensor, array, out, "TanBackward");
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
        return Generic_function_new_float(asinf, tensor, array, out, "ArcSinBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(asin, tensor, array, out, "ArcSinBackward");
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
        return Generic_function_new_float(acosf, tensor, array, out, "ArcCosBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(acos, tensor, array, out, "ArcCosBackward");
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
        return Generic_function_new_float(atanf, tensor, array, out, "ArcTanBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(atan, tensor, array, out, "ArcTanBackward");
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
        return Generic_function_new_float(sinhf, tensor, array, out, "SinhBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(sinh, tensor, array, out, "SinhBackward");
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
        return Generic_function_new_float(coshf, tensor, array, out, "CoshBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(cosh, tensor, array, out, "CoshBackward");
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
        return Generic_function_new_float(expf, tensor, array, out, "ExpBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(exp, tensor, array, out, "ExpBackward");
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
        return Generic_function_new_float(log10f, tensor, array, out, "Log10Backward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(log10, tensor, array, out, "Log10Backward");
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
        return Generic_function_new_float(logf, tensor, array, out, "LogBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(log, tensor, array, out, "LogBackward");
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
        return Generic_function_new_float(tanhf, tensor, array, out,"TanhBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(tanh, tensor, array, out,"TanhBackward");
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
        return Generic_function_new_float(asinhf, tensor, array, out,"ArcSinhBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(asinh, tensor, array, out,"ArcSinhBackward");
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
        return Generic_function_new_float(acoshf, tensor, array, out,"ArcCoshBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(acosh, tensor, array, out,"ArcCoshBackward");
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
        return Generic_function_new_float(atanhf, tensor, array, out,"ArcTanhBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(atanh, tensor, array, out,"ArcTanhBackward");
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
        return Generic_function_new_float(sqrtf, tensor, array, out,"SqrtBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(sqrt, tensor, array, out,"SqrtBackward");
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
        return Generic_function_new_float(asinhf, tensor, array, out,"ArcSinhBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(asinh, tensor, array, out,"ArcSinhBackward");
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
        return Generic_function_new_float(acoshf, tensor, array, out,"ArcCoshBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(acosh, tensor, array, out,"ArcCoshBackward");
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
        return Generic_function_new_float(atanhf, tensor, array, out,"ArcTanhBackward");
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double(atanh, tensor, array, out,"ArcTanhBackward");
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
    PyObject *result;
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
    Tensor *to_return = __new_Tensor(tensor, result, NULL, "PowBackward");
    Py_INCREF(power);
    store_power(to_return, power);
    return to_return;
}


static inline PyObject *internal_npy_cal_oneArgs(float(*func1)(float), double(*func2)(double), PyObject *args, PyObject *out){
    PyArrayObject *array = (PyArrayObject *)args;
    int typenum = PyArray_TYPE(array);
    if (typenum == NPY_FLOAT)
    {
        return Generic_function_new_float_internal(func1, array, out);
    }
    else if (typenum == NPY_DOUBLE)
    {
        return Generic_function_new_double_internal(func2, array, out);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Invalid type");
        return NULL;
    }
}


PyObject *_sin_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(sinf, sin, args, out);
}

PyObject *_cos_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(cosf, cos, args, out);
}

PyObject *_tan_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(tanf, tan, args, out);
}

PyObject *_asin_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(asinf, asin, args, out);
}

PyObject *_acos_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(acosf, acos, args, out);
}

PyObject *_atan_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(atanf, atan, args, out);
}

PyObject *_sinh_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(sinhf, sinh, args, out);
}

PyObject *_cosh_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(coshf, cosh, args, out);
}

PyObject *_exp_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(expf, exp, args, out);
}

PyObject *_log10_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(log10f, log10, args, out);
}

PyObject *_log_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(logf, log, args, out);
}

PyObject *_tanh_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(tanhf, tanh, args, out);
}

PyObject *_asinh_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(asinhf, asinh, args, out);
}

PyObject *_acosh_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(acoshf, acosh, args, out);
}

PyObject *_atanh_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(atanhf, atanh, args, out);
}

PyObject *_sqrt_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(sqrtf, sqrt, args, out);
}

PyObject *_arcsinh_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(asinhf, asinh, args, out);
}

PyObject *_arccosh_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(acoshf, acosh, args, out);
}

PyObject *_arctanh_internal(PyObject *args, PyObject *out)
{
    return internal_npy_cal_oneArgs(atanhf, atanh, args, out);
}

PyObject *_abs_internal(PyObject *args, PyObject *out)
{
    return Generic_function_internal(NP_METHOD->abs, args, out);
}