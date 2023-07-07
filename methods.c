#define PY_SSIZE_T_CLEAN
#define PY_ARRAY_UNIQUE_SYMBOL core_c
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "core.h"
#include "methods.h"
extern np_method *NP_METHOD;
extern PyTypeObject Tensor_type;

Tensor *Generic_function(PyObject *func, const char* grad_fn, PyObject *self, PyObject *const *args, size_t nargsf) {
    PyObject *result;
    Tensor *tensor = NULL;
    if (nargsf == 1)
    {
        tensor = (Tensor *)args;
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
    Tensor *to_return = __new_Tensor(&Tensor_type, tensor, result, grad_fn);
    return to_return;
}

PyObject *reshape(PyObject *self, PyObject *const *args, size_t nargsf, PyObject *kwnames)
{
    size_t nargs = PyVectorcall_NARGS(nargsf);
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
    PyArrayObject *array = (PyArrayObject *)tensor->data;
    PyObject *result = PyArray_Newshape(array, &shape, order);
    if (result == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Error in reshape");
        return NULL;
    }
    return result;
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
    Tensor *to_return = __new_Tensor(&Tensor_type, tensor, result, "");
    return to_return;
}

Tensor *_max(PyObject *self, PyObject *const *args, size_t nargsf)
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
            result = PyArray_Max(tmp, axis, out);
        else
            result = PyArray_Max(tmp, axis, NULL);
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
        result = PyArray_Max(tmp, axis, NULL);
    }
    if (result == NULL)
    {
        return NULL;
    }
    Tensor *to_return = __new_Tensor(&Tensor_type, tensor, result, "");
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
    Tensor *to_return = __new_Tensor(&Tensor_type, tensor, result, "");
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
    Tensor *to_return = __new_Tensor(&Tensor_type, tensor, result, "");
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
    Tensor *to_return = __new_Tensor(&Tensor_type, tensor, result, "");
    return to_return;
}

Tensor *_sin(PyObject *self, PyObject *const *args, size_t nargsf)
{
    return Generic_function(NP_METHOD->sin, "SinBackward", self, args, nargsf);
}

Tensor *_cos(PyObject *self, PyObject *const *args, size_t nargsf)
{
    return Generic_function(NP_METHOD->cos, "CosBackward", self, args, nargsf);
}

Tensor *_tan(PyObject *self, PyObject *const *args, size_t nargsf)
{
    return Generic_function(NP_METHOD->tan, "TanBackward", self, args, nargsf);
}

Tensor *_asin(PyObject *self, PyObject *const *args, size_t nargsf)
{
    return Generic_function(NP_METHOD->asin, "ArcSinBackward", self, args, nargsf);
}

Tensor *_acos(PyObject *self, PyObject *const *args, size_t nargsf)
{
    return Generic_function(NP_METHOD->acos, "ArcCosBackward", self, args, nargsf);
}

Tensor *_atan(PyObject *self, PyObject *const *args, size_t nargsf)
{
    return Generic_function(NP_METHOD->atan, "ArcTanBackward", self, args, nargsf);
}

Tensor *_sinh(PyObject *self, PyObject *const *args, size_t nargsf)
{
    return Generic_function(NP_METHOD->sinh, "SinhBackward", self, args, nargsf);
}

Tensor *_cosh(PyObject *self, PyObject *const *args, size_t nargsf)
{
    return Generic_function(NP_METHOD->cosh, "CoshBackward", self, args, nargsf);
}

Tensor *_exp(PyObject *self, PyObject *const *args, size_t nargsf)
{
    return Generic_function(NP_METHOD->exp, "ExphBackward", self, args, nargsf);
}

Tensor *_log10(PyObject *self, PyObject *const *args, size_t nargsf)
{
    return Generic_function(NP_METHOD->log10, "Log10Backward", self, args, nargsf);
}

Tensor *_log(PyObject *self, PyObject *const *args, size_t nargsf)
{
    return Generic_function(NP_METHOD->log, "LogBackward", self, args, nargsf);
}

Tensor *_tanh(PyObject *self, PyObject *const *args, size_t nargsf)
{
    return Generic_function(NP_METHOD->tanh, "TanhBackward", self, args, nargsf);
}

Tensor *_asinh(PyObject *self, PyObject *const *args, size_t nargsf)
{
    return Generic_function(NP_METHOD->arcsinh, "ArcsinhBackward", self, args, nargsf);
}

Tensor *_acosh(PyObject *self, PyObject *const *args, size_t nargsf)
{
    return Generic_function(NP_METHOD->arccosh, "ArcCoshBackward", self, args, nargsf);
}

Tensor *_atanh(PyObject *self, PyObject *const *args, size_t nargsf)
{
    return Generic_function(NP_METHOD->arctanh, "ArcTanBackward", self, args, nargsf);
}

Tensor *_sqrt(PyObject *self, PyObject *const *args, size_t nargsf)
{
    return Generic_function(NP_METHOD->sqrt, "SqrtBackward", self, args, nargsf);
}

Tensor *_arcsinh(PyObject *self, PyObject *const *args, size_t nargsf)
{
    return Generic_function(NP_METHOD->arcsinh, "ArcSinhBackward", self, args, nargsf);
}

Tensor *_arccosh(PyObject *self, PyObject *const *args, size_t nargsf)
{
    return Generic_function(NP_METHOD->arccosh, "ArcCoshBackward", self, args, nargsf);
}

Tensor *_arctanh(PyObject *self, PyObject *const *args, size_t nargsf)
{
    return Generic_function(NP_METHOD->arctanh, "ArcTanhBackward", self, args, nargsf);
}