

#ifndef PYTHON_MATH_MAGIC_H
#define PYTHON_MATH_MAGIC_H
#include "../tensor.h"
#include "../broadcast_ops/broadcast_impl.h"

PyObject *__new_Tensor(Tensor *tensor, PyObject *array, PyObject *to_y, const char *grad_fn);

PyObject *create_Tensor(Tensor *tensor, PyObject *other, PyObject *data,
                        const char *grad_fn);

PyObject *
Tensor__new__(PyTypeObject *type, PyObject *data);

PyObject *Tensor_Empty(PyObject *data);

PyObject *tensor_add(PyObject *self, PyObject *other);
PyObject *tensor_iadd(PyObject *self, PyObject *other);
PyObject *tensor_mul(PyObject *self, PyObject *other);
PyObject *tensor_imul(PyObject *self, PyObject *other);
PyObject *tensor_div(PyObject *self, PyObject *other);
PyObject *tensor_idiv(PyObject *self, PyObject *other);
PyObject *tensor_negative(PyObject *self);
PyObject *tensor_inegative(PyObject *self);
PyObject *tensor_sub(PyObject *self, PyObject *other);
PyObject *tensor_isub(PyObject *self, PyObject *other);
PyObject *tensor_pow(PyObject *self, PyObject *other);
PyObject *tensor_ipow(PyObject *self, PyObject *other);
PyObject *tensor_matmul(PyObject *self, PyObject *other);
PyObject *tensor_imatmul(PyObject *self, PyObject *other);
PyObject *tensor_positive(PyObject *self);
PyObject *tensor_absolute(PyObject *self);
PyObject *tensor_invert(PyObject *self);
PyObject *tensor_lshift(PyObject *self, PyObject *other);
PyObject *tensor_rshift(PyObject *self, PyObject *other);
PyObject *tensor_and(PyObject *self, PyObject *other);
PyObject *tensor_xor(PyObject *self, PyObject *other);
PyObject *tensor_or(PyObject *self, PyObject *other);
PyObject *tensor_int(PyObject *self);
PyObject *tensor_float(PyObject *self);
PyObject *tensor_remainder(PyObject *self, PyObject *other);
PyObject *tensor_ior(PyObject *self, PyObject *other);
PyObject *tensor_ixor(PyObject *self, PyObject *other);
PyObject *tensor_iand(PyObject *self, PyObject *other);
PyObject *tensor_ilshift(PyObject *self, PyObject *other);
PyObject *tensor_irshift(PyObject *self, PyObject *other);
PyObject *tensor_divmod(PyObject *self, PyObject *other);
PyObject *tensor_iremainder(PyObject *self, PyObject *other);
PyObject *tensor_floordiv(PyObject *self, PyObject *other);
PyObject *tensor_ifloordiv(PyObject *self, PyObject *other);

#define Generic_Binary_Operation(self, other, pynumber_method, op_enum, backward_name)                    \
    Tensor *tmp;                                                                                          \
    if (TRACK)                                                                                            \
    {                                                                                                     \
        PyObject *jaxarray = pynumber_method;                                                             \
        return jaxarray;                                                                                  \
    }                                                                                                     \
    PyObject *numpy_result = NULL;                                                                        \
    if (Py_IS_TYPE(other, Tensor_type) && Py_IS_TYPE(self, Tensor_type))                                  \
    {                                                                                                     \
        Tensor *_self = (Tensor *)self;                                                                   \
        PyArrayObject *a = (PyArrayObject *)_self->data;                                                  \
        tmp = (Tensor *)other;                                                                            \
        PyArrayObject *b = (PyArrayObject *)tmp->data;                                                    \
        bool equal = shape_isequal(PyArray_SHAPE(a), PyArray_SHAPE(b), PyArray_NDIM(a), PyArray_NDIM(b)); \
        if (!equal)                                                                                       \
        {                                                                                                 \
            numpy_result = (PyObject *)numboost_broadcast(a, b, op_enum);                                 \
        }                                                                                                 \
        else                                                                                              \
        {                                                                                                 \
            numpy_result = (PyObject *)numboost_binary(a, b, op_enum);                                    \
        }                                                                                                 \
        if (numpy_result == NULL)                                                                         \
            return NULL;                                                                                  \
        PyObject *new_tensor = create_Tensor(_self, other, numpy_result, backward_name);                       \
        return new_tensor;                                                                                \
    }                                                                                                     \
    else if (Py_IS_TYPE(other, Tensor_type) && PyArray_IsPythonNumber(self))                              \
    {                                                                                                     \
        tmp = (Tensor *)other;                                                                            \
        PyArrayObject *b = (PyArrayObject *)tmp->data;                                                    \
        numpy_result = (PyObject *)numboost_binary_scalar_left(self, b, op_enum);                         \
        if (numpy_result == NULL)                                                                         \
            return NULL;                                                                                  \
        PyObject *new_tensor = create_Tensor((Tensor *)other, other, numpy_result, backward_name);    \
        Py_DECREF(numpy_result);                                                                          \
        return new_tensor;                                                                                \
    }                                                                                                     \
    else if (Py_IS_TYPE(self, Tensor_type) && PyArray_IsPythonNumber(other))                              \
    {                                                                                                     \
        tmp = (Tensor *)self;                                                                             \
        PyArrayObject *a = (PyArrayObject *)tmp->data;                                                    \
        numpy_result = (PyObject *)numboost_binary_scalar_right(a, other, op_enum);                       \
        if (numpy_result == NULL)                                                                         \
            return NULL;                                                                                  \
        PyObject *new_tensor = create_Tensor(tmp, other, numpy_result, backward_name);                \
        Py_DECREF(numpy_result);                                                                          \
        return new_tensor;                                                                                \
    }                                                                                                     \
    else if (PyArray_Check(self) && PyArray_IsPythonNumber(other))                                        \
    {                                                                                                     \
        tmp = (Tensor *)self;                                                                                       \
        PyArrayObject *a = (PyArrayObject *)self;                                                         \
        numpy_result = (PyObject *)numboost_binary_scalar_right(a, other, op_enum);                       \
        if (numpy_result == NULL)                                                                         \
            return NULL;                                                                                  \
        return numpy_result;                                                                              \
    }                                                                                                     \
    else if (PyArray_Check(other) && PyArray_IsPythonNumber(self))                                        \
    {                                                                                                     \
        PyArrayObject *b = (PyArrayObject *)other;                                                        \
        numpy_result = (PyObject *)numboost_binary_scalar_left(self, b, op_enum);                         \
        if (numpy_result == NULL)                                                                         \
            return NULL;                                                                                  \
        return numpy_result;                                                                              \
    }                                                                                                     \
    else                                                                                                  \
    {                                                                                                     \
        PyErr_SetString(PyExc_TypeError, "not supported type");                                           \
        return NULL;                                                                                      \
    }

#endif