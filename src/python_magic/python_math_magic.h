

#ifndef PYTHON_MATH_MAGIC_H
#define PYTHON_MATH_MAGIC_H
#include "../tensor.h"

PyObject *create_tensor(Tensor *tensor, PyObject *other, PyObject *data,
                        const char *grad_fn);

PyObject *tensor_empty(PyObject *data);

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

#endif