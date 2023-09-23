#ifndef _TENSOR_METHODS_H
#define _TENSOR_METHODS_H
#include "tensor.h"

PyObject *astype(Tensor *self, PyObject *const *args, size_t nargsf);

PyObject *__str__(Tensor *self);

PyObject *__repr__(Tensor *self);

PyObject *__iter__(Tensor *self);

Py_ssize_t __len__(Tensor *self);

PyObject *rich_compare(PyObject *self, PyObject *other, int op);

PyObject *get_item(Tensor *self, PyObject *item);

Tensor *T(Tensor *self);

PyObject *dtype(Tensor *self);

PyObject *__new__(PyTypeObject *type, PyObject *args, PyObject *kwds);

PyObject *__tensor(PyObject *self, PyObject *args, PyObject *kwds);

Tensor *self_transpose(Tensor *self, PyObject *const *args, size_t nargsf);

Tensor *self_reshape(Tensor *self, PyObject *const *args, size_t nargsf);

PyObject *backward(PyObject *self, PyObject *args);

Tensor *copy(Tensor *self);

Py_hash_t __hash__(Tensor *self);

#endif