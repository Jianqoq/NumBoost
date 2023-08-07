#include "tensor.h"

PyObject *astype(Tensor *self, PyObject *const *args, size_t nargsf);

PyObject *__str__(Tensor *self);

PyObject *__repr__(Tensor *self);

PyObject *__len__(Tensor *self);

PyObject *__iter__(Tensor *self);

PyObject *__max__(Tensor *self);

PyObject *__min__(Tensor *self);

PyObject *get_item(Tensor *self, PyObject *item);

Tensor *T(Tensor *self);

PyObject *__new__(PyTypeObject *type, PyObject *args, PyObject *kwds);

Tensor *self_transpose(Tensor *self, PyObject *const *args, size_t nargsf, PyObject *kwnames);

Tensor *self_reshape(Tensor *self, PyObject *const *args, size_t nargsf, PyObject *kwnames);

PyObject *_Generic_backward(PyObject *self, PyObject *args);