#include "tensor.h"

PyObject *astype(Tensor *self, PyObject *const *args, size_t nargsf);

PyObject *__str__(Tensor *self);

PyObject *__repr__(Tensor *self);

PyObject *__len__(Tensor *self);

PyObject *__iter__(Tensor *self);

PyObject *__max__(Tensor *self);

PyObject *__min__(Tensor *self);