#include "numpy/arrayobject.h"
#include "tensor.h"

Tensor *
new_Tensor(PyTypeObject *type, Tensor *tensor, Tensor *tensor2, PyObject *data,
           PyObject *x, PyObject *y, int has_conv, uint64_t vars, bool require_grad,
           const char *grad_fn, PyObject *graph, PyObject *axis, int dim,
           PyObject *base);

Tensor *
new_Tensor_scalar(PyTypeObject *type, Tensor *self, PyObject *data, PyObject *y, int has_conv, uint64_t vars,
                  bool require_grad, const char *grad_fn, PyObject *graph,
                  PyObject *axis, int dim, PyObject *base);

Tensor *
new_Tensor_x(PyTypeObject *type, Tensor *self, PyObject *data, int has_conv, uint64_t vars,
             bool require_grad, const char *grad_fn, PyObject *graph,
             PyObject *axis, int dim, PyObject *base);

Tensor *
__new_Tensor(PyTypeObject *type, Tensor *self, PyObject *array, const char *grad_fn); //auto incref by 1