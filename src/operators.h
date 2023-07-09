
#ifndef TNESOR_H2
#define TNESOR_H2
#include "tensor.h"
#endif

Tensor * __new_Tensor(Tensor *tensor, PyObject *array, PyObject *to_y, const char *grad_fn);

Tensor *
new_Tensor(Tensor *tensor, Tensor *tensor2, PyObject *data,
           PyObject *x, PyObject *y, int has_conv, uint64_t vars, bool require_grad,
           const char *grad_fn, PyObject *graph, PyObject *axis, int dim,
           PyObject *base);

Tensor *
new_Tensor_scalar(Tensor *self, PyObject *data, PyObject *y, int has_conv, uint64_t vars,
                  bool require_grad, const char *grad_fn, PyObject *graph,
                  PyObject *axis, int dim, PyObject *base);

Tensor *
new_Tensor_x(Tensor *self, PyObject *data, int has_conv, uint64_t vars,
             bool require_grad, const char *grad_fn, PyObject *graph,
             PyObject *axis, int dim, PyObject *base);


Tensor *tensor_add(Tensor *self, PyObject *other);
PyObject *tensor_iadd(Tensor *self, PyObject *other);
PyObject *tensor_mul(Tensor *self, PyObject *other);
PyObject *tensor_imul(Tensor *self, PyObject *other);
PyObject *tensor_div(Tensor *self, PyObject *other);
PyObject *tensor_idiv(Tensor *self, PyObject *other);
PyObject *tensor_negative(Tensor *self);
PyObject *tensor_inegative(Tensor *self);
PyObject *tensor_sub(Tensor *self, PyObject *other);
PyObject *tensor_isub(Tensor *self, PyObject *other);
PyObject *tensor_pow(Tensor *self, PyObject *other);
PyObject *tensor_ipow(Tensor *self, PyObject *other);
PyObject *tensor_matmul(Tensor *self, Tensor *other);
PyObject *tensor_imatmul(Tensor *self, Tensor *other);
PyObject *tensor_positive(Tensor *self);
PyObject *tensor_absolute(Tensor *self);
PyObject *tensor_invert(Tensor *self);
PyObject *tensor_lshift(Tensor *self, PyObject *other);
PyObject *tensor_rshift(Tensor *self, PyObject *other);
PyObject *tensor_and(Tensor *self, PyObject *other);
PyObject *tensor_xor(Tensor *self, PyObject *other);
PyObject *tensor_or(Tensor *self, PyObject *other);
PyObject *tensor_int(Tensor *self);
PyObject *tensor_float(Tensor *self);
PyObject *tensor_remainder(Tensor *self, PyObject *other);
PyObject *tensor_ior(Tensor *self, PyObject *other);
PyObject *tensor_ixor(Tensor *self, PyObject *other);
PyObject *tensor_iand(Tensor *self, PyObject *other);
PyObject *tensor_ilshift(Tensor *self, PyObject *other);
PyObject *tensor_irshift(Tensor *self, PyObject *other);
PyObject *tensor_divmod(Tensor *self, PyObject *other);
PyObject *tensor_iremainder(Tensor *self, PyObject *other);
PyObject *tensor_floordiv(Tensor *self, PyObject *other);
PyObject *tensor_ifloordiv(Tensor *self, PyObject *other);