
#ifndef TNESOR_H2
#define TNESOR_H2
#include "tensor.h"
#endif

Tensor * __new_Tensor(Tensor *tensor, PyObject *array, PyObject *to_y, const char *grad_fn);

Tensor *
new_Tensor(Tensor *tensor, Tensor *tensor2, PyObject *data, const char *grad_fn);

Tensor *
new_Tensor_scalar(Tensor *self, PyObject *data, PyObject *y, const char *grad_fn);

Tensor *
new_Tensor_x(Tensor *self, PyObject *data, const char *grad_fn);


Tensor *tensor_add(Tensor *self, PyObject *other);
Tensor *tensor_iadd(Tensor *self, PyObject *other);
Tensor *tensor_mul(Tensor *self, PyObject *other);
Tensor *tensor_imul(Tensor *self, PyObject *other);
Tensor *tensor_div(Tensor *self, PyObject *other);
Tensor *tensor_idiv(Tensor *self, PyObject *other);
Tensor *tensor_negative(Tensor *self);
Tensor *tensor_inegative(Tensor *self);
Tensor *tensor_sub(Tensor *self, PyObject *other);
Tensor *tensor_isub(Tensor *self, PyObject *other);
Tensor *tensor_pow(Tensor *self, PyObject *other);
Tensor *tensor_ipow(Tensor *self, PyObject *other);
Tensor *tensor_matmul(Tensor *self, Tensor *other);
Tensor *tensor_imatmul(Tensor *self, Tensor *other);
Tensor *tensor_positive(Tensor *self);
Tensor *tensor_absolute(Tensor *self);
Tensor *tensor_invert(Tensor *self);
Tensor *tensor_lshift(Tensor *self, PyObject *other);
Tensor *tensor_rshift(Tensor *self, PyObject *other);
Tensor *tensor_and(Tensor *self, PyObject *other);
Tensor *tensor_xor(Tensor *self, PyObject *other);
Tensor *tensor_or(Tensor *self, PyObject *other);
PyObject *tensor_int(Tensor *self);
PyObject *tensor_float(Tensor *self);
Tensor *tensor_remainder(Tensor *self, PyObject *other);
Tensor *tensor_ior(Tensor *self, PyObject *other);
Tensor *tensor_ixor(Tensor *self, PyObject *other);
Tensor *tensor_iand(Tensor *self, PyObject *other);
Tensor *tensor_ilshift(Tensor *self, PyObject *other);
Tensor *tensor_irshift(Tensor *self, PyObject *other);
Tensor *tensor_divmod(Tensor *self, PyObject *other);
Tensor *tensor_iremainder(Tensor *self, PyObject *other);
Tensor *tensor_floordiv(Tensor *self, PyObject *other);
Tensor *tensor_ifloordiv(Tensor *self, PyObject *other);