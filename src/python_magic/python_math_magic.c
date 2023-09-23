#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "python_math_magic.h"
#include "../binary_ops/binary_op_def.h"
#include "../element_ops/element_ops_def.h"
#include "../numboost_api.h"
#include "../numboost_utils.h"
#include "../tensor_creation/creation_def.h"
#include "structmember.h"
#include <Python.h>
#include <numpy/arrayobject.h>


extern XLA_OPS *xla_ops;
extern jnp_method *JNP_METHOD;
extern bool TRACK;

PyObject *tensor_add(PyObject *self, PyObject *other) {
  PyObject *result = numboost_add(self, other, NULL);
  Numboost_AssertNULL(result);
  PyObject *to_return = NULL;
  if (Py_IS_TYPE(self, Tensor_type)) {
    to_return = tensor_new((Tensor *)self, other, result, "AddBackward");
  } else {
    to_return = tensor_new((Tensor *)other, self, result, "AddBackward");
  }
  Numboost_AssertNULL(to_return);
  return to_return;
}

PyObject *tensor_iadd(PyObject *self, PyObject *other) {
  if (Py_IS_TYPE(self, Tensor_type)) {
    Tensor *_self = (Tensor *)self;
    if (_self->require_grad) {
      PyErr_SetString(PyExc_RuntimeError,
                      "Inplace operation can't set require_grad to true on a "
                      "leaf variable");
      return NULL;
    }
    PyObject *result = numboost_add(_self->data, other, &_self->data);
    Numboost_AssertNULL(result);
    if (result != _self->data) {
      Py_DECREF(_self->data);
      _self->data = result;
    }
    Py_INCREF(self);
    return self;
  } else {
    PyErr_SetString(PyExc_RuntimeError,
                    "Left operands in inplace operation is not Tensor Object");
    return NULL;
  }
}

PyObject *tensor_mul(PyObject *self, PyObject *other) {
  PyObject *result = numboost_mul(self, other, NULL);
  Numboost_AssertNULL(result);
  PyObject *to_return = NULL;
  if (Py_IS_TYPE(self, Tensor_type)) {
    to_return = tensor_new((Tensor *)self, other, result, "MulBackward");
  } else {
    to_return = tensor_new((Tensor *)other, self, result, "MulBackward");
  }
  Numboost_AssertNULL(to_return);
  return to_return;
}

PyObject *tensor_imul(PyObject *self, PyObject *other) {
  if (Py_IS_TYPE(self, Tensor_type)) {
    Tensor *_self = (Tensor *)self;
    if (_self->require_grad) {
      PyErr_SetString(PyExc_RuntimeError,
                      "Inplace operation can't set require_grad to true on a "
                      "leaf variable");
      return NULL;
    }
    PyObject *result = numboost_mul(_self->data, other, &_self->data);
    Numboost_AssertNULL(result);
    if (result != _self->data) {
      Py_DECREF(_self->data);
      _self->data = result;
    }
    Py_INCREF(self);
    return self;
  } else {
    PyErr_SetString(PyExc_RuntimeError,
                    "Left operands in inplace operation is not Tensor Object");
    return NULL;
  }
}

PyObject *tensor_div(PyObject *self, PyObject *other) {
  PyObject *result = numboost_div(self, other, NULL);
  Numboost_AssertNULL(result);
  PyObject *to_return = NULL;
  if (Py_IS_TYPE(self, Tensor_type)) {
    Tensor *_self = (Tensor *)self;
    to_return = tensor_new(_self, other, result, "DivBackward");
  } else {
    Tensor *_other = (Tensor *)other;
    to_return = tensor_new(_other, self, result, "DivBackward");
  }
  Numboost_AssertNULL(to_return);
  return to_return;
}

PyObject *tensor_idiv(PyObject *self, PyObject *other) {
  if (Py_IS_TYPE(self, Tensor_type)) {
    Tensor *_self = (Tensor *)self;
    if (_self->require_grad) {
      PyErr_SetString(PyExc_RuntimeError,
                      "Inplace operation can't set require_grad to true on a "
                      "leaf variable");
      return NULL;
    }
    PyObject *result = numboost_div(self, other, &_self->data);
    Numboost_AssertNULL(result);
    if (result != _self->data) {
      Py_DECREF(_self->data);
      _self->data = result;
    }
    Py_INCREF(self);
    return self;
  } else {
    PyErr_SetString(PyExc_RuntimeError,
                    "Left operands in inplace operation is not Tensor Object");
    return NULL;
  }
}

PyObject *tensor_inegative(PyObject *self) {
  Tensor *_self = (Tensor *)self;
  if (_self->require_grad) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Inplace operation can't set require_grad to true on a leaf variable");
    return NULL;
  }
  PyObject *result = numboost_negative(_self->data, &_self->data);
  if (result == NULL) {
    return NULL;
  }
  Numboost_AssertNULL(result);
  if (result != _self->data) {
    Py_DECREF(_self->data);
    _self->data = result;
  }
  Py_INCREF(self);
  return self;
}

PyObject *tensor_negative(PyObject *self) // need to check
{
  Tensor *_self = (Tensor *)self;
  PyObject *numpy_result = numboost_negative(_self->data, NULL);
  if (numpy_result == NULL) {
    return NULL;
  }
  PyObject *new_tensor =
      tensor_new(_self, PyLong_FromLong(-1), numpy_result, "NegativeBackward");
  return new_tensor;
}

PyObject *tensor_sub(PyObject *self, PyObject *other) {
  PyObject *result = numboost_sub(self, other, NULL);
  Numboost_AssertNULL(result);
  PyObject *to_return = NULL;
  if (Py_IS_TYPE(self, Tensor_type)) {
    to_return = tensor_new((Tensor *)self, other, result, "SubBackward");
  } else {
    to_return = tensor_new((Tensor *)other, self, result, "SubBackward");
  }
  Numboost_AssertNULL(to_return);
  return to_return;
}

PyObject *tensor_isub(PyObject *self, PyObject *other) {
  if (Py_IS_TYPE(self, Tensor_type)) {
    Tensor *_self = (Tensor *)self;
    if (_self->require_grad) {
      PyErr_SetString(PyExc_RuntimeError,
                      "Inplace operation can't set require_grad to true on a "
                      "leaf variable");
      return NULL;
    }
    PyObject *result = numboost_sub(_self->data, other, &_self->data);
    Numboost_AssertNULL(result);
    if (result != _self->data) {
      Py_DECREF(_self->data);
      _self->data = result;
    }
    Py_INCREF(self);
    return self;
  } else {
    PyErr_SetString(PyExc_RuntimeError,
                    "Left operands in inplace operation is not Tensor Object");
    return NULL;
  }
}

PyObject *tensor_pow(PyObject *self, PyObject *other) {
  PyObject *result = numboost_pow(self, other, NULL);
  Numboost_AssertNULL(result);
  PyObject *to_return = NULL;
  if (Py_IS_TYPE(self, Tensor_type)) {
    Tensor *_self = (Tensor *)self;
    to_return = tensor_new(_self, other, result, "PowBackward");
    if (_self->require_grad)
      store_power((Tensor *)to_return, other);
  } else {
    Tensor *_other = (Tensor *)other;
    to_return = tensor_new(_other, self, result, "PowBackward");
    if (_other->require_grad)
      store_power((Tensor *)to_return, other);
  }
  Numboost_AssertNULL(to_return);
  return to_return;
}

PyObject *tensor_ipow(PyObject *self, PyObject *other) {
  if (Py_IS_TYPE(self, Tensor_type)) {
    Tensor *_self = (Tensor *)self;
    if (_self->require_grad) {
      PyErr_SetString(PyExc_RuntimeError,
                      "Inplace operation can't set require_grad to true on a "
                      "leaf variable");
      return NULL;
    }
    PyObject *result = numboost_pow(_self->data, other, &_self->data);
    Numboost_AssertNULL(result);
    if (result != _self->data) {
      Py_DECREF(_self->data);
      _self->data = result;
    }
    Py_INCREF(self);
    return self;
  } else {
    PyErr_SetString(PyExc_RuntimeError,
                    "Left operands in inplace operation is not Tensor Object");
    return NULL;
  }
}

PyObject *tensor_matmul(PyObject *self, PyObject *other) {
  Tensor *_self = (Tensor *)self;
  Tensor *_other = (Tensor *)other;
  PyObject *numpy_result = PyNumber_MatrixMultiply(_self->data, _other->data);
  if (numpy_result == NULL) {
    return NULL;
  }
  PyObject *new_tensor =
      tensor_new(_self, other, numpy_result, "MatMulBackward");
  Numboost_AssertNULL(new_tensor);
  return new_tensor;
}

PyObject *tensor_imatmul(PyObject *self, PyObject *other) {
  Tensor *_self = (Tensor *)self;
  Tensor *_other = (Tensor *)other;
  if (_self->require_grad) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Inplace operation can't set require_grad to true on a leaf variable");
    return NULL;
  }
  PyObject *numpy_result =
      PyNumber_InPlaceMatrixMultiply(_self->data, _other->data);
  if (numpy_result == NULL) {
    return NULL;
  }
  if (_other->require_grad) {
    Tensor_SetX(_self, (PyObject *)self);
    Tensor_SetY(_self, (PyObject *)other);
    _self->grad_fn = "InplaceMatMulBackward";
  }
  Tensor_SetData(_self, numpy_result);
  Py_INCREF(self);
  return self;
}

PyObject *tensor_positive(PyObject *self) {
  Py_INCREF(self);
  return self;
}

PyObject *tensor_absolute(PyObject *self) {
  Tensor *_self = (Tensor *)self;
  Numboost_AssertRequireGrad(
      _self, "shift operation auto backward not implemented yet");
  PyObject *result = numboost_abs(_self->data, NULL);
  Numboost_AssertNULL(result);
  PyObject *to_return = tensor_new(_self, Py_None, result, "AbsBackward");
  Numboost_AssertNULL(to_return);
  return to_return;
}

PyObject *tensor_invert(PyObject *self) {
  Tensor *_self = (Tensor *)self;
  if (_self->require_grad) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Logic invert operation auto backward not implemented yet");
    return NULL;
  }
  PyObject *numpy_result = PyNumber_Invert(_self->data);
  if (numpy_result == NULL) {
    return NULL;
  }
  PyObject *new_tensor = tensor_new(_self, Py_None, numpy_result, "");
  return new_tensor;
}

PyObject *tensor_lshift(PyObject *self, PyObject *other) {
  PyObject *result = numboost_lshift(self, other, NULL);
  Numboost_AssertNULL(result);
  PyObject *to_return = NULL;
  if (Py_IS_TYPE(self, Tensor_type)) {
    Tensor *_self = (Tensor *)self;
    to_return = tensor_new(_self, other, result, "");
  } else {
    Tensor *_other = (Tensor *)other;
    to_return = tensor_new(_other, self, result, "");
  }
  Numboost_AssertNULL(to_return);
  return to_return;
}

PyObject *tensor_ilshift(PyObject *self, PyObject *other) {
  if (Py_IS_TYPE(self, Tensor_type)) {
    Tensor *_self = (Tensor *)self;
    Numboost_AssertRequireGrad(
        _self, "shift operation auto backward not implemented yet");
    PyObject *result = numboost_lshift(_self->data, other, &_self->data);
    Numboost_AssertNULL(result);
    if (result != _self->data) {
      Py_DECREF(_self->data);
      _self->data = result;
    }
    Py_INCREF(self);
    return self;
  } else {
    PyErr_SetString(PyExc_RuntimeError,
                    "Left operands in inplace operation is not Tensor Object");
    return NULL;
  }
}

PyObject *tensor_rshift(PyObject *self, PyObject *other) {
  PyObject *result = numboost_rshift(self, other, NULL);
  Numboost_AssertNULL(result);
  PyObject *to_return = NULL;
  if (Py_IS_TYPE(self, Tensor_type)) {
    Tensor *_self = (Tensor *)self;
    to_return = tensor_new(_self, other, result, "");
  } else {
    Tensor *_other = (Tensor *)other;
    to_return = tensor_new(_other, self, result, "");
  }
  Numboost_AssertNULL(to_return);
  return to_return;
}

PyObject *tensor_irshift(PyObject *self, PyObject *other) {
  if (Py_IS_TYPE(self, Tensor_type)) {
    Tensor *_self = (Tensor *)self;
    Numboost_AssertRequireGrad(
        _self, "shift operation auto backward not implemented yet");
    PyObject *result = numboost_rshift(_self->data, other, &_self->data);
    Numboost_AssertNULL(result);
    if (result != _self->data) {
      Py_DECREF(_self->data);
      _self->data = result;
    }
    Py_INCREF(self);
    return self;
  } else {
    PyErr_SetString(PyExc_RuntimeError,
                    "Left operands in inplace operation is not Tensor Object");
    return NULL;
  }
}

PyObject *tensor_and(PyObject *self, PyObject *other) {
  PyObject *result = numboost_bitwise_and(self, other, NULL);
  Numboost_AssertNULL(result);
  PyObject *to_return = NULL;
  if (Py_IS_TYPE(self, Tensor_type)) {
    Tensor *_self = (Tensor *)self;
    to_return = tensor_new(_self, other, result, "");
  } else {
    Tensor *_other = (Tensor *)other;
    to_return = tensor_new(_other, self, result, "");
  }
  Numboost_AssertNULL(to_return);
  return to_return;
}

PyObject *tensor_xor(PyObject *self, PyObject *other) {
  PyObject *result = numboost_bitwise_xor(self, other, NULL);
  Numboost_AssertNULL(result);
  PyObject *to_return = NULL;
  if (Py_IS_TYPE(self, Tensor_type)) {
    Tensor *_self = (Tensor *)self;
    to_return = tensor_new(_self, other, result, "");
  } else {
    Tensor *_other = (Tensor *)other;
    to_return = tensor_new(_other, self, result, "");
  }
  Numboost_AssertNULL(to_return);
  return to_return;
}

PyObject *tensor_or(PyObject *self, PyObject *other) {
  PyObject *result = numboost_bitwise_or(self, other, NULL);
  Numboost_AssertNULL(result);
  PyObject *to_return = NULL;
  if (Py_IS_TYPE(self, Tensor_type)) {
    Tensor *_self = (Tensor *)self;
    to_return = tensor_new(_self, other, result, "");
  } else {
    Tensor *_other = (Tensor *)other;
    to_return = tensor_new(_other, self, result, "");
  }
  Numboost_AssertNULL(to_return);
  return to_return;
}

PyObject *tensor_int(PyObject *self) {
  if (TRACK) {
    return NULL;
  }
  Tensor *_self = (Tensor *)self;
  if (PyArray_SIZE((PyArrayObject *)_self->data) > 1) {
    PyErr_SetString(PyExc_TypeError,
                    "only size-1 arrays can be converted to Python scalars");
    return NULL;
  }
  PyObject *numpy_result =
      PyArray_Cast((PyArrayObject *)_self->data, NPY_INT64);
  long long *data = (npy_longlong *)PyArray_DATA((PyArrayObject *)numpy_result);
  if (numpy_result == NULL || data == NULL) {
    return NULL;
  }
  return PyLong_FromLongLong(data[0]);
}

PyObject *tensor_float(PyObject *self) {
  if (TRACK) {
    return NULL;
  }
  Tensor *_self = (Tensor *)self;
  if (PyArray_SIZE((PyArrayObject *)_self->data) > 1) {
    PyErr_SetString(PyExc_TypeError,
                    "only size-1 arrays can be converted to Python scalars");
    return NULL;
  }
  PyArrayObject *numpy_result = NULL;
  PyArrayObject *a = (PyArrayObject *)_self->data;
  as_type(&a, &numpy_result, NPY_FLOAT64);
  Numboost_AssertNULL(numpy_result);
  double *data = (double *)PyArray_DATA(numpy_result);
  PyObject *to_return = PyFloat_FromDouble(data[0]);
  Py_DECREF(numpy_result);
  return to_return;
}

PyObject *tensor_remainder(PyObject *self, PyObject *other) {
  PyObject *result = numboost_mod(self, other, NULL);
  Numboost_AssertNULL(result);
  PyObject *to_return = NULL;
  if (Py_IS_TYPE(self, Tensor_type)) {
    to_return = tensor_new((Tensor *)self, other, result, "");
  } else {
    to_return = tensor_new((Tensor *)other, self, result, "");
  }
  Numboost_AssertNULL(to_return);
  return to_return;
}

PyObject *tensor_iand(PyObject *self, PyObject *other) {
  if (Py_IS_TYPE(self, Tensor_type)) {
    Tensor *_self = (Tensor *)self;
    Numboost_AssertRequireGrad(_self,
                               "Logic operation not support auto backward");
    PyObject *result = numboost_bitwise_and(_self->data, other, &_self->data);
    Numboost_AssertNULL(result);
    if (result != _self->data) {
      Py_DECREF(_self->data);
      _self->data = result;
    }
    Py_INCREF(self);
    return self;
  } else {
    PyErr_SetString(PyExc_RuntimeError,
                    "Left operands in inplace operation is not Tensor Object");
    return NULL;
  }
}

PyObject *tensor_ior(PyObject *self, PyObject *other) {
  if (Py_IS_TYPE(self, Tensor_type)) {
    Tensor *_self = (Tensor *)self;
    Numboost_AssertRequireGrad(_self,
                               "Logic operation not support auto backward");
    PyObject *result = numboost_bitwise_or(_self->data, other, &_self->data);
    Numboost_AssertNULL(result);
    if (result != _self->data) {
      Py_DECREF(_self->data);
      _self->data = result;
    }
    Py_INCREF(self);
    return self;
  } else {
    PyErr_SetString(PyExc_RuntimeError,
                    "Left operands in inplace operation is not Tensor Object");
    return NULL;
  }
}

PyObject *tensor_ixor(PyObject *self, PyObject *other) {
  if (Py_IS_TYPE(self, Tensor_type)) {
    Tensor *_self = (Tensor *)self;
    Numboost_AssertRequireGrad(_self,
                               "Logic operation not support auto backward");
    PyObject *result = numboost_bitwise_xor(_self->data, other, &_self->data);
    Numboost_AssertNULL(result);
    if (result != _self->data) {
      Py_DECREF(_self->data);
      _self->data = result;
    }
    Py_INCREF(self);
    return self;
  } else {
    PyErr_SetString(PyExc_RuntimeError,
                    "Left operands in inplace operation is not Tensor Object");
    return NULL;
  }
}

PyObject *tensor_divmod(PyObject *self, PyObject *other) {
  PyObject **result = numboost_divmod(self, other, NULL, 0);
  Numboost_AssertNULL(result);
  PyObject *ret = (PyObject *)PyTuple_New(2);
  for (int i = 0; i < 2; i++) {
    PyObject *to_return = NULL;
    if (Py_IS_TYPE(self, Tensor_type)) {
      to_return = tensor_new((Tensor *)self, other, result[i], "");
    } else {
      to_return = tensor_new((Tensor *)other, self, result[i], "");
    }
    Numboost_AssertNULL(to_return);
    PyTuple_SET_ITEM(ret, i, to_return);
  }
  return ret;
}

PyObject *tensor_iremainder(PyObject *self, PyObject *other) {
  if (Py_IS_TYPE(self, Tensor_type)) {
    Tensor *_self = (Tensor *)self;
    Numboost_AssertRequireGrad(_self,
                               "Logic operation not support auto backward");
    PyObject *result = numboost_mod(self, other, &_self->data);
    Numboost_AssertNULL(result);
    if (result != _self->data) {
      Py_DECREF(_self->data);
      _self->data = result;
    }
    Py_INCREF(self);
    return self;
  } else {
    PyErr_SetString(PyExc_RuntimeError,
                    "Left operands in inplace operation is not Tensor Object");
    return NULL;
  }
}

PyObject *tensor_floordiv(PyObject *self, PyObject *other) {
  PyObject *result = numboost_fdiv(self, other, NULL);
  Numboost_AssertNULL(result);
  PyObject *to_return = NULL;
  if (Py_IS_TYPE(self, Tensor_type)) {
    to_return = tensor_new((Tensor *)self, other, result, "");
  } else {
    to_return = tensor_new((Tensor *)other, self, result, "");
  }
  Numboost_AssertNULL(to_return);
  return to_return;
}

PyObject *tensor_ifloordiv(PyObject *self, PyObject *other) {
  if (Py_IS_TYPE(self, Tensor_type)) {
    Tensor *_self = (Tensor *)self;
    Numboost_AssertRequireGrad(_self,
                               "Logic operation not support auto backward");
    PyObject *result = numboost_fdiv(_self->data, other, &_self->data);
    Numboost_AssertNULL(result);
    if (result != _self->data) {
      Py_DECREF(_self->data);
      _self->data = result;
    }
    Py_INCREF(self);
    return self;
  } else {
    PyErr_SetString(PyExc_RuntimeError,
                    "Left operands in inplace operation is not Tensor Object");
    return NULL;
  }
}