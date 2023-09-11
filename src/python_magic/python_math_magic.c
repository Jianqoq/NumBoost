#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "python_math_magic.h"
#include "../binary_ops/binary_op_def.h"
#include "../element_ops/element_ops_def.h"
#include "../numboost_api.h"
#include "../numboost_utils.h"
#include "../set_tensor_properties.h"
#include "mkl.h"
#include "structmember.h"
#include <Python.h>
#include <numpy/arrayobject.h>


extern XLA_OPS *xla_ops;
extern jnp_method *JNP_METHOD;
extern bool TRACK;

PyObject *__new_Tensor(Tensor *tensor, PyObject *array, PyObject *to_y,
                       const char *grad_fn) {
  Tensor *self = (Tensor *)Tensor_type->tp_alloc(Tensor_type, 0);
  if (self != NULL) {
    if (tensor->require_grad) {
      Tensor_SetGradFn(self, grad_fn);
      Tensor_SetRequireGrad(self, true);
      Tensor_SetVars(self, tensor->vars);
      Tensor_SetX_without_init_value(self, (PyObject *)tensor);
      if (to_y != NULL)
        Tensor_SetY_without_init_value(self, to_y);
      else {
        Py_INCREF(Py_None);
        Tensor_SetY_without_init_value(self, Py_None);
      }
    } else {
      Tensor_SetX_without_init_value(self, Py_None);
      Tensor_SetY_without_init_value(self, Py_None);
      Tensor_SetGradFn(self, "");
      Tensor_SetRequireGrad(self, false);
      Tensor_SetVars(self, 0);
    }
    self->data = array;
    Tensor_SetHasConv(self, tensor->has_conv);
    Tensor_SetGraph_without_init_value(self, tensor->graph);
    Tensor_SetDim(self, tensor->dim);
    Tensor_SetAxis_without_init_value(self, tensor->axis);
    self->grad = PyLong_FromLong(0);
    return (PyObject *)self;
  } else {
    return NULL;
  }
}

PyObject *new_Tensor(Tensor *tensor, Tensor *tensor2, PyObject *data,
                     const char *grad_fn) {
  Tensor *self = (Tensor *)Tensor_type->tp_alloc(Tensor_type, 0);
  if (self != NULL) {
    if (tensor->require_grad || tensor2->require_grad) {
      Tensor_SetX_without_init_value(self, (PyObject *)tensor);
      Tensor_SetY_without_init_value(self, (PyObject *)tensor2);
      Tensor_SetGradFn(self, grad_fn);
      Tensor_SetRequireGrad(self, true);
      Tensor_SetVars(self, tensor->vars + tensor2->vars + 1);
    } else {
      Tensor_SetX_without_init_value(self, Py_None);
      Tensor_SetY_without_init_value(self, Py_None);
      Tensor_SetGradFn(self, "");
      Tensor_SetRequireGrad(self, false);
      Tensor_SetVars(self, 0);
    }
    PyObject *zero = PyLong_FromLong(0);
    self->data = data;
    Tensor_SetHasConv(self, tensor->has_conv);
    Tensor_SetGraph_without_init_value(self, tensor->graph);
    Tensor_SetDim(self, tensor->dim);
    Tensor_SetAxis_without_init_value(self, tensor->axis);
    Tensor_SetGrad_without_init_value(self, zero);
    Py_DECREF(zero);
    return (PyObject *)self;
  } else {
    return NULL;
  }
}

PyObject *new_Tensor_scalar(Tensor *self, PyObject *data, PyObject *y,
                            const char *grad_fn) {
  Tensor *tensor;
  tensor = (Tensor *)Tensor_type->tp_alloc(Tensor_type, 0);
  if (tensor != NULL) {
    Tensor_SetData_startwone_without_init(tensor, data);
    if (self->require_grad) {
      Tensor_SetX_without_init_value(tensor, (PyObject *)self);
      Tensor_SetY_without_init_value(tensor, y);
      Tensor_SetRequireGrad(tensor, true);
      Tensor_SetGradFn(tensor, grad_fn);
      Tensor_SetVars(tensor, self->vars + 2);
    } else {
      Tensor_SetX_without_init_value(tensor, Py_None);
      Tensor_SetY_without_init_value(tensor, Py_None);
      Tensor_SetRequireGrad(tensor, false);
      Tensor_SetGradFn(tensor, grad_fn);
      Tensor_SetVars(tensor, 0);
    }
    PyObject *zero = PyLong_FromLong(0);
    Tensor_SetHasConv(tensor, self->has_conv);
    Tensor_SetGraph_without_init_value(tensor, self->graph);
    Tensor_SetDim(tensor, self->dim);
    Tensor_SetAxis_without_init_value(tensor, self->axis);
    Tensor_SetGrad_without_init_value(tensor, zero);
    Py_DECREF(zero);
    return (PyObject *)tensor;
  } else {
    return NULL;
  }
}

PyObject *create_tensor(Tensor *tensor, PyObject *other, PyObject *data,
                        const char *grad_fn) {
  Tensor *self = (Tensor *)Tensor_type->tp_alloc(Tensor_type, 0);
  if (self != NULL) {
    if (Py_IS_TYPE(other, Tensor_type)) {
      Tensor *tmp = (Tensor *)other;
      if (tensor->require_grad || tmp->require_grad) {
        Tensor_SetX_without_init_value(self, (PyObject *)tensor);
        Tensor_SetY_without_init_value(self, other);
        Tensor_SetGradFn(self, grad_fn);
        Tensor_SetRequireGrad(self, true);
        Tensor_SetVars(self, tensor->vars + tmp->vars + 1);
      } else {
        Tensor_SetX_without_init_value(self, Py_None);
        Tensor_SetY_without_init_value(self, Py_None);
        Tensor_SetGradFn(self, "");
        Tensor_SetRequireGrad(self, false);
        Tensor_SetVars(self, 0);
      }
    } else {
      if (tensor->require_grad) {
        Tensor_SetX_without_init_value(self, (PyObject *)tensor);
        Tensor_SetY_without_init_value(self, other);
        Tensor_SetRequireGrad(self, true);
        Tensor_SetGradFn(self, grad_fn);
        if (Py_IsNone(other)) {
          Tensor_SetVars(self, tensor->vars + 1);
        } else {
          Tensor_SetVars(self, tensor->vars + 2);
        }
      } else {
        Tensor_SetX_without_init_value(self, Py_None);
        Tensor_SetY_without_init_value(self, Py_None);
        Tensor_SetRequireGrad(self, false);
        Tensor_SetGradFn(self, grad_fn);
        Tensor_SetVars(self, 0);
      }
    }
    PyObject *zero = PyLong_FromLong(0);
    self->data = data;
    Tensor_SetHasConv(self, tensor->has_conv);
    Tensor_SetGraph_without_init_value(self, tensor->graph);
    Tensor_SetDim(self, tensor->dim);
    Tensor_SetAxis_without_init_value(self, tensor->axis);
    Tensor_SetGrad_without_init_value(self, zero);
    return (PyObject *)self;
  } else {
    PyErr_SetString(PyExc_MemoryError,
                    "Unable to allocate memory for Tensor Object");
    return NULL;
  }
}

PyObject *new_Tensor_x(Tensor *self, PyObject *data, const char *grad_fn) {
  Tensor *tensor;
  tensor = (Tensor *)Tensor_type->tp_alloc(Tensor_type, 0);
  if (tensor != NULL) {
    tensor->data = data;
    if (self->require_grad) {
      Tensor_SetX_without_init_value(tensor, (PyObject *)self);
      Tensor_SetY_without_init_value(tensor, Py_None);
      Tensor_SetRequireGrad(tensor, true);
      Tensor_SetGradFn(tensor, grad_fn);
      Tensor_SetVars(tensor, self->vars + 1);
    } else {
      Tensor_SetX_without_init_value(tensor, Py_None);
      Tensor_SetY_without_init_value(tensor, Py_None);
      Tensor_SetRequireGrad(tensor, false);
      Tensor_SetGradFn(tensor, "");
      Tensor_SetVars(tensor, 0);
    }
    PyObject *zero = PyLong_FromLong(0);
    Tensor_SetHasConv(tensor, self->has_conv);
    Tensor_SetGraph_without_init_value(tensor, self->graph);
    Tensor_SetDim(tensor, self->dim);
    Tensor_SetAxis_without_init_value(tensor, self->axis);
    Tensor_SetGrad_without_init_value(tensor, zero);
    Py_DECREF(zero);
    return (PyObject *)tensor;
  } else {
    return NULL;
  }
}

PyObject *Tensor__new__(PyTypeObject *type, PyObject *data) {
  Tensor *tensor;
  tensor = (Tensor *)type->tp_alloc(type, 0);
  if (tensor != NULL) {
    PyObject *zero = PyLong_FromLong(0);
    Tensor_SetData_startwone_without_init(tensor, data);
    Tensor_SetX_without_init_value(tensor, Py_None);
    Tensor_SetY_without_init_value(tensor, Py_None);
    Tensor_SetRequireGrad(tensor, false);
    Tensor_SetGradFn(tensor, "");
    Tensor_SetVars(tensor, 0);
    Tensor_SetHasConv(tensor, 0);
    Tensor_SetGraph_without_init_value(tensor, Py_None);
    Tensor_SetDim(tensor, 0);
    Tensor_SetAxis_without_init_value(tensor, Py_None);
    Tensor_SetGrad_without_init_value(tensor, zero);
    Py_DECREF(zero);
    return (PyObject *)tensor;
  } else {
    return NULL;
  }
}

PyObject *Tensor_Empty(PyObject *data) {
  Tensor *tensor = (Tensor *)(Tensor_type)->tp_alloc(Tensor_type, 0);
  if (tensor != NULL) {
    PyObject *zero = PyLong_FromLong(0);
    Tensor_SetData_startwone_without_init(tensor, data);
    Tensor_SetX_without_init_value(tensor, Py_None);
    Tensor_SetY_without_init_value(tensor, Py_None);
    Tensor_SetRequireGrad(tensor, false);
    Tensor_SetGradFn(tensor, "");
    Tensor_SetVars(tensor, 0);
    Tensor_SetHasConv(tensor, 0);
    Tensor_SetGraph_without_init_value(tensor, Py_None);
    Tensor_SetDim(tensor, 0);
    Tensor_SetAxis_without_init_value(tensor, Py_None);
    Tensor_SetGrad_without_init_value(tensor, zero);
    return (PyObject *)tensor;
  } else
    return NULL;
}

PyObject *tensor_add(PyObject *self, PyObject *other) {
  PyObject *result = numboost_add(((Tensor *)self)->data, other, NULL);
  Numboost_AssertNULL(result);
  PyObject *to_return =
      create_tensor((Tensor *)self, other, result, "AddBackward");
  Numboost_AssertNULL(to_return);
  return to_return;
}

PyObject *tensor_iadd(PyObject *self, PyObject *other) {
  Tensor *tmp;
  PyObject *numpy_result;
  Tensor *_self = (Tensor *)self;
  if (_self->require_grad) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Inplace operation can't set require_grad to true on a leaf variable");
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
}

PyObject *tensor_mul(PyObject *self, PyObject *other) {
  PyObject *result = numboost_mul(((Tensor *)self)->data, other, NULL);
  Numboost_AssertNULL(result);
  PyObject *to_return =
      create_tensor((Tensor *)self, other, result, "MulBackward");
  Numboost_AssertNULL(to_return);
  return to_return;
}

PyObject *tensor_imul(PyObject *self, PyObject *other) {
  Tensor *_self = (Tensor *)self;
  if (_self->require_grad) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Inplace operation can't set require_grad to true on a leaf variable");
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
}

PyObject *tensor_div(PyObject *self, PyObject *other) {
  PyObject *result = numboost_div(((Tensor *)self)->data, other, NULL);
  Numboost_AssertNULL(result);
  PyObject *to_return =
      create_tensor((Tensor *)self, other, result, "DivBackward");
  Numboost_AssertNULL(to_return);
  return to_return;
}

PyObject *tensor_idiv(PyObject *self, PyObject *other) {
  Tensor *_self = (Tensor *)self;
  if (_self->require_grad) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Inplace operation can't set require_grad to true on a leaf variable");
    return NULL;
  }
  PyObject *result = numboost_div(_self->data, other, &_self->data);
  Numboost_AssertNULL(result);
  if (result != _self->data) {
    Py_DECREF(_self->data);
    _self->data = result;
  }
  Py_INCREF(self);
  return self;
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
  PyObject *new_tensor = create_tensor(_self, PyLong_FromLong(-1), numpy_result,
                                       "NegativeBackward");
  return new_tensor;
}

PyObject *tensor_sub(PyObject *self, PyObject *other) {
  PyObject *result = numboost_sub(((Tensor *)self)->data, other, NULL);
  Numboost_AssertNULL(result);
  PyObject *to_return =
      create_tensor((Tensor *)self, other, result, "SubBackward");
  Numboost_AssertNULL(to_return);
  return to_return;
}

PyObject *tensor_isub(PyObject *self, PyObject *other) {
  Tensor *_self = (Tensor *)self;
  if (_self->require_grad) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Inplace operation can't set require_grad to true on a leaf variable");
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
}

PyObject *tensor_pow(PyObject *self, PyObject *other) {
  PyObject *result = numboost_pow(((Tensor *)self)->data, other, NULL);
  Numboost_AssertNULL(result);
  PyObject *to_return =
      create_tensor((Tensor *)self, Py_None, result, "PowBackward");
  Numboost_AssertNULL(to_return);
  if (((Tensor *)self)->require_grad)
    store_power((Tensor *)to_return, other);
  return to_return;
}

PyObject *tensor_ipow(PyObject *self, PyObject *other) {
  Tensor *_self = (Tensor *)self;
  if (_self->require_grad) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Inplace operation can't set require_grad to true on a leaf variable");
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
}

PyObject *tensor_matmul(PyObject *self, PyObject *other) {
  Tensor *_self = (Tensor *)self;
  Tensor *_other = (Tensor *)other;
  PyObject *numpy_result = PyNumber_MatrixMultiply(_self->data, _other->data);
  if (numpy_result == NULL) {
    return NULL;
  }
  PyObject *new_tensor =
      create_tensor(_self, other, numpy_result, "MatMulBackward");
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
  PyObject *to_return = create_tensor(_self, Py_None, result, "AbsBackward");
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
  PyObject *new_tensor = new_Tensor_x(_self, numpy_result, "");
  return new_tensor;
}

PyObject *tensor_lshift(PyObject *self, PyObject *other) {
  Tensor *_self = (Tensor *)self;
  Numboost_AssertRequireGrad(
      _self, "shift operation auto backward not implemented yet");
  PyObject *result = numboost_lshift(_self->data, other, NULL);
  Numboost_AssertNULL(result);
  PyObject *to_return = create_tensor(_self, other, result, "");
  Numboost_AssertNULL(to_return);
  return to_return;
}

PyObject *tensor_ilshift(PyObject *self, PyObject *other) {
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
}

PyObject *tensor_rshift(PyObject *self, PyObject *other) {
  Tensor *_self = (Tensor *)self;
  Numboost_AssertRequireGrad(
      _self, "shift operation auto backward not implemented yet");
  PyObject *result = numboost_rshift(_self->data, other, NULL);
  Numboost_AssertNULL(result);
  PyObject *to_return = create_tensor(_self, other, result, "");
  Numboost_AssertNULL(to_return);
  return to_return;
}

PyObject *tensor_irshift(PyObject *self, PyObject *other) {
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
}

PyObject *tensor_and(PyObject *self, PyObject *other) {
  Tensor *_self = (Tensor *)self;
  Numboost_AssertRequireGrad(_self, "Logic operation is not differentiable");
  PyObject *result = numboost_bitwise_and(_self->data, other, NULL);
  Numboost_AssertNULL(result);
  PyObject *to_return = create_tensor(_self, other, result, "");
  Numboost_AssertNULL(to_return);
  return to_return;
}

PyObject *tensor_xor(PyObject *self, PyObject *other) {
  Tensor *_self = (Tensor *)self;
  Numboost_AssertRequireGrad(_self, "Logic operation is not differentiable");
  PyObject *result = numboost_bitwise_xor(_self->data, other, NULL);
  Numboost_AssertNULL(result);
  PyObject *to_return = create_tensor(_self, other, result, "");
  Numboost_AssertNULL(to_return);
  return to_return;
}

PyObject *tensor_or(PyObject *self, PyObject *other) {
  Tensor *_self = (Tensor *)self;
  Numboost_AssertRequireGrad(_self, "Logic operation is not differentiable");
  PyObject *result = numboost_bitwise_or(_self->data, other, NULL);
  Numboost_AssertNULL(result);
  PyObject *to_return = create_tensor(_self, other, result, "");
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
  Tensor *_self = (Tensor *)self;
  Numboost_AssertRequireGrad(
      _self, "remainder operation auto backward not implemented yet");
  PyObject *result = numboost_mod(_self->data, other, NULL);
  Numboost_AssertNULL(result);
  PyObject *to_return = create_tensor(_self, other, result, "");
  Numboost_AssertNULL(to_return);
  return to_return;
}

PyObject *tensor_iand(PyObject *self, PyObject *other) {
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
}

PyObject *tensor_ior(PyObject *self, PyObject *other) {
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
}

PyObject *tensor_ixor(PyObject *self, PyObject *other) {
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
}

PyObject *tensor_divmod(PyObject *self, PyObject *other) {
  Tensor *_self = (Tensor *)self;
  Numboost_AssertRequireGrad(_self,
                             "DivMod operation not support auto backward");
  PyObject **result = numboost_divmod(_self->data, other, NULL, 0);
  Numboost_AssertNULL(result);
  PyObject *ret = (PyObject *)PyTuple_New(2);
  for (int i = 0; i < 2; i++) {
    PyObject *to_return = create_tensor(_self, other, result[i], "");
    Numboost_AssertNULL(to_return);
    PyTuple_SET_ITEM(ret, i, to_return);
  }
  return ret;
}

PyObject *tensor_iremainder(PyObject *self, PyObject *other) {
  Tensor *_self = (Tensor *)self;
  Numboost_AssertRequireGrad(
      _self, "Inplace remainder operation doesn't support auto backward");
  PyObject *result = numboost_mod(_self->data, other, &_self->data);
  Numboost_AssertNULL(result);
  if (result != _self->data) {
    Py_DECREF(_self->data);
    _self->data = result;
  }
  Py_INCREF(self);
  return self;
}

PyObject *tensor_floordiv(PyObject *self, PyObject *other) {
  Tensor *_self = (Tensor *)self;
  Numboost_AssertRequireGrad(_self, "Floor div not support auto backward");
  PyObject *result = numboost_fdiv(_self->data, other, NULL);
  Numboost_AssertNULL(result);
  PyObject *to_return = create_tensor(_self, other, result, "");
  Numboost_AssertNULL(to_return);
  return to_return;
}

PyObject *tensor_ifloordiv(PyObject *self, PyObject *other) {
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
}