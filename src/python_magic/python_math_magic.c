#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "python_math_magic.h"
#include "../binary_ops/binary_op_def.h"
#include "../numboost_api.h"
#include "../op.h"
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

PyObject *create_Tensor(Tensor *tensor, PyObject *other, PyObject *data,
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
  PyObject *result = numboost_add(((Tensor *)self)->data, other);
  Numboost_AssertNULL(result);
  PyObject *to_return =
      create_Tensor((Tensor *)self, other, result, "AddBackward");
  return to_return;
}

PyObject *tensor_iadd(PyObject *self, PyObject *other) {
  Tensor *tmp;
  PyObject *numpy_result;
  if (TRACK) {
    PyObject *jaxarray = PyNumber_InPlaceAdd(self, other);
    return jaxarray;
  }
  Tensor *_self = (Tensor *)self;
  if (_self->require_grad) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Inplace operation can't set require_grad to true on a leaf variable");
    return NULL;
  }
  if (Py_TYPE(other) == Tensor_type) {
    tmp = (Tensor *)other;
    numpy_result = PyNumber_InPlaceAdd(_self->data, tmp->data);
    if (numpy_result == NULL) {
      return NULL;
    }
    if (tmp->require_grad) {
      Tensor_SetX(_self, (PyObject *)_self);
      Tensor_SetY(_self, other);
      _self->grad_fn = "AddBackward";
    }
  } else {
    numpy_result = PyNumber_InPlaceAdd(_self->data, other);
    if (numpy_result == NULL) {
      return NULL;
    }
  }
  Tensor_SetData(_self, numpy_result);
  Py_INCREF(self);
  return self;
}

PyObject *tensor_mul(PyObject *self, PyObject *other) {
  PyObject *result = numboost_mul(((Tensor *)self)->data, other);
  Numboost_AssertNULL(result);
  PyObject *to_return =
      create_Tensor((Tensor *)self, other, result, "MulBackward");
  return to_return;
}

PyObject *tensor_imul(PyObject *self, PyObject *other) {
  Tensor *tmp;
  PyObject *numpy_result;
  if (TRACK) {
    PyObject *jaxarray = PyNumber_InPlaceMultiply(self, other);
    return jaxarray;
  }
  Tensor *_self = (Tensor *)self;
  if (_self->require_grad) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Inplace operation can't set require_grad to true on a leaf variable");
    return NULL;
  }
  if (Py_TYPE(other) == Tensor_type) {
    tmp = (Tensor *)other;
    numpy_result = PyNumber_InPlaceMultiply(_self->data, tmp->data);
    if (numpy_result == NULL) {
      return NULL;
    }
    if (tmp->require_grad) {
      Tensor_SetX(_self, self);
      Tensor_SetY(_self, other);
      _self->grad_fn = "MulBackward";
    }
  } else {
    numpy_result = PyNumber_InPlaceMultiply(_self->data, other);
    if (numpy_result == NULL) {
      return NULL;
    }
  }
  Tensor_SetData(_self, numpy_result);
  Py_INCREF(self);
  return self;
}

PyObject *tensor_div(PyObject *self, PyObject *other) {
  PyObject *result = numboost_div(((Tensor *)self)->data, other);
  Numboost_AssertNULL(result);
  PyObject *to_return =
      create_Tensor((Tensor *)self, other, result, "DivBackward");
  return to_return;
}

PyObject *tensor_idiv(PyObject *self, PyObject *other) {
  Tensor *tmp;
  PyObject *numpy_result;
  if (TRACK) {
    PyObject *jaxarray = PyNumber_InPlaceTrueDivide(self, other);
    return jaxarray;
  }
  Tensor *_self = (Tensor *)self;
  if (_self->require_grad) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Inplace operation can't set require_grad to true on a leaf variable");
    return NULL;
  }
  if (Py_TYPE(other) == Tensor_type) {
    tmp = (Tensor *)other;
    numpy_result = PyNumber_InPlaceTrueDivide(_self->data, tmp->data);
    if (numpy_result == NULL) {
      return NULL;
    }
    if (tmp->require_grad) {
      Tensor_SetX(_self, self);
      Tensor_SetY(_self, other);
      _self->grad_fn = "DivBackward";
    }
  } else {
    numpy_result = PyNumber_InPlaceTrueDivide(_self->data, other);
    if (numpy_result == NULL) {
      return NULL;
    }
  }
  Tensor_SetData(_self, numpy_result);
  Py_INCREF(self);
  return self;
}

PyObject *tensor_inegative(PyObject *self) {
  if (TRACK) {
    PyObject *jaxarray = PyNumber_Negative(self);
    return jaxarray;
  }
  Tensor *_self = (Tensor *)self;
  if (_self->require_grad) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Inplace operation can't set require_grad to true on a leaf variable");
    return NULL;
  }
  PyObject *negative_1 = PyLong_FromLong(-1);
  PyObject *numpy_result = PyNumber_InPlaceMultiply(_self->data, negative_1);
  if (numpy_result == NULL) {
    Py_DECREF(negative_1);
    return NULL;
  }
  Tensor_SetData(_self, numpy_result);
  Py_DECREF(negative_1);
  return self;
}

PyObject *tensor_negative(PyObject *self) // need to check
{
  if (TRACK) {
    PyObject *jaxarray = PyNumber_Negative(((Tensor *)self)->data);
    return jaxarray;
  }
  Tensor *_self = (Tensor *)self;
  PyObject *negative_1 = PyLong_FromLong(-1);
  PyObject *numpy_result = PyNumber_Negative(_self->data);
  if (numpy_result == NULL) {
    return NULL;
  }
  PyObject *new_tensor =
      create_Tensor(_self, negative_1, numpy_result, "NegativeBackward");
  return new_tensor;
}

PyObject *tensor_sub(PyObject *self, PyObject *other) {
  PyObject *result = numboost_sub(((Tensor *)self)->data, other);
  Numboost_AssertNULL(result);
  PyObject *to_return =
      create_Tensor((Tensor *)self, other, result, "SubBackward");
  return to_return;
}

PyObject *tensor_isub(PyObject *self, PyObject *other) {
  Tensor *tmp;
  PyObject *numpy_result;
  if (TRACK) {
    PyObject *jaxarray = PyNumber_InPlaceSubtract(self, other);
    return jaxarray;
  }
  Tensor *_self = (Tensor *)self;
  if (_self->require_grad) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Inplace operation can't set require_grad to true on a leaf variable");
    return NULL;
  }
  if (Py_TYPE(other) == Tensor_type) {
    tmp = (Tensor *)other;
    numpy_result = PyNumber_InPlaceSubtract(_self->data, tmp->data);
    if (numpy_result == NULL) {
      return NULL;
    }
    if (tmp->require_grad) {
      Tensor_SetX(_self, (PyObject *)self);
      Tensor_SetY(_self, other);
      _self->grad_fn = "InplaceSubBackward";
    }
  } else {
    numpy_result = PyNumber_InPlaceSubtract(_self->data, other);
    if (numpy_result == NULL) {
      return NULL;
    }
  }
  Tensor_SetData(_self, numpy_result);
  Py_INCREF(self);
  return self;
}

PyObject *tensor_pow(PyObject *self, PyObject *other) {
  PyObject *result = numboost_pow(((Tensor *)self)->data, other);
  Numboost_AssertNULL(result);
  PyObject *to_return =
      create_Tensor((Tensor *)self, other, result, "PowBackward");
  return to_return;
}

PyObject *tensor_ipow(PyObject *self, PyObject *other) {
  PyObject *numpy_result;
  Tensor *tmp;
  if (TRACK) {
    PyObject *jaxarray = PyNumber_InPlacePower(self, other, Py_None);
    return jaxarray;
  }
  Tensor *_self = (Tensor *)self;
  if (_self->require_grad) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Inplace operation can't set require_grad to true on a leaf variable");
    return NULL;
  }
  if (Py_TYPE(other) == Tensor_type) {
    tmp = (Tensor *)other;
    numpy_result = PyNumber_InPlacePower(_self->data, tmp->data, Py_None);
    if (numpy_result == NULL) {
      return NULL;
    }
    if (tmp->require_grad) {
      Tensor_SetX(_self, (PyObject *)self);
      Tensor_SetY(_self, other);
      _self->grad_fn = "InplacePowerBackward";
    }
  } else {
    numpy_result = PyNumber_InPlacePower(_self->data, other, Py_None);
    if (numpy_result == NULL) {
      return NULL;
    }
  }
  Tensor_SetData(_self, numpy_result);
  Py_INCREF(self);
  return self;
}

PyObject *tensor_matmul(PyObject *self, PyObject *other) {
  if (TRACK) {
    PyObject *jaxarray = PyNumber_MatrixMultiply(self, other);
    return jaxarray;
  }
  Tensor *_self = (Tensor *)self;
  Tensor *_other = (Tensor *)other;
  PyObject *numpy_result = PyNumber_MatrixMultiply(_self->data, _other->data);
  if (numpy_result == NULL) {
    return NULL;
  }
  PyObject *new_tensor =
      new_Tensor(_self, _other, numpy_result, "MatMulBackward");
  return new_tensor;
}

PyObject *tensor_imatmul(PyObject *self, PyObject *other) {
  if (TRACK) {
    PyObject *jaxarray = PyNumber_InPlaceMatrixMultiply(self, other);
    return jaxarray;
  }
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
  if (TRACK) {
    PyObject *jaxarray = PyNumber_Absolute(self);
    return jaxarray;
  }
  Tensor *_self = (Tensor *)self;
  PyObject *numpy_result = PyNumber_Absolute(_self->data);

  if (numpy_result == NULL) {
    return NULL;
  }
  PyObject *new_tensor = new_Tensor_x(_self, numpy_result, "");
  return new_tensor;
}

PyObject *tensor_invert(PyObject *self) {
  if (TRACK) {
    PyObject *jaxarray = PyNumber_Invert(self);
    return jaxarray;
  }
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
  PyObject *result = numboost_lshift(((Tensor *)self)->data, other);
  Numboost_AssertNULL(result);
  PyObject *to_return = create_Tensor((Tensor *)self, other, result, "");
  return to_return;
}

PyObject *tensor_ilshift(PyObject *self, PyObject *other) {
  if (TRACK) {
    PyObject *jaxarray = PyNumber_InPlaceLshift(self, other);
    return jaxarray;
  }
  Tensor *_self = (Tensor *)self;
  if (_self->require_grad) {
    PyErr_SetString(PyExc_RuntimeError,
                    "shift operation auto backward not implemented yet");
    return NULL;
  }
  if (Py_TYPE(other) == Tensor_type) {
    Tensor *tmp = (Tensor *)other;
    PyObject *numpy_result = PyNumber_InPlaceLshift(_self->data, tmp->data);
    if (numpy_result == NULL) {
      return NULL;
    }
    Py_INCREF(self);
    Tensor_SetData(_self, numpy_result);
    return self;
  } else {
    PyObject *numpy_result = PyNumber_InPlaceLshift(_self->data, other);
    if (numpy_result == NULL) {
      return NULL;
    }
    Py_INCREF(self);
    Tensor_SetData(_self, numpy_result);
    return self;
  }
}

PyObject *tensor_rshift(PyObject *self, PyObject *other) {
  PyObject *result = numboost_rshift(((Tensor *)self)->data, other);
  Numboost_AssertNULL(result);
  PyObject *to_return = create_Tensor((Tensor *)self, other, result, "");
  return to_return;
}

PyObject *tensor_irshift(PyObject *self, PyObject *other) {
  if (TRACK) {
    PyObject *jaxarray = PyNumber_InPlaceRshift(self, other);
    return jaxarray;
  }
  Tensor *_self = (Tensor *)self;
  if (_self->require_grad) {
    PyErr_SetString(PyExc_RuntimeError,
                    "shift operation auto backward not implemented yet");
    return NULL;
  }
  if (Py_TYPE(other) == Tensor_type) {
    Tensor *tmp = (Tensor *)other;
    PyObject *numpy_result = PyNumber_InPlaceRshift(_self->data, tmp->data);
    if (numpy_result == NULL) {
      return NULL;
    }
    Py_INCREF(self);
    Tensor_SetData(_self, numpy_result);
    return self;
  } else {
    PyObject *numpy_result = PyNumber_InPlaceRshift(_self->data, other);
    if (numpy_result == NULL) {
      return NULL;
    }
    Py_INCREF(self);
    Tensor_SetData(_self, numpy_result);
    return self;
  }
}

PyObject *tensor_and(PyObject *self, PyObject *other) {
  Tensor *tmp;
  if (TRACK) {
    PyObject *jaxarray = PyNumber_And(self, other);
    return jaxarray;
  }
  Tensor *_self = (Tensor *)self;
  if (_self->require_grad) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Logic operation is not differentiable");
    return NULL;
  }
  if (Py_TYPE(other) == Tensor_type) {
    tmp = (Tensor *)other;
    PyObject *numpy_result = PyNumber_And(_self->data, tmp->data);
    if (numpy_result == NULL) {
      return NULL;
    }
    PyObject *new_tensor = new_Tensor(_self, tmp, numpy_result, "");
    return new_tensor;
  } else {
    PyObject *numpy_result = PyNumber_Rshift(_self->data, other);

    if (numpy_result == NULL) {
      return NULL;
    }
    PyObject *new_tensor = create_Tensor(_self, other, numpy_result, "");
    Py_DECREF(numpy_result);
    return new_tensor;
  }
}

PyObject *tensor_xor(PyObject *self, PyObject *other) {
  Tensor *tmp;
  if (TRACK) {
    PyObject *jaxarray = PyNumber_Xor(self, other);
    return jaxarray;
  }
  Tensor *_self = (Tensor *)self;
  if (_self->require_grad) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Logic operation is not differentiable");
    return NULL;
  }
  if (Py_TYPE(other) == Tensor_type) {
    tmp = (Tensor *)other;
    PyObject *numpy_result = PyNumber_Xor(_self->data, tmp->data);
    if (numpy_result == NULL) {
      return NULL;
    }
    PyObject *new_tensor = new_Tensor(_self, tmp, numpy_result, "");
    return new_tensor;
  } else {
    PyObject *numpy_result = PyNumber_Xor(_self->data, other);

    if (numpy_result == NULL) {
      return NULL;
    }
    PyObject *new_tensor = create_Tensor(_self, other, numpy_result, "");
    Py_DECREF(numpy_result);
    return new_tensor;
  }
}

PyObject *tensor_or(PyObject *self, PyObject *other) {
  Tensor *tmp;
  if (TRACK) {
    PyObject *jaxarray = PyNumber_Or(self, other);
    return jaxarray;
  }
  Tensor *_self = (Tensor *)self;
  if (_self->require_grad) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Logic operation is not differentiable");
    return NULL;
  }
  if (Py_TYPE(other) == Tensor_type) {
    tmp = (Tensor *)other;
    PyObject *numpy_result = PyNumber_Or(_self->data, tmp->data);
    if (numpy_result == NULL) {
      return NULL;
    }
    PyObject *new_tensor = new_Tensor(_self, tmp, numpy_result, "");
    return new_tensor;
  } else {
    PyObject *numpy_result = PyNumber_Or(_self->data, other);

    if (numpy_result == NULL) {
      return NULL;
    }
    PyObject *new_tensor =
        create_Tensor(_self, other, numpy_result, "OrBackward");
    Py_DECREF(numpy_result);
    return new_tensor;
  }
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
  long long *data = (long long *)PyArray_DATA((PyArrayObject *)numpy_result);
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
  PyObject *result = numboost_mod(((Tensor *)self)->data, other);
  Numboost_AssertNULL(result);
  PyObject *to_return = create_Tensor((Tensor *)self, other, result, "");
  return to_return;
}

PyObject *tensor_iand(PyObject *self, PyObject *other) {
  Tensor *tmp;
  PyObject *numpy_result;
  if (TRACK) {
    PyObject *jaxarray = PyNumber_InPlaceAnd(self, other);
    return jaxarray;
  }
  Tensor *_self = (Tensor *)self;
  if (_self->require_grad) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Logic operation is not differentiable");
    return NULL;
  }
  if (Py_TYPE(other) == Tensor_type) {
    tmp = (Tensor *)other;
    numpy_result = PyNumber_InPlaceAnd(_self->data, tmp->data);
    if (numpy_result == NULL) {
      return NULL;
    }
  } else {
    numpy_result = PyNumber_InPlaceAnd(_self->data, other);
    if (numpy_result == NULL) {
      return NULL;
    }
  }
  _self->grad_fn = "";
  Tensor_SetData(_self, numpy_result);
  Py_INCREF(self);
  return self;
}

PyObject *tensor_ior(PyObject *self, PyObject *other) {
  Tensor *tmp;
  PyObject *numpy_result;
  if (TRACK) {
    PyObject *jaxarray = PyNumber_InPlaceOr(self, other);
    return jaxarray;
  }
  Tensor *_self = (Tensor *)self;
  if (_self->require_grad) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Logic operation is not differentiable");
    return NULL;
  }
  if (Py_TYPE(other) == Tensor_type) {
    tmp = (Tensor *)other;
    numpy_result = PyNumber_InPlaceOr(_self->data, tmp->data);
    if (numpy_result == NULL) {
      return NULL;
    }
  } else {
    numpy_result = PyNumber_InPlaceOr(_self->data, other);
    if (numpy_result == NULL) {
      return NULL;
    }
  }
  _self->grad_fn = "";
  Tensor_SetData(_self, numpy_result);
  Py_INCREF(self);
  return self;
}

PyObject *tensor_ixor(PyObject *self, PyObject *other) {
  Tensor *tmp;
  PyObject *numpy_result;
  if (TRACK) {
    PyObject *jaxarray = PyNumber_InPlaceXor(self, other);
    return jaxarray;
  }
  Tensor *_self = (Tensor *)self;
  if (_self->require_grad) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Logic operation is not differentiable");
    return NULL;
  }
  if (Py_TYPE(other) == Tensor_type) {
    tmp = (Tensor *)other;
    numpy_result = PyNumber_InPlaceXor(_self->data, tmp->data);
    if (numpy_result == NULL) {
      return NULL;
    }
    Tensor_SetY(_self, (PyObject *)tmp);
  } else {
    numpy_result = PyNumber_InPlaceXor(_self->data, other);
    if (numpy_result == NULL) {
      return NULL;
    }
    Tensor_SetY(_self, other);
  }
  _self->grad_fn = "";
  Tensor_SetX(_self, _self->data);
  Tensor_SetData(_self, numpy_result);
  Py_INCREF(self);
  return self;
}

PyObject *tensor_divmod(PyObject *self, PyObject *other) {
  Tensor *tmp;
  if (TRACK) {
    PyObject *jaxarray = PyNumber_Divmod(self, other);
    return jaxarray;
  }
  Tensor *_self = (Tensor *)self;
  if (_self->require_grad) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Divmod operation is not differentiable");
    return NULL;
  }
  if (Py_TYPE(other) == Tensor_type) {
    tmp = (Tensor *)other;
    PyObject *numpy_result = PyNumber_Divmod(_self->data, tmp->data);
    if (numpy_result == NULL) {
      return NULL;
    }
    PyObject *new_tensor = new_Tensor(_self, tmp, numpy_result, "");
    return new_tensor;
  } else {
    PyObject *numpy_result = PyNumber_Divmod(_self->data, other);

    if (numpy_result == NULL) {
      return NULL;
    }
    PyObject *new_tensor = create_Tensor(_self, other, numpy_result, "");
    return new_tensor;
  }
}

PyObject *tensor_iremainder(PyObject *self, PyObject *other) {
  Tensor *tmp;
  PyObject *numpy_result;
  if (TRACK) {
    PyObject *jaxarray = PyNumber_InPlaceRemainder(self, other);
    return jaxarray;
  }
  Tensor *_self = (Tensor *)self;
  if (_self->require_grad) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Inplace remainder operation doesn't support auto backward");
    return NULL;
  }
  if (Py_TYPE(other) == Tensor_type) {
    tmp = (Tensor *)other;
    numpy_result = PyNumber_InPlaceRemainder(_self->data, tmp->data);
    if (numpy_result == NULL) {
      return NULL;
    }
    Tensor_SetY(_self, (PyObject *)tmp);
  } else {
    numpy_result = PyNumber_InPlaceRemainder(_self->data, other);
    if (numpy_result == NULL) {
      return NULL;
    }
    Tensor_SetY(_self, other);
  }
  Tensor_SetX(_self, _self->data);
  Py_INCREF(self);
  return self;
}

PyObject *tensor_floordiv(PyObject *self, PyObject *other) {
  PyObject *result = numboost_fdiv(((Tensor *)self)->data, other);
  Numboost_AssertNULL(result);
  PyObject *to_return = create_Tensor((Tensor *)self, other, result, "");
  return to_return;
}

PyObject *tensor_ifloordiv(PyObject *self, PyObject *other) {
  Tensor *tmp;
  PyObject *numpy_result;
  if (TRACK) {
    PyObject *jaxarray = PyNumber_InPlaceFloorDivide(self, other);
    return jaxarray;
  }
  Tensor *_self = (Tensor *)self;
  if (_self->require_grad) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Floor divide operation is not differentiable");
    return NULL;
  }
  if (Py_TYPE(other) == Tensor_type) {
    tmp = (Tensor *)other;
    numpy_result = PyNumber_InPlaceFloorDivide(_self->data, tmp->data);
    if (numpy_result == NULL) {
      return NULL;
    }
  } else {
    numpy_result = PyNumber_InPlaceFloorDivide(_self->data, other);
    if (numpy_result == NULL) {
      return NULL;
    }
  }
  Tensor_SetData(_self, numpy_result);
  Py_INCREF(self);
  return self;
}