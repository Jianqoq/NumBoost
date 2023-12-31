#ifndef SET_TENSOR_PROPERTIES_H
#define SET_TENSOR_PROPERTIES_H
#include "tensor.h"

inline static void Tensor_SetData(Tensor *self, PyObject *data) {
  Py_DECREF(self->data);
  self->data = data;
  Py_INCREF(self->data);
  if (self->data == NULL) {
    Py_DECREF(self);
    PyErr_SetString(PyExc_RuntimeError,
                    "Unable to allocate memory for Tensor object");
  }
}
inline static void Tensor_SetX(Tensor *self, PyObject *x) {
  Py_DECREF(self->x);
  self->x = x;
  Py_INCREF(self->x);
  if (self->x == NULL) {
    Py_DECREF(self);
    PyErr_SetString(PyExc_RuntimeError,
                    "Unable to allocate memory for Tensor object");
  }
}
inline static void Tensor_SetY(Tensor *self, PyObject *y) {
  Py_INCREF(y);
  Py_DECREF(self->y);
  self->y = y;
  if (self->y == NULL) {
    Py_DECREF(self);
    PyErr_SetString(PyExc_RuntimeError,
                    "Unable to allocate memory for Tensor object");
  }
}

inline static void Tensor_SetGrad(Tensor *self, PyObject *grad) {
  Py_INCREF(grad);
  Py_DECREF(self->grad);
  self->grad = grad;
  if (self->grad == NULL) {
    Py_DECREF(self);
    PyErr_SetString(PyExc_RuntimeError,
                    "Unable to allocate memory for Tensor object");
  }
}
inline static void Tensor_SetGraph(Tensor *self, PyObject *graph) {
  Py_INCREF(graph);
  Py_DECREF(self->graph);
  self->graph = graph;
  if (self->graph == NULL) {
    Py_DECREF(self);
    PyErr_SetString(PyExc_RuntimeError,
                    "Unable to allocate memory for Tensor object");
  }
}

inline static void Tensor_SetGradFn(Tensor *self, const char *grad_fn) {
  self->grad_fn = grad_fn;
}
inline static void Tensor_SetRequireGrad(Tensor *self, bool require_grad) {
  self->require_grad = require_grad;
}
inline static void Tensor_SetVars(Tensor *self, unsigned long long vars) {
  self->vars = vars;
}
inline static void Tensor_SetDim(Tensor *self, int dim) { self->dim = dim; }
inline static void Tensor_SetAxis(Tensor *self, PyObject *axis) {
  Py_INCREF(axis);
  Py_DECREF(self->axis);
  self->axis = axis;
  if (self->axis == NULL) {
    Py_DECREF(self);
    PyErr_SetString(PyExc_RuntimeError,
                    "Unable to allocate memory for Tensor object");
  }
}
inline static void Tensor_SetHasConv(Tensor *self, int has_conv) {
  self->has_conv = has_conv;
}

inline static void Tensor_SetData_without_init_value(Tensor *self, PyObject *data) {
  self->data = data;
  Py_INCREF(data);
  if (self->data == NULL) {
    Py_DECREF(self);
    PyErr_SetString(PyExc_RuntimeError,
                    "Unable to allocate memory for Tensor object");
  }
}
inline static void Tensor_SetX_without_init_value(Tensor *self, PyObject *x) {
  self->x = x;
  Py_INCREF(x);
  if (self->x == NULL) {
    Py_DECREF(self);
    PyErr_SetString(PyExc_RuntimeError,
                    "Unable to allocate memory for Tensor object");
  }
}
inline static void Tensor_SetY_without_init_value(Tensor *self, PyObject *y) {
  Py_INCREF(y);
  self->y = y;
  if (self->y == NULL) {
    Py_DECREF(self);
    PyErr_SetString(PyExc_RuntimeError,
                    "Unable to allocate memory for Tensor object");
  }
}
inline static void Tensor_SetGrad_without_init_value(Tensor *self, PyObject *grad) {
  Py_INCREF(grad);
  self->grad = grad;
  if (self->grad == NULL) {
    Py_DECREF(self);
    PyErr_SetString(PyExc_RuntimeError,
                    "Unable to allocate memory for Tensor object");
  }
}
inline static void Tensor_SetGraph_without_init_value(Tensor *self, PyObject *graph) {
  Py_INCREF(graph);
  self->graph = graph;
  if (self->graph == NULL) {
    Py_DECREF(self);
    PyErr_SetString(PyExc_RuntimeError,
                    "Unable to allocate memory for Tensor object");
  }
}

inline static void Tensor_SetAxis_without_init_value(Tensor *self, PyObject *axis) {
  Py_INCREF(axis);
  self->axis = axis;
  if (self->axis == NULL) {
    Py_DECREF(self);
    PyErr_SetString(PyExc_RuntimeError,
                    "Unable to allocate memory for Tensor object");
  }
}

inline static void Tensor_SetData_startwone(Tensor *self, PyObject *data) {
  Py_DECREF(self->data);
  self->data = data;
  if (self->data == NULL) {
    Py_DECREF(self);
    PyErr_SetString(PyExc_RuntimeError,
                    "Unable to allocate memory for Tensor object");
  }
}
inline static void Tensor_SetX_startwone(Tensor *self, PyObject *x) {
  Py_DECREF(self->x);
  self->x = x;
  if (self->x == NULL) {
    Py_DECREF(self);
    PyErr_SetString(PyExc_RuntimeError,
                    "Unable to allocate memory for Tensor object");
  }
}
inline static void Tensor_SetY_startwone(Tensor *self, PyObject *y) {
  Py_DECREF(self->y);
  self->y = y;
  if (self->y == NULL) {
    Py_DECREF(self);
    PyErr_SetString(PyExc_RuntimeError,
                    "Unable to allocate memory for Tensor object");
  }
}
inline static  void Tensor_SetGrad_startwone(Tensor *self, PyObject *grad) {
  Py_DECREF(self->grad);
  self->grad = grad;
  if (self->grad == NULL) {
    Py_DECREF(self);
    PyErr_SetString(PyExc_RuntimeError,
                    "Unable to allocate memory for Tensor object");
  }
}
inline static void Tensor_SetGraph_startwone(Tensor *self, PyObject *graph) {
  Py_DECREF(self->graph);
  self->graph = graph;
  if (self->graph == NULL) {
    Py_DECREF(self);
    PyErr_SetString(PyExc_RuntimeError,
                    "Unable to allocate memory for Tensor object");
  }
}

inline static void Tensor_SetAxis_startwone(Tensor *self, PyObject *axis) {
  Py_DECREF(self->axis);
  self->axis = axis;
  if (self->axis == NULL) {
    Py_DECREF(self);
    PyErr_SetString(PyExc_RuntimeError,
                    "Unable to allocate memory for Tensor object");
  }
}

inline static void Tensor_SetData_startwone_without_init(Tensor *self,
                                                  PyObject *data) {
  self->data = data;
  Py_INCREF(data);
  if (self->data == NULL) {
    Py_DECREF(self);
    PyErr_SetString(PyExc_RuntimeError,
                    "Unable to allocate memory for Tensor object");
  }
}
inline static void Tensor_SetX_startwone_without_init(Tensor *self, PyObject *x) {
  self->x = x;
  Py_INCREF(x);
  if (self->x == NULL) {
    Py_DECREF(self);
    PyErr_SetString(PyExc_RuntimeError,
                    "Unable to allocate memory for Tensor object");
  }
}
inline static void Tensor_SetY_startwone_without_init(Tensor *self, PyObject *y) {
  self->y = y;
  Py_INCREF(y);
  if (self->y == NULL) {
    Py_DECREF(self);
    PyErr_SetString(PyExc_RuntimeError,
                    "Unable to allocate memory for Tensor object");
  }
}
inline static void Tensor_SetGrad_startwone_without_init(Tensor *self,
                                                  PyObject *grad) {
  self->grad = grad;
  Py_INCREF(grad);
  if (self->grad == NULL) {
    Py_DECREF(self);
    PyErr_SetString(PyExc_RuntimeError,
                    "Unable to allocate memory for Tensor object");
  }
}
inline static void Tensor_SetGraph_startwone_without_init(Tensor *self,
                                                   PyObject *graph) {
  self->graph = graph;
  Py_INCREF(graph);
  if (self->graph == NULL) {
    Py_DECREF(self);
    PyErr_SetString(PyExc_RuntimeError,
                    "Unable to allocate memory for Tensor object");
  }
}

inline static void Tensor_SetAxis_startwone_without_init(Tensor *self,
                                                  PyObject *axis) {
  self->axis = axis;
  Py_INCREF(axis);
  if (self->axis == NULL) {
    Py_DECREF(self);
    PyErr_SetString(PyExc_RuntimeError,
                    "Unable to allocate memory for Tensor object");
  }
}

#endif