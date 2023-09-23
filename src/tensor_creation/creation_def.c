#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "creation_def.h"
#include "../numboost_api.h"
#include "numpy/arrayobject.h"
#include "omp.h"

Register_Arange_Method(bool);
Register_Arange_Method(byte);
Register_Arange_Method(ubyte);
Register_Arange_Method(short);
Register_Arange_Method(ushort);
Register_Arange_Method(int);
Register_Arange_Method(uint);
Register_Arange_Method(long);
Register_Arange_Method(ulong);
Register_Arange_Method(longlong);
Register_Arange_Method(ulonglong);
Register_Arange_Method(half);
Register_Arange_Method(float);
Register_Arange_Method(double);
Register_Arange_Method(longdouble);
Register_Arange_Not_Support_Type(cfloat);
Register_Arange_Not_Support_Type(cdouble);
Register_Arange_Not_Support_Type(clongdouble);
Register_Arange_Not_Support_Type(object);
Register_Arange_Not_Support_Type(string);
Register_Arange_Not_Support_Type(unicode);
Register_Arange_Not_Support_Type(void);
Register_Arange_Not_Support_Type(datetime);
Register_Arange_Not_Support_Type(timedelta);
Register_Arange_Array();

PyObject *tensor_new(Tensor *tensor, PyObject *other, PyObject *data,
                     const char *grad_fn) {
  Tensor *ret = (Tensor *)Tensor_type->tp_alloc(Tensor_type, 0);
  if (ret != NULL) {
    if (Py_IS_TYPE(other, Tensor_type)) {
      Tensor *tmp = (Tensor *)other;
      if (tensor->require_grad || tmp->require_grad) {
        Tensor_SetX_without_init_value(ret, (PyObject *)tensor);
        Tensor_SetY_without_init_value(ret, other);
        Tensor_SetGradFn(ret, grad_fn);
        Tensor_SetRequireGrad(ret, true);
        Tensor_SetVars(ret, tensor->vars + tmp->vars + 1);
      } else {
        Tensor_SetX_without_init_value(ret, Py_None);
        Tensor_SetY_without_init_value(ret, Py_None);
        Tensor_SetGradFn(ret, "");
        Tensor_SetRequireGrad(ret, false);
        Tensor_SetVars(ret, 0);
      }
    } else {
      if (tensor->require_grad) {
        Tensor_SetX_without_init_value(ret, (PyObject *)tensor);
        Tensor_SetY_without_init_value(ret, other);
        Tensor_SetRequireGrad(ret, true);
        Tensor_SetGradFn(ret, grad_fn);
        if (Py_IsNone(other)) {
          Tensor_SetVars(ret, tensor->vars + 1);
        } else {
          Tensor_SetVars(ret, tensor->vars + 2);
        }
      } else {
        Tensor_SetX_without_init_value(ret, Py_None);
        Tensor_SetY_without_init_value(ret, Py_None);
        Tensor_SetRequireGrad(ret, false);
        Tensor_SetGradFn(ret, grad_fn);
        Tensor_SetVars(ret, 0);
      }
    }
    PyObject *zero = PyLong_FromLong(0);
    ret->data = data;
    Tensor_SetHasConv(ret, tensor->has_conv);
    Tensor_SetGraph_without_init_value(ret, tensor->graph);
    Tensor_SetDim(ret, tensor->dim);
    Tensor_SetAxis_without_init_value(ret, tensor->axis);
    Tensor_SetGrad_without_init_value(ret, zero);
    return (PyObject *)ret;
  } else {
    PyErr_SetString(PyExc_MemoryError,
                    "Unable to allocate memory for Tensor Object");
    return NULL;
  }
}

PyObject *tensor_empty(PyObject *data) {
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

PyObject *arange(PyObject *self, PyObject *args, PyObject *kwds) {
  int start = 0, stop = -1, step = -1, dtype = -1;
  (void)self;
  char *kwds_ls[] = {"start", "stop", "step", "dtype", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "i|iii", kwds_ls, &start, &stop,
                                   &step, &dtype)) {
    return NULL;
  }
  int start_;
  int stop_;
  int step_;
  npy_intp size = 0;
  if (dtype == -1) {
    dtype = NPY_LONGLONG;
  } else if (dtype > NPY_HALF || dtype < -1) {
    PyErr_SetString(PyExc_RuntimeError, "dtype value is not valid");
    return NULL;
  }
  if (stop == -1) {
    start_ = 0;
    stop_ = start;
    if (step == -1) {
      step_ = 1;
    } else {
      step_ = step;
    }
  } else {
    stop_ = stop;
    start_ = start;
    if (step == -1) {
      step_ = 1;
    } else {
      step_ = step;
    }
  }
  size = abs(stop_ - start_ / step_);
  npy_intp shape[1] = {size};
  PyArrayObject *result = (PyArrayObject *)PyArray_EMPTY(1, shape, dtype, 0);
  Numboost_AssertNULL(result);
  arange_operations[dtype](result, start_, step_, size);
  PyObject *ret = tensor_empty((PyObject *)result);
  return ret;
}
