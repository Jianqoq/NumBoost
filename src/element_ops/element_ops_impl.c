#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "../binary_ops/binary_op_def.h"
#include "../import_module_methods.h"
#include "../python_magic/python_math_magic.h"
#include "../set_tensor_properties.h"
#include "../tensor.h"
#include "../type_convertor/type_convertor.h"
#include "../utils.h"
#include "element_ops_def.h"
#include "mkl_vml_functions.h"
#include "numpy/ndarraytypes.h"
#include <math.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <omp.h>
#include <stdlib.h>

void store_base(Tensor *key, PyObject *base) {
  Log_Dict *s = NULL;
  if (LOG_DICT != NULL)
    HASH_FIND_PTR(LOG_DICT, &key, s);
  if (s == NULL) {
    s = (Log_Dict *)malloc(sizeof(Power_Dict));
    s->key = key;
    s->base = base;
    HASH_ADD_PTR(LOG_DICT, key, s);
  }
}

PyObject *get_base(Tensor *key) {
  Log_Dict *s;
  HASH_FIND_PTR(LOG_DICT, &key, s);
  if (s == NULL) {
    PyErr_SetString(PyExc_KeyError, "Base not found in dict");
    return NULL;
  }
  return s->base;
}

void store_power(Tensor *key, PyObject *power) {
  Power_Dict *s = NULL;
  if (POWER_DICT != NULL)
    HASH_FIND_PTR(POWER_DICT, &key, s);
  if (s == NULL) {
    s = (Power_Dict *)malloc(sizeof(Power_Dict));
    s->key = key;
    s->prev_power = power;
    HASH_ADD_PTR(POWER_DICT, key, s);
  }
}

PyObject *get_power(Tensor *key) {
  Power_Dict *s;
  HASH_FIND_PTR(POWER_DICT, &key, s);
  if (s == NULL) {
    PyErr_SetString(PyExc_KeyError, "Power not found in dict");
    return NULL;
  }
  return s->prev_power;
}

void store_array_shape(Tensor *key, npy_intp *shape, int len) {
  Array_Shape *s = NULL;
  if (ARRAY_SHAPE != NULL)
    HASH_FIND_PTR(ARRAY_SHAPE, &key, s);
  if (s == NULL) {
    s = (Array_Shape *)malloc(sizeof(Array_Shape));
    s->key = key;
    s->shape = shape;
    s->len = len;
    HASH_ADD_PTR(ARRAY_SHAPE, key, s);
  }
}

npy_intp *get_array_shape(Tensor *key) {
  Array_Shape *s;
  HASH_FIND_PTR(ARRAY_SHAPE, &key, s);
  if (s == NULL) {
    PyErr_SetString(PyExc_KeyError, "Array shape not found in dict");
    return NULL;
  }
  return s->shape;
}

npy_intp get_shape_len(Tensor *key) {
  Array_Shape *s;
  HASH_FIND_PTR(ARRAY_SHAPE, &key, s);
  if (s == NULL) {
    PyErr_SetString(PyExc_KeyError, "Array shape not found in dict");
    return -1;
  }
  return s->len;
}

void store_tensordot_data(Tensor *key, Tensordot_Metadata *metadata) {
  Tensordot_Dict *s = NULL;
  if (TENSORDOT_DICT != NULL)
    HASH_FIND_PTR(TENSORDOT_DICT, &key, s);
  if (s == NULL) {
    s = (Tensordot_Dict *)malloc(sizeof(Tensordot_Dict));
    s->key = key;
    s->metadata = metadata;
    Py_INCREF(metadata->matmul_result);
    Py_INCREF(metadata->transposed_reshape_a);
    Py_INCREF(metadata->transposed_reshape_b);
    HASH_ADD_PTR(TENSORDOT_DICT, key, s);
  }
}

Tensordot_Metadata *get_tensordot_data(Tensor *key) {
  Tensordot_Dict *s;
  HASH_FIND_PTR(TENSORDOT_DICT, &key, s);
  if (s == NULL) {
    PyErr_SetString(PyExc_KeyError, "Tensordot data not found in dict");
    return NULL;
  }
  return s->metadata;
}

inline static Tensor *Generic_function_new_float(
    void (*vect_func)(const int, const float *, float *), float (*func)(float),
    Tensor *self, PyArrayObject *array, Tensor *out, const char *grad_fn) {
  npy_intp ndims = PyArray_NDIM(array);
  npy_intp *shape = PyArray_SHAPE(array);
  npy_intp size = 1;
  for (npy_intp i = 0; i < ndims; i++) {
    size *= shape[i];
  }
  if (out == NULL) {
    PyArrayObject *array2 =
        (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT, 0);
    float *data2 = (float *)PyArray_DATA(array2);
    const float *data = (const float *)PyArray_DATA(array);
    vect_func((const int)size, data,
              data2); // need benchmark to see if needed to release GIL
    return (Tensor *)__new_Tensor(self, (PyObject *)array2, NULL,
                                  self->require_grad ? grad_fn : "");
  } else {
    long long i;

    npy_float *data = (npy_float *)PyArray_DATA(
        (PyArrayObject *)out->data); // ONLY WHEN MEM SIZE IS SAME OR SMALLER
#pragma omp parallel for // need benchmark to see if needed to release GIL
    for (i = 0; i < size; i++) {
      data[i] = func(data[i]);
    }
    out->grad_fn = out->require_grad ? grad_fn : "";
    Py_INCREF(out);
    return out;
  }
}

inline static Tensor *Generic_function_new_double(
    void (*vect_func)(const int, const double *, double *),
    double (*func)(double), Tensor *self, PyArrayObject *array, Tensor *out,
    const char *grad_fn) {
  npy_intp ndims = PyArray_NDIM(array);
  npy_intp *shape = PyArray_SHAPE(array);
  npy_intp size = 1;
  for (npy_intp i = 0; i < ndims; i++) {
    size *= shape[i];
  }
  if (out == NULL) {
    PyArrayObject *array2 =
        (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_DOUBLE, 0);
    double *data2 = (double *)PyArray_DATA(array2);
    const double *data = (const double *)PyArray_DATA(array);
    vect_func((const int)size, data,
              data2); // need benchmark to see if needed to release GIL
    return (Tensor *)__new_Tensor(self, (PyObject *)array2, NULL,
                                  self->require_grad ? grad_fn : "");
  } else {
    long long i;
    double *data = (double *)PyArray_DATA(
        (PyArrayObject *)out->data); // ONLY WHEN MEM SIZE IS SAME OR SMALLER
#pragma omp parallel for
    for (i = 0; i < size; i++) {
      data[i] = func(data[i]);
    }
    out->grad_fn = out->require_grad ? grad_fn : "";
    Py_INCREF(out);
    return out;
  }
}

inline static Tensor *Generic_function(PyObject *func, const char *grad_fn,
                                       PyObject *const *args, size_t nargsf) {
  PyObject *result;
  Tensor *tensor = NULL;
  if (nargsf == 1) {
    tensor = (Tensor *)args[0];
    result = PyObject_CallOneArg(func, tensor->data);
    if (result == NULL) {
      return NULL;
    }
  } else {
    tensor = (Tensor *)args[0];
    Tensor *out = NULL;
    if (Py_IsNone(args[1]))
      out = (Tensor *)args[1];
    if (out == NULL) {
      result = PyObject_CallFunctionObjArgs(func, tensor->data, Py_None, NULL);
    } else {
      result =
          PyObject_CallFunctionObjArgs(func, tensor->data, out->data, NULL);
    }
    if (result == NULL) {
      return NULL;
    }
  }
  if (!tensor->require_grad)
    grad_fn = "";
  Tensor *to_return = (Tensor *)__new_Tensor(tensor, result, NULL, grad_fn);
  return to_return;
}

inline static PyObject *
Generic_function_internal(PyObject *func, PyObject *args, PyObject *out) {
  PyObject *result;
  if (out == NULL) {
    result = PyObject_CallOneArg(func, args);
  } else {
    result = PyObject_CallFunctionObjArgs(func, args, out, NULL);
  }
  if (result == NULL) {
    return NULL;
  }
  return result;
}

inline static PyObject *Generic_function_new_float_internal(
    void (*vect_func)(const int, const float *, float *), float (*func)(float),
    PyArrayObject *array, PyObject *out) {
  npy_intp ndims = PyArray_NDIM(array);
  npy_intp *shape = PyArray_SHAPE(array);
  npy_intp size = 1;
  for (npy_intp i = 0; i < ndims; i++) {
    size *= shape[i];
  }
  if (out == NULL) {
    PyArrayObject *array2 =
        (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT, 0);
    float *data2 = (float *)PyArray_DATA(array2);
    const float *data = (const float *)PyArray_DATA(array);
    vect_func((const int)size, data,
              data2); // need benchmark to see if needed to release GIL
    return (PyObject *)array2;
  } else {
    npy_intp i;
    npy_float *data = (npy_float *)PyArray_DATA(
        (PyArrayObject *)out); // ONLY WHEN MEM SIZE IS SAME OR SMALLER
#pragma omp parallel for       // need benchmark to see if needed to release GIL
    for (i = 0; i < size; i++) {
      data[i] = func(data[i]);
    }
    Py_INCREF(out);
    return out;
  }
}

inline static PyObject *Generic_function_new_double_internal(
    void (*vect_func)(const int, const double *, double *),
    double (*func)(double), PyArrayObject *array, PyObject *out) {
  npy_intp ndims = PyArray_NDIM(array);
  npy_intp *shape = PyArray_SHAPE(array);
  npy_intp size = 1;
  for (npy_intp i = 0; i < ndims; i++) {
    size *= shape[i];
  }
  if (out == NULL) {
    PyArrayObject *array2 =
        (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_DOUBLE, 0);
    double *data2 = (double *)PyArray_DATA(array2);
    const double *data = (const double *)PyArray_DATA(array);
    vect_func((const int)size, data, data2);
    PyObject *ret = (PyObject *)array2;
    return ret;
  } else {
    long long i;
    npy_float64 *data = (npy_float64 *)PyArray_DATA(
        (PyArrayObject *)out); // ONLY WHEN MEM SIZE IS SAME OR SMALLER
#pragma omp parallel for
    for (i = 0; i < size; i++) {
      data[i] = func(data[i]);
    }
    Py_INCREF(out);
    return out;
  }
}

Tensor *reshape(PyObject *self, PyObject *const *args, size_t nargsf,
                PyObject *kwnames) {
  (void)self;
  size_t nargs = PyVectorcall_NARGS(nargsf);
  PyArrayObject *array;
  npy_intp *pre_shape = NULL;
  npy_intp *pre_shape2 = malloc(sizeof(npy_intp) * NPY_MAXDIMS);
  bool isNULL = kwnames == NULL;
  int order = 0;
  if (nargs < 2 && isNULL) {
    PyErr_SetString(PyExc_TypeError,
                    "Expected at least 2 positional arguments");
    return NULL;
  }

  Tensor *tensor = (Tensor *)args[0];
  int length = (int)PyTuple_GET_SIZE(args[1]);
  npy_intp dims[NPY_MAXDIMS] = {0};
  for (uint8_t i = 0; i < length; i++) {
    long item = PyLong_AsLong(PyTuple_GET_ITEM(args[1], i));
    dims[i] = item;
  }
  PyArray_Dims shape = {dims, length};
  array = (PyArrayObject *)tensor->data;
  PyObject *result = PyArray_Newshape(array, &shape, order);
  if (result == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Error in reshape");
    return NULL;
  }
  const char *grad_fn = "ReshapeBackward";
  if (!tensor->require_grad)
    grad_fn = "";
  else {
    pre_shape = PyArray_SHAPE(array);
  }
  int ndim = (int)PyArray_NDIM(array);
  for (npy_intp i = 0; i < NPY_MAXDIMS; i++) {
    if (i < ndim)
      pre_shape2[i] = pre_shape[i];
    else
      pre_shape2[i] = 0;
  }
  Tensor *to_return = (Tensor *)__new_Tensor(tensor, result, NULL, grad_fn);
  if (pre_shape != NULL) {
    store_array_shape(to_return, pre_shape2, ndim);
  }
  return to_return;
}

inline void tensordot_axes_(int ndim, long *axes_, long n_len, long *_len,
                            npy_intp *shape, npy_intp *newshape,
                            npy_intp **newaxes, npy_intp **oldshape,
                            long *axes_len, bool a) {
  long real_len = 0;
  long *__notin = range_excluding_list(0, ndim, axes_, -100, n_len, &real_len);
  *_len = real_len;
  // get len
  long *notin = malloc(sizeof(long) * (real_len));
  int index = 0;
  for (int i = 0; i < ndim; i++)
    if (__notin[i] != -100) {
      notin[index] = __notin[i];
      index++;
    }
  free(__notin);
#ifdef DEBUG
  DEBUG_PRINT("notin = [");
  for (int i = 0; i < real_len; i++) {
    DEBUG_PRINT("%ld ", notin[i]);
  }
  DEBUG_PRINT("]\n");
#endif
  // newaxes_a
  DEBUG_PRINT("newaxes length: %ld\n", n_len + real_len);
  *axes_len = n_len + real_len;
  npy_intp *newaxes_ = malloc(sizeof(npy_intp) * (*axes_len));
  *newaxes = newaxes_;
  if (a) {
    int j = 0;
    index = 0;
    for (j = 0; j < real_len; j++)
      newaxes_[j] = notin[j];
    for (; j < *axes_len; j++)
      newaxes_[j] = axes_[index++];
  } else // b
  {
    int j = 0;
    index = 0;
    for (j = 0; j < n_len; j++)
      newaxes_[j] = axes_[j];
    for (; j < *axes_len; j++)
      newaxes_[j] = notin[index++];
  }
#ifdef DEBUG
  DEBUG_PRINT("newaxes_ = [");
  for (int i = 0; i < *axes_len; i++) {
    DEBUG_PRINT("%ld ", newaxes_[i]);
  }
  DEBUG_PRINT("]\n");
#endif
  npy_intp N2 = 1;
  for (long i = 0; i < n_len; i++) {
    long axis = axes_[i];
    N2 *= shape[axis];
  }
  // newshape_a
  npy_intp multiply_reduce = 1;
  for (int i = 0; i < real_len; i++)
    multiply_reduce *= shape[notin[i]];
  if (!a) {
    newshape[0] = N2;
    newshape[1] = multiply_reduce;
  } else {
    newshape[0] = multiply_reduce;
    newshape[1] = N2;
  }
  // old_a
  npy_intp *oldshape_a = malloc(sizeof(npy_intp) * real_len);
  for (int i = 0; i < real_len; i++)
    oldshape_a[i] = shape[notin[i]];
  free(notin);
  DEBUG_PRINT("REAL_LEN: %ld\n", real_len);
  *oldshape = oldshape_a;
}

static inline void *handle_axes(long **axes_, PyObject *axes_tuple,
                                long *ndim) {
  if (*axes_ == NULL && axes_tuple != NULL && PySequence_Check(axes_tuple)) {
    long nd = (long)PyObject_Length(axes_tuple);
    *ndim = nd;
    *axes_ = malloc(sizeof(long) * nd);
    PyObject **ptr = PySequence_Fast_ITEMS(axes_tuple);
    DEBUG_PRINT("ndim: %ld\n", nd);
    for (Py_ssize_t i = 0; i < nd; i++) {
      (*axes_)[i] = PyLong_AsLong(ptr[i]);
      if ((*axes_)[i] == -1 && PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Invalid data type for axes");
        return NULL;
      }
    }
    Py_DECREF(axes_tuple);
  } else if (*axes_ == NULL && axes_tuple != NULL) {
    *axes_ = malloc(sizeof(long) * 1);
    (*axes_)[0] = PyLong_AsLong(axes_tuple);
    DEBUG_PRINT("axes_tuple != NULL, axes: %ld\n", (*axes_)[0]);
    if ((*axes_)[0] == -1 && PyErr_Occurred()) {
      PyErr_SetString(PyExc_TypeError, "Invalid data type for axes");
      return NULL;
    }
    Py_DECREF(axes_tuple);
  }
  return ndim;
}

Tensor *tensordot(PyObject *self, PyObject *const *args, size_t nargsf,
                  PyObject *kwnames) {
  (void)self;
  if (nargsf != 3) {
    PyErr_SetString(PyExc_TypeError, "Expected 3 positional arguments");
    return NULL;
  }
  Tensor *tensor1 = (Tensor *)args[0];
  Tensor *tensor2 = (Tensor *)args[1];
  npy_intp *a_shape = NULL, *b_shape = NULL;
  long na = 1, nb = 1;
  long axes = 1;
  long *axes_a = NULL, *axes_b = NULL;
  PyObject *axes_a_tuple = NULL;
  PyObject *axes_b_tuple = NULL;
  if (PySequence_Check(args[2])) {
    axes_a_tuple = PySequence_GetItem(args[2], 0);
    axes_b_tuple = PySequence_GetItem(args[2], 1);
  } else {
    axes = PyLong_AsLong(args[2]);
    long axes_abs = abs(axes);
    na = axes_abs;
    nb = axes_abs;
    if (axes == -1 && PyErr_Occurred()) {
      PyErr_SetString(PyExc_TypeError, "Invalid data type for axes");
      return NULL;
    } else {
      axes_a = malloc(sizeof(long) * axes_abs);
      axes_b = malloc(sizeof(long) * axes_abs);
      if (axes < 0) {
        for (long i = 0; i < axes_abs; i++)
          axes_a[i] = axes_abs - i;
        for (long i = 0; i < axes_abs; i++)
          axes_b[i] = -i + axes_abs; // (+ axes_abs) means when (-axes + i) is
                                     // -1, list[-axes + i] can be last element
      } else if (axes > 0) {
        for (long i = 0; i < axes_abs; i++)
          axes_a[i] =
              -axes + i + axes_abs; // (+ axes_abs) means when (-axes + i) is
                                    // -1, list[-axes + i] can be last element
        for (long i = 0; i < axes_abs; i++)
          axes_b[i] = i;
      } else {
        na = 0;
        nb = 0;
        axes_a = NULL;
        axes_b = NULL;
      }
    }
  }
  if (handle_axes(&axes_a, axes_a_tuple, &na) == NULL)
    return NULL;
  if (handle_axes(&axes_b, axes_b_tuple, &nb) == NULL)
    return NULL;
  PyObject *a =
      PyArray_FromAny(tensor1->data, NULL, 0, 0, NPY_ARRAY_DEFAULT, NULL);
  PyObject *b =
      PyArray_FromAny(tensor2->data, NULL, 0, 0, NPY_ARRAY_DEFAULT, NULL);

  if (a == NULL || b == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "connot convert tensor to numpy array");
    return NULL;
  }
  PyArrayObject *a_ = (PyArrayObject *)a;
  PyArrayObject *b_ = (PyArrayObject *)b;
  a_shape = PyArray_SHAPE(a_);
  b_shape = PyArray_SHAPE(b_);
  int ndim_a = ((PyArrayObject_fields *)a_)->nd;
  int ndim_b = ((PyArrayObject_fields *)b_)->nd;
  bool shape_equal = true;
  if (na != nb) {
    shape_equal = false;
  } else if (axes_a != NULL && axes_b != NULL) {
    for (int i = 0; i < na; i++) {
      if (a_shape[axes_a[i]] != b_shape[axes_b[i]]) {
        shape_equal = false;
        break;
      }
      if (axes_a[i] < 0)
        axes_a[i] += ndim_a;
      if (axes_b[i] < 0)
        axes_b[i] += ndim_b;
    }
  }
  if (!shape_equal) {
    PyErr_SetString(PyExc_TypeError, "shape-mismatch for sum");
    return NULL;
  }
  long a_len = 0, newaxes_a_len = 0;
  npy_intp *newshape_a = malloc(sizeof(npy_intp) * 2);
  npy_intp *newaxes_a = NULL, *oldshape_a = NULL;
  tensordot_axes_(ndim_a, axes_a, na, &a_len, a_shape, newshape_a, &newaxes_a,
                  &oldshape_a, &newaxes_a_len, true);
  PyArray_Dims at_dims = {newshape_a, 2};
  PyArray_Dims at_new_dims = {newaxes_a, newaxes_a_len};

  long b_len = 0, newaxes_b_len = 0;
  npy_intp *newshape_b = malloc(sizeof(npy_intp) * 2);
  npy_intp *newaxes_b = NULL, *oldshape_b = NULL;
  tensordot_axes_(ndim_b, axes_b, nb, &b_len, b_shape, newshape_b, &newaxes_b,
                  &oldshape_b, &newaxes_b_len, false);
  PyArray_Dims bt_dims = {newshape_b, 2};
  PyArray_Dims bt_new_dims = {newaxes_b, newaxes_b_len};

  PyObject *at_ = PyArray_Transpose(a_, &at_new_dims);
  PyObject *bt_ = PyArray_Transpose(b_, &bt_new_dims);
  if (at_ == NULL || bt_ == NULL) {
    PyErr_SetString(PyExc_TypeError, "transpose error");
    return NULL;
  }
  PyArrayObject *at_arr = (PyArrayObject *)at_;
  PyArrayObject *bt_arr = (PyArrayObject *)bt_;
  PyObject *at = PyArray_Newshape(at_arr, &at_dims, 0);
  PyObject *bt = PyArray_Newshape(bt_arr, &bt_dims, 0);
  if (at == NULL || bt == NULL) {
    return NULL;
  }
  PyObject *res = PyArray_MatrixProduct(at, bt);
  if (res == NULL) {
    PyErr_SetString(PyExc_TypeError, "matmul error");
    return NULL;
  }
  int total_len = a_len + b_len;
  npy_intp *olds_merge_shape = malloc(sizeof(npy_intp) * (total_len));
  int j = 0;
  for (; j < total_len; j++) {
    if (j < a_len)
      olds_merge_shape[j] = oldshape_a[j];
    else
      olds_merge_shape[j] = oldshape_b[j - a_len];
  }
  PyArray_Dims olds_merge_dims = {olds_merge_shape, total_len};
  PyObject *result =
      PyArray_Newshape((PyArrayObject *)res, &olds_merge_dims, 0);
  Tensor *to_return = (Tensor *)create_Tensor((Tensor *)args[0], (Tensor *)args[1],
                                           result, "TensordotBackward");
  Py_DECREF(a);
  Py_DECREF(b);
  free(oldshape_a);
  free(oldshape_b);
  free(axes_a);
  free(axes_b);
  free(olds_merge_shape);
  free(newshape_a);
  free(newshape_b);
  if (to_return->require_grad) {
    Tensordot_Metadata *metadata = malloc(sizeof(Tensordot_Metadata));
    metadata->newshape_a.ptr = PyArray_SHAPE(at_arr);
    metadata->newshape_a.len = PyArray_NDIM(at_arr);
    metadata->newshape_b.ptr = PyArray_SHAPE(bt_arr);
    metadata->newshape_b.len = PyArray_NDIM(bt_arr);
    metadata->newaxes_a.ptr = newaxes_a;
    metadata->newaxes_a.len = newaxes_a_len;
    metadata->newaxes_b.ptr = newaxes_b;
    metadata->newaxes_b.len = newaxes_b_len;
    metadata->matmul_result = res;
    metadata->matmul_result_shape.ptr = PyArray_SHAPE((PyArrayObject *)res);
    metadata->matmul_result_shape.len = PyArray_NDIM((PyArrayObject *)res);
    metadata->transposed_shape_a.ptr = PyArray_SHAPE(at_arr);
    metadata->transposed_shape_a.len = at_new_dims.len;
    metadata->transposed_shape_b.ptr = PyArray_SHAPE(bt_arr);
    metadata->transposed_shape_b.len = bt_new_dims.len;
    metadata->transposed_reshape_a = at;
    metadata->transposed_reshape_b = bt;
    store_tensordot_data(to_return, metadata);
    Py_DECREF(at_);
    Py_DECREF(bt_);
    Py_DECREF(at);
    Py_DECREF(bt);
    Py_DECREF(res);
    return to_return;
  }
  Py_DECREF(at_);
  Py_DECREF(bt_);
  Py_DECREF(at);
  Py_DECREF(bt);
  Py_DECREF(res);
  free(newaxes_a);
  free(newaxes_b);
  return to_return;
}

PyObject *transpose(PyObject *self, PyObject *const *args, size_t nargsf,
                    PyObject *kwnames) {
  (void)self;
  int nargs = (int)PyVectorcall_NARGS(nargsf);
  DEBUG_PRINT("nargs: %d\n", nargs);
  if (nargs < 2) {
    PyErr_SetString(PyExc_TypeError,
                    "Expected at least 2 positional arguments");
    return NULL;
  }
  Tensor *tensor = (Tensor *)args[0];
  DEBUG_PyObject_Print(tensor);
  PyArrayObject *array = (PyArrayObject *)tensor->data;
  int length = nargs - 1;
  npy_intp *dims = malloc(sizeof(npy_intp) * length);
  DEBUG_PRINT("dims: ");
  for (uint8_t i = 1; i < nargs; i++) {
    long item = PyLong_AsLong(args[i]);
    if (item < 0) // axis input cannot be negative
      return NULL;
    DEBUG_PRINT("%d ", item);
    dims[i - 1] = item;
  }
  DEBUG_PRINT("\n");
  PyArray_Dims shape = {dims, length};
  PyObject *result = PyArray_Transpose(array, &shape);
  if (result == NULL) {
    return NULL;
  }
  PyObject *to_return = create_Tensor(tensor, Py_None, result, "TransposeBackward");
  if (tensor->require_grad)
    store_array_shape((Tensor *)to_return, dims, length);
  else
    free(dims);
  return to_return;
}

Tensor *_sum(PyObject *self, PyObject *const *args, size_t nargsf) {
  (void)self;
  (void)nargsf;
  Tensor *tensor = (Tensor *)args[0];
  PyArrayObject *tmp = (PyArrayObject *)tensor->data;
  int axis = NPY_MAXDIMS;
  PyArray_Descr *descr = NULL;
  if (args[2] != Py_None) {
    PyArray_DescrConverter(args[2], &descr);
  } else {
    PyArrayObject_fields *fields = (PyArrayObject_fields *)tmp;
    descr = fields->descr;
  }
  if (descr == NULL)
    return NULL;
  int dtype_enum = descr->type_num;
  int ndims;
  uint8_t i;
  PyObject *result = NULL;
  PyArrayObject *out = NULL;
  if (args[1] != Py_None)
    axis = PyLong_AsLong(args[1]);
  if (args[3] != Py_None)
    out = (PyArrayObject *)args[3];
  if (PyArray_CheckAxis(tmp, &axis, 0) == NULL) {
    return NULL;
  };
  if (PyObject_IsTrue(args[4])) {
    npy_intp new_shape[NPY_MAXDIMS] = {0};
    if (out != NULL)
      result = PyArray_Sum(tmp, axis, dtype_enum, out);
    else
      result = PyArray_Sum(tmp, axis, dtype_enum, NULL);
    if (result == NULL)
      return NULL;
    PyArrayObject *r = (PyArrayObject *)result;
    npy_intp *shape = PyArray_SHAPE(r);
    ndims = PyArray_NDIM(r);
    for (i = 0; i < axis; i++) {
      new_shape[i] = shape[i];
    }
    new_shape[axis] = 1;
    for (i = 0; i < ndims - axis; i++) {
      new_shape[i + axis + 1] = shape[axis];
      axis++;
    }
    PyArray_Dims d = {new_shape, ndims + 1};
    result = PyArray_Newshape(r, &d, 0);
  } else {
    result = PyArray_Sum(tmp, axis, dtype_enum, out);
  }
  if (result == NULL) {
    return NULL;
  }
  Tensor *to_return = (Tensor *)__new_Tensor(tensor, result, NULL, "");
  return to_return;
}

Tensor *_max(PyObject *self, PyObject *const *args, size_t nargsf) {
  (void)nargsf;
  (void)self;
  Tensor *tensor = (Tensor *)args[0];
  PyArrayObject *tmp = (PyArrayObject *)tensor->data;
  int axis = NPY_MAXDIMS;
  PyArray_Descr *descr = NULL;
  if (args[2] != Py_None) {
    PyArray_DescrConverter(args[2], &descr);
  } else {
    PyArrayObject_fields *fields = (PyArrayObject_fields *)tmp;
    descr = fields->descr;
  }
  if (descr == NULL)
    return NULL;
  int dtype_enum = descr->type_num;
  int ndims;
  uint8_t i;
  PyObject *result = NULL;
  PyArrayObject *out = NULL;
  if (args[1] != Py_None)
    axis = PyLong_AsLong(args[1]);
  if (args[3] != Py_None)
    out = (PyArrayObject *)args[3];
  if (PyArray_CheckAxis(tmp, &axis, 0) == NULL) {
    return NULL;
  };
  if (PyObject_IsTrue(args[4])) {
    npy_intp new_shape[NPY_MAXDIMS] = {0};
    if (out != NULL)
      result = PyArray_Sum(tmp, axis, dtype_enum, out);
    else
      result = PyArray_Sum(tmp, axis, dtype_enum, NULL);
    if (result == NULL)
      return NULL;
    PyArrayObject *r = (PyArrayObject *)result;
    npy_intp *shape = PyArray_SHAPE(r);
    ndims = PyArray_NDIM(r);
    for (i = 0; i < axis; i++) {
      new_shape[i] = shape[i];
    }
    new_shape[axis] = 1;
    for (i = 0; i < ndims - axis; i++) {
      new_shape[i + axis + 1] = shape[axis];
      axis++;
    }
    PyArray_Dims d = {new_shape, ndims + 1};
    result = PyArray_Newshape(r, &d, 0);
  } else {
    result = PyArray_Sum(tmp, axis, dtype_enum, out);
  }
  if (result == NULL) {
    return NULL;
  }
  Tensor *to_return = (Tensor *)__new_Tensor(tensor, result, NULL, "");
  return to_return;
}

Tensor *_min(PyObject *self, PyObject *const *args, size_t nargsf) {
  (void)nargsf;
  (void)self;
  Tensor *tensor = (Tensor *)args[0];
  PyArrayObject *tmp = (PyArrayObject *)tensor->data;
  int axis = NPY_MAXDIMS;
  int ndims;
  uint8_t i;
  PyObject *result = NULL;
  PyArrayObject *out = NULL;
  if (args[1] != Py_None)
    axis = PyLong_AsLong(args[1]);
  if (args[3] != Py_None)
    out = (PyArrayObject *)args[3];
  if (PyArray_CheckAxis(tmp, &axis, 0) == NULL) {
    return NULL;
  };
  if (PyObject_IsTrue(args[2])) {
    npy_intp new_shape[NPY_MAXDIMS] = {0};
    if (out != NULL)
      result = PyArray_Min(tmp, axis, out);
    else
      result = PyArray_Min(tmp, axis, NULL);
    if (result == NULL)
      return NULL;
    PyArrayObject *r = (PyArrayObject *)result;
    npy_intp *shape = PyArray_SHAPE(r);
    ndims = PyArray_NDIM(r);
    for (i = 0; i < axis; i++) {
      new_shape[i] = shape[i];
    }
    new_shape[axis] = 1;
    for (i = 0; i < ndims - axis; i++) {
      new_shape[i + axis + 1] = shape[axis];
      axis++;
    }
    PyArray_Dims d = {new_shape, ndims + 1};
    result = PyArray_Newshape(r, &d, 0);
  } else {
    result = PyArray_Min(tmp, axis, NULL);
  }
  if (result == NULL) {
    return NULL;
  }
  Tensor *to_return = (Tensor *)__new_Tensor(tensor, result, NULL, "");
  return to_return;
}

Tensor *_mean(PyObject *self, PyObject *const *args, size_t nargsf) {
  (void)self;
  Tensor *tensor = (Tensor *)args[0];
  PyArrayObject *tmp = (PyArrayObject *)tensor->data;
  int axis = NPY_MAXDIMS;
  PyArray_Descr *descr = NULL;
  if (args[2] != Py_None) {
    PyArray_DescrConverter(args[2], &descr);
  } else {
    PyArrayObject_fields *fields = (PyArrayObject_fields *)tmp;
    descr = fields->descr;
  }
  if (descr == NULL)
    return NULL;
  int dtype_enum = descr->type_num;
  int ndims;
  uint8_t i;
  PyObject *result = NULL;
  PyArrayObject *out = NULL;
  if (args[1] != Py_None)
    axis = PyLong_AsLong(args[1]);
  if (args[3] != Py_None)
    out = (PyArrayObject *)args[3];
  if (PyArray_CheckAxis(tmp, &axis, 0) == NULL) {
    return NULL;
  };
  if (PyObject_IsTrue(args[4])) {
    npy_intp new_shape[NPY_MAXDIMS] = {0};
    if (out != NULL)
      result = PyArray_Mean(tmp, axis, dtype_enum, out);
    else
      result = PyArray_Mean(tmp, axis, dtype_enum, NULL);
    if (result == NULL)
      return NULL;
    PyArrayObject *r = (PyArrayObject *)result;
    npy_intp *shape = PyArray_SHAPE(r);
    ndims = PyArray_NDIM(r);
    for (i = 0; i < axis; i++) {
      new_shape[i] = shape[i];
    }
    new_shape[axis] = 1;
    for (i = 0; i < ndims - axis; i++) {
      new_shape[i + axis + 1] = shape[axis];
      axis++;
    }
    PyArray_Dims d = {new_shape, ndims + 1};
    result = PyArray_Newshape(r, &d, 0);
  } else {
    result = PyArray_Mean(tmp, axis, dtype_enum, out);
  }
  if (result == NULL) {
    return NULL;
  }
  Tensor *to_return = (Tensor *)__new_Tensor(tensor, result, NULL, "");
  return to_return;
}

Tensor *_argmax_wrapper(PyObject *self, PyObject *const *args, size_t nargsf) {
  (void)nargsf;
  (void)self;
  Tensor *tensor = (Tensor *)args[0];
  PyArrayObject *tmp = (PyArrayObject *)tensor->data;
  int axis = NPY_MAXDIMS;
  int ndims;
  uint8_t i;
  PyObject *result = NULL;
  PyArrayObject *out = NULL;
  if (args[1] != Py_None)
    axis = PyLong_AsLong(args[1]);
  if (args[3] != Py_None)
    out = (PyArrayObject *)args[3];
  if (PyArray_CheckAxis(tmp, &axis, 0) == NULL) {
    return NULL;
  };
  if (PyObject_IsTrue(args[2])) {
    npy_intp new_shape[NPY_MAXDIMS] = {0};
    if (out != NULL)
      result = PyArray_ArgMax(tmp, axis, out);
    else
      result = PyArray_ArgMax(tmp, axis, NULL);
    if (result == NULL)
      return NULL;
    PyArrayObject *r = (PyArrayObject *)result;
    npy_intp *shape = PyArray_SHAPE(r);
    ndims = PyArray_NDIM(r);
    for (i = 0; i < axis; i++) {
      new_shape[i] = shape[i];
    }
    new_shape[axis] = 1;
    for (i = 0; i < ndims - axis; i++) {
      new_shape[i + axis + 1] = shape[axis];
      axis++;
    }
    PyArray_Dims d = {new_shape, ndims + 1};
    result = PyArray_Newshape(r, &d, 0);
  } else {
    result = PyArray_ArgMax(tmp, axis, NULL);
  }
  if (result == NULL) {
    return NULL;
  }
  Tensor *to_return = (Tensor *)__new_Tensor(tensor, result, NULL, "");
  return to_return;
}

Tensor *_argmin_wrapper(PyObject *self, PyObject *const *args, size_t nargsf) {
  (void)nargsf;
  (void)self;
  Tensor *tensor = (Tensor *)args[0];
  PyArrayObject *tmp = (PyArrayObject *)tensor->data;
  int axis = NPY_MAXDIMS;
  int ndims;
  uint8_t i;
  PyObject *result = NULL;
  PyArrayObject *out = NULL;
  if (args[1] != Py_None)
    axis = PyLong_AsLong(args[1]);
  if (args[3] != Py_None)
    out = (PyArrayObject *)args[3];
  if (PyArray_CheckAxis(tmp, &axis, 0) == NULL) {
    return NULL;
  };
  if (PyObject_IsTrue(args[2])) {
    npy_intp new_shape[NPY_MAXDIMS] = {0};
    if (out != NULL)
      result = PyArray_ArgMin(tmp, axis, out);
    else
      result = PyArray_ArgMin(tmp, axis, NULL);
    if (result == NULL)
      return NULL;
    PyArrayObject *r = (PyArrayObject *)result;
    npy_intp *shape = PyArray_SHAPE(r);
    ndims = PyArray_NDIM(r);
    for (i = 0; i < axis; i++) {
      new_shape[i] = shape[i];
    }
    new_shape[axis] = 1;
    for (i = 0; i < ndims - axis; i++) {
      new_shape[i + axis + 1] = shape[axis];
      axis++;
    }
    PyArray_Dims d = {new_shape, ndims + 1};
    result = PyArray_Newshape(r, &d, 0);
  } else {
    result = PyArray_ArgMin(tmp, axis, NULL);
  }
  if (result == NULL) {
    return NULL;
  }
  Tensor *to_return = (Tensor *)__new_Tensor(tensor, result, NULL, "");
  return to_return;
}

Tensor *_sin(PyObject *self, PyObject *tensor) {
  (void)self;
  PyObject *result = numboost_sin(((Tensor *)tensor)->data);
  return (Tensor *)__new_Tensor((Tensor *)tensor, result, NULL, "SinBackward");
}

Tensor *_cos(PyObject *self, PyObject *tensor) {
  (void)self;
  PyObject *result = numboost_cos(((Tensor *)tensor)->data);
  free(result);
  return (Tensor *)__new_Tensor((Tensor *)tensor, result, NULL, "CosBackward");
}

Tensor *_tan(PyObject *self, PyObject *tensor) {
  (void)self;
  PyObject *result = numboost_tan(((Tensor *)tensor)->data);
  free(result);
  return (Tensor *)__new_Tensor((Tensor *)tensor, result, NULL, "TanBackward");
}

Tensor *_asin(PyObject *self, PyObject *tensor) {
  (void)self;
  PyObject *result = numboost_asin(((Tensor *)tensor)->data);
  free(result);
  return (Tensor *)__new_Tensor((Tensor *)tensor, result, NULL, "ArcSinBackward");
}

Tensor *_acos(PyObject *self, PyObject *tensor) {
  (void)self;
  PyObject *result = numboost_acos(((Tensor *)tensor)->data);
  free(result);
  return (Tensor *)__new_Tensor((Tensor *)tensor, result, NULL, "ArcCosBackward");
}

Tensor *_atan(PyObject *self, PyObject *tensor) {
  (void)self;
  PyObject *result = numboost_atan(((Tensor *)tensor)->data);
  free(result);
  return (Tensor *)__new_Tensor((Tensor *)tensor, result, NULL, "ArcTanBackward");
}

Tensor *_sinh(PyObject *self, PyObject *tensor) {
  (void)self;
  PyObject *result = numboost_sinh(((Tensor *)tensor)->data);
  free(result);
  return (Tensor *)__new_Tensor((Tensor *)tensor, result, NULL, "SinhBackward");
}

Tensor *_cosh(PyObject *self, PyObject *tensor) {
  (void)self;
  PyObject *result = numboost_cosh(((Tensor *)tensor)->data);
  free(result);
  return (Tensor *)__new_Tensor((Tensor *)tensor, result, NULL, "CoshBackward");
}

Tensor *_exp(PyObject *self, PyObject *const *args, size_t nargsf) {
  (void)self;
  Tensor *tensor = (Tensor *)args[0];
  PyArrayObject *array = (PyArrayObject *)tensor->data;
  Tensor *out = nargsf > 1 && !Py_IsNone(args[1]) ? (Tensor *)args[1] : NULL;
  int typenum = PyArray_TYPE(array);
  if (typenum == NPY_FLOAT) {
    return Generic_function_new_float(vsExp, expf, tensor, array, out,
                                      "ExpBackward");
  } else if (typenum == NPY_DOUBLE) {
    return Generic_function_new_double(vdExp, exp, tensor, array, out,
                                       "ExpBackward");
  } else {
    Any_to_Float(&array, NULL, typenum);
    return Generic_function_new_float(vsExp, expf, tensor, array, out,
                                      "ExpBackward");
  }
}

Tensor *_log10(PyObject *self, PyObject *const *args, size_t nargsf) {
  (void)self;
  Tensor *tensor = (Tensor *)args[0];
  PyArrayObject *array = (PyArrayObject *)tensor->data;
  Tensor *out = nargsf > 1 && !Py_IsNone(args[1]) ? (Tensor *)args[1] : NULL;
  int typenum = PyArray_TYPE(array);
  if (typenum == NPY_FLOAT) {
    return Generic_function_new_float(vsLog10, log10f, tensor, array, out,
                                      "Log10Backward");
  } else if (typenum == NPY_DOUBLE) {
    return Generic_function_new_double(vdLog10, log10, tensor, array, out,
                                       "Log10Backward");
  } else {
    Any_to_Float(&array, NULL, typenum);
    return Generic_function_new_float(vsLog10, log10f, tensor, array, out,
                                      "Log10Backward");
  }
}

Tensor *_log(PyObject *self, PyObject *const *args, size_t nargsf) {
  (void)self;
  Tensor *tensor = (Tensor *)args[0];
  PyArrayObject *array = (PyArrayObject *)tensor->data;
  Tensor *out = nargsf > 1 && !Py_IsNone(args[1]) ? (Tensor *)args[1] : NULL;
  int typenum = PyArray_TYPE(array);
  if (typenum == NPY_FLOAT) {
    return Generic_function_new_float(vsLn, logf, tensor, array, out,
                                      "LogBackward");
  } else if (typenum == NPY_DOUBLE) {
    return Generic_function_new_double(vdLn, log, tensor, array, out,
                                       "LogBackward");
  } else {
    Any_to_Float(&array, NULL, typenum);
    return Generic_function_new_float(vsLn, logf, tensor, array, out,
                                      "LogBackward");
  }
}

Tensor *_tanh(PyObject *self, PyObject *tensor) {
  (void)self;
  PyObject *result = numboost_tanh(((Tensor *)tensor)->data);
  free(result);
  Tensor *to_return =
      (Tensor *)__new_Tensor((Tensor *)tensor, result, NULL, "TanhBackward");
  return to_return;
}

Tensor *_asinh(PyObject *self, PyObject *tensor) {
  (void)self;
  PyObject *result = numboost_asinh(((Tensor *)tensor)->data);
  free(result);
  Tensor *to_return =
      (Tensor *)__new_Tensor((Tensor *)tensor, result, NULL, "ArcSinhBackward");
  return to_return;
}

Tensor *_acosh(PyObject *self, PyObject *tensor) {
  (void)self;
  PyObject *result = numboost_acosh(((Tensor *)tensor)->data);
  free(result);
  Tensor *to_return =
      (Tensor *)__new_Tensor((Tensor *)tensor, result, NULL, "ArcCoshBackward");
  return to_return;
}

Tensor *_atanh(PyObject *self, PyObject *tensor) {
  (void)self;
  PyObject *result = numboost_atanh(((Tensor *)tensor)->data);
  free(result);
  Tensor *to_return =
      (Tensor *)__new_Tensor((Tensor *)tensor, result, NULL, "ArcTanhBackward");
  return to_return;
}

Tensor *_sqrt(PyObject *self, PyObject *const *args, size_t nargsf) {
  (void)self;
  Tensor *tensor = (Tensor *)args[0];
  PyArrayObject *array = (PyArrayObject *)tensor->data;
  Tensor *out = nargsf > 1 && !Py_IsNone(args[1]) ? (Tensor *)args[1] : NULL;
  int typenum = PyArray_TYPE(array);
  if (typenum == NPY_FLOAT) {
    return Generic_function_new_float(vsSqrt, sqrtf, tensor, array, out,
                                      "SqrtBackward");
  } else if (typenum == NPY_DOUBLE) {
    return Generic_function_new_double(vdSqrt, sqrt, tensor, array, out,
                                       "SqrtBackward");
  } else {
    Any_to_Float(&array, NULL, typenum);
    return Generic_function_new_float(vsSqrt, sqrtf, tensor, array, out,
                                      "SqrtBackward");
  }
}

Tensor *_abs(PyObject *self, PyObject *const *args, size_t nargsf) {
  (void)nargsf;
  (void)self;
  PyObject *result = numboost_abs(((Tensor *)args[0])->data);
  Tensor *to_return =
      (Tensor *)__new_Tensor((Tensor *)args[0], result, NULL, "AbsBackward");
  return to_return;
}

Tensor *_pow(PyObject *self, PyObject *const *args, size_t nargsf) {
  (void)nargsf;
  (void)self;
  PyObject *result = NULL;
  Tensor *tensor = NULL;
  PyObject *power = NULL;
  tensor = (Tensor *)args[0];
  power = args[1]; /*power could be python number or PyArrayObject*/
  if (Py_IS_TYPE(args[1], Tensor_type)) {
    Tensor *tmp = (Tensor *)args[1];
    power = tmp->data;
  }
  result = numboost_pow(tensor->data, power);
  Tensor *to_return = (Tensor *)__new_Tensor(tensor, result, NULL, "PowBackward");
  if (tensor->require_grad)
    store_power(to_return, power);
  return to_return;
}

static inline PyObject *internal_npy_cal_oneArgs(
    void (*vect_func1)(const int, const double *, double *),
    void (*vect_func2)(const int, const float *, float *),
    float (*func1)(float), double (*func2)(double), PyObject *args,
    PyObject *out) {
  PyArrayObject *array = (PyArrayObject *)args;
  PyObject *ret;
  int typenum = PyArray_TYPE(array);
  if (typenum == NPY_FLOAT) {
    DEBUG_PRINT("internal_npy_cal_oneArgs: NPY_FLOAT\n")
    ret = Generic_function_new_float_internal(vect_func2, func1, array, out);
    DEBUG_PRINT("Got ret: %p\n", ret)
    return ret;
  } else if (typenum == NPY_DOUBLE) {
    DEBUG_PRINT("internal_npy_cal_oneArgs: NPY_DOUBLE\n")
    ret = Generic_function_new_double_internal(vect_func1, func2, array, out);
    DEBUG_PRINT("Got ret: %p\n", ret)
    return ret;
  } else {
    DEBUG_PRINT("internal_npy_cal_oneArgs: Any_to_Float\n")
    Any_to_Float(&array, NULL, typenum);
    ret = Generic_function_new_float_internal(vect_func2, func1, array, out);
    DEBUG_PRINT("Got ret: %p\n", ret)
    return ret;
  }
}

PyObject *_sin_internal(PyObject *args, PyObject *out) {
  return internal_npy_cal_oneArgs(vdSin, vsSin, sinf, sin, args, out);
}

PyObject *_cos_internal(PyObject *args, PyObject *out) {
  return internal_npy_cal_oneArgs(vdCos, vsCos, cosf, cos, args, out);
}

PyObject *_tan_internal(PyObject *args, PyObject *out) {
  return internal_npy_cal_oneArgs(vdTan, vsTan, tanf, tan, args, out);
}

PyObject *_asin_internal(PyObject *args, PyObject *out) {
  return internal_npy_cal_oneArgs(vdAsin, vsAsin, asinf, asin, args, out);
}

PyObject *_acos_internal(PyObject *args, PyObject *out) {
  return internal_npy_cal_oneArgs(vdAcos, vsAcos, acosf, acos, args, out);
}

PyObject *_atan_internal(PyObject *args, PyObject *out) {
  return internal_npy_cal_oneArgs(vdAtan, vsAtan, atanf, atan, args, out);
}

PyObject *_sinh_internal(PyObject *args, PyObject *out) {
  return internal_npy_cal_oneArgs(vdSinh, vsSinh, sinhf, sinh, args, out);
}

PyObject *_cosh_internal(PyObject *args, PyObject *out) {
  return internal_npy_cal_oneArgs(vdCosh, vsCosh, coshf, cosh, args, out);
}

PyObject *_exp_internal(PyObject *args, PyObject *out) {
  return internal_npy_cal_oneArgs(vdExp, vsExp, expf, exp, args, out);
}

PyObject *_log10_internal(PyObject *args, PyObject *out) {
  return internal_npy_cal_oneArgs(vdLog10, vsLog10, log10f, log10, args, out);
}

PyObject *_log_internal(PyObject *args, PyObject *out) {
  return internal_npy_cal_oneArgs(vdLn, vsLn, logf, log, args, out);
}

PyObject *_tanh_internal(PyObject *args, PyObject *out) {
  return internal_npy_cal_oneArgs(vdTanh, vsTanh, tanhf, tanh, args, out);
}

PyObject *_asinh_internal(PyObject *args, PyObject *out) {
  return internal_npy_cal_oneArgs(vdAsinh, vsAsinh, asinhf, asinh, args, out);
}

PyObject *_acosh_internal(PyObject *args, PyObject *out) {
  return internal_npy_cal_oneArgs(vdAcosh, vsAcosh, acoshf, acosh, args, out);
}

PyObject *_atanh_internal(PyObject *args, PyObject *out) {
  return internal_npy_cal_oneArgs(vdAtanh, vsAtanh, atanhf, atanh, args, out);
}

PyObject *_sqrt_internal(PyObject *args, PyObject *out) {
  return internal_npy_cal_oneArgs(vdSqrt, vsSqrt, sqrtf, sqrt, args, out);
}

PyObject *_abs_internal(PyObject *args, PyObject *out) {
  return Generic_function_internal(NP_METHOD->abs, args, out);
}