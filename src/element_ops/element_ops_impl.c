#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "../binary_ops/binary_op_def.h"
#include "../import_module_methods.h"
#include "../numboost_api.h"
#include "../python_magic/python_math_magic.h"
#include "../reduction_ops/reduction_ops_def.h"
#include "../shape.h"
#include "../tensor.h"
#include "../type_convertor/type_convertor.h"
#include "../utils.h"
#include "element_ops_def.h"
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include "../tensor_creation/creation_def.h"

static char *keyword_list[] = {"a", "b", "out", NULL};

int compare(const void *a, const void *b) {
  int int_a = *((int *)a);
  int int_b = *((int *)b);

  if (int_a == int_b)
    return 0;
  else if (int_a < int_b)
    return -1;
  else
    return 1;
}

bool preprocess_axes(PyArrayObject *a, PyObject *axis, int *axes,
                     int *axis_len) {
  if (axis == NULL) {
    for (int i = 0; i < PyArray_NDIM(a); i++) {
      axes[i] = i;
    }
    *axis_len = PyArray_NDIM(a);
    return true;
  } else if (PyArray_IsAnyScalar(axis)) {
#define Assert_Axis_Valid(idx, axes, array)                                    \
  if (axes[idx] >= PyArray_NDIM(array)) {                                      \
    PyErr_SetString(PyExc_TypeError, "Invalid axis");                          \
    return false;                                                              \
  }
    axes[0] = (int)PyLong_AsLong(axis);
    Assert_Axis_Valid(0, axes, a);
    return true;
  } else if (PyTuple_Check(axis)) {
    *axis_len = (int)PyTuple_GET_SIZE(axis);
    for (int i = 0; i < *axis_len; i++) {
      axes[i] = (int)PyLong_AsLong(PyTuple_GET_ITEM(axis, i));
      Assert_Axis_Valid(i, axes, a);
    }
    qsort(axes, *axis_len, sizeof(int), compare);
    return true;
  } else {
    PyErr_SetString(PyExc_TypeError, "Invalid type for axis");
    return false;
  }
}

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
  Tensor *to_return = (Tensor *)tensor_new(tensor, Py_None, result, grad_fn);
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
  Tensor *to_return = (Tensor *)tensor_new((Tensor *)args[0], args[1],
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
  PyObject *to_return =
      tensor_new(tensor, Py_None, result, "TransposeBackward");
  if (tensor->require_grad)
    store_array_shape((Tensor *)to_return, dims, length);
  else
    free(dims);
  return to_return;
}

Tensor *_sum(PyObject *self, PyObject *args, PyObject *kwds) {
  (void)self;
  char *kwds_ls[] = {"a", "axis", "keepdims", "out", NULL};
  PyObject *_a = NULL;
  PyObject *axis = NULL;
  PyObject *keepdims = NULL;
  PyObject *out = NULL;
  bool keepdims_c = true;
  int axes[NPY_MAXDIMS] = {NULL};
  int axis_len = 1;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|OO", kwds_ls, &_a, &axis,
                                   &keepdims, &out)) {
    return NULL;
  }
  if (keepdims == NULL) {
    keepdims_c = false;
  } else {
    keepdims_c = PyObject_IsTrue(keepdims);
  }
  if (_a == NULL) {
    PyErr_SetString(PyExc_TypeError, "Expected at least an array");
    return NULL;
  }
  if (!Py_IS_TYPE(_a, Tensor_type)) {
    PyErr_SetString(PyExc_TypeError, "Expected a as Tensor obj");
    return NULL;
  }

  PyArrayObject *a = (PyArrayObject *)((Tensor *)_a)->data;
  if (!preprocess_axes(a, axis, axes, &axis_len)) {
    return NULL;
  }

  PyObject *result =
      numboost_sum((PyObject *)a, &out, axes, axis_len, keepdims_c);
  Numboost_AssertNULL(result);
  Tensor *to_return =
      (Tensor *)tensor_new((Tensor *)_a, Py_None, result, "");
  return to_return;
}

Tensor *_max(PyObject *self, PyObject *args, PyObject *kwds) {
  (void)self;
  char *kwds_ls[] = {"a", "axis", "keepdims", "out", NULL};
  PyObject *_a = NULL;
  PyObject *axis = NULL;
  PyObject *keepdims = NULL;
  PyObject *out = NULL;
  bool keepdims_c = true;
  int axes[NPY_MAXDIMS] = {NULL};
  int axis_len = 1;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|OO", kwds_ls, &_a, &axis,
                                   &keepdims, &out)) {
    return NULL;
  }
  if (keepdims == NULL) {
    keepdims_c = false;
  } else {
    keepdims_c = PyObject_IsTrue(keepdims);
  }
  if (_a == NULL) {
    PyErr_SetString(PyExc_TypeError, "Expected at least an array");
    return NULL;
  }
  if (!Py_IS_TYPE(_a, Tensor_type)) {
    PyErr_SetString(PyExc_TypeError, "Expected a as Tensor obj");
    return NULL;
  }

  PyArrayObject *a = (PyArrayObject *)((Tensor *)_a)->data;
  if (!preprocess_axes(a, axis, axes, &axis_len)) {
    return NULL;
  }

  PyObject *result =
      numboost_max((PyObject *)a, &out, axes, axis_len, keepdims_c);
  Numboost_AssertNULL(result);
  Tensor *to_return =
      (Tensor *)tensor_new((Tensor *)_a, Py_None, result, "");
  return to_return;
}

Tensor *_min(PyObject *self, PyObject *args, PyObject *kwds) {
  (void)self;
  char *kwds_ls[] = {"a", "axis", "keepdims", "out", NULL};
  PyObject *_a = NULL;
  PyObject *axis = NULL;
  PyObject *keepdims = NULL;
  PyObject *out = NULL;
  bool keepdims_c = true;
  int axes[NPY_MAXDIMS] = {NULL};
  int axis_len = 1;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|OO", kwds_ls, &_a, &axis,
                                   &keepdims, &out)) {
    return NULL;
  }
  if (keepdims == NULL) {
    keepdims_c = false;
  } else {
    keepdims_c = PyObject_IsTrue(keepdims);
  }
  if (_a == NULL) {
    PyErr_SetString(PyExc_TypeError, "Expected at least an array");
    return NULL;
  }
  if (!Py_IS_TYPE(_a, Tensor_type)) {
    PyErr_SetString(PyExc_TypeError, "Expected a as Tensor obj");
    return NULL;
  }

  PyArrayObject *a = (PyArrayObject *)((Tensor *)_a)->data;
  if (!preprocess_axes(a, axis, axes, &axis_len)) {
    return NULL;
  }

  PyObject *result =
      numboost_min((PyObject *)a, &out, axes, axis_len, keepdims_c);
  Numboost_AssertNULL(result);
  Tensor *to_return =
      (Tensor *)tensor_new((Tensor *)_a, Py_None, result, "");
  return to_return;
}

Tensor *_mean(PyObject *self, PyObject *args, PyObject *kwds) {
  (void)self;
  char *kwds_ls[] = {"a", "axis", "keepdims", "out", NULL};
  PyObject *_a = NULL;
  PyObject *axis = NULL;
  PyObject *keepdims = NULL;
  PyObject *out = NULL;
  bool keepdims_c = true;
  int axes[NPY_MAXDIMS] = {NULL};
  int axis_len = 1;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|OO", kwds_ls, &_a, &axis,
                                   &keepdims, &out)) {
    return NULL;
  }
  if (keepdims == NULL) {
    keepdims_c = false;
  } else {
    keepdims_c = PyObject_IsTrue(keepdims);
  }
  if (_a == NULL) {
    PyErr_SetString(PyExc_TypeError, "Expected at least an array");
    return NULL;
  }
  if (!Py_IS_TYPE(_a, Tensor_type)) {
    PyErr_SetString(PyExc_TypeError, "Expected a as Tensor obj");
    return NULL;
  }

  PyArrayObject *a = (PyArrayObject *)((Tensor *)_a)->data;
  if (!preprocess_axes(a, axis, axes, &axis_len)) {
    return NULL;
  }

  PyObject *result =
      numboost_mean((PyObject *)a, &out, axes, axis_len, keepdims_c);
  Numboost_AssertNULL(result);
  Tensor *to_return =
      (Tensor *)tensor_new((Tensor *)_a, Py_None, result, "");
  return to_return;
}

Tensor *_argmax(PyObject *self, PyObject *args, PyObject *kwds) {
  (void)self;
  char *kwds_ls[] = {"a", "axis", "keepdims", "out", NULL};
  PyObject *_a = NULL;
  PyObject *axis = NULL;
  PyObject *keepdims = NULL;
  PyObject *out = NULL;
  bool keepdims_c = true;
  int axes[NPY_MAXDIMS] = {NULL};
  int axis_len = 1;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|OO", kwds_ls, &_a, &axis,
                                   &keepdims, &out)) {
    return NULL;
  }
  if (keepdims == NULL) {
    keepdims_c = false;
  } else {
    keepdims_c = PyObject_IsTrue(keepdims);
  }
  if (_a == NULL) {
    PyErr_SetString(PyExc_TypeError, "Expected at least an array");
    return NULL;
  }
  if (!Py_IS_TYPE(_a, Tensor_type)) {
    PyErr_SetString(PyExc_TypeError, "Expected a as Tensor obj");
    return NULL;
  }

  PyArrayObject *a = (PyArrayObject *)((Tensor *)_a)->data;
  if (!preprocess_axes(a, axis, axes, &axis_len)) {
    return NULL;
  }

  PyObject *result =
      numboost_argmax((PyObject *)a, &out, axes, axis_len, keepdims_c);
  Numboost_AssertNULL(result);
  Tensor *to_return =
      (Tensor *)tensor_new((Tensor *)_a, Py_None, result, "");
  return to_return;
}

Tensor *_argmin(PyObject *self, PyObject *args, PyObject *kwds) {
  (void)self;
  char *kwds_ls[] = {"a", "axis", "keepdims", "out", NULL};
  PyObject *_a = NULL;
  PyObject *axis = NULL;
  PyObject *keepdims = NULL;
  PyObject *out = NULL;
  bool keepdims_c = true;
  int axes[NPY_MAXDIMS] = {NULL};
  int axis_len = 1;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|OO", kwds_ls, &_a, &axis,
                                   &keepdims, &out)) {
    return NULL;
  }
  if (keepdims == NULL) {
    keepdims_c = false;
  } else {
    keepdims_c = PyObject_IsTrue(keepdims);
  }
  if (_a == NULL) {
    PyErr_SetString(PyExc_TypeError, "Expected at least an array");
    return NULL;
  }
  if (!Py_IS_TYPE(_a, Tensor_type)) {
    PyErr_SetString(PyExc_TypeError, "Expected a as Tensor obj");
    return NULL;
  }

  PyArrayObject *a = (PyArrayObject *)((Tensor *)_a)->data;
  if (!preprocess_axes(a, axis, axes, &axis_len)) {
    return NULL;
  }

  PyObject *result =
      numboost_argmin((PyObject *)a, &out, axes, axis_len, keepdims_c);
  Numboost_AssertNULL(result);
  Tensor *to_return =
      (Tensor *)tensor_new((Tensor *)_a, Py_None, result, "");
  return to_return;
}

Register_mudule_elementwise_methods(sin, "SinBackward");
Register_mudule_elementwise_methods(cos, "CosBackward");
Register_mudule_elementwise_methods(tan, "TanBackward");
Register_mudule_elementwise_methods(asin, "ArcSinBackward");
Register_mudule_elementwise_methods(acos, "ArcCosBackward");
Register_mudule_elementwise_methods(atan, "ArcTanBackward");
Register_mudule_elementwise_methods(sinh, "SinhBackward");
Register_mudule_elementwise_methods(cosh, "CoshBackward");
Register_mudule_elementwise_methods(tanh, "TanhBackward");
Register_mudule_elementwise_methods(asinh, "ArcSinhBackward");
Register_mudule_elementwise_methods(acosh, "ArcCoshBackward");
Register_mudule_elementwise_methods(atanh, "ArcTanhBackward");
Register_mudule_elementwise_methods(abs, "AbsBackward");
Register_mudule_elementwise_methods(sqrt, "SqrtBackward");
Register_mudule_elementwise_methods(log, "LogBackward");
Register_mudule_elementwise_methods(log10, "Log10Backward");
Register_mudule_elementwise_methods(exp, "ExpBackward");