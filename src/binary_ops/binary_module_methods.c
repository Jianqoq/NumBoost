#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "binary_module_methods.h"
#include "../python_magic/python_math_magic.h"
#include "../tensor.h"
#include "../tensor_creation/creation_def.h"
#include "../ufunc_ops/ufunc_def.h"
#include "binary_op_def.h"

static char *keyword_list[] = {"a", "b", "out", NULL};

Register_mudule_methods(add, "AddBackward");
Register_mudule_methods(sub, "SubBackward");
Register_mudule_methods(mul, "MulBackward");
Register_mudule_methods(div, "DivBackward");
Register_mudule_methods(mod, "");
Register_mudule_methods(fdiv, "");

PyObject *nb_module_pow(PyObject *numboost_module, PyObject *args,
                        PyObject *kwds) {
  (void)numboost_module;
  PyObject *a = NULL, *power = NULL, *out = NULL;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|O", keyword_list, &a, &power,
                                   &out)) {
    return NULL;
  }
  if (!Py_IS_TYPE(a, Tensor_type)) {
    PyErr_SetString(PyExc_TypeError, "a must be Tensor");
    return NULL;
  }
  PyObject *outs;
  Tensor *to_replace = NULL;
  Tensor *a_ = (Tensor *)a;
  if (out == Py_None || out == NULL) {
    outs = NULL;
  } else if (Py_IS_TYPE(out, Tensor_type)) {
    to_replace = (Tensor *)out;
    outs = ((Tensor *)out)->data;
  } else {
    PyErr_SetString(PyExc_TypeError, "out must be None or Tensor");
    return NULL;
  }
  PyObject *result = numboost_pow(a, power, &outs);
  Numboost_AssertNULL(result);
  if (outs) {
    if (result != to_replace->data) {
      Py_DECREF(to_replace->data);
      to_replace->data = result;
      Py_INCREF(to_replace);
      return (PyObject *)to_replace;
    } else {
      Py_INCREF(to_replace);
      return (PyObject *)to_replace;
    }
  } else {
    PyObject *to_return = tensor_new(a_, Py_None, result, "PowBackward");
    if (a_->require_grad)
      store_power((Tensor *)to_return, power);
    return to_return;
  }
}

PyObject *nb_module_where(PyObject *numboost_module, PyObject *args,
                          PyObject *kwds) {
  (void)numboost_module;
  PyObject *condition = NULL, *x = NULL, *y = NULL, *exact_indice = NULL;
  char *nb_where_kws_list[] = {"condition", "x", "y", "exact_indice", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOp", nb_where_kws_list,
                                   &condition, &x, &y, &exact_indice)) {
    return NULL;
  }
  PyArrayObject *condition_ = NULL;
  if (Py_IS_TYPE(condition, Tensor_type)) {
    condition_ = (PyArrayObject *)((Tensor *)condition)->data;
  } else {
    PyErr_SetString(PyExc_TypeError, "condition must be Tensor");
    return NULL;
  }
  if (x == NULL && y == NULL) {
    npy_bool *condition_data = (npy_bool *)PyArray_DATA(condition_);
    npy_intp size = PyArray_SIZE(condition_);
    npy_intp cnt = 0;
    npy_intp i;
    int thread_num = 1;
    npy_intp *threads_cnt = (npy_intp *)malloc(sizeof(npy_intp) * thread_num);
#pragma omp parallel num_threads(thread_num) reduction(+ : cnt)
    {
      int tid = omp_get_thread_num();
      threads_cnt[tid] = 0;
#pragma omp for
      for (i = 0; i < size; ++i) {
        if (condition_data[i]) {
          threads_cnt[tid]++;
          cnt++;
        }
      }
    }
    npy_intp start_idx = 0;
    for (i = 0; i < thread_num; ++i) {
      npy_intp tmp_cnt = threads_cnt[i];
      threads_cnt[i] = start_idx;
      start_idx += tmp_cnt;
    }
    if (exact_indice == NULL || exact_indice == Py_False) {
      PyArrayObject *true_index =
          (PyArrayObject *)PyArray_EMPTY(1, &cnt, NPY_INTP, 0);
      npy_longlong *true_index_data = (npy_longlong *)PyArray_DATA(true_index);
#pragma omp parallel num_threads(thread_num)
      {
        int tid = omp_get_thread_num();
        npy_intp start = threads_cnt[tid];
        npy_intp k;
#pragma omp for
        for (k = 0; k < size; ++k) {
          if (condition_data[k]) {
            true_index_data[start++] = k;
          }
        }
      }
      return tensor_empty((PyObject *)true_index);
    } else {
      PyArrayObject_fields *fields = (PyArrayObject_fields *)condition_;
      PyArrayObject **true_index_arr =
          (PyArrayObject **)malloc(sizeof(PyArrayObject *) * fields->nd);
      npy_longlong **true_index_data =
          (npy_longlong **)malloc(sizeof(npy_longlong *) * fields->nd);
      for (i = 0; i < fields->nd; ++i) {
        PyArrayObject *tmp =
            (PyArrayObject *)PyArray_EMPTY(1, &cnt, NPY_INTP, 0);
        Numboost_AssertNULL(tmp);
        true_index_arr[i] = tmp;
        true_index_data[i] = (npy_longlong *)PyArray_DATA(tmp);
      }
      npy_intp *shape = fields->dimensions;
#pragma omp parallel num_threads(thread_num)
      {
        int tid = omp_get_thread_num();
        npy_intp start = threads_cnt[tid];
        npy_intp k;
        npy_intp i;
#pragma omp for
        for (k = 0; k < size; ++k) {
          if (condition_data[k]) {
            npy_intp prg = k;
            for (i = fields->nd - 1; i >= 0; i--) {
              true_index_data[i][start] = prg % shape[i];
              prg /= shape[i];
            }
            start++;
          }
        }
      }
      PyObject *result = PyDict_New();
      for (i = fields->nd - 1; i >= 0; i--) {
        PyObject *ret = tensor_empty((PyObject *)true_index_arr[i]);
        char key[100];
        sprintf(key, "axis: %lld", (long long)i);
        Numboost_AssertNULL(ret);
        PyObject *key_str = PyUnicode_FromString(key);
        Numboost_AssertNULL(key_str);
        PyDict_SetItem(result, key_str, ret);
      }
      return result;
    }
  } else {
    if (x == NULL || y == NULL) {
      PyErr_SetString(PyExc_TypeError, "x and y both must be present");
      return NULL;
    } else {
      int a_type = any_to_type_enum(x);
      int b_type = any_to_type_enum(y);
      int a_size = type_2_size[a_type];
      int b_size = type_2_size[b_type];
      int result_type =
          binary_result_type(WHERE, a_type, a_size, b_type, b_size);
      PyObject **result = numboost_where(condition, x, y, NULL, 1, result_type);
      Numboost_AssertNULL(result);
      return tensor_empty(result[0]);
    }
  }
  return NULL;
}

PyObject *nb_module_any(PyObject *numboost_module, PyObject *args,
                        PyObject *kwds) {
  (void)numboost_module;
  PyObject *a = NULL, *axis = NULL, *out = NULL, *keepdims = NULL,
           *where = NULL;
  char *nb_any_kws_list[] = {"a", "axis", "out", "keepdims", "where", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOp", nb_any_kws_list, &a,
                                   &axis, &out, &keepdims, &where)) {
    return NULL;
  }
  PyArrayObject *a_ = NULL;
  if (Py_IS_TYPE(a, Tensor_type)) {
    a_ = (PyArrayObject *)((Tensor *)a)->data;
  } else {
    PyErr_SetString(PyExc_TypeError, "a must be Tensor");
    return NULL;
  }
  Py_ssize_t *axes = NULL;
  if (axis == NULL || axis == Py_None) {
    axes = NULL;
  } else if (PyTuple_Check(axis) || PyList_Check(axis)) {
    Py_ssize_t len = PySequence_Size(axis);
    axes = (Py_ssize_t *)malloc(sizeof(Py_ssize_t) * len);
    Py_ssize_t i;
    for (i = 0; i < len; ++i) {
      PyObject *tmp = PySequence_GetItem(axis, i);
      Numboost_AssertNULL(tmp);
      if (!PyLong_Check(tmp)) {
        PyErr_SetString(PyExc_TypeError, "axis must be int");
        return NULL;
      }
      axes[i] = PyLong_AsLong(tmp);
      Py_DECREF(tmp);
    }
  } else {
    PyErr_SetString(PyExc_TypeError, "axis must be list or tuple");
    return NULL;
  }
  PyObject *out_ = NULL;
  if (out == NULL || out == Py_None) {
    out_ = NULL;
  } else if (Py_IS_TYPE(out, Tensor_type)) {
    out_ = ((Tensor *)out)->data;
  } else {
    PyErr_SetString(PyExc_TypeError, "out must be None or Tensor");
    return NULL;
  }
  bool keepdims_ = false;
  if (keepdims == NULL || keepdims == Py_None) {
    keepdims_ = false;
  } else if (PyBool_Check(keepdims)) {
    keepdims_ = (keepdims == Py_True);
  } else {
    PyErr_SetString(PyExc_TypeError, "keepdims must be bool");
    return NULL;
  }
  PyObject *where_ = NULL;
  if (where == NULL || where == Py_None) {
    where_ = NULL;
  } else if (Py_IS_TYPE(where, Tensor_type)) {
    where_ = ((Tensor *)where)->data;
    int where_nd = PyArray_NDIM(where_);
    npy_intp *where_shape = PyArray_SHAPE((PyArrayObject *)where_);
    if (where_nd != PyArray_NDIM(a_)) {
      PyErr_SetString(PyExc_TypeError, "where must have same shape with a");
      return NULL;
    }
    npy_intp *a_shape = PyArray_SHAPE(a_);
    int i;
    for (i = 0; i < where_nd; ++i) {
      if (where_shape[i] != a_shape[i]) {
        PyErr_SetString(PyExc_TypeError, "where must have same shape with a");
        return NULL;
      }
    }
  } else {
    PyErr_SetString(PyExc_TypeError, "where must be None or Tensor");
    return NULL;
  }
  return NULL;
  // PyObject *result = numboost_any(a_, where_, out_, axes, keepdims_);
}