#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "../numboost_api.h"
#include "../numboost_math.h"
#include "../numboost_sort_utils.h"
#include "../numboost_utils.h"
#include "../python_magic/python_math_magic.h"
#include "binary_op_def.h"

PyObject *numboost_add_test(PyObject *a, PyObject *b, PyObject **outs_arr) {
  PyArrayObject **return_arr =
      (PyArrayObject **)malloc(sizeof(PyArrayObject *) * 1);
  bool shape_equal = 1;
  bool all_scalar = 1;
  PyArrayObject *handler_a = NULL;
  PyArrayObject *a_ = NULL;
  if (Py_IS_TYPE(a, Tensor_type)) {
    a = ((Tensor *)a)->data;
  }
  if (Py_IS_TYPE(a, &PyArray_Type)) {
    PyArrayObject *tmp = (PyArrayObject *)a;
    if (PyArray_TYPE(tmp) != NPY_FLOAT) {
      as_type(&tmp, &a_, NPY_FLOAT);
      handler_a = a_;
    } else {
      a_ = (PyArrayObject *)a;
    }
  } else if (PyArray_IsAnyScalar(a)) {
    if (PyArray_IsPythonNumber(a)) {
      npy_intp const dims[] = {1};
      a_ = (PyArrayObject *)PyArray_EMPTY(0, dims, NPY_FLOAT, 0);
      if (PyFloat_Check(a)) {
        *((npy_float *)PyArray_DATA(a_)) = (npy_float)PyDouble_AsHalf(
            npy_float, PyFloat_AsDouble((PyObject *)a));
      } else if (PyLong_Check(a)) {
        *((npy_float *)PyArray_DATA(a_)) =
            (npy_float)PyLong_AsHalf(npy_float, PyLong_AsLong((PyObject *)a));
      } else if (PyBool_Check(a)) {
        *((npy_float *)PyArray_DATA(a_)) =
            (npy_float)PyBool_AsHalf(type, Py_IsTrue((PyObject *)a));
      } else {
        PyErr_SetString(PyExc_TypeError, "Scalar type not supported");
        return NULL;
      }
      handler_a = a_;
    } else if (Py_IS_TYPE(a, &PyArray_Type)) {
      PyArrayObject *tmp = (PyArrayObject *)a;
      as_type(&tmp, &a_, NPY_FLOAT);
      handler_a = a_;
    } else {
      PyErr_SetString(PyExc_TypeError, "npy_float is not supported");
      return ((void *)0);
    }
    assert(PyArray_STRIDES(a_) == NULL);
  } else {
    PyErr_SetString(PyExc_TypeError, "type not supported");
    return ((void *)0);
  }
  PyArrayObject *handler_b = NULL;
  PyArrayObject *b_ = NULL;
  if (Py_IS_TYPE(b, Tensor_type)) {
    b = ((Tensor *)b)->data;
  }
  if (Py_IS_TYPE(b, &PyArray_Type)) {
    PyArrayObject *tmp = (PyArrayObject *)b;
    if (PyArray_TYPE(tmp) != NPY_FLOAT) {
      as_type(&tmp, &b_, NPY_FLOAT);
      handler_b = b_;
    } else {
      b_ = (PyArrayObject *)b;
    }
  } else if (PyArray_IsAnyScalar(b)) {
    if (PyArray_IsPythonNumber(b)) {
      npy_intp const dims[] = {1};
      b_ = (PyArrayObject *)PyArray_EMPTY(0, dims, NPY_FLOAT, 0);
      if (PyFloat_Check(b)) {
        *((npy_float *)PyArray_DATA(b_)) = (npy_float)PyDouble_AsHalf(
            npy_float, PyFloat_AsDouble((PyObject *)b));
      } else if (PyLong_Check(b)) {
        *((npy_float *)PyArray_DATA(b_)) =
            (npy_float)PyLong_AsHalf(npy_float, PyLong_AsLong((PyObject *)b));
      } else if (PyBool_Check(b)) {
        *((npy_float *)PyArray_DATA(b_)) =
            (npy_float)PyBool_AsHalf(type, Py_IsTrue((PyObject *)b));
      } else {
        PyErr_SetString(PyExc_TypeError, "Scalar type not supported");
        return NULL;
      }
      handler_b = b_;
    } else if (Py_IS_TYPE(b, &PyArray_Type)) {
      PyArrayObject *tmp = (PyArrayObject *)b;
      as_type(&tmp, &b_, NPY_FLOAT);
      handler_b = b_;
    } else {
      PyErr_SetString(PyExc_TypeError, "npy_float is not supported");
      return NULL;
    }
    assert(PyArray_STRIDES(b_) == NULL);
  } else {
    PyErr_SetString(PyExc_TypeError, "type not supported");
    return NULL;
  }
  npy_intp *shapes[] = {PyArray_SHAPE(a_), PyArray_SHAPE(b_)};
  int ndims[] = {(((PyArrayObject_fields *)(a_))->nd),
                 (((PyArrayObject_fields *)(b_))->nd)};
  npy_intp *shape_ref = shapes[0];
  int ndim_ref = ndims[0];
  for (int i = 0; i < 1 + 1; i++) {
    if (!shape_isequal(shape_ref, shapes[i], ndim_ref, ndims[i])) {
      shape_equal = 0;
      break;
    }
  }
  PyArrayObject *arr[] = {a_, b_};
  npy_intp sizes[1 + 1] = {0};
  int biggest_index = 0;
  PyArrayObject *biggest_array = NULL;
  npy_intp biggest_size = 0;
  for (int i = 0; i < (1 + 1); i++) {
    npy_intp size = 1;
    for (int j = 0; j < ndims[i]; j++) {
      size *= shapes[i][j];
    }
    sizes[i] = size;
    if (size > biggest_size) {
      biggest_size = size;
      biggest_array = arr[i];
      biggest_index = i;
    }
  }
  for (int j = 0; j < (1 + 1); j++) {
    if (j != biggest_index && sizes[j] != 1) {
      all_scalar = 0;
      break;
    }
  }
  PyObject *out_arr[1] = {NULL};
  if (outs_arr != NULL) {
    memcpy(out_arr, outs_arr, sizeof(PyArrayObject *) * 1);
  }
  if ((!shape_equal && !all_scalar) ||
      (!PyArray_ISCONTIGUOUS(b) && !PyArray_ISCONTIGUOUS(a))) {
    npy_intp *new_shapes[1 + 1] = {NULL};
    for (int i = 0; i < (1 + 1); i++) {
      if (!shape_isbroadcastable_to_ex(
              shapes[i], PyArray_SHAPE(biggest_array), ndims[i],
              (((PyArrayObject_fields *)(biggest_array))->nd),
              &new_shapes[i])) {
        PyErr_SetString(PyExc_ValueError, "Cannot broadcast shapes");
        return NULL;
      }
    }
    npy_intp stride_last =
        PyArray_STRIDES(biggest_array)[PyArray_NDIM(biggest_array) - 1];
    npy_intp *broadcast_shape = NULL;
    npy_intp broadcast_size = 1;
    for (int j = 0; j < 2; j++) {
      npy_intp *broadcast_shape_tmp = NULL;
      predict_broadcast_shape(PyArray_SHAPE(biggest_array), new_shapes[j],
                              (((PyArrayObject_fields *)(biggest_array))->nd),
                              &broadcast_shape_tmp);
      npy_intp tmp = 1;
      for (int k = 0; k < (((PyArrayObject_fields *)(biggest_array))->nd);
           k++) {
        tmp *= broadcast_shape_tmp[k];
      }
      if (tmp < broadcast_size) {
        free(broadcast_shape_tmp);
      } else {
        broadcast_shape = broadcast_shape_tmp;
        broadcast_size = tmp;
      }
    }
    PyArrayObject *__result_result = NULL;
    if (outs_arr[0] == NULL) {
      __result_result = (PyArrayObject *)PyArray_EMPTY(
          PyArray_NDIM(biggest_array), broadcast_shape, NPY_FLOAT, 0);
      if (__result_result == NULL) {
        return NULL;
      }
    } else {
      if ((((PyArrayObject_fields *)((PyArrayObject *)outs_arr[0]))->nd) !=
          PyArray_NDIM(biggest_array)) {
        PyErr_SetString(PyExc_RuntimeError, "out ndim not correct");
        return ((void *)0);
      } else {
        for (int i = 0;
             i < (((PyArrayObject_fields *)((PyArrayObject *)outs_arr[0]))->nd);
             i++) {
          if (PyArray_SHAPE((PyArrayObject *)outs_arr[0])[i] !=
              broadcast_shape[i]) {
            PyErr_SetString(PyExc_RuntimeError, "out ndim not correct");
            return ((void *)0);
          }
        }
        if ((((PyArrayObject_fields *)((PyArrayObject *)outs_arr[0]))
                 ->descr->type_num) != NPY_FLOAT) {
          PyArrayObject *tmp = (PyArrayObject *)outs_arr[0];
          as_type(&tmp, &__result_result, NPY_FLOAT);
        } else {
          __result_result = (PyArrayObject *)outs_arr[0];
          Py_INCREF(outs_arr[0]);
        }
      }
    }
    do {
      int ndim = (((PyArrayObject_fields *)(__result_result))->nd);
      npy_intp max_dim = ndim - 1;
      npy_intp *__strides_a = (((PyArrayObject_fields *)(a_))->strides);
      npy_intp *strides_a = ((void *)0);
      npy_intp *indice_a_cache = (npy_intp *)malloc(sizeof(npy_intp) * ndim);
      npy_float *a_data_ptr_saved =
          (npy_float *)((void *)((PyArrayObject_fields *)(a_))->data);
      if (__strides_a == ((void *)0)) {
        strides_a = (npy_intp *)calloc(
            (((PyArrayObject_fields *)(__result_result))->nd),
            sizeof(npy_intp));
      } else {
        preprocess_strides(new_shapes[0], stride_last, ndim, &strides_a);
      }
      npy_intp *__strides_b = (((PyArrayObject_fields *)(b_))->strides);
      npy_intp *strides_b = ((void *)0);
      npy_intp *indice_b_cache = (npy_intp *)malloc(sizeof(npy_intp) * ndim);
      npy_float *b_data_ptr_saved =
          (npy_float *)((void *)((PyArrayObject_fields *)(b_))->data);
      if (__strides_b == ((void *)0)) {
        strides_b =
            (npy_intp *)calloc(PyArray_NDIM(__result_result), sizeof(npy_intp));
      } else {
        preprocess_strides(new_shapes[0 + 1], stride_last, ndim, &strides_b);
      }
      for (int i = 0; i < ndim; i++) {
        strides_a[i] /= sizeof(npy_float);
        strides_b[i] /= sizeof(npy_float);
      }
      npy_intp stride_a_last = strides_a[max_dim];
      npy_intp stride_b_last = strides_b[max_dim];
      npy_intp _size = PyArray_SIZE(__result_result);
      npy_intp *shape_cpy = (npy_intp *)malloc(sizeof(npy_intp) *
                                               (PyArray_NDIM(__result_result)));
      npy_intp *__shape = PyArray_SHAPE(__result_result);
      memcpy(shape_cpy, PyArray_SHAPE(__result_result),
             sizeof(npy_intp) * (PyArray_NDIM(__result_result)));
      int axis_sep = ndim - 1;
      npy_intp inner_loop_size = PyArray_SHAPE(__result_result)[axis_sep];
      npy_intp outter_loop_size = _size / inner_loop_size;
      npy_intp outer_start = max_dim - 1;
      npy_intp *shape_copy = (npy_intp *)malloc(sizeof(npy_intp) * ndim);
      npy_float *result___result_result_ptr = PyArray_DATA(__result_result);
      npy_float *result___result_result_ptr_saved =
          PyArray_DATA(__result_result);
      npy_float *result___result_result_ptr_cpy = PyArray_DATA(__result_result);
      for (int i = 0; i < ndim; i++) {
        shape_cpy[i]--;
        shape_copy[i] = 0;
        indice_a_cache[i] = strides_a[i] * shape_cpy[i];
        indice_b_cache[i] = strides_b[i] * shape_cpy[i];
      }
      npy_intp num_threads = outter_loop_size < omp_get_max_threads()
                                 ? outter_loop_size
                                 : omp_get_max_threads();
      npy_float **result___result_result_ptr_arr =
          (npy_float **)malloc(sizeof(npy_float *) * num_threads);
      npy_intp **current_shape_process_ =
          (npy_intp **)malloc(sizeof(npy_intp *) * num_threads);
      for (npy_intp id = 0; id < num_threads; id++) {
        npy_intp start_index = id * (outter_loop_size / num_threads) +
                               (((id) < (outter_loop_size % num_threads))
                                    ? (id)
                                    : (outter_loop_size % num_threads));
        npy_intp end_index = start_index + outter_loop_size / num_threads +
                             (id < outter_loop_size % num_threads);
        result___result_result_ptr_arr[id] = result___result_result_ptr_cpy;
        result___result_result_ptr_cpy +=
            (end_index - start_index) * inner_loop_size;
        npy_intp prd___result_result = result___result_result_ptr_arr[id] -
                                       result___result_result_ptr_saved;
        npy_intp *current_shape_process =
            (npy_intp *)calloc(ndim, sizeof(npy_intp));
        for (npy_intp j = max_dim; j >= 0; j--) {
          current_shape_process[j] = prd___result_result % __shape[j];
          prd___result_result /= __shape[j];
        }
        current_shape_process_[id] = current_shape_process;
      }
      npy_intp k = 0;
      {
        int thread_id = omp_get_thread_num();
        result___result_result_ptr = result___result_result_ptr_arr[thread_id];
        npy_intp *current_process = current_shape_process_[thread_id];
        for (npy_intp j = max_dim; j >= 0; j--) {
          a_data_ptr_saved += current_process[j] * strides_a[j];
          b_data_ptr_saved += current_process[j] * strides_b[j];
        }
        for (k = 0; k < outter_loop_size; k++) {
          npy_intp i;
          for (i = 0; i < inner_loop_size; i++) {
            npy_float a_val = (a_data_ptr_saved[i * stride_a_last]);
            npy_float b_val = (b_data_ptr_saved[i * stride_b_last]);
            npy_float result = a_val + b_val;
            result___result_result_ptr[i] = (result);
          }
          result___result_result_ptr += inner_loop_size;
          for (npy_intp j = outer_start; j >= 0; j--) {
            if (current_process[j] < shape_cpy[j]) {
              current_process[j]++;
              a_data_ptr_saved += strides_a[j];
              b_data_ptr_saved += strides_b[j];
              ;
              break;
            } else {
              current_process[j] = 0;
              a_data_ptr_saved -= indice_a_cache[j];
              b_data_ptr_saved -= indice_b_cache[j];
              ;
            }
          }
        }
        free(current_process);
      }
      free(current_shape_process_);
      free(result___result_result_ptr_arr);
      ;
      free(indice_a_cache);
      free(strides_a);
      free(indice_b_cache);
      free(strides_b);
      ;
      free(shape_cpy);
      free(shape_copy);
    } while (0);
    ;
    PyArrayObject *results_arr[] = {__result_result};
    if (handler_a)
      Py_DECREF(handler_a);
    if (handler_b)
      Py_DECREF(handler_b);
    memcpy(return_arr, results_arr, sizeof(PyArrayObject *) * (1));
  } else {
    npy_intp *__strides_a = PyArray_STRIDES(a_);
    npy_intp stride_a_last = 0;
    if (__strides_a == NULL) {
      stride_a_last = 0;
    } else {
      stride_a_last =
          PyArray_STRIDES(a_)[PyArray_NDIM(a_) - 1] / sizeof(npy_float);
    }
    npy_intp *__strides_b = PyArray_STRIDES(b_);
    npy_intp stride_b_last = 0;
    if (__strides_b == NULL) {
      stride_b_last = 0;
    } else {
      stride_b_last =
          PyArray_STRIDES(b_)[PyArray_NDIM(b_) - 1] / sizeof(npy_float);
    };
    PyArrayObject *__result_result = NULL;
    if (out_arr[0] == NULL) {
      __result_result = PyArray_EMPTY(PyArray_NDIM(biggest_array),
                                      PyArray_SHAPE(biggest_array), NPY_FLOAT,
                                      PyArray_ISFORTRAN(biggest_array));
      if (__result_result == NULL) {
        return NULL;
      }
    } else {
      if (PyArray_NDIM(out_arr[0]) !=
          (((PyArrayObject_fields *)(biggest_array))->nd)) {
        PyErr_SetString(PyExc_RuntimeError, "out ndim not correct");
        return NULL;
      } else {
        for (int i = 0; i < PyArray_NDIM(out_arr[0]); i++) {
          if (PyArray_SHAPE((PyArrayObject *)out_arr[0])[i] !=
              PyArray_SHAPE(biggest_array)[i]) {
            PyErr_SetString(PyExc_RuntimeError, "out ndim not correct");
            return NULL;
          }
        }
        if (PyArray_TYPE(out_arr[0]) != NPY_FLOAT) {
          PyArrayObject *tmp = (PyArrayObject *)out_arr[0];
          as_type(&tmp, &__result_result, NPY_FLOAT);
        } else {
          __result_result = (PyArrayObject *)out_arr[0];
          Py_INCREF(out_arr[0]);
        }
      }
    }
    do {
      npy_float *a_data_ptr_saved = (npy_float *)PyArray_DATA(a_);
      npy_float *b_data_ptr_saved = (npy_float *)PyArray_DATA(b_);
      npy_intp _size = (*(npy_intp(*)(npy_intp const *, int))tensor_c[158])(
          (((PyArrayObject_fields *)(__result_result))->dimensions),
          (((PyArrayObject_fields *)(__result_result))->nd));
      npy_float *result___result_result_ptr_saved =
          (npy_float *)((void *)((PyArrayObject_fields *)(__result_result))
                            ->data);
      ;
      npy_intp i;
      for (i = 0; i < _size; i++) {
        npy_float a_val = a_data_ptr_saved[i * stride_a_last];
        npy_float b_val = b_data_ptr_saved[i * stride_b_last];
        npy_float result = a_val + b_val;
        result___result_result_ptr_saved[i] = (result);
      };
    } while (0);
    if (handler_a)
      Py_DECREF(handler_a);
    if (handler_b)
      Py_DECREF(handler_b);
    ;
    PyArrayObject *results_arr[] = {__result_result};
    memcpy(return_arr, results_arr, sizeof(PyArrayObject *) * (1));
  }
  if (return_arr == NULL) {
    return NULL;
  } else {
    PyObject *ret = (PyObject *)return_arr[0];
    free(return_arr);
    return ret;
  }
}

Tensor *reduction_test(PyObject *self, PyObject *args, PyObject *kwds) {
  (void)self;
  char *kwds_ls[] = {"a", "axis", "keepdims", NULL};
  PyObject *_a = NULL;
  PyObject *axis = NULL;
  PyObject *keepdims = NULL;
  bool keepdims_c = true;
  int axes[NPY_MAXDIMS] = {NULL};
  int axis_len = 1;
  bool is_left = true;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|O", kwds_ls, &_a, &axis,
                                   &keepdims)) {
    return NULL;
  }
  if (keepdims == NULL) {
    keepdims_c = false;
  } else {
    keepdims_c = PyObject_IsTrue(keepdims);
  }
  if (_a == NULL || axis == NULL) {
    PyErr_SetString(PyExc_TypeError,
                    "Expected at least 2 positional arguments");
    return NULL;
  }
  if (!Py_IS_TYPE(_a, Tensor_type)) {
    PyErr_SetString(PyExc_TypeError, "Expected a as Tensor obj");
    return NULL;
  }
  PyArrayObject *a = (PyArrayObject *)((Tensor *)_a)->data;
  if (PyArray_IsAnyScalar(axis)) {
#define Assert_Axis_Valid(idx, axes, array)                                    \
  if (axes[idx] >= PyArray_NDIM(array)) {                                      \
    PyErr_SetString(PyExc_TypeError, "Invalid axis");                          \
    return NULL;                                                               \
  }
    axes[0] = (int)PyLong_AsLong(axis);
    Assert_Axis_Valid(0, axes, a);
  } else if (PyTuple_Check(axis)) {
    axis_len = (int)PyTuple_GET_SIZE(axis);
    for (int i = 0; i < axis_len; i++) {
      axes[i] = (int)PyLong_AsLong(PyTuple_GET_ITEM(axis, i));
      Assert_Axis_Valid(i, axes, a);
    }
    qsort(axes, axis_len, sizeof(int), compare);
  } else {
    PyErr_SetString(PyExc_TypeError, "Invalid type for axis");
    return NULL;
  }
  /*check if the axes are in the innermost*/
  for (int i = 0; i < axis_len; i++) {
    if (is_in((int *)axes, axis_len, PyArray_NDIM(a) - 1)) {
      is_left = false;
      break;
    }
  }
  /*transpose the array along the axes which need to do reduction operation*/
  int a_ndim = PyArray_NDIM(a);
  npy_intp *a_shape = PyArray_SHAPE(a);
  npy_intp *a_shape_cpy = (npy_intp *)malloc(sizeof(npy_intp) * a_ndim);
  npy_intp *transposed_axis = (npy_intp *)malloc(sizeof(npy_intp) * a_ndim);
  memcpy(a_shape_cpy, a_shape, sizeof(npy_intp) * a_ndim);
  move_axes_to_innermost(axes, axis_len, a_shape_cpy, a_ndim, transposed_axis);

  PyArray_Dims d = {transposed_axis, a_ndim};
  PyArrayObject *transposed_arr = (PyArrayObject *)PyArray_Transpose(a, &d);
  free(transposed_axis);
  npy_intp *transposed_strides = PyArray_STRIDES(transposed_arr);
  npy_intp *transposed_strides_cpy = malloc(sizeof(npy_intp) * a_ndim);
  memcpy(transposed_strides_cpy, transposed_strides, sizeof(npy_intp) * a_ndim);

  /*normalize the transposed strides*/
  for (int i = 0; i < a_ndim; i++) {
    transposed_strides_cpy[i] /= sizeof(npy_double);
  }
  assert(transposed_strides_cpy[a_ndim - 1] == 1);
  npy_intp *transposed_shape = PyArray_SHAPE(transposed_arr);
  npy_intp *transposed_shape_cpy = malloc(sizeof(npy_intp) * a_ndim);
  memcpy(transposed_shape_cpy, transposed_shape, sizeof(npy_intp) * a_ndim);
  for (int i = 0; i < a_ndim; i++) {
    transposed_shape_cpy[i]--;
  }

  npy_double *a_data = PyArray_DATA(a);
  npy_intp *result_shape = malloc(sizeof(npy_intp) * (a_ndim - axis_len));
  int k = 0;
  for (int i = 0; i < a_ndim; i++) {
    if (a_shape_cpy[i] != 0) {
      result_shape[k++] = a_shape_cpy[i];
    }
  }
  free(a_shape_cpy);
  PyArrayObject *result = (PyArrayObject *)PyArray_EMPTY(
      a_ndim - axis_len, result_shape, NPY_DOUBLE, 0);
  npy_double *result_data = PyArray_DATA(result);

  npy_intp init_idx;
#pragma omp parallel for
  for (init_idx = 0; init_idx < PyArray_SIZE(result); init_idx++) {
    result_data[init_idx] = NPY_INFINITY;
  }

  if (a_ndim == axis_len) {
    npy_intp size = PyArray_SIZE(a);
    int num_threads =
        size < omp_get_max_threads() ? size : omp_get_max_threads();
    npy_double *min_vals = malloc(sizeof(npy_double) * num_threads);
#pragma omp parallel /*need to improve when omp is upgraded*/
    {
      int thread_id = omp_get_thread_num();
      npy_double min_val = NPY_INFINITY;
      npy_intp i;
#pragma omp for schedule(static)
      for (i = 0; i < size; i++) {
        if (a_data[i] < min_val) {
          min_val = a_data[i];
        }
      }
      min_vals[thread_id] = min_val;
    }
    npy_double min_value = NPY_INFINITY;
    for (int i = 0; i < num_threads; i++) {
      if (min_vals[i] < min_value) {
        min_value = min_vals[i];
      }
    }
    result_data[0] = min_value;
    free(min_vals);
  } else {
    /*most inner axis is the one that could be sequential*/
    npy_intp a_last_index = a_ndim - 1;
    int result_nd = PyArray_NDIM(result);
    int result_nd_except_last = result_nd - 1;
    int a_ndim_except_last = a_last_index;
    npy_intp inner_loop_size = a_shape[a_last_index];
    npy_intp result_size = PyArray_SIZE(result);
    npy_intp a_size = PyArray_SIZE(a);
    npy_double *a_data_ptr_cpy = a_data;
    npy_double *result_data_cpy = result_data;
    npy_intp last_stride = PyArray_STRIDE(a, a_ndim - 1) / sizeof(npy_double);

    if (!is_left) {
      npy_intp outer_loop_size = a_size / inner_loop_size;
      npy_intp inner_loop_size_2 = outer_loop_size / result_size;
      npy_intp num_threads = result_size < omp_get_max_threads()
                                 ? result_size
                                 : omp_get_max_threads();
      npy_intp task_amount = 0;

      npy_double **result_ptr_arr =
          (npy_double **)calloc(num_threads, sizeof(npy_double *));
      npy_double **a_data_ptr_arr =
          (npy_double **)malloc(sizeof(npy_double *) * num_threads);
      npy_intp **progress_init_a_data_arr =
          malloc(sizeof(npy_intp *) * num_threads);
      npy_intp *progress_init_a_data = calloc(result_nd, sizeof(npy_intp));
      npy_intp **prg_arr = malloc(sizeof(npy_intp *) * num_threads);

      /*init ptrs for different threads*/
      for (npy_intp id = 0; id < num_threads; id++) {

        npy_intp start_index = id * (result_size / num_threads) +
                               min(id, result_size % num_threads);
        npy_intp end_index = start_index + result_size / num_threads +
                             (id < result_size % num_threads);

        a_data_ptr_cpy = a_data;
        for (npy_intp k = result_nd - 1; k >= 0; k--) {
          a_data_ptr_cpy += progress_init_a_data[k] * transposed_strides_cpy[k];
        }

        npy_intp *progress_init_a_data_cpy =
            malloc(sizeof(npy_intp) * result_nd);
        memcpy(progress_init_a_data_cpy, progress_init_a_data,
               sizeof(npy_intp) * result_nd);
        progress_init_a_data_arr[id] = progress_init_a_data_cpy;
        npy_intp tmp1 = task_amount * inner_loop_size_2;
        npy_intp *prg = calloc(a_ndim_except_last, sizeof(npy_intp));
        prg_arr[id] = prg;
        for (npy_intp j = a_ndim_except_last - 1; j >= 0; j--) {
          prg[j] = tmp1 % transposed_shape[j];
          tmp1 /= transposed_shape[j];
        }
        a_data_ptr_arr[id] = a_data_ptr_cpy;
        task_amount += (end_index - start_index);
        result_ptr_arr[id] = result_data_cpy;
        result_data_cpy += end_index - start_index;
        npy_intp tmp2 = task_amount;
        for (npy_intp j = result_nd - 1; j >= 0; j--) {
          progress_init_a_data[j] = tmp2 % result_shape[j];
          tmp2 /= result_shape[j];
        }
      }

#pragma omp parallel num_threads(num_threads)
      {
        int thread_id = omp_get_thread_num();
        npy_double *result_data_ptr = result_ptr_arr[thread_id];
        npy_double *a_data_ptr = a_data_ptr_arr[thread_id];
        npy_intp *_prg = prg_arr[thread_id];
        npy_intp p = 0;
#pragma omp for schedule(static)
        for (p = 0; p < result_size; p++) {
          for (npy_intp j = 0; j < inner_loop_size_2; j++) {
            for (npy_intp i = 0; i < inner_loop_size; i++) {
              npy_double a_val = a_data_ptr[i * last_stride];
              if (a_val < *result_data_ptr)
                *result_data_ptr = a_val;
            }
            for (int h = a_ndim_except_last - 1; h >= 0; h--) {
              if (_prg[h] < transposed_shape_cpy[h]) {
                _prg[h]++;
                a_data_ptr += transposed_strides_cpy[h];
                break;
              } else {
                _prg[h] = 0;
                a_data_ptr -=
                    transposed_shape_cpy[h] * transposed_strides_cpy[h];
              }
            }
          }
          result_data_ptr++;
        }
      }
      free(progress_init_a_data);
      for (npy_intp id = 0; id < num_threads; id++) {
        free(progress_init_a_data_arr[id]);
        free(prg_arr[id]);
      }
      free(prg_arr);
      free(progress_init_a_data_arr);
      free(a_data_ptr_arr);
      free(result_ptr_arr);
      free(transposed_shape_cpy);
      free(transposed_strides_cpy);
    } else {
      npy_intp outer_loop_size = result_size / inner_loop_size;
      npy_intp inner_loop_size_2 = a_size / result_size;

      npy_intp num_threads = outer_loop_size < omp_get_max_threads()
                                 ? outer_loop_size
                                 : omp_get_max_threads();
      npy_intp task_amount = 0;
      npy_double **result_ptr_arr =
          (npy_double **)calloc(num_threads, sizeof(npy_double *));
      npy_double **a_data_ptr_arr =
          (npy_double **)malloc(sizeof(npy_double *) * num_threads);
      npy_intp *progress_init_a_data = calloc(result_nd, sizeof(npy_intp));
      npy_intp **progress_init_a_data_arr =
          malloc(sizeof(npy_intp *) * num_threads);
      for (npy_intp id = 0; id < num_threads; id++) {
        npy_intp start_index = id * (outer_loop_size / num_threads) +
                               min(id, outer_loop_size % num_threads);
        npy_intp end_index = start_index + outer_loop_size / num_threads +
                             (id < outer_loop_size % num_threads);
        a_data_ptr_cpy = a_data;
        for (npy_intp k = result_nd_except_last - 1; k >= 0; k--) {
          a_data_ptr_cpy += progress_init_a_data[k] * transposed_strides_cpy[k];
        }

        npy_intp *progress_init_a_data_cpy =
            malloc(sizeof(npy_intp) * result_nd);
        memcpy(progress_init_a_data_cpy, progress_init_a_data,
               sizeof(npy_intp) * result_nd);

        progress_init_a_data_arr[id] = progress_init_a_data_cpy;
        a_data_ptr_arr[id] = a_data_ptr_cpy;
        task_amount += (end_index - start_index);
        result_ptr_arr[id] = result_data_cpy;
        result_data_cpy += (end_index - start_index) * inner_loop_size;
        npy_intp tmp = task_amount;
        for (npy_intp j = result_nd_except_last - 1; j >= 0; j--) {
          progress_init_a_data[j] = tmp % result_shape[j];
          tmp /= result_shape[j];
        }
      }
#pragma omp parallel num_threads(num_threads)
      {
        int thread_id = omp_get_thread_num();
        npy_double *result_data_ptr = result_ptr_arr[thread_id];
        npy_double *a_data_ptr = a_data_ptr_arr[thread_id];
        npy_intp *a_data_progess = progress_init_a_data_arr[thread_id];
        npy_intp *prg = calloc(a_ndim, sizeof(npy_intp));
        npy_intp p2;
#pragma omp for schedule(static)
        for (p2 = 0; p2 < outer_loop_size; p2++) {
          for (npy_intp j = 0; j < inner_loop_size_2; j++) {
            for (npy_intp idx = 0; idx < inner_loop_size; idx++) {
              npy_double a_val = a_data_ptr[idx * last_stride];
              npy_double result_val = result_data_ptr[idx];
              if (a_val < result_val) {
                result_data_ptr[idx] = a_val;
              }
            }
            for (int h = a_last_index; h >= result_nd; h--) {
              if (prg[h] < transposed_shape_cpy[h]) {
                prg[h]++;
                a_data_ptr += transposed_strides_cpy[h];
                break;
              } else {
                prg[h] = 0;
                a_data_ptr -=
                    transposed_shape_cpy[h] * transposed_strides_cpy[h];
              }
            }
          }
          for (npy_intp t = result_nd_except_last - 1; t >= 0; t--) {
            if (a_data_progess[t] < transposed_shape_cpy[t]) {
              a_data_progess[t]++;
              a_data_ptr += transposed_strides_cpy[t];
              break;
            } else {
              a_data_progess[t] = 0;
              a_data_ptr -= transposed_shape_cpy[t] * transposed_strides_cpy[t];
            }
          }
          result_data_ptr += inner_loop_size;
          memset(prg, 0, sizeof(npy_intp) * a_ndim);
        }
        free(prg);
      }
      free(progress_init_a_data);
      free(a_data_ptr_arr);
      free(result_ptr_arr);
      for (npy_intp id = 0; id < num_threads; id++) {
        free(progress_init_a_data_arr[id]);
      }
      free(progress_init_a_data_arr);
      free(transposed_shape_cpy);
      free(transposed_strides_cpy);
    }
  }
  Tensor *to_return =
      (Tensor *)tensor_new((Tensor *)_a, Py_None, (PyObject *)result, "");
  return to_return;
}

PyObject *argmax_test(PyObject *a, PyObject **out_arr, int *axes, int axis_len,
                      bool keepdims) {
  PyArrayObject *out = ((void *)0);
  if (*out_arr != ((void *)0)) {
    if (Py_IS_TYPE(*out_arr, Tensor_type)) {
      Tensor *to_replace = (Tensor *)*out_arr;
      out = (PyArrayObject *)to_replace->data;
    } else if (Py_IS_TYPE(*out_arr, &PyArray_Type)) {
      out = (PyArrayObject *)*out_arr;
    } else {
      PyErr_SetString(PyExc_TypeError, "out type not supported");
      return ((void *)0);
    }
  }
  PyArrayObject *a_ = (PyArrayObject *)a;
  PyArrayObject *result = ((void *)0);
  int a_ndim = (((PyArrayObject_fields *)(a_))->nd);
  npy_intp *a_shape = PyArray_SHAPE(a_);
  npy_intp *a_shape_cpy = (npy_intp *)malloc(sizeof(npy_intp) * a_ndim);
  npy_intp *transposed_axis = (npy_intp *)malloc(sizeof(npy_intp) * a_ndim);
  memcpy(a_shape_cpy, a_shape, sizeof(npy_intp) * a_ndim);
  move_axes_to_innermost(axes, axis_len, a_shape_cpy, a_ndim, transposed_axis);
  PyArray_Dims d = {transposed_axis, a_ndim};
  PyArrayObject *transposed_arr = (PyArrayObject *)(*(
      PyObject * (*)(PyArrayObject *, PyArray_Dims *)) tensor_c[123])(a_, &d);
  free(transposed_axis);
  npy_intp *transposed_strides =
      (((PyArrayObject_fields *)(transposed_arr))->strides);
  npy_intp *transposed_strides_cpy = malloc(sizeof(npy_intp) * a_ndim);
  memcpy(transposed_strides_cpy, transposed_strides, sizeof(npy_intp) * a_ndim);
  for (int i = 0; i < a_ndim; i++) {
    transposed_strides_cpy[i] /= sizeof(npy_float);
  }
  assert(transposed_strides_cpy[a_ndim - 1] == 1);
  npy_intp *transposed_shape = PyArray_SHAPE(transposed_arr);
  npy_intp *transposed_shape_cpy = malloc(sizeof(npy_intp) * a_ndim);
  memcpy(transposed_shape_cpy, transposed_shape, sizeof(npy_intp) * a_ndim);
  for (int i = 0; i < a_ndim; i++) {
    transposed_shape_cpy[i]--;
  }
  npy_float *a_data = ((void *)((PyArrayObject_fields *)(a_))->data);
  npy_intp *result_shape = malloc(sizeof(npy_intp) * (a_ndim - axis_len));
  int k = 0;
  for (int i = 0; i < a_ndim; i++) {
    if (a_shape_cpy[i] != 0) {
      result_shape[k++] = a_shape_cpy[i];
    }
  }
  free(a_shape_cpy);
  npy_intp *new_shape = ((void *)0);
  if (keepdims) {
    new_shape = malloc(sizeof(npy_intp) * a_ndim);
    for (int i = 0; i < a_ndim; i++) {
      if (a_shape_cpy[i] != 0) {
        new_shape[i] = a_shape_cpy[i];
      } else {
        new_shape[i] = 1;
      }
    }
  }
  if (out != ((void *)0)) {
    if (!keepdims) {
      for (int i = 0; i < a_ndim - axis_len; i++) {
        if (PyArray_SHAPE(out)[i] != result_shape[i]) {
          PyErr_SetString(PyExc_ValueError, "Output array has incorrect shape");
          return ((void *)0);
        }
      }
    } else {
      for (int i = 0; i < a_ndim; i++) {
        if (PyArray_SHAPE(out)[i] != new_shape[i]) {
          PyErr_SetString(PyExc_ValueError, "Output array has incorrect shape");
          return ((void *)0);
        }
      }
    }
    if ((((PyArrayObject_fields *)(out))->descr->type_num) != NPY_FLOAT) {
      npy_intp out_mem_size =
          (*(npy_intp(*)(npy_intp const *, int))tensor_c[158])(
              (((PyArrayObject_fields *)(out))->dimensions),
              (((PyArrayObject_fields *)(out))->nd)) *
          ((PyArrayObject_fields *)out)->descr->elsize;
      npy_intp result_size = 1;
      for (int i = 0; i < a_ndim - axis_len; i++) {
        result_size *= result_shape[i];
      }
      if (out_mem_size != result_size * sizeof(npy_float)) {
        PyErr_SetString(PyExc_ValueError,
                        "Output array has incorrect type or total mem size "
                        "!= result mem size");
        return ((void *)0);
      }
    }
    result = out;
  } else {
    result = (PyArrayObject *)PyArray_EMPTY(a_ndim - axis_len, result_shape,
                                            NPY_LONGLONG, 0);
    if (result == NULL) {
      return NULL;
    };
  }
  npy_longlong *result_data =
      ((void *)((PyArrayObject_fields *)(result))->data);
  npy_intp a_last_index = a_ndim - 1;
  int result_nd = (((PyArrayObject_fields *)(result))->nd);
  int result_nd_except_last = result_nd - 1;
  int a_ndim_except_last = a_last_index;
  npy_intp inner_loop_size = transposed_shape[a_last_index];
  npy_intp result_size = PyArray_SIZE(result);
  npy_intp a_size = PyArray_SIZE(a_);
  npy_float *a_data_ptr_cpy = a_data;
  npy_longlong *result_data_cpy = result_data;
  npy_intp last_stride = transposed_strides_cpy[a_last_index];
  npy_intp outer_loop_size = a_size / inner_loop_size;
  npy_intp inner_loop_size_2 = outer_loop_size / result_size;
  npy_intp num_threads =
      result_size < omp_get_max_threads() ? result_size : omp_get_max_threads();
  npy_intp task_amount = 0;
  npy_longlong **result_ptr_arr =
      (npy_longlong **)calloc(num_threads, sizeof(npy_longlong *));
  npy_float **a_data_ptr_arr =
      (npy_float **)malloc(sizeof(npy_float *) * num_threads);
  npy_intp **progress_init_a_data_arr =
      malloc(sizeof(npy_intp *) * num_threads);
  npy_intp *progress_init_a_data = calloc(result_nd, sizeof(npy_intp));
  npy_intp **prg_arr = malloc(sizeof(npy_intp *) * num_threads);
  for (npy_intp id = 0; id < num_threads; id++) {
    npy_intp start_index =
        id * (result_size / num_threads) + (((id) < (result_size % num_threads))
                                                ? (id)
                                                : (result_size % num_threads));
    npy_intp end_index = start_index + result_size / num_threads +
                         (id < result_size % num_threads);
    a_data_ptr_cpy = a_data;
    for (npy_intp k = result_nd - 1; k >= 0; k--) {
      a_data_ptr_cpy += progress_init_a_data[k] * transposed_strides_cpy[k];
    }
    npy_intp *progress_init_a_data_cpy = malloc(sizeof(npy_intp) * result_nd);
    memcpy(progress_init_a_data_cpy, progress_init_a_data,
           sizeof(npy_intp) * result_nd);
    progress_init_a_data_arr[id] = progress_init_a_data_cpy;
    npy_intp tmp1 = task_amount;
    npy_intp *prg = calloc(a_ndim_except_last, sizeof(npy_intp));
    prg_arr[id] = prg;
    for (npy_intp j = a_ndim_except_last - 1; j >= 0; j--) {
      prg[j] = tmp1 % transposed_shape[j];
      tmp1 /= transposed_shape[j];
    }
    a_data_ptr_arr[id] = a_data_ptr_cpy;
    task_amount += (end_index - start_index);
    result_ptr_arr[id] = result_data_cpy;
    result_data_cpy += end_index - start_index;
    npy_intp tmp2 = task_amount;
    for (npy_intp j = result_nd - 1; j >= 0; j--) {
      progress_init_a_data[j] = tmp2 % result_shape[j];
      tmp2 /= result_shape[j];
    }
  }
  {
    int thread_id = omp_get_thread_num();
    npy_longlong *result_data_ptr = result_ptr_arr[thread_id];
    npy_float *a_data_ptr = a_data_ptr_arr[thread_id];
    npy_intp *_prg = prg_arr[thread_id];
    npy_intp p = 0;
    for (p = 0; p < result_size; p++) {
      npy_longlong index = 0;
      npy_float val = -__npy_inff();
      for (npy_intp i = 0; i < inner_loop_size; i++) {
        npy_float a_val = (a_data_ptr[i * last_stride]);
        if (a_val > val) {
          val = a_val;
          *result_data_ptr = i;
        }
      }
      for (int h = a_ndim_except_last - 1; h >= 0; h--) {
        if (_prg[h] < transposed_shape_cpy[h]) {
          _prg[h]++;
          a_data_ptr += transposed_strides_cpy[h];
          break;
        } else {
          _prg[h] = 0;
          a_data_ptr -= transposed_shape_cpy[h] * transposed_strides_cpy[h];
        }
      }
      result_data_ptr++;
    }
  }
  free(progress_init_a_data);
  for (npy_intp id = 0; id < num_threads; id++) {
    free(progress_init_a_data_arr[id]);
    free(prg_arr[id]);
  }
  free(prg_arr);
  free(progress_init_a_data_arr);
  free(a_data_ptr_arr);
  free(result_ptr_arr);
  free(transposed_shape_cpy);
  free(transposed_strides_cpy);
  return (PyObject *)result;
}