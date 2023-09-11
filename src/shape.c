#define NO_IMPORT_ARRAY
#include "shape.h"
#include "utils.h"

// this method does not check for a specific Tensor, it only checks if the shape
// is broadcastable use shape_isbroadcastable_to if you want to check if a
// Tensor is broadcastable to another Tensor it assumes the shape is not equal
bool shape_isbroadcastable(npy_intp *a_shape, npy_intp *b_shape, int a_ndim,
                           int b_ndim) {
  int i;
  if (a_ndim == b_ndim) {
    bool isbroad =
        isbroadcastable_same_ndim(a_shape, b_shape, a_ndim, NULL, NULL);
    return isbroad;
  } else {
    npy_intp *new_shape = NULL;
    int ndim_diff = a_ndim - b_ndim;
    if (ndim_diff > 0) {
      new_shape = (npy_intp *)malloc(a_ndim * sizeof(npy_intp));
      for (i = 0; i < a_ndim; i++) {
        if (i < ndim_diff)
          new_shape[i] = 1;
        else
          new_shape[i] = b_shape[i - ndim_diff];
      }
      b_ndim = a_ndim;
    } else {
      ndim_diff = -ndim_diff;
      new_shape = (npy_intp *)malloc(b_ndim * sizeof(npy_intp));
      for (i = 0; i < b_ndim; i++) {
        if (i < ndim_diff)
          new_shape[i] = 1;
        else
          new_shape[i] = a_shape[i + ndim_diff];
      }
      a_ndim = b_ndim;
    }
    bool isbroad =
        isbroadcastable_same_ndim(a_shape, new_shape, a_ndim, NULL, NULL);
    free(new_shape);
    return isbroad;
  }
}

// ex support return new shape
bool shape_isbroadcastable_ex(npy_intp *a_shape, npy_intp *b_shape, int a_ndim,
                              int b_ndim, npy_intp **a_new_shape,
                              npy_intp **b_new_shape) {
  int i;
  if (a_ndim == b_ndim) {
    bool isbroad = isbroadcastable_same_ndim(a_shape, b_shape, a_ndim,
                                             a_new_shape, b_new_shape);
    return isbroad;
  } else {
    npy_intp *new_shape = NULL;
    int ndim_diff = a_ndim - b_ndim;
    if (ndim_diff > 0) {
      new_shape = (npy_intp *)malloc(a_ndim * sizeof(npy_intp));
      for (i = 0; i < a_ndim; i++) {
        if (i < ndim_diff)
          new_shape[i] = 1;
        else
          new_shape[i] = b_shape[i - ndim_diff];
      }
      b_ndim = a_ndim;
    } else {
      ndim_diff = -ndim_diff;
      new_shape = (npy_intp *)malloc(b_ndim * sizeof(npy_intp));
      for (i = 0; i < b_ndim; i++) {
        if (i < ndim_diff)
          new_shape[i] = 1;
        else
          new_shape[i] = a_shape[i - ndim_diff];
      }
      a_ndim = b_ndim;
    }
    bool isbroad =
        isbroadcastable_same_ndim(ndim_diff > 0 ? a_shape : b_shape, new_shape,
                                  a_ndim, a_new_shape, b_new_shape);
    if (!isbroad)
      free(new_shape);
    return isbroad;
  }
}

// this method check for a specific Tensor, it checks if the shape is
// broadcastable to another Tensor it assumes the shape is not equal and a_ndim
// <= b_ndim
bool shape_isbroadcastable_to(npy_intp *a_shape, npy_intp *b_shape, int a_ndim,
                              int b_ndim) {
  int i;
  if (a_ndim == b_ndim) {
    bool isbroad =
        isbroadcastable_same_ndim(a_shape, b_shape, a_ndim, NULL, NULL);
    return isbroad;
  } else {
    npy_intp *new_shape = NULL;
    int ndim_diff = a_ndim - b_ndim;
    if (ndim_diff > 0) {
      return false;
    } else {
      ndim_diff = -ndim_diff;
      new_shape = (npy_intp *)malloc(b_ndim * sizeof(npy_intp));
      for (i = 0; i < b_ndim; i++) {
        if (i < ndim_diff)
          new_shape[i] = 1;
        else
          new_shape[i] = a_shape[i - ndim_diff];
      }
      a_ndim = b_ndim;
      bool isbroad =
          isbroadcastable_same_ndim(b_shape, new_shape, a_ndim, NULL, NULL);
      if (!isbroad)
        free(new_shape);
      return isbroad;
    }
  }
}

/*a is a samll array, a broadcast to b.
 *this method will check if two array is broadcastable, if not return false,
 *else will malloc a padded new shape and attach to a_new_shape.
 *If a_shape is (1, 5) and b_shape is (5, 1), it will also malloc a new shape
 *and attach to a_new_shape.*/
bool shape_isbroadcastable_to_ex(npy_intp *a_shape, npy_intp *b_shape,
                                 int a_ndim, int b_ndim,
                                 npy_intp **a_new_shape) {
  int i;
  npy_intp *new_shape = NULL;
  if (a_ndim == b_ndim) {
    new_shape = (npy_intp *)malloc(b_ndim * sizeof(npy_intp));
    memcpy(new_shape, a_shape, b_ndim * sizeof(npy_intp));
    bool isbroad = isbroadcastable_same_ndim(new_shape, b_shape, a_ndim,
                                             a_new_shape, NULL);
    return isbroad;
  } else {
    int ndim_diff = a_ndim - b_ndim;
    if (ndim_diff > 0) {
      return false;
    } else {
      ndim_diff = -ndim_diff;
      new_shape = (npy_intp *)malloc(b_ndim * sizeof(npy_intp));
      for (i = 0; i < b_ndim; i++) {
        if (i < ndim_diff)
          new_shape[i] = 1;
        else
          new_shape[i] = a_shape[i - ndim_diff];
      }
      a_ndim = b_ndim;
      bool isbroad = isbroadcastable_same_ndim(new_shape, b_shape, a_ndim,
                                               a_new_shape, NULL);
      if (!isbroad)
        free(new_shape);
      return isbroad;
    }
  }
}

int newshape_except_num(npy_intp except, npy_intp *shape, int ndim,
                        npy_intp **new_shape) {
  int i;
  int cnt = 0;
  for (i = 0; i < ndim; i++)
    if (shape[i] == 1)
      cnt++;
  int new_ndim = ndim - cnt;
  npy_intp *_new_shape = (npy_intp *)malloc(new_ndim * sizeof(npy_intp));
  int track = 0;
  for (i = 0; i < ndim; i++) {
    if (i != except) {
      _new_shape[track] = shape[i];
      track++;
    }
  }
  *new_shape = _new_shape;
  return new_ndim;
}

npy_intp shape_prod(npy_intp *shape, int ndim) {
  npy_intp prod = 1;
  for (int i = 0; i < ndim; i++)
    prod *= shape[i];
  return prod;
}

int shape_count_one(npy_intp *shape, int ndim) {
  int cnt = 0;
  for (int i = 0; i < ndim; i++)
    if (shape[i] == 1)
      cnt++;
  return cnt;
}

// given a shape (1, 2, 1, 1, 3, 4) we will find the 2 and its corresponding
// stride
void find_special_one(npy_intp *shape, int ndim, npy_intp *strides,
                      npy_intp *stride, npy_intp *left_prod,
                      npy_intp *right_prod) {
  int special = 0;
  for (int i = ndim - 1; i > 1; i--) {
    if (shape[i] == 1 && shape[i - 1] != 1) {
      special = i - 1;
      break;
    }
  }
  npy_intp right = 1;
  npy_intp left = 1;
  for (int i = special + 1; i < ndim; i++)
    right *= shape[i];
  for (int i = 0; i <= special; i++)
    left *= shape[i];
  *left_prod = left;
  *right_prod = right;
  *stride = strides[special];
}

npy_intp find_innerloop_size(npy_intp *shape_a, npy_intp *shape_b, int ndim) {
  npy_intp prod = 1;
  for (int i = ndim - 1; i > 1; i--) {
    if (shape_a[i] == shape_b[i]) {
      prod *= shape_a[i];
    } else {
      break;
    }
  }
  return prod;
}

void predict_broadcast_shape(npy_intp *a_shape, npy_intp *b_shape, int ndim,
                             npy_intp **predict_shape) {
  npy_intp *shape = (npy_intp *)malloc(ndim * sizeof(npy_intp));
  for (int i = 0; i < ndim; i++) {
    if (a_shape[i] == b_shape[i]) {
      shape[i] = a_shape[i];
    } else if (a_shape[i] == 1) {
      shape[i] = b_shape[i];
    } else if (b_shape[i] == 1) {
      shape[i] = a_shape[i];
    } else {
      PyErr_SetString(PyExc_ValueError, "Cannot broadcast shapes");
      return;
    }
  }
  *predict_shape = shape;
}

void preprocess_strides(npy_intp *a_shape, npy_intp last_stride, int ndim,
                        npy_intp **strides) {
  npy_intp *strides_ = (npy_intp *)malloc(ndim * sizeof(npy_intp));
  for (int i = ndim - 1; i >= 0; i--) {
    if (a_shape[i] == 1)
      strides_[i] = 0;
    else {
      strides_[i] = last_stride;
      last_stride *= a_shape[i];
    }
  }
  *strides = strides_;
}

npy_intp rightprod_non_one(npy_intp *shape, int ndim, int *axis) {
  npy_intp prod = 1;
  int i;
  for (i = ndim - 1; i > 0; i--)
    if (shape[i] != 1)
      prod *= shape[i];
    else
      break;
  *axis = i;
  return prod;
}

PyArrayObject *arry_to_broadcast(npy_intp *shape_a, npy_intp *shape_b,
                                 PyArrayObject *a, PyArrayObject *b, int axis) {
  if (shape_a[axis] > shape_b[axis])
    return b;
  else
    return a;
}