#ifndef _SHAPE_H
#define _SHAPE_H
#include "tensor.h"

inline bool shape_isequal(npy_intp *a_shape, npy_intp *b_shape, int a_ndim,
                          int b_ndim) {
  if (a_ndim != b_ndim)
    return false;
  for (int i = 0; i < a_ndim; i++)
    if (a_shape[i] != b_shape[i])
      return false;
  return true;
}

inline bool isbroadcastable_same_ndim(npy_intp *a_shape, npy_intp *b_shape,
                                      int ndim, npy_intp **a_new_shape,
                                      npy_intp **b_new_shape) {
  int i;
  for (i = 0; i < ndim; i++)
    if (a_shape[i] != b_shape[i]) {
      if (a_shape[i] == 1 || b_shape[i] == 1)
        continue;
      else
        return false;
    }
  if (a_new_shape != NULL) {
    *a_new_shape = a_shape;
  }
  if (b_new_shape != NULL) {
    *b_new_shape = b_shape;
  }
  return true;
}

// this method does not check for a specific Tensor, it only checks if the shape
// is broadcastable use shape_isbroadcastable_to if you want to check if a
// Tensor is broadcastable to another Tensor it assumes the shape is not equal
bool shape_isbroadcastable(npy_intp *a_shape, npy_intp *b_shape, int a_ndim,
                           int b_ndim);

// ex support return new shape
bool shape_isbroadcastable_ex(npy_intp *a_shape, npy_intp *b_shape, int a_ndim,
                              int b_ndim, npy_intp **a_new_shape,
                              npy_intp **b_new_shape);

// this method check for a specific Tensor, it checks if the shape is
// broadcastable to another Tensor it assumes the shape is not equal and a_ndim
// <= b_ndim
bool shape_isbroadcastable_to(npy_intp *a_shape, npy_intp *b_shape, int a_ndim,
                              int b_ndim);

// ex support return new shape
bool shape_isbroadcastable_to_ex(npy_intp *a_shape, npy_intp *b_shape,
                                 int a_ndim, int b_ndim,
                                 npy_intp **a_new_shape);

int newshape_except_num(npy_intp except, npy_intp *shape, int ndim,
                        npy_intp **new_shape);

npy_intp shape_prod(npy_intp *shape, int ndim);

int shape_count_one(npy_intp *shape, int ndim);

inline bool shape_smaller(npy_intp *a_shape, npy_intp *b_shape, int a_ndim,
                          int b_ndim) {

  if (a_ndim < b_ndim)
    return true;
  else if (a_ndim > b_ndim)
    return false;
  else {
    npy_intp prod_a = 1;
    npy_intp prod_b = 1;
    for (int i = 0; i < a_ndim; i++) {
      prod_a *= a_shape[i];
      prod_b *= b_shape[i];
    }
    if (prod_a < prod_b)
      return true;
    else
      return false;
  }
}

inline bool is_in(int *arr, int len, int target) {
  for (int i = 0; i < len; i++) {
    if (arr[i] == target)
      return true;
  }
  return false;
}

inline void move_axes_to_innermost(int *axises, int axises_len, npy_intp *shape,
                         int a_ndim, npy_intp *shape_to_transpose) {
  for (int i = 0; i < axises_len; i++) {
    shape[axises[i]] = 0;
  }
  int j = a_ndim - axises_len;
  int k = 0;
  int track_idx = 0;
  for (int i = 0; i < a_ndim; i++) {
    if (shape[i] != 0) {
      shape_to_transpose[k++] = i;
    } else {
      shape_to_transpose[j++] = axises[track_idx++];
    }
  }
}

npy_intp find_innerloop_size(npy_intp *shape_a, npy_intp *shape_b, int ndim);

void find_special_one(npy_intp *shape, int ndim, npy_intp *strides,
                      npy_intp *stride, npy_intp *left_prod,
                      npy_intp *right_prod);

void predict_broadcast_shape(npy_intp *a_shape, npy_intp *b_shape, int ndim,
                             npy_intp **predict_shape);

npy_intp rightprod_non_one(npy_intp *shape, int ndim, int *axis);

void preprocess_strides(npy_intp *a_shape, npy_intp last_stride, int ndim,
                        npy_intp **strides);

PyArrayObject *arry_to_broadcast(npy_intp *shape_a, npy_intp *shape_b,
                                 PyArrayObject *a, PyArrayObject *b, int axis);

#endif