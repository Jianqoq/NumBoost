#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "../binary_ops/binary_op_impl.h"
#include "../numboost_api.h"
#include "../tensor.h"
#include "ufunc_backward_def.h"
#include <numpy/arrayobject.h>

extern Array_Shape *ARRAY_SHAPE;
extern jnp_method *JNP_METHOD;
extern Zeros_Array_Dict *ZEROS_ARRAY_DICT;

double time_spent = 0.0;

void power_backward_fn(Tensor *self, PyObject *grad, PyObject **out,
                       PyObject **null) {
  Tensor *tmp1 = (Tensor *)self->x;
  *null = NULL;
  PyObject *power = get_power(self);
  if (TRACK) {
    PyObject *one = PyLong_FromLong(1);
    PyObject *sub = PyNumber_Subtract(power, one);
    PyObject *tmp = PyNumber_Power(tmp1->data, sub, Py_None);
    PyObject *mul = PyNumber_Multiply(grad, tmp);
    PyObject *grad2 = PyNumber_Multiply(power, mul);
    Py_DECREF(tmp);
    Py_DECREF(sub);
    Py_DECREF(one);
    Py_DECREF(mul);
    *out = grad2;
  } else {
    int power_type = PyArray_IsPythonNumber(power)
                         ? PyFloat_Check(power)  ? NPY_DOUBLE
                           : PyLong_Check(power) ? NPY_LONG
                           : PyBool_Check(power) ? NPY_BOOL
                                                 : -1
                         : PyArray_TYPE((PyArrayObject *)power);
    if (power_type == -1) {
      PyErr_SetString(PyExc_TypeError, "power type error");
      return;
    }
    int result_type = binary_result_type(
        SUB, power_type, type_2_size[power_type], NPY_LONG, sizeof(npy_long));
    result_type =
        binary_result_type(POW, PyArray_TYPE((PyArrayObject *)tmp1->data),
                           PyArray_DESCR((PyArrayObject *)tmp1->data)->elsize,
                           result_type, type_2_size[result_type]);
    result_type =
        binary_result_type(MUL, PyArray_TYPE((PyArrayObject *)grad),
                           PyArray_DESCR((PyArrayObject *)grad)->elsize,
                           result_type, type_2_size[result_type]);
    result_type = binary_result_type(MUL, power_type, type_2_size[power_type],
                                     result_type, type_2_size[result_type]);
    *out =
        (PyObject *)power_backward_fusefn[result_type](tmp1->data, power, grad);
  }
}

void sin_backward_fn(Tensor *self, PyObject *grad, PyObject **out,
                     PyObject **null) {
  Tensor *tmp1 = (Tensor *)self->x;
  *null = NULL;
  if (TRACK) {
    Tensor *tmp1 = (Tensor *)self->x;
    PyObject *cos = _cos_internal(tmp1->data, NULL);
    *out = PyNumber_Multiply(grad, cos);
    Py_DECREF(cos);
  } else {
    int result_type =
        elementwise_result_type(COS, PyArray_TYPE((PyArrayObject *)tmp1->data));
    result_type =
        binary_result_type(MUL, PyArray_TYPE((PyArrayObject *)grad),
                           PyArray_DESCR((PyArrayObject *)grad)->elsize,
                           result_type, type_2_size[result_type]);
    *out = (PyObject *)sin_backward_fusefn[result_type](tmp1->data, grad);
  }
};

void cos_backward_fn(Tensor *self, PyObject *grad, PyObject **out,
                     PyObject **null) {
  Tensor *tmp1 = (Tensor *)self->x;
  *null = NULL;
  if (TRACK) {
    Tensor *tmp1 = (Tensor *)self->x;
    PyObject *sin = _sin_internal(tmp1->data, NULL);
    PyObject *neg = PyNumber_Negative(grad);
    *out = PyNumber_Multiply(neg, sin);
    Py_DECREF(sin);
    Py_DECREF(neg);
  } else {
    int result_type =
        elementwise_result_type(SIN, PyArray_TYPE((PyArrayObject *)tmp1->data));
    result_type =
        binary_result_type(MUL, PyArray_TYPE((PyArrayObject *)grad),
                           PyArray_DESCR((PyArrayObject *)grad)->elsize,
                           result_type, type_2_size[result_type]);
    *out = (PyObject *)cos_backward_fusefn[result_type](tmp1->data, grad);
  }
};

void tan_backward_fn(Tensor *self, PyObject *grad, PyObject **out,
                     PyObject **null) {
  Tensor *tmp1 = (Tensor *)self->x;
  *null = NULL;
  if (TRACK) {
    PyObject *cos = _cos_internal(tmp1->data, NULL);
    PyObject *one = PyLong_FromLong(1);
    PyObject *sec = PyNumber_TrueDivide(one, cos);
    PyObject *mul = PyNumber_Multiply(sec, sec);
    *out = PyNumber_Multiply(grad, mul);
    Py_DECREF(sec);
    Py_DECREF(cos);
    Py_DECREF(one);
    Py_DECREF(mul);
    return;
  } else {
    int result_type =
        elementwise_result_type(COS, PyArray_TYPE((PyArrayObject *)tmp1->data));
    result_type = binary_result_type(MUL, result_type, type_2_size[result_type],
                                     result_type, type_2_size[result_type]);
    result_type =
        binary_result_type(DIV, PyArray_TYPE((PyArrayObject *)grad),
                           PyArray_DESCR((PyArrayObject *)grad)->elsize,
                           result_type, type_2_size[result_type]);
    *out = (PyObject *)tan_backward_fusefn[result_type](tmp1->data, grad);
  }
};

void arcsin_backward_fn(Tensor *self, PyObject *grad, PyObject **out,
                        PyObject **null) {
  Tensor *tmp1 = (Tensor *)self->x;
  *null = NULL;
  if (TRACK) {
    PyObject *square = PyNumber_Multiply(tmp1->data, tmp1->data);
    PyObject *point_5 = PyFloat_FromDouble(0.5);
    PyObject *one = PyLong_FromLong(1);
    PyObject *sub = PyNumber_Subtract(one, square);
    PyObject *power = PyNumber_Power(sub, point_5, Py_None);
    PyObject *divide = PyNumber_TrueDivide(one, power);
    PyObject *result = PyNumber_Multiply(grad, divide);
    *out = result;
    Py_DECREF(square);
    Py_DECREF(point_5);
    Py_DECREF(one);
    Py_DECREF(sub);
    Py_DECREF(power);
    Py_DECREF(divide);
  } else {
    int result_type =
        binary_result_type(SUB, NPY_LONG, sizeof(npy_long),
                           PyArray_TYPE((PyArrayObject *)tmp1->data),
                           PyArray_DESCR((PyArrayObject *)tmp1->data)->elsize);
    result_type = binary_result_type(POW, result_type, type_2_size[result_type],
                                     NPY_FLOAT, sizeof(npy_float));
    result_type = binary_result_type(DIV, NPY_FLOAT, sizeof(npy_float),
                                     result_type, type_2_size[result_type]);
    result_type =
        binary_result_type(MUL, PyArray_TYPE((PyArrayObject *)grad),
                           PyArray_DESCR((PyArrayObject *)grad)->elsize,
                           result_type, type_2_size[result_type]);
    *out = (PyObject *)arcsin_backward_fusefn[result_type](tmp1->data, grad);
  }
};

void arccos_backward_fn(Tensor *self, PyObject *grad, PyObject **out,
                        PyObject **null) {
  Tensor *tmp1 = (Tensor *)self->x;
  *null = NULL;
  if (TRACK) {
    PyObject *square = PyNumber_Multiply(tmp1->data, tmp1->data);
    PyObject *one = PyLong_FromLong(1);
    PyObject *sub = PyNumber_Subtract(one, square);
    PyObject *point_5 = PyFloat_FromDouble(0.5);
    PyObject *power = PyNumber_Power(sub, point_5, Py_None);
    PyObject *negative_one = PyLong_FromLong(-1);
    PyObject *divide = PyNumber_TrueDivide(negative_one, power);
    PyObject *result = PyNumber_Multiply(grad, divide);
    *out = result;
    Py_DECREF(square);
    Py_DECREF(one);
    Py_DECREF(sub);
    Py_DECREF(point_5);
    Py_DECREF(power);
    Py_DECREF(negative_one);
    Py_DECREF(divide);
  } else {
    int result_type =
        binary_result_type(SUB, NPY_LONG, sizeof(npy_long),
                           PyArray_TYPE((PyArrayObject *)tmp1->data),
                           PyArray_DESCR((PyArrayObject *)tmp1->data)->elsize);
    result_type = binary_result_type(POW, result_type, type_2_size[result_type],
                                     NPY_FLOAT, sizeof(npy_float));
    result_type = binary_result_type(DIV, NPY_LONG, sizeof(npy_long),
                                     result_type, type_2_size[result_type]);
    result_type =
        binary_result_type(MUL, PyArray_TYPE((PyArrayObject *)grad),
                           PyArray_DESCR((PyArrayObject *)grad)->elsize,
                           result_type, type_2_size[result_type]);
    *out = (PyObject *)arccos_backward_fusefn[result_type](tmp1->data, grad);
    *null = NULL;
  }
};

void arctan_backward_fn(Tensor *self, PyObject *grad, PyObject **out,
                        PyObject **null) {
  Tensor *tmp1 = (Tensor *)self->x;
  *null = NULL;
  if (TRACK) {
    PyObject *square = PyNumber_Multiply(tmp1->data, tmp1->data);
    PyObject *one = PyLong_FromLong(1);
    PyObject *add = PyNumber_Add(one, square);
    PyObject *divide = PyNumber_TrueDivide(one, add);
    PyObject *result = PyNumber_Multiply(grad, divide);
    *out = result;
    Py_DECREF(square);
    Py_DECREF(one);
    Py_DECREF(add);
    Py_DECREF(divide);
  } else {
    int result_type =
        binary_result_type(ADD, NPY_LONG, sizeof(npy_long),
                           PyArray_TYPE((PyArrayObject *)tmp1->data),
                           PyArray_DESCR((PyArrayObject *)tmp1->data)->elsize);
    result_type = binary_result_type(DIV, NPY_LONG, sizeof(npy_long),
                                     result_type, type_2_size[result_type]);
    result_type =
        binary_result_type(MUL, PyArray_TYPE((PyArrayObject *)grad),
                           PyArray_DESCR((PyArrayObject *)grad)->elsize,
                           result_type, type_2_size[result_type]);
    *out = (PyObject *)arctan_backward_fusefn[result_type](tmp1->data, grad);
    *null = NULL;
  }
};

void sinh_backward_fn(Tensor *self, PyObject *grad, PyObject **out,
                      PyObject **null) {
  Tensor *tmp1 = (Tensor *)self->x;
  *null = NULL;
  if (TRACK) {
    PyObject *cosh = _cosh_internal(tmp1->data, NULL);
    PyObject *result = PyNumber_Multiply(grad, cosh);
    *out = result;
    Py_DECREF(cosh);
    return;
  } else {
    int result_type = elementwise_result_type(
        COSH, PyArray_TYPE((PyArrayObject *)tmp1->data));
    result_type =
        binary_result_type(MUL, PyArray_TYPE((PyArrayObject *)grad),
                           PyArray_DESCR((PyArrayObject *)grad)->elsize,
                           result_type, type_2_size[result_type]);
    *out = (PyObject *)sinh_backward_fusefn[result_type](tmp1->data, grad);
    *null = NULL;
  }
};

void cosh_backward_fn(Tensor *self, PyObject *grad, PyObject **out,
                      PyObject **null) {
  Tensor *tmp1 = (Tensor *)self->x;
  *null = NULL;
  if (TRACK) {
    PyObject *sinh = _sinh_internal(tmp1->data, NULL);
    PyObject *result = PyNumber_Multiply(grad, sinh);
    *out = result;
    Py_DECREF(sinh);
  } else {
    int result_type = elementwise_result_type(
        SINH, PyArray_TYPE((PyArrayObject *)tmp1->data));
    result_type =
        binary_result_type(MUL, PyArray_TYPE((PyArrayObject *)grad),
                           PyArray_DESCR((PyArrayObject *)grad)->elsize,
                           result_type, type_2_size[result_type]);
    *out = (PyObject *)cosh_backward_fusefn[result_type](tmp1->data, grad);
    *null = NULL;
  }
}

void tanh_backward_fn(Tensor *self, PyObject *grad, PyObject **out,
                      PyObject **null) {
  *null = NULL;
  if (TRACK) {
    PyObject *self_val_square = PyNumber_Multiply(self->data, self->data);
    PyObject *one = PyFloat_FromDouble(1.0);
    PyObject *sub = PyNumber_Subtract(one, self_val_square);
    PyObject *result = PyNumber_Multiply(grad, sub);
    *out = result;
    Py_DECREF(self_val_square);
    Py_DECREF(one);
    Py_DECREF(sub);
  } else {
    int result_type =
        binary_result_type(SUB, NPY_FLOAT, type_2_size[NPY_FLOAT],
                           PyArray_TYPE((PyArrayObject *)self->data),
                           PyArray_DESCR((PyArrayObject *)self->data)->elsize);
    result_type =
        binary_result_type(MUL, PyArray_TYPE((PyArrayObject *)grad),
                           PyArray_DESCR((PyArrayObject *)grad)->elsize,
                           result_type, type_2_size[result_type]);
    *out = (PyObject *)tanh_backward_fusefn[result_type](self->data, grad);
  }
};

void arcsinh_backward_fn(Tensor *self, PyObject *grad, PyObject **out,
                         PyObject **null) {
  Tensor *tmp1 = (Tensor *)self->x;
  *null = NULL;
  if (TRACK) {
    PyObject *square = PyNumber_Multiply(tmp1->data, tmp1->data);
    PyObject *one = PyFloat_FromDouble(1.0);
    PyObject *tmp = PyNumber_Add(one, square);
    PyObject *point_5 = PyFloat_FromDouble(0.5);
    PyNumber_InPlacePower(tmp, point_5, Py_None);
    PyObject *result = PyNumber_TrueDivide(grad, tmp);
    *out = result;
    Py_DECREF(square);
    Py_DECREF(one);
    Py_DECREF(tmp);
    Py_DECREF(point_5);
  } else {
    int result_type =
        binary_result_type(ADD, NPY_LONG, sizeof(npy_long),
                           PyArray_TYPE((PyArrayObject *)tmp1->data),
                           PyArray_DESCR((PyArrayObject *)tmp1->data)->elsize);
    result_type =
        binary_result_type(SQRT, result_type, type_2_size[result_type],
                           NPY_FLOAT, sizeof(npy_float));
    result_type =
        binary_result_type(DIV, PyArray_TYPE((PyArrayObject *)grad),
                           PyArray_DESCR((PyArrayObject *)grad)->elsize,
                           result_type, type_2_size[result_type]);
    *out = (PyObject *)arcsinh_backward_fusefn[result_type](tmp1->data, grad);
  }
};

void arccosh_backward_fn(Tensor *self, PyObject *grad, PyObject **out,
                         PyObject **null) {
  Tensor *tmp1 = (Tensor *)self->x;
  *null = NULL;
  if (TRACK) {
    PyObject *square = PyNumber_Multiply(tmp1->data, tmp1->data);
    PyObject *one = PyFloat_FromDouble(1.0);
    PyObject *tmp = PyNumber_Subtract(square, one);
    PyObject *point_5 = PyFloat_FromDouble(0.5);
    PyNumber_InPlacePower(tmp, point_5, Py_None);
    PyObject *result = PyNumber_TrueDivide(grad, tmp);
    *out = result;
    Py_DECREF(tmp);
    Py_DECREF(square);
    Py_DECREF(one);
    Py_DECREF(point_5);
  } else {
    int result_type =
        binary_result_type(SUB, PyArray_TYPE((PyArrayObject *)tmp1->data),
                           PyArray_DESCR((PyArrayObject *)tmp1->data)->elsize,
                           NPY_LONG, sizeof(npy_long));
    result_type =
        binary_result_type(SQRT, result_type, type_2_size[result_type],
                           NPY_FLOAT, sizeof(npy_float));
    result_type =
        binary_result_type(DIV, PyArray_TYPE((PyArrayObject *)grad),
                           PyArray_DESCR((PyArrayObject *)grad)->elsize,
                           result_type, type_2_size[result_type]);
    *out = (PyObject *)arccosh_backward_fusefn[result_type](tmp1->data, grad);
  }
};

void arctanh_backward_fn(Tensor *self, PyObject *grad, PyObject **out,
                         PyObject **null) {
  Tensor *tmp1 = (Tensor *)self->x;
  *null = NULL;
  if (TRACK) {
    PyObject *square = PyNumber_Multiply(tmp1->data, tmp1->data);
    PyObject *one = PyFloat_FromDouble(1.0);
    PyObject *tmp = PyNumber_Subtract(one, square);
    PyObject *grad2 = PyNumber_TrueDivide(grad, tmp);
    *out = grad2;
    Py_DECREF(one);
    Py_DECREF(tmp);
    Py_DECREF(square);
  } else {
    int result_type =
        binary_result_type(SUB, NPY_LONG, sizeof(npy_long),
                           PyArray_TYPE((PyArrayObject *)tmp1->data),
                           PyArray_DESCR((PyArrayObject *)tmp1->data)->elsize);
    result_type =
        binary_result_type(DIV, PyArray_TYPE((PyArrayObject *)grad),
                           PyArray_DESCR((PyArrayObject *)grad)->elsize,
                           result_type, type_2_size[result_type]);
    *out = (PyObject *)arctanh_backward_fusefn[result_type](tmp1->data, grad);
  }
};

void exp_backward_fn(Tensor *self, PyObject *grad, PyObject **out,
                     PyObject **null) {
  Tensor *tmp1 = (Tensor *)self->x;
  *null = NULL;
  if (TRACK) {
    PyObject *exp = _exp_internal(tmp1->data, NULL);
    PyObject *result = PyNumber_Multiply(grad, exp);
    Py_DECREF(exp);
    *out = result;
  } else {
    int result_type =
        elementwise_result_type(EXP, PyArray_TYPE((PyArrayObject *)tmp1->data));
    result_type =
        binary_result_type(MUL, PyArray_TYPE((PyArrayObject *)grad),
                           PyArray_DESCR((PyArrayObject *)grad)->elsize,
                           result_type, type_2_size[result_type]);
    *out = (PyObject *)exp_backward_fusefn[result_type](tmp1->data, grad);
  }
};

void log_backward_fn(Tensor *self, PyObject *grad, PyObject **out,
                     PyObject **null) {
  Tensor *tmp1 = (Tensor *)self->x;
  *null = NULL;
  if (TRACK) {
    PyObject *result = PyNumber_TrueDivide(grad, tmp1->data);
    *out = result;
    return;
  } else {
    int result_type =
        binary_result_type(DIV, PyArray_TYPE((PyArrayObject *)grad),
                           PyArray_DESCR((PyArrayObject *)grad)->elsize,
                           PyArray_TYPE((PyArrayObject *)tmp1->data),
                           PyArray_DESCR((PyArrayObject *)tmp1->data)->elsize);
    *out = (PyObject *)log_backward_fusefn[result_type](grad, tmp1->data);
    *null = NULL;
  };
}

void log10_backward_fn(Tensor *self, PyObject *grad, PyObject **out,
                       PyObject **null) {
  Tensor *tmp1 = (Tensor *)self->x;
  *null = NULL;
  if (TRACK) {
    npy_intp dims[1] = {1};
    double data[1] = {10.0};
    PyObject *ten = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, data);
    PyObject *ln = _log_internal(ten, NULL);
    PyObject *mul = PyNumber_Multiply(tmp1->data, ln);
    PyObject *grad2 = PyNumber_TrueDivide(grad, mul);
    *out = grad2;
    Py_DECREF(ten);
    Py_DECREF(ln);
    Py_DECREF(mul);
  } else {
    int result_type =
        binary_result_type(MUL, PyArray_TYPE((PyArrayObject *)tmp1->data),
                           PyArray_DESCR((PyArrayObject *)tmp1->data)->elsize,
                           PyArray_TYPE((PyArrayObject *)grad),
                           PyArray_DESCR((PyArrayObject *)grad)->elsize);
    result_type =
        binary_result_type(DIV, PyArray_TYPE((PyArrayObject *)grad),
                           PyArray_DESCR((PyArrayObject *)grad)->elsize,
                           result_type, type_2_size[result_type]);
    *out = (PyObject *)log10_backward_fusefn[result_type](grad, tmp1->data);
  }
};

void sqrt_backward_fn(Tensor *self, PyObject *grad, PyObject **out,
                      PyObject **null) {
  Tensor *tmp1 = (Tensor *)self->x;
  *null = NULL;
  if (TRACK) {
    PyObject *point_5 = PyFloat_FromDouble(0.5);
    PyObject *negative_5 = PyFloat_FromDouble(-0.5);
    PyObject *pow = PyNumber_Power(tmp1->data, negative_5, Py_None);
    PyObject *mul = PyNumber_Multiply(point_5, pow);
    PyObject *result = PyNumber_Multiply(grad, mul);
    *out = result;
    *null = NULL;
    Py_DECREF(point_5);
    Py_DECREF(negative_5);
    Py_DECREF(pow);
    Py_DECREF(mul);
  } else {
    int result_type =
        binary_result_type(MUL, PyArray_TYPE((PyArrayObject *)self->data),
                           PyArray_DESCR((PyArrayObject *)self->data)->elsize,
                           NPY_FLOAT, sizeof(npy_float));
    result_type =
        binary_result_type(DIV, PyArray_TYPE((PyArrayObject *)grad),
                           PyArray_DESCR((PyArrayObject *)grad)->elsize,
                           result_type, type_2_size[result_type]);
    *out = (PyObject *)sqrt_backward_fusefn[result_type](self->data, grad);
  }
};

void abs_backward_fn(Tensor *self, PyObject *grad, PyObject **out,
                     PyObject **null) {
  Tensor *tmp1 = (Tensor *)self->x;
  *null = NULL;
  if (TRACK) {
    PyObject *zero = PyLong_FromLong(0);
    PyObject *one = PyLong_FromLong(1);
    PyObject *negative_one = PyLong_FromLong(-1);
    PyObject *less_than_zero_mask =
        PyObject_RichCompare(tmp1->data, zero, Py_LT);
    PyObject *reverted = PyNumber_Multiply(less_than_zero_mask, negative_one);
    PyObject *less_than_zero = PyNumber_Multiply(reverted, grad);
    PyObject *sub = PyNumber_Subtract(less_than_zero_mask, one);
    PyObject *greater_than_zero_mask = PyNumber_Multiply(sub, negative_one);
    PyObject *greater_than_zero =
        PyNumber_Multiply(greater_than_zero_mask, grad);
    PyObject *result = PyNumber_Add(less_than_zero, greater_than_zero);
    *out = result;
    Py_DECREF(less_than_zero_mask);
    Py_DECREF(reverted);
    Py_DECREF(less_than_zero);
    Py_DECREF(sub);
    Py_DECREF(greater_than_zero_mask);
    Py_DECREF(greater_than_zero);
    Py_DECREF(zero);
    Py_DECREF(one);
    Py_DECREF(negative_one);
  } else {
    *out = (PyObject *)
        abs_backward_fusefn[PyArray_TYPE((PyArrayObject *)tmp1->data)](
            tmp1->data, grad);
  }
};

void reshape_backward_fn(Tensor *self, PyObject *grad, PyObject **out,
                         PyObject **null) {
  npy_intp *shape = get_array_shape(self);
  npy_intp len = get_shape_len(self);
  if (len == -1) {
    *out = NULL;
    *null = NULL;
    return;
  }
  PyArray_Dims prev_shape = {shape, (int)len};
  if (TRACK) {
    PyObject *tuple = PyTuple_New(len);
    for (int i = 0; i < len; i++) {
      PyTuple_SetItem(tuple, i, PyLong_FromLongLong(shape[i]));
    }
    *out = PyObject_CallFunctionObjArgs(JNP_METHOD->reshape, grad, tuple, NULL);
    Py_DECREF(tuple);
    *null = NULL;
    return;
  }
  PyObject *result =
      PyArray_Newshape((PyArrayObject *)grad, &prev_shape, NPY_CORDER);
  *out = result;
  *null = NULL;
};

void transpose_backward_fn(Tensor *self, PyObject *grad, PyObject **out,
                           PyObject **null) {
  npy_intp *axes = get_array_shape(self);
  npy_intp len = get_shape_len(self);
  if (len == -1) {
    *out = NULL;
    *null = NULL;
    return;
  }
  npy_intp *new_axes = (npy_intp *)malloc(sizeof(npy_intp) * len);
  for (npy_intp i = 0; i < len; i++) {
    new_axes[i] = search_num(axes, len, i);
  }
  PyArray_Dims prev_axes = {new_axes, (int)len};
  PyObject *result = PyArray_Transpose((PyArrayObject *)grad, &prev_axes);
  free(new_axes);
  *out = result;
  *null = NULL;
};

void slice_backward_fn(Tensor *self, PyObject *grad, PyObject **out,
                       PyObject **null) {
  npy_intp *origin_shape = NULL;
  PyObject *slice_obj = NULL, *zeros = NULL;
  int nd = 0;
  get_slice_objs(self, &origin_shape, &slice_obj, &nd,
                 &zeros); // zeros is ndarray
  if (zeros == NULL || slice_obj == NULL || origin_shape == NULL) {
    *out = NULL;
    *null = NULL;
    return;
  }
  PyObject *sliced = PyObject_GetItem(zeros, slice_obj);
  PyObject *result = PyNumber_InPlaceAdd(sliced, grad);
  *out = zeros;
  *null = NULL;
  Py_INCREF(zeros);
  Py_DECREF(sliced);
  Py_DECREF(result);
};