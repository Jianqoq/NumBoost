#define PY_ARRAY_UNIQUE_SYMBOL core_c
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "tensor.h"
#include "core.h"
extern Array_Shape *ARRAY_SHAPE;

void power_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    Tensor *tmp1 = (Tensor *)self->x;
    PyObject *power = get_power(self);
    PyObject *sub = PyNumber_Subtract(power, PyLong_FromLong(1));
    PyObject *tmp = PyNumber_Power(tmp1->data, sub, Py_None);
    PyObject *grad2 = PyNumber_Multiply(power, PyNumber_Multiply(grad, tmp));
    Py_DECREF(tmp);
    Py_DECREF(sub);
    *out = grad2;
    *null = NULL;
};

void sin_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    Tensor *tmp1 = (Tensor *)self->x;
    PyObject *cos = _cos_internal(tmp1->data, NULL);
    *out = PyNumber_Multiply(grad, cos);
    Py_DECREF(cos);
    *null = NULL;
};

void cos_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    Tensor *tmp1 = (Tensor *)self->x;
    PyObject *sin = _sin_internal(tmp1->data, NULL);
    *out = PyNumber_Multiply(PyNumber_Negative(grad), sin);
    Py_DECREF(sin);
    *null = NULL;
};

void tan_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    Tensor *tmp1 = (Tensor *)self->x;
    PyObject *cos = _cos_internal(tmp1->data, NULL);

    PyObject *sec = PyNumber_TrueDivide(PyLong_FromLong(1), cos);
    *out = PyNumber_Multiply(grad, PyNumber_Multiply(sec, sec));
    Py_DECREF(sec);
    *null = NULL;
};

void arcsin_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    Tensor *tmp1 = (Tensor *)self->x;
    PyObject *square = PyNumber_Multiply(tmp1->data, tmp1->data);
    PyObject *point_5 = PyFloat_FromDouble(0.5);
    PyObject *one = PyLong_FromLong(1);
    PyObject *sub = PyNumber_Subtract(one, square);
    PyObject *power = PyNumber_Power(sub, point_5, Py_None);
    PyObject *divide = PyNumber_TrueDivide(one, power);
    PyObject * result = PyNumber_Multiply(grad, divide);
    *out = result;
    *null = NULL;
    Py_DECREF(square);
    Py_DECREF(point_5);
    Py_DECREF(one);
    Py_DECREF(sub);
    Py_DECREF(power);
    Py_DECREF(divide);
};

void arccos_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    Tensor *tmp1 = (Tensor *)self->x;
    PyObject *square = PyNumber_Multiply(tmp1->data, tmp1->data);
    PyObject *one = PyLong_FromLong(1);
    PyObject *sub = PyNumber_Subtract(one, square);
    PyObject *point_5 = PyFloat_FromDouble(0.5);
    PyObject *power = PyNumber_Power(sub, point_5, Py_None);
    PyObject *negative_one = PyLong_FromLong(-1);
    PyObject *divide = PyNumber_TrueDivide(negative_one, power);
    PyObject * result = PyNumber_Multiply(grad, divide);
    *out = result;
    *null = NULL;
    Py_DECREF(square);
    Py_DECREF(one);
    Py_DECREF(sub);
    Py_DECREF(point_5);
    Py_DECREF(power);
    Py_DECREF(negative_one);
    Py_DECREF(divide);
};

void arctan_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    Tensor *tmp1 = (Tensor *)self->x;
    PyObject *square = PyNumber_Multiply(tmp1->data, tmp1->data);
    PyObject *one = PyLong_FromLong(1);
    PyObject *add = PyNumber_Add(one, square);
    PyObject *divide = PyNumber_TrueDivide(one, add);
    PyObject * result = PyNumber_Multiply(grad, divide);
    *out = result;
    *null = NULL;
    Py_DECREF(square);
    Py_DECREF(one);
    Py_DECREF(add);
    Py_DECREF(divide);
};

void sinh_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    Tensor *tmp1 = (Tensor *)self->x;
    PyObject *cosh = _cosh_internal(tmp1->data, NULL);
    PyObject * result = PyNumber_Multiply(grad, cosh);
    *out = result;
    *null = NULL;
    Py_DECREF(cosh);
};

void cosh_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    Tensor *tmp1 = (Tensor *)self->x;
    PyObject *sinh = _sinh_internal(tmp1->data, NULL);
    PyObject * result = PyNumber_Multiply(grad, sinh);
    *out = result;
    *null = NULL;
    Py_DECREF(sinh);
};

void tanh_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    PyObject *self_val_square = PyNumber_Multiply(self->data, self->data);
    PyObject *one = PyFloat_FromDouble(1.0);
    PyObject *sub = PyNumber_Subtract(one, self_val_square);
    PyObject * result = PyNumber_Multiply(grad, sub);
    *out = result;
    *null = NULL;
    Py_DECREF(self_val_square);
    Py_DECREF(one);
    Py_DECREF(sub);
};

void arcsinh_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    Tensor *tmp1 = (Tensor *)self->x;
    PyObject *square = PyNumber_Multiply(tmp1->data, tmp1->data);
    PyObject *one = PyFloat_FromDouble(1.0);
    PyObject *tmp = PyNumber_Add(one, square);
    PyObject *point_5 = PyFloat_FromDouble(0.5);
    PyNumber_InPlacePower(tmp, point_5, Py_None);
    PyObject * result = PyNumber_TrueDivide(grad, tmp);
    *out = result;
    *null = NULL;
    Py_DECREF(square);
    Py_DECREF(one);
    Py_DECREF(tmp);
    Py_DECREF(point_5);
};

void arccosh_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    Tensor *tmp1 = (Tensor *)self->x;
    PyObject *square = PyNumber_Multiply(tmp1->data, tmp1->data);
    PyObject *one = PyFloat_FromDouble(1.0);
    PyObject *tmp = PyNumber_Subtract(square, one);
    PyObject *point_5 = PyFloat_FromDouble(0.5);
    PyNumber_InPlacePower(tmp, point_5, Py_None);
    PyObject * result = PyNumber_TrueDivide(grad, tmp);
    *out = result;
    *null = NULL;
    Py_DECREF(tmp);
    Py_DECREF(square);
    Py_DECREF(one);
    Py_DECREF(point_5);
};

void arctanh_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    Tensor *tmp1 = (Tensor *)self->x;
    PyObject *square = PyNumber_Multiply(tmp1->data, tmp1->data);
    PyObject *one = PyFloat_FromDouble(1.0);
    PyObject *tmp = PyNumber_Subtract(one, square);
    PyObject *grad2 = PyNumber_TrueDivide(grad, tmp);
    *out = grad2;
    *null = NULL;
    Py_DECREF(one);
    Py_DECREF(tmp);
    Py_DECREF(square);
};

void exp_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    Tensor *tmp1 = (Tensor *)self->x;
    PyObject *exp = _exp_internal(tmp1->data, NULL);
    PyObject * result = PyNumber_Multiply(grad, exp);
    Py_DECREF(exp);
    *out = result;
    *null = NULL;
};

void log_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    Tensor *tmp1 = (Tensor *)self->x;
    PyObject * result = PyNumber_TrueDivide(grad, tmp1->data);
    *out = result;
    *null = NULL;
};

void log10_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null) // not implemaent yet
{
    Tensor *tmp1 = (Tensor *)self->x;
    PyNumber_InPlacePower(tmp1->data, PyNumber_Subtract(self->y, PyLong_FromLong(1)), Py_None);
    PyObject *grad2 = PyNumber_Multiply(self->y, PyNumber_Multiply(grad, tmp1->data));
    *out = grad2;
    *null = NULL;
};

void sqrt_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    Tensor *tmp1 = (Tensor *)self->x;
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
};

void abs_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    Tensor *tmp1 = (Tensor *)self->x;
    PyObject *one = PyLong_FromLong(1);
    PyObject *sub = PyNumber_Subtract(self->y, one);
    PyObject *pow = PyNumber_Power(tmp1->data, sub, Py_None);
    PyObject *mul = PyNumber_Multiply(grad, pow);
    PyObject *grad2 = PyNumber_Multiply(self->y, mul);
    *out = grad2;
    *null = NULL;
    Py_DECREF(one);
    Py_DECREF(sub);
    Py_DECREF(pow);
    Py_DECREF(mul);
};

void reshape_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    npy_intp *get = get_array_shape(self);
    int len = *get_shape_len(self);
    PyArray_Dims prev_shape = {get, len};
    PyObject * result = PyArray_Newshape((PyArrayObject*)grad, &prev_shape, NPY_CORDER);
    free(get);
    *out = result;
    *null = NULL;
};