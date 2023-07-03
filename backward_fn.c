#include "tensor.h"

void add_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2)
{
    import_array();
    if (PyErr_Occurred())
    {
        PyErr_Print();
        PyErr_Clear();
    }
    PyObject *grad1 = (PyObject *)self->data;
    PyArrayObject *tmp = (PyArrayObject *)grad;
    PyObject *grad2 = PyArray_Copy(tmp);
    *out1 = grad1;
    Py_INCREF(grad1);
    *out2 = grad2;
    Py_INCREF(grad2);
    if (grad1 == NULL)
    {
        PyErr_Print();
        PyErr_Clear();
    }
};

void sub_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2)
{
    import_array();
    PyObject *grad2 = PyNumber_Negative(grad);
    *out1 = self->data;
    Py_INCREF(self->data);
    Py_INCREF(grad2);
    *out2 = grad2;
    if (grad2 == NULL)
    {
        PyErr_Print();
        PyErr_Clear();
    }
};

void mul_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2)
{
    import_array();
    Tensor *tmp1 = (Tensor *)self->x;
    Tensor *tmp2 = (Tensor *)self->y;
    PyObject *grad1 = PyNumber_Multiply(grad, tmp1->data);
    PyObject *grad2 = PyNumber_Multiply(grad, tmp2->data);
    if (grad1 == NULL || grad2 == NULL)
    {
        PyErr_Print();
        PyErr_Clear();
    }
    *out1 = grad1;
    Py_INCREF(grad1);
    *out2 = grad2;
    Py_INCREF(grad2);
};

void div_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2)
{
    import_array();
    Tensor *tmp1 = (Tensor *)self->x;
    Tensor *tmp2 = (Tensor *)self->y;
    PyObject *grad1 = PyNumber_TrueDivide(grad, tmp2->data);
    PyObject *midle = PyNumber_Power(tmp2->data, PyLong_FromLong(2), Py_None);
    PyObject *midle2 = PyNumber_Negative(tmp1->data);
    PyObject *tmp = PyNumber_TrueDivide(midle2, midle);
    PyObject *grad2 = PyNumber_Multiply(grad, tmp1->data);
    if (grad1 == NULL || grad2 == NULL)
    {
        PyErr_Print();
        PyErr_Clear();
    }
    *out1 = grad1;
    Py_INCREF(grad1);
    *out2 = grad2;
    Py_INCREF(grad2);
};
