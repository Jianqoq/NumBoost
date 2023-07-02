#include "tensor.h"

void
add_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2)
{   
    import_array();
    if (PyErr_Occurred()) {
        PyErr_Print();
        PyErr_Clear();
    }
    PyObject *grad1 = (PyObject*)self->data;
    PyArrayObject *tmp = (PyArrayObject*)grad;
    PyObject *grad2 = PyArray_Copy(tmp);
    *out1 = grad1;
    Py_INCREF(grad1);
    *out2 = grad2;
    Py_INCREF(grad2);
    if (grad1 == NULL) {
        PyErr_Print();
        PyErr_Clear();
    }
};

void
sub_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2)
{
    PyObject *grad2 = PyNumber_Negative((PyObject*)grad);
    PyObject *tmp = (PyObject*)self;
    out1 = &tmp;
    Py_INCREF(tmp);
    Py_INCREF(grad2);
    out2 = &grad2;
    if (grad2 == NULL) {
        PyErr_Print();
        PyErr_Clear();
    }
};


void
mul_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2)
{
    PyObject *grad1 = PyNumber_Multiply(grad, self->y);
    PyObject *grad2 = PyNumber_Multiply(grad, self->x);
    if (grad1 == NULL || grad2 == NULL) {
        PyErr_Print();
        PyErr_Clear();
    }
    out1 = &grad1;
    Py_INCREF(grad1);
    out2 = &grad2;
    Py_INCREF(grad2);
};

void
div_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2)
{
    PyObject *grad1 = PyNumber_TrueDivide(grad, self->y);
    PyObject *tmp = PyNumber_TrueDivide(PyNumber_Negative(self->x), PyNumber_Power(self->y, PyLong_FromLong(2), NULL));
    PyObject *grad2 = PyNumber_Multiply(grad, self->x);
    if (grad1 == NULL || grad2 == NULL) {
        PyErr_Print();
        PyErr_Clear();
    }
    out1 = &grad1;
    Py_INCREF(grad1);
    out2 = &grad2;
    Py_INCREF(grad2);
};

void
matmul_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2)
{
    PyArrayObject *y = (PyArrayObject *)self->y;
    int nd = y->nd;
    npy_intp* dims = malloc(nd * sizeof(npy_intp));
    for (int i = 0; i < nd; i++) {
        dims[i] = i;
    }
    dims[nd - 2] = nd - 1;
    dims[nd - 1] = nd - 2;
    PyArray_Dims permute = {dims, nd};
    PyObject *grad1 = PyNumber_MatrixMultiply(grad, PyArray_Transpose((PyArrayObject*)self->y, &permute));
    PyObject *grad2 = PyNumber_MatrixMultiply(PyArray_Transpose((PyArrayObject*)self->x, &permute), grad);
    if (grad1 == NULL || grad2 == NULL) {
        PyErr_Print();
        PyErr_Clear();
    }
    out1 = &grad1;
    Py_INCREF(grad1);
    out2 = &grad2;
    Py_INCREF(grad2);
};

