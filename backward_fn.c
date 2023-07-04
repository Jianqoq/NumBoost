#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#include "numpy/arrayobject.h"
#include "tensor.h"

void add_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2)
{
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
    Tensor *tmp1 = (Tensor *)self->x;
    Tensor *tmp2 = (Tensor *)self->y;
    PyObject *grad1 = PyNumber_TrueDivide(grad, tmp2->data);
    PyObject *midle = PyNumber_Power(tmp2->data, PyLong_FromLong(2), Py_None);
    PyObject *midle2 = PyNumber_Negative(tmp1->data);
    PyObject *tmp = PyNumber_TrueDivide(midle2, midle);
    PyObject *grad2 = PyNumber_Multiply(grad, tmp);
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

void matmul_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2)
{
    PyObject *transposed1 = NULL;
    PyObject *transposed2 = NULL;
    Tensor *tmp1 = (Tensor *)self->y;
    PyArrayObject *tmp2 = (PyArrayObject *)tmp1->data;
    PyArrayObject_fields *fields = (PyArrayObject_fields *)tmp2;
    Tensor *tmp3 = (Tensor *)self->x;
    PyArrayObject *tmp4 = (PyArrayObject *)tmp3->data;
    int nd = fields->nd;
    npy_intp *dims = NULL;
    if (0 < nd && nd < 2)
    {
        nd = 1;
        transposed1 = (PyObject *)tmp2;
        transposed2 = (PyObject *)tmp4;
    }
    else if (nd >= 2)
    {
        dims = malloc(nd * sizeof(npy_intp));
        dims[nd - 2] = nd - 1;
        dims[nd - 1] = nd - 2;
        for (int i = 0; i < nd; i++)
        {
            dims[i] = i;
        }
        PyArray_Dims permute = {dims, nd};
        transposed1 = PyArray_Transpose(tmp2, &permute);
        transposed2 = PyArray_Transpose(tmp4, &permute);
    }
    else
    {
        PyErr_Print();
        PyErr_Clear();
        Py_Finalize();
    }
    PyObject *grad1 = PyNumber_MatrixMultiply(grad, transposed1);
    PyObject *grad2 = PyNumber_MatrixMultiply(transposed2, grad);

    if (grad1 == NULL || grad2 == NULL)
    {
        PyErr_Print();
        PyErr_Clear();
        free(dims);
    }
    free(dims);
    *out1 = grad1;
    Py_INCREF(grad1);
    *out2 = grad2;
    Py_INCREF(grad2);
};
