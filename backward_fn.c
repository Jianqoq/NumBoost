#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#include "numpy/arrayobject.h"
#include "tensor.h"

static bool shape_is_equal(npy_intp *dims1, npy_intp *dims2, int *nd)
{
    for (uint8_t i = 0; i < *nd; i++)
    {
        if (dims1[i] != dims2[i])
        {
            return 0;
        }
    }
    return 1;
}

void check_shape(PyArrayObject *grad, PyObject *origin_data, PyObject **out)
{
    PyArrayObject_fields *fields = (PyArrayObject_fields *)grad;
    PyTypeObject *type = Py_TYPE(origin_data);
    npy_intp dims1[NPY_MAXDIMS] = {0}; // grad shape
    npy_intp dims2[NPY_MAXDIMS] = {0}; // original data shape
    int nd1 = fields->nd;              // grad dimension
    uint8_t i = 0;
    for (i = 0; i < nd1; i++)
    {
        dims1[i] = fields->dimensions[i];
    }
    int nd2; // original data dimension
    npy_intp new_dims[NPY_MAXDIMS] = {0};

    if (type == &PyArray_Type)
    {
        PyArrayObject_fields *fields2 = (PyArrayObject_fields *)origin_data;
        nd2 = fields2->nd;
        for (i = 0; i < nd2; i++)
        {
            dims2[i] = fields2->dimensions[i];
        }
    }
    else if (type == &Tensor_type)
    {
        Tensor *tensor = (Tensor *)origin_data;
        PyArrayObject_fields *fields2 = (PyArrayObject_fields *)tensor->data;
        nd2 = fields2->nd;
        for (i = 0; i < nd2; i++)
        {
            dims2[i] = fields2->dimensions[i];
        }
    }
    else
    {
        nd2 = 0;
    }
    if (nd1 == nd2)
    {
        if (shape_is_equal(dims1, dims2, &nd1))
        {
            *out = (PyObject *)grad;
            return;
        }
        else
        {
            PyObject *g = NULL;
            uint8_t new_axis = 0;
            for (i = 0; i < NPY_MAXDIMS; i++)
            {
                if (dims1[i] != dims2[i])
                {
                    g = PyArray_Sum(grad, i, NPY_DOUBLE, NULL);
                    new_dims[new_axis] = 1;
                    Py_DECREF(grad);
                    grad = (PyArrayObject *)g;
                }
                else
                {
                    new_dims[new_axis] = dims1[i];
                }
                new_axis++;
            }
            PyArray_Dims shape = {new_dims, nd1};                          // new shape with keep dims
            PyObject *result = PyArray_Newshape(grad, &shape, NPY_CORDER); // reshape to original shape
            Py_DECREF(grad);
            *out = (PyObject *)result;
            return;
        }
    }
    else
    {
        PyObject *g = NULL;
        uint8_t range = nd1 - nd2;
        npy_intp dims3[NPY_MAXDIMS] = {0};
        for (i = 0; i < range; i++)
        {
            dims3[i] = 0;
        }
        for (i = range; i < nd1; i++)
        {
            dims3[i] = dims1[i];
        }
        for (i = 0; i < nd1; i++)
        {
            if (dims3[i] == 0)
            {
                g = PyArray_Sum(grad, i, NPY_DOUBLE, NULL);
                Py_DECREF(grad);
                grad = (PyArrayObject *)g;
            }
        }
        *out = (PyObject *)grad;
        return;
    }
}

int vaild_shape(PyArrayObject *grad, PyArrayObject *a, const char *error_msg)
{
    PyArrayObject_fields *fields = (PyArrayObject_fields *)grad;
    PyArrayObject_fields *fields2 = (PyArrayObject_fields *)a;
    for (uint8_t i = 0; i < fields->nd; i++)
    {
        if (fields->dimensions[i] != fields2->dimensions[i])
        {
            PyErr_SetString(PyExc_ValueError, error_msg);
            return 0;
        }
    }
    return 1;
}

void add_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2)
{
    if (!vaild_shape((PyArrayObject *)grad, (PyArrayObject *)self->data, "grad shape not equal to previous output shape"))
    {
        *out1 = NULL;
        *out2 = NULL;
        return;
    }
    PyObject *grad1 = (PyObject *)self->data;
    PyArrayObject *tmp = (PyArrayObject *)grad;
    PyObject *grad2 = PyArray_Copy(tmp);
};

void sub_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2)
{
    if (!vaild_shape((PyArrayObject *)grad, (PyArrayObject *)self->data, "grad shape not equal to previous output shape"))
    {
        *out1 = NULL;
        *out2 = NULL;
        return;
    }
    PyObject *grad2 = PyNumber_Negative(grad);
    *out1 = grad;
    Py_INCREF(self->data);
    *out2 = grad2;
};

void mul_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2)
{
    if (!vaild_shape((PyArrayObject *)grad, (PyArrayObject *)self->data, "grad shape not equal to previous output shape"))
    {
        *out1 = NULL;
        *out2 = NULL;
        return;
    }
    Tensor *tmp1 = (Tensor *)self->x;
    Tensor *tmp2 = (Tensor *)self->y;

    PyObject *grad1 = PyNumber_Multiply(grad, tmp2->data);
    PyObject *grad2 = PyNumber_Multiply(grad, tmp1->data);
    check_shape((PyArrayObject *)grad1, tmp1->data, out1);
    check_shape((PyArrayObject *)grad2, tmp2->data, out2);
};

void div_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2)
{
    if (!vaild_shape((PyArrayObject *)grad, (PyArrayObject *)self->data, "grad shape not equal to previous output shape"))
    {
        *out1 = NULL;
        *out2 = NULL;
        return;
    }
    Tensor *tmp1 = (Tensor *)self->x;
    Tensor *tmp2 = (Tensor *)self->y;
    PyObject *grad1 = PyNumber_TrueDivide(grad, tmp2->data);
    PyObject *midle = PyNumber_Power(tmp2->data, PyLong_FromLong(2), Py_None);
    PyObject *midle2 = PyNumber_Negative(tmp1->data);
    PyObject *tmp = PyNumber_TrueDivide(midle2, midle);
    PyObject *grad2 = PyNumber_Multiply(grad, tmp);
    check_shape((PyArrayObject *)grad1, tmp1->data, out1);
    check_shape((PyArrayObject *)grad2, tmp2->data, out2);
};

void matmul_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2)
{
    if (!vaild_shape((PyArrayObject *)grad, (PyArrayObject *)self->data, "grad shape not equal to previous output shape"))
    {
        *out1 = NULL;
        *out2 = NULL;
        return;
    }
    PyArrayObject *ga = (PyArrayObject *)grad;
    PyArrayObject *a = (PyArrayObject *)self->data;
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
        for (uint8_t i = 0; i < nd; i++)
        {
            dims[i] = i;
        }
        dims[nd - 2] = nd - 1;
        dims[nd - 1] = nd - 2;
        PyArray_Dims permute = {dims, nd};
        transposed1 = PyArray_Transpose(tmp2, &permute);
        transposed2 = PyArray_Transpose(tmp4, &permute);
    }
    else
    {
        PyErr_Print();
        PyErr_Clear();
        Py_Finalize();
        *out1 = NULL;
        *out2 = NULL;
        return;
    }
    PyObject *grad1 = PyNumber_MatrixMultiply(grad, transposed1);
    PyObject *grad2 = PyNumber_MatrixMultiply(transposed2, grad);
    free(dims);
    check_shape((PyArrayObject *)grad1, tmp3->data, out1);
    check_shape((PyArrayObject *)grad2, tmp1->y, out2);
};

void negative_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2)
{
    Tensor *tmp1 = (Tensor *)self->x;
    Tensor *tmp2 = (Tensor *)self->y;
    PyObject *grad1 = PyNumber_Multiply(grad, tmp2->data);
    PyObject *grad2 = PyNumber_Multiply(grad, tmp1->data);

    PyObject_Print(grad1, stdout, 0);
    PyObject_Print(grad2, stdout, 0);
    check_shape((PyArrayObject *)grad1, tmp1->data, out1);
    check_shape((PyArrayObject *)grad2, tmp2->data, out2);
};
