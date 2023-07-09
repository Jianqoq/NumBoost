#define PY_ARRAY_UNIQUE_SYMBOL core_c
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "tensor.h"
#include "core.h"
extern Array_Shape *ARRAY_SHAPE;

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

void check_shape(PyArrayObject *grad, PyObject *origin_data, PyObject **out, const char *error_msg)
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
        PyObject_Print(origin_data, stdout, 0);
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
            if (!vaild_shape((PyArrayObject *)result, (PyArrayObject *)origin_data, error_msg))
            {
                *out = NULL;
                return;
            }
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

void add_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2)
{
    if (!vaild_shape((PyArrayObject *)grad, (PyArrayObject *)self->data, "grad shape not equal to previous output shape"))
    {
        *out1 = NULL;
        *out2 = NULL;
        return;
    }
    PyArrayObject *tmp = (PyArrayObject *)grad;
    PyObject *grad2 = PyArray_Copy(tmp);
    *out1 = grad;
    *out2 = grad2;
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
    check_shape((PyArrayObject *)grad1, tmp1->data, out1, "grad1 shape not equal to previous data shape in mulbackward");
    check_shape((PyArrayObject *)grad2, tmp2->data, out2, "grad2 shape not equal to previous data shape in mulbackward");
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
    check_shape((PyArrayObject *)grad1, tmp1->data, out1, "grad1 shape not equal to previous data shape in divbackward");
    check_shape((PyArrayObject *)grad2, tmp2->data, out2, "grad2 shape not equal to previous data shape in divbackward");
};

void matmul_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2)
{
    if (!vaild_shape((PyArrayObject *)grad, (PyArrayObject *)self->data, "grad shape not equal to previous output shape"))
    {
        *out1 = NULL;
        *out2 = NULL;
        return;
    }
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
    check_shape((PyArrayObject *)grad1, tmp3->data, out1, "grad1 shape not equal to previous data shape in matmulbackward");
    check_shape((PyArrayObject *)grad2, tmp1->data, out2, "grad2 shape not equal to previous data shape in matmulbackward");
};

void negative_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2)
{
    Tensor *tmp1 = (Tensor *)self->x;
    Tensor *tmp2 = (Tensor *)self->y;
    PyObject *grad1 = PyNumber_Multiply(grad, tmp2->data);
    PyObject *grad2 = PyNumber_Multiply(grad, tmp1->data);

    PyObject_Print(grad1, stdout, 0);
    PyObject_Print(grad2, stdout, 0);
    check_shape((PyArrayObject *)grad1, tmp1->data, out1, "grad1 shape not equal to previous data shape in negativebackward");
    check_shape((PyArrayObject *)grad2, tmp2->data, out2, "grad2 shape not equal to previous data shape in negativebackward");
};

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