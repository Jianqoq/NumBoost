#include "tensor.h"

Tensor *
new_Tensor(PyTypeObject *type, Tensor *tensor, Tensor *tensor2, PyObject *data,
           PyObject *x, PyObject *y, int has_conv, uint64_t vars, bool require_grad,
           const char *grad_fn, PyObject *graph, PyObject *axis, int dim,
           PyObject *stride, PyObject *base)
{
    Tensor *self;
    self = (Tensor *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->data = data;
        Py_INCREF(self->data);
        if (tensor->require_grad || tensor2->require_grad)
        {
            self->x = (PyObject *)tensor; //decreased performance
            self->y = (PyObject *)tensor2; //decreased performance
            self->require_grad = true;
            self->grad_fn = grad_fn;
            self->vars = tensor->vars + tensor2->vars;
        }
        else
        {
            self->x = Py_None;
            self->y = Py_None;
            self->require_grad = false;
            self->grad_fn = "";
            self->vars = 0;
        }
        Py_INCREF(self->x);
        Py_INCREF(self->y);
        self->has_conv = has_conv;
        self->axis = axis;
        Py_INCREF(self->axis);
        self->dim = dim;
        self->stride = stride;
        Py_INCREF(self->stride);
        self->base = base;
        Py_INCREF(self->base);
        return self;
    }
    else
    {
        PyErr_Print();
        PyErr_Clear();
        return NULL;
    }
}

Tensor *
new_Tensor_scalar(PyTypeObject *type, Tensor *tensor, PyObject *data,
                  PyObject *x, PyObject *y, int has_conv, uint64_t vars,
                  bool require_grad, const char *grad_fn, PyObject *graph,
                  PyObject *axis, int dim, PyObject *stride, PyObject *base)
{
    Tensor *self;
    self = (Tensor *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->data = data;
        Py_INCREF(self->data);
        if (tensor->require_grad)
        {
            self->x = x;
            Py_INCREF(self->x);
            self->y = y;
            Py_INCREF(self->y);
            self->require_grad = true;
            self->grad_fn = grad_fn;
        }
        else
        {
            self->x = Py_None;
            Py_INCREF(self->x);
            self->y = Py_None;
            Py_INCREF(self->y);
            self->require_grad = false;
            self->grad_fn = grad_fn;
        }
        self->has_conv = has_conv;
        self->vars = vars;
        self->graph = graph;
        Py_INCREF(self->graph);
        self->axis = axis;
        Py_INCREF(self->axis);
        self->dim = dim;
        self->stride = stride;
        Py_INCREF(self->stride);
        self->base = base;
        Py_INCREF(self->base);
        return self;
    }
    else
    {
        PyErr_Print();
        PyErr_Clear();
        return NULL;
    }
}

PyObject *
tensor_add(Tensor *self, PyObject *other)
{
    Tensor *tmp;
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        PyObject *numpy_result = PyNumber_Add(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            PyErr_Print();
            PyErr_Clear();
            return NULL;
        }
        Tensor *new_tensor = new_Tensor(
            &Tensor_type, self, tmp, numpy_result, self->data,
            tmp->data, self->has_conv, self->vars, self->require_grad,
            "AddBackward", self->graph, self->axis, self->dim,
            self->stride, self->base);
        Py_INCREF(new_tensor);
        return (PyObject *)new_tensor;
    }
    else
    {
        PyObject *numpy_result = PyNumber_Add(self->data, other);

        if (numpy_result == NULL)
        {
            PyErr_Print();
            PyErr_Clear();
            return NULL;
        }
        Tensor *new_tensor = new_Tensor_scalar(
            &Tensor_type, self, numpy_result, self->data, other,
            self->has_conv, self->vars, self->require_grad, "AddBackward",
            self->graph, self->axis, self->dim, self->stride, self->base);
        Py_INCREF(new_tensor);
        return (PyObject *)new_tensor;
    }
}

PyObject *
tensor_iadd(Tensor *self, PyObject *other)
{
    Tensor *tmp;
    PyObject *numpy_result;
    Py_DECREF(self->x);
    Py_DECREF(self->y);
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        numpy_result = PyNumber_InPlaceAdd(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            PyErr_Print();
            PyErr_Clear();
            return NULL;
        }
        self->y = tmp->data;
    }
    else
    {
        numpy_result = PyNumber_InPlaceAdd(self->data, other);
        if (numpy_result == NULL)
        {
            PyErr_Print();
            PyErr_Clear();
            return NULL;
        }
        self->y = other;
    }
    if (self->require_grad)
    {
        self->grad_fn = "AddBackward";
    }
    self->x = self->data;
    Py_INCREF(self->x);
    Py_INCREF(self->y);
    Py_DECREF(self->data);
    self->data = numpy_result;
    Py_INCREF(self->data);
    Py_INCREF(self);
    return (PyObject *)self;
}

PyObject *
tensor_mul(Tensor *self, PyObject *other)
{
    Tensor *tmp;
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        PyObject *numpy_result = PyNumber_Multiply(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            PyErr_Print();
            PyErr_Clear();
            return NULL;
        }
        Tensor *new_tensor = new_Tensor(
            &Tensor_type, self, tmp, numpy_result, self->data,
            tmp->data, self->has_conv, self->vars, self->require_grad,
            "MulBackward", self->graph, self->axis, self->dim,
            self->stride, self->base);
        Py_INCREF(new_tensor);
        return (PyObject *)new_tensor;
    }
    else
    {
        PyObject *numpy_result = PyNumber_Multiply(self->data, other);

        if (numpy_result == NULL)
        {
            PyErr_Print();
            PyErr_Clear();
            return NULL;
        }
        Tensor *new_tensor = new_Tensor_scalar(
            &Tensor_type, self, numpy_result, self->data, other,
            self->has_conv, self->vars, self->require_grad, "MulBackward",
            self->graph, self->axis, self->dim, self->stride, self->base);
        Py_INCREF(new_tensor);
        return (PyObject *)new_tensor;
    }
}

PyObject *
tensor_imul(Tensor *self, PyObject *other)
{
    Tensor *tmp;
    PyObject *numpy_result;
    Py_DECREF(self->x);
    Py_DECREF(self->y);
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        numpy_result = PyNumber_InPlaceMultiply(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            PyErr_Print();
            PyErr_Clear();
            return NULL;
        }
        self->y = tmp->data;
    }
    else
    {
        numpy_result = PyNumber_InPlaceMultiply(self->data, other);
        if (numpy_result == NULL)
        {
            PyErr_Print();
            PyErr_Clear();
            return NULL;
        }
        self->y = other;
    }
    if (self->require_grad)
    {
        self->grad_fn = "MulBackward";
    }
    self->x = self->data;
    Py_INCREF(self->x);
    Py_INCREF(self->y);
    Py_DECREF(self->data);
    self->data = numpy_result;
    Py_INCREF(self->data);
    Py_INCREF(self);
    return (PyObject *)self;
}

PyObject *
tensor_div(Tensor *self, PyObject *other)
{
    Tensor *tmp;
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        PyObject *numpy_result = PyNumber_TrueDivide(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            PyErr_Print();
            PyErr_Clear();
            return NULL;
        }
        Tensor *new_tensor = new_Tensor(
            &Tensor_type, self, tmp, numpy_result, self->data,
            tmp->data, self->has_conv, self->vars, self->require_grad,
            "DivBackward", self->graph, self->axis, self->dim,
            self->stride, self->base);
        Py_INCREF(new_tensor);
        return (PyObject *)new_tensor;
    }
    else
    {
        PyObject *numpy_result = PyNumber_TrueDivide(self->data, other);

        if (numpy_result == NULL)
        {
            PyErr_Print();
            PyErr_Clear();
            return NULL;
        }
        Tensor *new_tensor = new_Tensor_scalar(
            &Tensor_type, self, numpy_result, self->data, other,
            self->has_conv, self->vars, self->require_grad, "DivBackward",
            self->graph, self->axis, self->dim, self->stride, self->base);
        Py_INCREF(new_tensor);
        return (PyObject *)new_tensor;
    }
}

PyObject *
tensor_idiv(Tensor *self, PyObject *other)
{
    Tensor *tmp;
    PyObject *numpy_result;
    Py_DECREF(self->x);
    Py_DECREF(self->y);
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        numpy_result = PyNumber_InPlaceTrueDivide(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            PyErr_Print();
            PyErr_Clear();
            return NULL;
        }
        self->y = tmp->data;
    }
    else
    {
        numpy_result = PyNumber_InPlaceTrueDivide(self->data, other);
        if (numpy_result == NULL)
        {
            PyErr_Print();
            PyErr_Clear();
            return NULL;
        }
        self->y = other;
    }
    if (self->require_grad)
    {
        self->grad_fn = "DivBackward";
    }
    self->x = self->data;
    Py_INCREF(self->x);
    Py_INCREF(self->y);
    Py_DECREF(self->data);
    self->data = numpy_result;
    Py_INCREF(self->data);
    Py_INCREF(self);
    return (PyObject *)self;
}

PyObject *
tensor_negative(Tensor *self)
{
    PyObject *numpy_result =
        PyNumber_InPlaceMultiply(self->data, PyLong_FromLong(-1));
    if (numpy_result == NULL)
    {
        PyErr_Print();
        PyErr_Clear();
        return NULL;
    }
    if (self->require_grad)
    {
        self->grad_fn = "NegativeBackward";
        Py_DECREF(self->x);
        Py_DECREF(self->y);
        self->x = self->data;
        Py_INCREF(self->x);
    }
    Py_DECREF(self->data);
    self->data = numpy_result;
    Py_INCREF(self->data);
    Py_INCREF(self);
    return (PyObject *)self;
}

PyObject *
tensor_sub(Tensor *self, PyObject *other)
{
    Tensor *tmp;
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        PyObject *numpy_result = PyNumber_Subtract(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            PyErr_Print();
            PyErr_Clear();
            return NULL;
        }
        Tensor *new_tensor = new_Tensor(
            &Tensor_type, self, tmp, numpy_result, self->data,
            tmp->data, self->has_conv, self->vars, self->require_grad,
            "SubBackward", self->graph, self->axis, self->dim,
            self->stride, self->base);
        Py_INCREF(new_tensor);
        return (PyObject *)new_tensor;
    }
    else
    {
        PyObject *numpy_result = PyNumber_Subtract(self->data, other);

        if (numpy_result == NULL)
        {
            PyErr_Print();
            PyErr_Clear();
            return NULL;
        }
        Tensor *new_tensor = new_Tensor_scalar(
            &Tensor_type, self, numpy_result, self->data, other,
            self->has_conv, self->vars, self->require_grad, "SubBackward",
            self->graph, self->axis, self->dim, self->stride, self->base);
        Py_INCREF(new_tensor);
        return (PyObject *)new_tensor;
    }
}

PyObject *
tensor_isub(Tensor *self, PyObject *other)
{
    Tensor *tmp;
    PyObject *numpy_result;
    Py_DECREF(self->x);
    Py_DECREF(self->y);
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        numpy_result = PyNumber_InPlaceSubtract(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            PyErr_Print();
            PyErr_Clear();
            return NULL;
        }
        self->y = tmp->data;
    }
    else
    {
        numpy_result = PyNumber_InPlaceSubtract(self->data, other);
        if (numpy_result == NULL)
        {
            PyErr_Print();
            PyErr_Clear();
            return NULL;
        }
        self->y = other;
    }
    if (self->require_grad)
    {
        self->grad_fn = "SubBackward";
    }
    self->x = self->data;
    Py_INCREF(self->x);
    Py_INCREF(self->y);
    Py_DECREF(self->data);
    self->data = numpy_result;
    Py_INCREF(self->data);
    Py_INCREF(self);
    return (PyObject *)self;
}

PyObject *
tensor_pow(Tensor *self, PyObject *other)
{
    Tensor *temp;
    if (Py_TYPE(other) == &Tensor_type)
    {
        temp = (Tensor *)other;
        PyObject *numpy_result =
            PyNumber_Power(self->data, temp->data, Py_None);
        if (numpy_result == NULL)
        {
            PyErr_Print();
            PyErr_Clear();
            return NULL;
        }
        Tensor *new_tensor = new_Tensor(
            &Tensor_type, self, temp, numpy_result, self->data, temp->data,
            self->has_conv, self->vars, self->require_grad, "PowBackward",
            self->graph, self->axis, self->dim, self->stride, self->base);
        Py_INCREF(new_tensor);
        return (PyObject *)new_tensor;
    }
    else
    {
        PyObject *numpy_result = PyNumber_Power(self->data, other, Py_None);
        if (numpy_result == NULL)
        {
            PyErr_Print();
            PyErr_Clear();
            return NULL;
        }
        Tensor *new_tensor = new_Tensor_scalar(
            &Tensor_type, self, numpy_result, self->data, other,
            self->has_conv, self->vars, self->require_grad, "PowBackward",
            self->graph, self->axis, self->dim, self->stride, self->base);
        Py_INCREF(new_tensor);
        return (PyObject *)new_tensor;
    }
}

PyObject *
tensor_ipow(Tensor *self, PyObject *other)
{
    PyObject *numpy_result;
    Tensor *temp;
    Py_DECREF(self->x);
    Py_DECREF(self->y);
    if (Py_TYPE(other) == &Tensor_type)
    {
        temp = (Tensor *)other;
        numpy_result = PyNumber_InPlacePower(self->data, temp->data, Py_None);
        if (numpy_result == NULL)
        {
            PyErr_Print();
            PyErr_Clear();
            return NULL;
        }
        if (!self->require_grad && !temp->require_grad)
        {
            Py_DECREF(self->data);
            self->data = numpy_result;
            Py_INCREF(self->data);
            Py_INCREF(self);
            return (PyObject *)self;
        }
        self->y = temp->data;
    }
    else
    {
        numpy_result = PyNumber_InPlacePower(self->data, other, Py_None);
        if (numpy_result == NULL)
        {
            PyErr_Print();
            PyErr_Clear();
            return NULL;
        }
        if (!self->require_grad)
        {
            Py_DECREF(self->data);
            self->data = numpy_result;
            Py_INCREF(self->data);
            Py_INCREF(self);
            return (PyObject *)self;
        }
        self->y = other;
    }
    self->grad_fn = "PowBackward";
    self->x = self->data;
    Py_INCREF(self->x);
    Py_INCREF(self->y);
    Py_DECREF(self->data);
    self->data = numpy_result;
    Py_INCREF(self->data);
    Py_INCREF(self);
    return (PyObject *)self;
}

PyObject *
tensor_matmul(Tensor *self, Tensor *other)
{
    PyObject *numpy_result = PyNumber_MatrixMultiply(self->data, other->data);
    if (numpy_result == NULL)
    {
        PyErr_Print();
        PyErr_Clear();
        return NULL;
    }
    Tensor *new_tensor = new_Tensor(
        &Tensor_type, self, other, numpy_result, self->data, other->data,
        self->has_conv, self->vars, self->require_grad, "MatMulBackward",
        self->graph, self->axis, self->dim, self->stride, self->base);
    Py_INCREF(new_tensor);
    return (PyObject *)new_tensor;
}

PyObject *
tensor_imatmul(Tensor *self, Tensor *other)
{
    PyObject *numpy_result =
        PyNumber_InPlaceMatrixMultiply(self->data, other->data);
    if (numpy_result == NULL)
    {
        PyErr_Print();
        PyErr_Clear();
        return NULL;
    }
    if (self->require_grad)
    {
        Py_DECREF(self->x);
        Py_DECREF(self->y);
        self->x = self->data;
        self->y = other->data;
        Py_INCREF(self->x);
        Py_INCREF(self->y);
        self->grad_fn = "MatMulBackward";
    }
    Py_DECREF(self->data);
    self->data = numpy_result;
    Py_INCREF(self->data);
    Py_INCREF(self);
    return (PyObject *)self;
}