#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#include "tensor.h"

Tensor *
new_Tensor(PyTypeObject *type, Tensor *tensor, Tensor *tensor2, PyObject *data,
           PyObject *x, PyObject *y, int has_conv, uint64_t vars, bool require_grad,
           const char *grad_fn, PyObject *graph, PyObject *axis, int dim,
           PyObject *base)
{
    Tensor *self = (Tensor *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->data = data;
        Py_INCREF(self->data);
        if (tensor->require_grad || tensor2->require_grad)
        {
            self->x = (PyObject *)tensor;  // decreased performance
            self->y = (PyObject *)tensor2; // decreased performance
            self->require_grad = true;
            self->grad_fn = grad_fn;
            self->vars = tensor->vars + tensor2->vars + 1;
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
        self->dim = dim;
        self->axis = axis;
        Py_INCREF(self->axis);
        self->base = base;
        Py_INCREF(self->base);
        self->grad = Py_None;
        Py_INCREF(self->grad);
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
new_Tensor_scalar(PyTypeObject *type, Tensor *self, PyObject *data, PyObject *y, int has_conv, uint64_t vars,
                  bool require_grad, const char *grad_fn, PyObject *graph,
                  PyObject *axis, int dim, PyObject *base)
{
    Tensor *tensor;
    tensor = (Tensor *)type->tp_alloc(type, 0);
    if (tensor != NULL)
    {
        tensor->data = data;
        Py_INCREF(tensor->data);
        if (self->require_grad)
        {
            tensor->x = (PyObject *)self;
            Py_INCREF(tensor->x);
            tensor->y = y;
            Py_INCREF(tensor->y);
            tensor->require_grad = true;
            tensor->grad_fn = grad_fn;
            tensor->vars = self->vars + 2;
        }
        else
        {
            tensor->x = Py_None;
            Py_INCREF(tensor->x);
            tensor->y = Py_None;
            Py_INCREF(tensor->y);
            tensor->require_grad = false;
            tensor->grad_fn = grad_fn;
            tensor->vars = 0;
        }
        tensor->has_conv = has_conv;
        tensor->vars = vars;
        tensor->dim = dim;
        tensor->graph = graph;
        Py_INCREF(tensor->graph);
        tensor->axis = axis;
        Py_INCREF(tensor->axis);
        tensor->base = base;
        Py_INCREF(tensor->base);
        tensor->grad = Py_None;
        Py_INCREF(tensor->grad);
        return tensor;
    }
    else
    {
        PyErr_Print();
        PyErr_Clear();
        return NULL;
    }
}

Tensor *
new_Tensor_x(PyTypeObject *type, Tensor *self, PyObject *data, int has_conv, uint64_t vars,
             bool require_grad, const char *grad_fn, PyObject *graph,
             PyObject *axis, int dim, PyObject *base)
{
    Tensor *tensor;
    tensor = (Tensor *)type->tp_alloc(type, 0);
    if (tensor != NULL)
    {
        tensor->data = data;
        Py_INCREF(tensor->data);
        if (self->require_grad)
        {
            tensor->x = (PyObject *)self;
            Py_INCREF(tensor->x);
            tensor->y = Py_None;
            Py_INCREF(tensor->y);
            tensor->require_grad = true;
            tensor->grad_fn = grad_fn;
            tensor->vars = self->vars + 1;
        }
        else
        {
            tensor->x = Py_None;
            Py_INCREF(tensor->x);
            tensor->y = Py_None;
            Py_INCREF(tensor->y);
            tensor->require_grad = false;
            tensor->grad_fn = grad_fn;
            tensor->vars = 0;
        }
        tensor->has_conv = has_conv;
        tensor->vars = vars;
        tensor->dim = dim;
        tensor->graph = graph;
        Py_INCREF(tensor->graph);
        tensor->axis = axis;
        Py_INCREF(tensor->axis);
        tensor->base = base;
        Py_INCREF(tensor->base);
        tensor->grad = Py_None;
        Py_INCREF(tensor->grad);
        return tensor;
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
            self->base);
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
            &Tensor_type, self, numpy_result, other,
            self->has_conv, self->vars, self->require_grad, "AddBackward",
            self->graph, self->axis, self->dim, self->base);
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
            self->base);
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
            &Tensor_type, self, numpy_result, other,
            self->has_conv, self->vars, self->require_grad, "MulBackward",
            self->graph, self->axis, self->dim, self->base);
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
            self->base);
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
            &Tensor_type, self, numpy_result, other,
            self->has_conv, self->vars, self->require_grad, "DivBackward",
            self->graph, self->axis, self->dim, self->base);
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
    return (PyObject *)self;
}

PyObject *
tensor_inegative(Tensor *self)
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
    return (PyObject *)self;
}

PyObject *
tensor_negative(Tensor *self)
{
    PyObject *negative_1 = PyLong_FromLong(-1);
    PyObject *numpy_result =
        PyNumber_Multiply(self->data, negative_1);
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
        self->y = negative_1;
        Py_INCREF(self->x);
        Py_INCREF(self->y);
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
            self->base);
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
            &Tensor_type, self, numpy_result, other,
            self->has_conv, self->vars, self->require_grad, "SubBackward",
            self->graph, self->axis, self->dim, self->base);
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
            self->graph, self->axis, self->dim, self->base);
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
            &Tensor_type, self, numpy_result, other,
            self->has_conv, self->vars, self->require_grad, "PowBackward",
            self->graph, self->axis, self->dim, self->base);
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
        self->graph, self->axis, self->dim, self->base);
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
    return (PyObject *)self;
}

PyObject *
tensor_positive(Tensor *self)
{
    Py_INCREF(self);
    return (PyObject *)self;
}

PyObject *
tensor_absolute(Tensor *self)
{
    PyObject *numpy_result = PyNumber_Absolute(self->data);
    if (numpy_result == NULL)
    {
        PyErr_Print();
        PyErr_Clear();
        return NULL;
    }
    Tensor *new_tensor = new_Tensor_x(
        &Tensor_type, self, numpy_result,
        self->has_conv, self->vars, self->require_grad, "AbsoluteBackward",
        self->graph, self->axis, self->dim, self->base);
    return (PyObject *)new_tensor;
}

PyObject *
tensor_invert(Tensor *self)
{
    PyObject *numpy_result = PyNumber_Invert(self->data);
    if (numpy_result == NULL)
    {
        PyErr_Print();
        PyErr_Clear();
        return NULL;
    }
    Tensor *new_tensor = new_Tensor_x(
        &Tensor_type, self, numpy_result,
        self->has_conv, self->vars, self->require_grad, "InvertBackward",
        self->graph, self->axis, self->dim, self->base);
    return (PyObject *)new_tensor;
}

PyObject *
tensor_lshift(Tensor *self, PyObject *other)
{
    Tensor *tmp;
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        PyObject *numpy_result = PyNumber_Lshift(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            PyErr_Print();
            PyErr_Clear();
            return NULL;
        }
        Tensor *new_tensor = new_Tensor(
            &Tensor_type, self, tmp, numpy_result, self->data,
            tmp->data, self->has_conv, self->vars, self->require_grad,
            "LshiftBackward", self->graph, self->axis, self->dim,
            self->base);
        return (PyObject *)new_tensor;
    }
    else
    {
        PyObject *numpy_result = PyNumber_Lshift(self->data, other);

        if (numpy_result == NULL)
        {
            PyErr_Print();
            PyErr_Clear();
            return NULL;
        }
        Tensor *new_tensor = new_Tensor_scalar(
            &Tensor_type, self, numpy_result, other,
            self->has_conv, self->vars, self->require_grad, "LshiftBackward",
            self->graph, self->axis, self->dim, self->base);
        return (PyObject *)new_tensor;
    }
}

PyObject *tensor_rshift(Tensor *self, PyObject *other){
        Tensor *tmp;
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        PyObject *numpy_result = PyNumber_Rshift(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            PyErr_Print();
            PyErr_Clear();
            return NULL;
        }
        Tensor *new_tensor = new_Tensor(
            &Tensor_type, self, tmp, numpy_result, self->data,
            tmp->data, self->has_conv, self->vars, self->require_grad,
            "RshiftBackward", self->graph, self->axis, self->dim,
            self->base);
        return (PyObject *)new_tensor;
    }
    else
    {
        PyObject *numpy_result = PyNumber_Rshift(self->data, other);

        if (numpy_result == NULL)
        {
            PyErr_Print();
            PyErr_Clear();
            return NULL;
        }
        Tensor *new_tensor = new_Tensor_scalar(
            &Tensor_type, self, numpy_result, other,
            self->has_conv, self->vars, self->require_grad, "RshiftBackward",
            self->graph, self->axis, self->dim, self->base);
        return (PyObject *)new_tensor;
    }
}

PyObject *tensor_and(Tensor *self, PyObject *other){
        Tensor *tmp;
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        PyObject *numpy_result = PyNumber_And(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            PyErr_Print();
            PyErr_Clear();
            return NULL;
        }
        Tensor *new_tensor = new_Tensor(
            &Tensor_type, self, tmp, numpy_result, self->data,
            tmp->data, self->has_conv, self->vars, self->require_grad,
            "AndBackward", self->graph, self->axis, self->dim,
            self->base);
        return (PyObject *)new_tensor;
    }
    else
    {
        PyObject *numpy_result = PyNumber_Rshift(self->data, other);

        if (numpy_result == NULL)
        {
            PyErr_Print();
            PyErr_Clear();
            return NULL;
        }
        Tensor *new_tensor = new_Tensor_scalar(
            &Tensor_type, self, numpy_result, other,
            self->has_conv, self->vars, self->require_grad, "AndBackward",
            self->graph, self->axis, self->dim, self->base);
        return (PyObject *)new_tensor;
    }
}

PyObject *tensor_xor(Tensor *self, PyObject *other){
    Tensor *tmp;
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        PyObject *numpy_result = PyNumber_Xor(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            PyErr_Print();
            PyErr_Clear();
            return NULL;
        }
        Tensor *new_tensor = new_Tensor(
            &Tensor_type, self, tmp, numpy_result, self->data,
            tmp->data, self->has_conv, self->vars, self->require_grad,
            "XorBackward", self->graph, self->axis, self->dim,
            self->base);
        return (PyObject *)new_tensor;
    }
    else
    {
        PyObject *numpy_result = PyNumber_Xor(self->data, other);

        if (numpy_result == NULL)
        {
            PyErr_Print();
            PyErr_Clear();
            return NULL;
        }
        Tensor *new_tensor = new_Tensor_scalar(
            &Tensor_type, self, numpy_result, other,
            self->has_conv, self->vars, self->require_grad, "XorBackward",
            self->graph, self->axis, self->dim, self->base);
        return (PyObject *)new_tensor;
    }
}

PyObject *tensor_or(Tensor *self, PyObject *other){
    Tensor *tmp;
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        PyObject *numpy_result = PyNumber_Or(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            PyErr_Print();
            PyErr_Clear();
            return NULL;
        }
        Tensor *new_tensor = new_Tensor(
            &Tensor_type, self, tmp, numpy_result, self->data,
            tmp->data, self->has_conv, self->vars, self->require_grad,
            "OrBackward", self->graph, self->axis, self->dim,
            self->base);
        return (PyObject *)new_tensor;
    }
    else
    {
        PyObject *numpy_result = PyNumber_Or(self->data, other);

        if (numpy_result == NULL)
        {
            PyErr_Print();
            PyErr_Clear();
            return NULL;
        }
        Tensor *new_tensor = new_Tensor_scalar(
            &Tensor_type, self, numpy_result, other,
            self->has_conv, self->vars, self->require_grad, "OrBackward",
            self->graph, self->axis, self->dim, self->base);
        return (PyObject *)new_tensor;
    }
}

PyObject *tensor_bool(Tensor *self){
    PyObject *numpy_result = Py_IsTrue(self->data);

    if (numpy_result == NULL)
    {
        PyErr_Print();
        PyErr_Clear();
        return NULL;
    }
    Tensor *new_tensor = new_Tensor_scalar(
        &Tensor_type, self, numpy_result, NULL,
        self->has_conv, self->vars, self->require_grad, "BoolBackward",
        self->graph, self->axis, self->dim, self->base);
    return (PyObject *)new_tensor;
}

PyObject *tensor_int(Tensor *self){
    if (PyArray_SIZE((PyArrayObject*)self->data) > 1) {
        PyErr_SetString(PyExc_TypeError, "only size-1 arrays can be converted to Python scalars");
        return NULL;
    }
    PyObject *numpy_result = PyArray_Cast((PyArrayObject*)self->data, NPY_INT64);
    long long *data = (long long *)PyArray_DATA((PyArrayObject*)numpy_result);
    if (numpy_result == NULL || data == NULL)
    {
        PyErr_Print();
        PyErr_Clear();
        return NULL;
    }
    return PyLong_FromLongLong(data[0]);
}

PyObject *tensor_float(Tensor *self){
    if (PyArray_SIZE((PyArrayObject*)self->data) > 1) {
        PyErr_SetString(PyExc_TypeError, "only size-1 arrays can be converted to Python scalars");
        return NULL;
    }
    PyObject *numpy_result = PyArray_Cast((PyArrayObject*)self->data, NPY_FLOAT64);
    double *data = (double *)PyArray_DATA((PyArrayObject*)numpy_result);
    if (numpy_result == NULL || data == NULL)
    {
        PyErr_Print();
        PyErr_Clear();
        return NULL;
    }
    return PyFloat_FromDouble(data[0]);
}



