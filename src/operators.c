#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "operators.h"
#include "structmember.h"
#include <numpy/arrayobject.h>
#include "set_Tensor_properties.h"
#include <Python.h>

Tensor *
__new_Tensor(Tensor *tensor, PyObject *array, PyObject *to_y, const char *grad_fn)
{
    PyTypeObject *Tensor_Type = &Tensor_type;
    Tensor *self = (Tensor *)Tensor_Type->tp_alloc(Tensor_Type, 0);
    if (self != NULL)
    {
        if (tensor->require_grad)
        {
            Tensor_SetGradFn(self, grad_fn);
            Tensor_SetRequireGrad(self, true);
            Tensor_SetVars(self, tensor->vars);
            Tensor_SetX_without_init_value(self, (PyObject *)tensor);
            if (to_y != NULL)
                Tensor_SetY_without_init_value(self, to_y);
            else
                Tensor_SetY_without_init_value(self, Py_None);
        }
        else
        {
            Tensor_SetX_without_init_value(self, Py_None);
            Tensor_SetY_without_init_value(self, Py_None);
            Tensor_SetGradFn(self, "");
            Tensor_SetRequireGrad(self, false);
            Tensor_SetVars(self, 0);
        }
        PyObject *zero = PyLong_FromLong(0);
        Tensor_SetData_startwone_without_init(self, array);
        Tensor_SetHasConv(self, tensor->has_conv);
        Tensor_SetGraph_without_init_value(self, tensor->graph);
        Tensor_SetDim(self, tensor->dim);
        Tensor_SetAxis_without_init_value(self, tensor->axis);
        Tensor_SetBase_without_init_value(self, tensor->base);
        Tensor_SetGrad_without_init_value(self, zero);
        Py_DECREF(zero);
        return self;
    }
    else
    {
        return NULL;
    }
}

Tensor *
new_Tensor(Tensor *tensor, Tensor *tensor2, PyObject *data, const char *grad_fn)
{
    PyTypeObject *Tensor_Type = &Tensor_type;
    Tensor *self = (Tensor *)Tensor_Type->tp_alloc(Tensor_Type, 0);
    if (self != NULL)
    {
        if (tensor->require_grad || tensor2->require_grad)
        {
            Tensor_SetX_without_init_value(self, (PyObject *)tensor);
            Tensor_SetY_without_init_value(self, (PyObject *)tensor2);
            Tensor_SetGradFn(self, grad_fn);
            Tensor_SetRequireGrad(self, true);
            Tensor_SetVars(self, tensor->vars + tensor2->vars + 1);
        }
        else
        {
            Tensor_SetX_without_init_value(self, Py_None);
            Tensor_SetY_without_init_value(self, Py_None);
            Tensor_SetGradFn(self, "");
            Tensor_SetRequireGrad(self, false);
            Tensor_SetVars(self, 0);
        }
        PyObject *zero = PyLong_FromLong(0);
        Tensor_SetData_startwone_without_init(self, data);
        Tensor_SetHasConv(self, tensor->has_conv);
        Tensor_SetGraph_without_init_value(self, tensor->graph);
        Tensor_SetDim(self, tensor->dim);
        Tensor_SetAxis_without_init_value(self, tensor->axis);
        Tensor_SetBase_without_init_value(self, tensor->base);
        Tensor_SetGrad_without_init_value(self, zero);
        Py_DECREF(zero);
        return self;
    }
    else
    {
        return NULL;
    }
}

Tensor *
new_Tensor_scalar(Tensor *self, PyObject *data, PyObject *y, const char *grad_fn)
{
    Tensor *tensor;
    PyTypeObject *Tensor_Type = &Tensor_type;
    tensor = (Tensor *)Tensor_Type->tp_alloc(Tensor_Type, 0);
    if (tensor != NULL)
    {
        Tensor_SetData_startwone_without_init(tensor, data);
        if (self->require_grad)
        {
            Tensor_SetX_without_init_value(tensor, (PyObject *)self);
            Tensor_SetY_without_init_value(tensor, y);
            Tensor_SetRequireGrad(tensor, true);
            Tensor_SetGradFn(tensor, grad_fn);
            Tensor_SetVars(tensor, self->vars + 2);
        }
        else
        {
            Tensor_SetX_without_init_value(tensor, Py_None);
            Tensor_SetY_without_init_value(tensor, Py_None);
            Tensor_SetRequireGrad(tensor, false);
            Tensor_SetGradFn(tensor, grad_fn);
            Tensor_SetVars(tensor, 0);
        }
        PyObject *zero = PyLong_FromLong(0);
        Tensor_SetHasConv(tensor, self->has_conv);
        Tensor_SetGraph_without_init_value(tensor, self->graph);
        Tensor_SetDim(tensor, self->dim);
        Tensor_SetAxis_without_init_value(tensor, self->axis);
        Tensor_SetBase_without_init_value(tensor, self->base);
        Tensor_SetGrad_without_init_value(tensor, zero);
        Py_DECREF(zero);
        return tensor;
    }
    else
    {
        return NULL;
    }
}

Tensor *
new_Tensor_x(Tensor *self, PyObject *data, const char *grad_fn)
{
    Tensor *tensor;
    PyTypeObject *Tensor_Type = &Tensor_type;
    tensor = (Tensor *)Tensor_Type->tp_alloc(Tensor_Type, 0);
    if (tensor != NULL)
    {
        Tensor_SetData_startwone_without_init(tensor, data);
        if (self->require_grad)
        {
            Tensor_SetX_without_init_value(tensor, (PyObject *)self);
            Tensor_SetY_without_init_value(tensor, Py_None);
            Tensor_SetRequireGrad(tensor, true);
            Tensor_SetGradFn(tensor, grad_fn);
            Tensor_SetVars(tensor, self->vars + 1);
        }
        else
        {
            Tensor_SetX_without_init_value(tensor, Py_None);
            Tensor_SetY_without_init_value(tensor, Py_None);
            Tensor_SetRequireGrad(tensor, false);
            Tensor_SetGradFn(tensor, grad_fn);
            Tensor_SetVars(tensor, 0);
        }
        PyObject *zero = PyLong_FromLong(0);
        Tensor_SetHasConv(tensor, self->has_conv);
        Tensor_SetGraph_without_init_value(tensor, self->graph);
        Tensor_SetDim(tensor, self->dim);
        Tensor_SetAxis_without_init_value(tensor, self->axis);
        Tensor_SetBase_without_init_value(tensor, self->base);
        Tensor_SetGrad_without_init_value(tensor, zero);
        Py_DECREF(zero);
        return tensor;
    }
    else
    {
        return NULL;
    }
}

Tensor *
Tensor__new__(PyTypeObject *type, PyObject *data)
{
    Tensor *tensor;
    tensor = (Tensor *)type->tp_alloc(type, 0);
    if (tensor != NULL)
    {
        PyObject *zero = PyLong_FromLong(0);
        Tensor_SetData_startwone_without_init(tensor, data);
        Tensor_SetX_without_init_value(tensor, Py_None);
        Tensor_SetY_without_init_value(tensor, Py_None);
        Tensor_SetRequireGrad(tensor, false);
        Tensor_SetGradFn(tensor, "");
        Tensor_SetVars(tensor, 0);
        Tensor_SetHasConv(tensor, 0);
        Tensor_SetGraph_without_init_value(tensor, Py_None);
        Tensor_SetDim(tensor, 0);
        Tensor_SetAxis_without_init_value(tensor, Py_None);
        Tensor_SetBase_without_init_value(tensor, Py_None);
        Tensor_SetGrad_without_init_value(tensor, zero);
        Py_DECREF(zero);
        return tensor;
    }
    else
    {
        return NULL;
    }
}

Tensor *
tensor_add(Tensor *self, PyObject *other)
{
    Tensor *tmp;
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        PyObject *numpy_result = PyNumber_Add(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            return NULL;
        }
        Tensor *new_tensor = new_Tensor(self, tmp, numpy_result, "AddBackward");
        Py_DECREF(numpy_result);
        return new_tensor;
    }
    else
    {
        PyObject *numpy_result = PyNumber_Add(self->data, other);
        if (numpy_result == NULL)
        {
            return NULL;
        }
        Tensor *new_tensor = new_Tensor_scalar(self, numpy_result, other, "AddBackward");
        Py_DECREF(numpy_result);
        return new_tensor;
    }
}

Tensor *
tensor_iadd(Tensor *self, PyObject *other)
{
    Tensor *tmp;
    PyObject *numpy_result;
    if (self->require_grad)
    {
        PyErr_SetString(PyExc_RuntimeError, "Inplace operation can't set require_grad to true on a leaf variable");
        return NULL;
    }
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        numpy_result = PyNumber_InPlaceAdd(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            return NULL;
        }
        if (tmp->require_grad)
        {
            Tensor_SetX(self, (PyObject *)self);
            Tensor_SetY(self, other);
            self->grad_fn = "AddBackward";
        }
    }
    else
    {
        numpy_result = PyNumber_InPlaceAdd(self->data, other);
        if (numpy_result == NULL)
        {
            return NULL;
        }
    }
    Tensor_SetData(self, numpy_result);
    Py_INCREF(self);
    return self;
}

Tensor *
tensor_mul(Tensor *self, PyObject *other)
{
    Tensor *tmp;
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        PyObject *numpy_result = PyNumber_Multiply(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            return NULL;
        }
        Tensor *new_tensor = new_Tensor(self, tmp, numpy_result, "MulBackward");
        return new_tensor;
    }
    else
    {
        PyObject *numpy_result = PyNumber_Multiply(self->data, other);

        if (numpy_result == NULL)
        {
            return NULL;
        }
        Tensor *new_tensor = new_Tensor_scalar(self, numpy_result, other, "MulBackward");
        return new_tensor;
    }
}

Tensor *
tensor_imul(Tensor *self, PyObject *other)
{
    Tensor *tmp;
    PyObject *numpy_result;
    if (self->require_grad)
    {
        PyErr_SetString(PyExc_RuntimeError, "Inplace operation can't set require_grad to true on a leaf variable");
        return NULL;
    }
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        numpy_result = PyNumber_InPlaceMultiply(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            return NULL;
        }
        if (tmp->require_grad)
        {
            Tensor_SetX(self, (PyObject *)self);
            Tensor_SetY(self, other);
            self->grad_fn = "MulBackward";
        }
    }
    else
    {
        numpy_result = PyNumber_InPlaceMultiply(self->data, other);
        if (numpy_result == NULL)
        {
            return NULL;
        }
    }
    Tensor_SetData(self, numpy_result);
    Py_INCREF(self);
    return self;
}

Tensor *
tensor_div(Tensor *self, PyObject *other)
{
    Tensor *tmp;
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        PyObject *numpy_result = PyNumber_TrueDivide(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            return NULL;
        }
        Tensor *new_tensor = new_Tensor(self, tmp, numpy_result, "DivBackward");
        Py_DECREF(numpy_result);
        return new_tensor;
    }
    else
    {
        PyObject *numpy_result = PyNumber_TrueDivide(self->data, other);

        if (numpy_result == NULL)
        {
            return NULL;
        }
        Tensor *new_tensor = new_Tensor_scalar(self, numpy_result, other, "DivBackward");
        Py_DECREF(numpy_result);
        return new_tensor;
    }
}

Tensor *
tensor_idiv(Tensor *self, PyObject *other)
{
    Tensor *tmp;
    PyObject *numpy_result;
    if (self->require_grad)
    {
        PyErr_SetString(PyExc_RuntimeError, "Inplace operation can't set require_grad to true on a leaf variable");
        return NULL;
    }
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        numpy_result = PyNumber_InPlaceTrueDivide(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            return NULL;
        }
        if (tmp->require_grad)
        {
            Tensor_SetX(self, (PyObject *)self);
            Tensor_SetY(self, other);
            self->grad_fn = "DivBackward";
        }
    }
    else
    {
        numpy_result = PyNumber_InPlaceTrueDivide(self->data, other);
        if (numpy_result == NULL)
        {
            return NULL;
        }
    }
    Tensor_SetData(self, numpy_result);
    Py_INCREF(self);
    return self;
}

Tensor *
tensor_inegative(Tensor *self)
{
    if (self->require_grad)
    {
        PyErr_SetString(PyExc_RuntimeError, "Inplace operation can't set require_grad to true on a leaf variable");
        return NULL;
    }
    PyObject *negative_1 = PyLong_FromLong(-1);
    PyObject *numpy_result = PyNumber_InPlaceMultiply(self->data, negative_1);
    if (numpy_result == NULL)
    {
        Py_DECREF(negative_1);
        return NULL;
    }
    Tensor_SetData(self, numpy_result);
    Py_DECREF(negative_1);
    return self;
}

Tensor *
tensor_negative(Tensor *self) // need to check
{
    PyObject *negative_1 = PyLong_FromLong(-1);
    PyObject *numpy_result;
    if (!self->require_grad)
    {
        numpy_result = PyNumber_InPlaceMultiply(self->data, negative_1);
        Tensor_SetData(self, numpy_result);
        Py_DECREF(negative_1);
        Py_DECREF(numpy_result);
        return self;
    }
    else
    {
        numpy_result = PyNumber_Multiply(self->data, negative_1);

        if (numpy_result == NULL)
        {
            return NULL;
        }
        Tensor *new_tensor = new_Tensor_scalar(self, numpy_result, negative_1, "NegativeBackward");
        Py_DECREF(negative_1);
        Py_DECREF(numpy_result);
        return new_tensor;
    }
}

Tensor *
tensor_sub(Tensor *self, PyObject *other)
{
    Tensor *tmp;
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        PyObject *numpy_result = PyNumber_Subtract(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            return NULL;
        }
        Tensor *new_tensor = new_Tensor(self, tmp, numpy_result, "SubBackward");
        Py_DECREF(numpy_result);
        return new_tensor;
    }
    else
    {
        PyObject *numpy_result = PyNumber_Subtract(self->data, other);

        if (numpy_result == NULL)
        {
            return NULL;
        }
        Tensor *new_tensor = new_Tensor_scalar(self, numpy_result, other, "SubBackward");
        Py_DECREF(numpy_result);
        return new_tensor;
    }
}

Tensor *
tensor_isub(Tensor *self, PyObject *other)
{
    Tensor *tmp;
    PyObject *numpy_result;
    if (self->require_grad)
    {
        PyErr_SetString(PyExc_RuntimeError, "Inplace operation can't set require_grad to true on a leaf variable");
        return NULL;
    }
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        numpy_result = PyNumber_InPlaceSubtract(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            return NULL;
        }
        if (tmp->require_grad)
        {
            Tensor_SetX(self, (PyObject *)self);
            Tensor_SetY(self, other);
            self->grad_fn = "InplaceSubBackward";
        }
    }
    else
    {
        numpy_result = PyNumber_InPlaceSubtract(self->data, other);
        if (numpy_result == NULL)
        {
            return NULL;
        }
    }
    Tensor_SetData(self, numpy_result);
    Py_INCREF(self);
    return self;
}

Tensor *
tensor_pow(Tensor *self, PyObject *other)
{
    Tensor *temp;
    if (Py_TYPE(other) == &Tensor_type)
    {
        temp = (Tensor *)other;
        PyObject *numpy_result = PyNumber_Power(self->data, temp->data, Py_None);
        if (numpy_result == NULL)
        {
            return NULL;
        }
        Tensor *new_tensor = new_Tensor(self, temp, numpy_result, "PowBackward");
        Py_DECREF(numpy_result);
        return new_tensor;
    }
    else
    {
        PyObject *numpy_result = PyNumber_Power(self->data, other, Py_None);
        if (numpy_result == NULL)
        {
            return NULL;
        }
        Tensor *new_tensor = new_Tensor_scalar(self, numpy_result, other, "PowBackward");
        Py_DECREF(numpy_result);
        return new_tensor;
    }
}

Tensor *
tensor_ipow(Tensor *self, PyObject *other)
{
    PyObject *numpy_result;
    Tensor *tmp;
    if (self->require_grad)
    {
        PyErr_SetString(PyExc_RuntimeError, "Inplace operation can't set require_grad to true on a leaf variable");
        return NULL;
    }
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        numpy_result = PyNumber_InPlacePower(self->data, tmp->data, Py_None);
        if (numpy_result == NULL)
        {
            return NULL;
        }
        if (tmp->require_grad)
        {
            Tensor_SetX(self, (PyObject *)self);
            Tensor_SetY(self, other);
            self->grad_fn = "InplacePowerBackward";
        }
    }
    else
    {
        numpy_result = PyNumber_InPlacePower(self->data, other, Py_None);
        if (numpy_result == NULL)
        {
            return NULL;
        }
    }
    Tensor_SetData(self, numpy_result);
    Py_INCREF(self);
    return self;
}

Tensor *
tensor_matmul(Tensor *self, Tensor *other)
{
    PyObject *numpy_result = PyNumber_MatrixMultiply(self->data, other->data);
    if (numpy_result == NULL)
    {
        return NULL;
    }
    Tensor *new_tensor = new_Tensor(self, other, numpy_result, "MatMulBackward");
    Py_DECREF(numpy_result);
    return new_tensor;
}

Tensor *
tensor_imatmul(Tensor *self, Tensor *other)
{
    if (self->require_grad)
    {
        PyErr_SetString(PyExc_RuntimeError, "Inplace operation can't set require_grad to true on a leaf variable");
        return NULL;
    }
    PyObject *numpy_result =
        PyNumber_InPlaceMatrixMultiply(self->data, other->data);
    if (numpy_result == NULL)
    {
        return NULL;
    }
    if (other->require_grad)
    {
        Tensor_SetX(self, (PyObject *)self);
        Tensor_SetY(self, (PyObject *)other);
        self->grad_fn = "InplaceMatMulBackward";
    }
    Tensor_SetData(self, numpy_result);
    Py_INCREF(self);
    return self;
}

Tensor *
tensor_positive(Tensor *self)
{
    return self;
}

Tensor *
tensor_absolute(Tensor *self)
{
    PyObject *numpy_result = PyNumber_Absolute(self->data);

    if (numpy_result == NULL)
    {
        return NULL;
    }
    Tensor *new_tensor = new_Tensor_x(self, numpy_result, "");
    Py_DECREF(numpy_result);
    return new_tensor;
}

Tensor *
tensor_invert(Tensor *self)
{
    if (self->require_grad)
    {
        PyErr_SetString(PyExc_RuntimeError, "Logic invert operation auto backward not implemented yet");
        return NULL;
    }
    PyObject *numpy_result = PyNumber_Invert(self->data);
    if (numpy_result == NULL)
    {
        return NULL;
    }
    Tensor *new_tensor = new_Tensor_x(self, numpy_result, "");
    Py_DECREF(numpy_result);
    return new_tensor;
}

Tensor *
tensor_lshift(Tensor *self, PyObject *other)
{
    Tensor *tmp;
    if (self->require_grad)
    {
        PyErr_SetString(PyExc_RuntimeError, "shift operation auto backward not implemented yet");
        return NULL;
    }
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        PyObject *numpy_result = PyNumber_Lshift(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            return NULL;
        }
        Tensor *new_tensor = new_Tensor(self, tmp, numpy_result, "");
        Py_DECREF(numpy_result);
        return new_tensor;
    }
    else
    {
        PyObject *numpy_result = PyNumber_Lshift(self->data, other);

        if (numpy_result == NULL)
        {
            return NULL;
        }
        Tensor *new_tensor = new_Tensor_scalar(self, numpy_result, other, "");
        Py_DECREF(numpy_result);
        return new_tensor;
    }
}

Tensor *
tensor_ilshift(Tensor *self, PyObject *other)
{
    if (self->require_grad)
    {
        PyErr_SetString(PyExc_RuntimeError, "shift operation auto backward not implemented yet");
        return NULL;
    }
    if (Py_TYPE(other) == &Tensor_type)
    {
        Tensor *tmp = (Tensor *)other;
        PyObject *numpy_result = PyNumber_InPlaceLshift(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            return NULL;
        }
        Py_INCREF(self);
        Tensor_SetData(self, numpy_result);
        return self;
    }
    else
    {
        PyObject *numpy_result = PyNumber_InPlaceLshift(self->data, other);
        if (numpy_result == NULL)
        {
            return NULL;
        }
        Py_INCREF(self);
        Tensor_SetData(self, numpy_result);
        return self;
    }
}

Tensor *
tensor_rshift(Tensor *self, PyObject *other)
{
    Tensor *tmp;
    if (self->require_grad)
    {
        PyErr_SetString(PyExc_RuntimeError, "shift operation auto backward not implemented yet");
        return NULL;
    }
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        PyObject *numpy_result = PyNumber_Rshift(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            return NULL;
        }
        Tensor *new_tensor = new_Tensor(self, tmp, numpy_result, "");
        Py_DECREF(numpy_result);
        return new_tensor;
    }
    else
    {
        PyObject *numpy_result = PyNumber_Rshift(self->data, other);
        if (numpy_result == NULL)
        {
            return NULL;
        }
        Tensor *new_tensor = new_Tensor_scalar(self, numpy_result, other, "");
        Py_DECREF(numpy_result);
        return new_tensor;
    }
}

Tensor *
tensor_irshift(Tensor *self, PyObject *other)
{
    if (self->require_grad)
    {
        PyErr_SetString(PyExc_RuntimeError, "shift operation auto backward not implemented yet");
        return NULL;
    }
    if (Py_TYPE(other) == &Tensor_type)
    {
        Tensor *tmp = (Tensor *)other;
        PyObject *numpy_result = PyNumber_InPlaceRshift(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            return NULL;
        }
        Py_INCREF(self);
        Tensor_SetData(self, numpy_result);
        return self;
    }
    else
    {
        PyObject *numpy_result = PyNumber_InPlaceRshift(self->data, other);
        if (numpy_result == NULL)
        {
            return NULL;
        }
        Py_INCREF(self);
        Tensor_SetData(self, numpy_result);
        return self;
    }
}

Tensor *
tensor_and(Tensor *self, PyObject *other)
{
    Tensor *tmp;
    if (self->require_grad)
    {
        PyErr_SetString(PyExc_RuntimeError, "Logic operation is not differentiable");
        return NULL;
    }
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        PyObject *numpy_result = PyNumber_And(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            return NULL;
        }
        Tensor *new_tensor = new_Tensor(self, tmp, numpy_result, "");
        Py_DECREF(numpy_result);
        return new_tensor;
    }
    else
    {
        PyObject *numpy_result = PyNumber_Rshift(self->data, other);

        if (numpy_result == NULL)
        {
            return NULL;
        }
        Tensor *new_tensor = new_Tensor_scalar(self, numpy_result, other, "");
        Py_DECREF(numpy_result);
        return new_tensor;
    }
}

Tensor *
tensor_xor(Tensor *self, PyObject *other)
{
    Tensor *tmp;
    if (self->require_grad)
    {
        PyErr_SetString(PyExc_RuntimeError, "Logic operation is not differentiable");
        return NULL;
    }
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        PyObject *numpy_result = PyNumber_Xor(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            return NULL;
        }
        Tensor *new_tensor = new_Tensor(self, tmp, numpy_result, "");
        Py_DECREF(numpy_result);
        return new_tensor;
    }
    else
    {
        PyObject *numpy_result = PyNumber_Xor(self->data, other);

        if (numpy_result == NULL)
        {
            return NULL;
        }
        Tensor *new_tensor = new_Tensor_scalar(self, numpy_result, other, "");
        Py_DECREF(numpy_result);
        return new_tensor;
    }
}

Tensor *
tensor_or(Tensor *self, PyObject *other)
{
    Tensor *tmp;
    if (self->require_grad)
    {
        PyErr_SetString(PyExc_RuntimeError, "Logic operation is not differentiable");
        return NULL;
    }
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        PyObject *numpy_result = PyNumber_Or(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            return NULL;
        }
        Tensor *new_tensor = new_Tensor(self, tmp, numpy_result, "");
        Py_DECREF(numpy_result);
        return new_tensor;
    }
    else
    {
        PyObject *numpy_result = PyNumber_Or(self->data, other);

        if (numpy_result == NULL)
        {
            return NULL;
        }
        Tensor *new_tensor = new_Tensor_scalar(self, numpy_result, other, "OrBackward");
        Py_DECREF(numpy_result);
        return new_tensor;
    }
}

PyObject *
tensor_int(Tensor *self)
{
    if (PyArray_SIZE((PyArrayObject *)self->data) > 1)
    {
        PyErr_SetString(PyExc_TypeError, "only size-1 arrays can be converted to Python scalars");
        return NULL;
    }
    PyObject *numpy_result = PyArray_Cast((PyArrayObject *)self->data, NPY_INT64);
    long long *data = (long long *)PyArray_DATA((PyArrayObject *)numpy_result);
    if (numpy_result == NULL || data == NULL)
    {
        return NULL;
    }
    return PyLong_FromLongLong(data[0]);
}

PyObject *
tensor_float(Tensor *self)
{
    if (PyArray_SIZE((PyArrayObject *)self->data) > 1)
    {
        PyErr_SetString(PyExc_TypeError, "only size-1 arrays can be converted to Python scalars");
        return NULL;
    }
    PyObject *numpy_result = PyArray_Cast((PyArrayObject *)self->data, NPY_FLOAT64);
    double *data = (double *)PyArray_DATA((PyArrayObject *)numpy_result);
    if (numpy_result == NULL || data == NULL)
    {
        return NULL;
    }
    return PyFloat_FromDouble(data[0]);
}

Tensor *
tensor_remainder(Tensor *self, PyObject *other)
{
    Tensor *tmp;
    if (self->require_grad)
    {
        PyErr_SetString(PyExc_RuntimeError, "Remainder operation is not differentiable");
        return NULL;
    }
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        PyObject *numpy_result = PyNumber_Remainder(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            return NULL;
        }
        Tensor *new_tensor = new_Tensor(self, tmp, numpy_result, "");
        Py_DECREF(numpy_result);
        return new_tensor;
    }
    else
    {
        PyObject *numpy_result = PyNumber_Remainder(self->data, other);

        if (numpy_result == NULL)
        {
            return NULL;
        }
        Tensor *new_tensor = new_Tensor_scalar(self, numpy_result, other, "");
        Py_DECREF(numpy_result);
        return new_tensor;
    }
}

Tensor *
tensor_iand(Tensor *self, PyObject *other)
{
    Tensor *tmp;
    PyObject *numpy_result;
    if (self->require_grad)
    {
        PyErr_SetString(PyExc_RuntimeError, "Logic operation is not differentiable");
        return NULL;
    }
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        numpy_result = PyNumber_InPlaceAnd(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            return NULL;
        }
    }
    else
    {
        numpy_result = PyNumber_InPlaceAnd(self->data, other);
        if (numpy_result == NULL)
        {
            return NULL;
        }
    }
    self->grad_fn = "";
    Tensor_SetData(self, numpy_result);
    Py_INCREF(self);
    return self;
}

Tensor *
tensor_ior(Tensor *self, PyObject *other)
{
    Tensor *tmp;
    PyObject *numpy_result;
    if (self->require_grad)
    {
        PyErr_SetString(PyExc_RuntimeError, "Logic operation is not differentiable");
        return NULL;
    }
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        numpy_result = PyNumber_InPlaceOr(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            return NULL;
        }
    }
    else
    {
        numpy_result = PyNumber_InPlaceOr(self->data, other);
        if (numpy_result == NULL)
        {
            return NULL;
        }
    }
    self->grad_fn = "";
    Tensor_SetData(self, numpy_result);
    Py_INCREF(self);
    return self;
}

Tensor *
tensor_ixor(Tensor *self, PyObject *other)
{
    Tensor *tmp;
    PyObject *numpy_result;
    if (self->require_grad)
    {
        PyErr_SetString(PyExc_RuntimeError, "Logic operation is not differentiable");
        return NULL;
    }
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        numpy_result = PyNumber_InPlaceXor(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            return NULL;
        }
        Tensor_SetY(self, (PyObject *)tmp);
    }
    else
    {
        numpy_result = PyNumber_InPlaceXor(self->data, other);
        if (numpy_result == NULL)
        {
            return NULL;
        }
        Tensor_SetY(self, other);
    }
    self->grad_fn = "";
    Tensor_SetX(self, self->data);
    Tensor_SetData(self, numpy_result);
    Py_INCREF(self);
    return self;
}

Tensor *
tensor_divmod(Tensor *self, PyObject *other)
{
    Tensor *tmp;
    if (self->require_grad)
    {
        PyErr_SetString(PyExc_RuntimeError, "Divmod operation is not differentiable");
        return NULL;
    }
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        PyObject *numpy_result = PyNumber_Divmod(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            return NULL;
        }
        Tensor *new_tensor = new_Tensor(self, tmp, numpy_result, "");
        Py_DECREF(numpy_result);
        return new_tensor;
    }
    else
    {
        PyObject *numpy_result = PyNumber_Divmod(self->data, other);

        if (numpy_result == NULL)
        {
            return NULL;
        }
        Tensor *new_tensor = new_Tensor_scalar(self, numpy_result, other, "");
        Py_DECREF(numpy_result);
        return new_tensor;
    }
}

Tensor *
tensor_iremainder(Tensor *self, PyObject *other)
{
    Tensor *tmp;
    PyObject *numpy_result;
    if (self->require_grad)
    {
        PyErr_SetString(PyExc_RuntimeError, "Inplace remainder operation doesn't support auto backward");
        return NULL;
    }
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        numpy_result = PyNumber_InPlaceRemainder(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            return NULL;
        }
        Tensor_SetY(self, (PyObject *)tmp);
    }
    else
    {
        numpy_result = PyNumber_InPlaceRemainder(self->data, other);
        if (numpy_result == NULL)
        {
            return NULL;
        }
        Tensor_SetY(self, other);
    }
    Tensor_SetX(self, self->data);
    Py_INCREF(self);
    return self;
}

Tensor *
tensor_floordiv(Tensor *self, PyObject *other)
{
    Tensor *tmp;
    if (self->require_grad)
    {
        PyErr_SetString(PyExc_RuntimeError, "Floor divide operation is not differentiable");
        return NULL;
    }
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        PyObject *numpy_result = PyNumber_FloorDivide(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            return NULL;
        }
        Tensor *new_tensor = new_Tensor(self, tmp, numpy_result, "");
        Py_DECREF(numpy_result);
        return new_tensor;
    }
    else
    {
        PyObject *numpy_result = PyNumber_FloorDivide(self->data, other);

        if (numpy_result == NULL)
        {
            return NULL;
        }
        Tensor *new_tensor = new_Tensor_scalar(self, numpy_result, other, "");
        Py_DECREF(numpy_result);
        return new_tensor;
    }
}

Tensor *
tensor_ifloordiv(Tensor *self, PyObject *other)
{
    Tensor *tmp;
    PyObject *numpy_result;
    if (self->require_grad)
    {
        PyErr_SetString(PyExc_RuntimeError, "Floor divide operation is not differentiable");
        return NULL;
    }
    if (Py_TYPE(other) == &Tensor_type)
    {
        tmp = (Tensor *)other;
        numpy_result = PyNumber_InPlaceFloorDivide(self->data, tmp->data);
        if (numpy_result == NULL)
        {
            return NULL;
        }
    }
    else
    {
        numpy_result = PyNumber_InPlaceFloorDivide(self->data, other);
        if (numpy_result == NULL)
        {
            return NULL;
        }
    }
    Tensor_SetData(self, numpy_result);
    Py_INCREF(self);
    return self;
}