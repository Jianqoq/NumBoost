#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "../tensor.h"
#include "../binary_ops/binary_op_impl.h"
#include "../numboost_api.h"
#include "ufunc_backward_def.h"
extern Array_Shape *ARRAY_SHAPE;
extern jnp_method *JNP_METHOD;
extern Zeros_Array_Dict *ZEROS_ARRAY_DICT;

double time_spent = 0.0;

void power_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    Tensor *tmp1 = (Tensor *)self->x;
    PyObject *power = get_power(self);
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
    *null = NULL;
};

void sin_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    Tensor *tmp1 = (Tensor *)self->x;
    *null = NULL;
    if (TRACK)
    {
        Tensor *tmp1 = (Tensor *)self->x;
        PyObject *cos = _cos_internal(tmp1->data, NULL);
        *out = PyNumber_Multiply(grad, cos);
        Py_DECREF(cos);
        return;
    }
    /*======================================== predict type =============================================================*/
    int result_type = elementwise_result_type(COS, PyArray_TYPE((PyArrayObject *)tmp1->data));
    result_type = binary_result_type(MUL, PyArray_TYPE((PyArrayObject *)grad), PyArray_DESCR((PyArrayObject *)grad)->elsize,
                                     result_type, type_2_size[result_type]);
    /*======================================== predict type =============================================================*/
    *out = sin_backward_fusefn[result_type]((PyArrayObject *)tmp1->data, (PyArrayObject *)grad);
};

void cos_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    Tensor *tmp1 = (Tensor *)self->x;
    *null = NULL;
    if (TRACK)
    {
        Tensor *tmp1 = (Tensor *)self->x;
        PyObject *sin = _sin_internal(tmp1->data, NULL);
        PyObject *neg = PyNumber_Negative(grad);
        *out = PyNumber_Multiply(neg, sin);
        Py_DECREF(sin);
        Py_DECREF(neg);
        return;
    }
    /*======================================== predict type =============================================================*/
    int result_type = elementwise_result_type(SIN, PyArray_TYPE((PyArrayObject *)tmp1->data));
    result_type = binary_result_type(MUL, PyArray_TYPE((PyArrayObject *)grad), PyArray_DESCR((PyArrayObject *)grad)->elsize,
                                     result_type, type_2_size[result_type]);
    /*======================================== predict type =============================================================*/
    *out = cos_backward_fusefn[result_type]((PyArrayObject *)tmp1->data, (PyArrayObject *)grad);
};

void tan_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    Tensor *tmp1 = (Tensor *)self->x;
    *null = NULL;
    if (TRACK)
    {
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
    }
    /*======================================== predict type =============================================================*/
    int result_type = elementwise_result_type(COS, PyArray_TYPE((PyArrayObject *)tmp1->data));
    result_type = binary_result_type(MUL, result_type, type_2_size[result_type], result_type, type_2_size[result_type]);
    result_type = binary_result_type(DIV, PyArray_TYPE((PyArrayObject *)grad), PyArray_DESCR((PyArrayObject *)grad)->elsize,
                                     result_type, type_2_size[result_type]);
    /*======================================== predict type =============================================================*/
    *out = tan_backward_fusefn[result_type]((PyArrayObject *)tmp1->data, (PyArrayObject *)grad);
};

void arcsin_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    Tensor *tmp1 = (Tensor *)self->x;
    *null = NULL;
    if (TRACK)
    {
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
        return;
    }
    /*======================================== predict type =============================================================*/
    /*here we directly use sub because square operation doesn't change the data type*/
    int result_type = binary_result_type(SUB, NPY_LONG, sizeof(npy_long), PyArray_TYPE((PyArrayObject *)tmp1->data),
                                         PyArray_DESCR((PyArrayObject *)tmp1->data)->elsize);
    result_type = binary_result_type(POW, result_type, type_2_size[result_type], NPY_FLOAT, sizeof(npy_float));
    result_type = binary_result_type(DIV, NPY_FLOAT, sizeof(npy_float), result_type, type_2_size[result_type]);
    result_type = binary_result_type(MUL, PyArray_TYPE((PyArrayObject *)grad), PyArray_DESCR((PyArrayObject *)grad)->elsize,
                                     result_type, type_2_size[result_type]);
    /*======================================== predict type =============================================================*/
    *out = arcsin_backward_fusefn[result_type]((PyArrayObject *)tmp1->data, (PyArrayObject *)grad);
};

void arccos_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    Tensor *tmp1 = (Tensor *)self->x;
    *null = NULL;
    if (TRACK)
    {
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
        return;
    }
    /*======================================== predict type =============================================================*/
    /*here we directly use sub because square operation doesn't change the data type*/
    int result_type = binary_result_type(SUB, NPY_LONG, sizeof(npy_long), PyArray_TYPE((PyArrayObject *)tmp1->data),
                                         PyArray_DESCR((PyArrayObject *)tmp1->data)->elsize);
    result_type = binary_result_type(POW, result_type, type_2_size[result_type], NPY_FLOAT, sizeof(npy_float));
    result_type = binary_result_type(DIV, NPY_LONG, sizeof(npy_long), result_type, type_2_size[result_type]);
    result_type = binary_result_type(MUL, PyArray_TYPE((PyArrayObject *)grad), PyArray_DESCR((PyArrayObject *)grad)->elsize,
                                     result_type, type_2_size[result_type]);
    /*======================================== predict type =============================================================*/
    *out = arccos_backward_fusefn[result_type]((PyArrayObject *)tmp1->data, (PyArrayObject *)grad);
    *null = NULL;
};

void arctan_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    Tensor *tmp1 = (Tensor *)self->x;
    *null = NULL;
    if (TRACK)
    {
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
        return;
    }
    /*======================================== predict type =============================================================*/
    /*here we directly use add because square operation doesn't change the data type*/
    int result_type = binary_result_type(ADD, NPY_LONG, sizeof(npy_long), PyArray_TYPE((PyArrayObject *)tmp1->data),
                                         PyArray_DESCR((PyArrayObject *)tmp1->data)->elsize);
    result_type = binary_result_type(DIV, NPY_LONG, sizeof(npy_long), result_type, type_2_size[result_type]);
    result_type = binary_result_type(MUL, PyArray_TYPE((PyArrayObject *)grad), PyArray_DESCR((PyArrayObject *)grad)->elsize,
                                     result_type, type_2_size[result_type]);
    /*======================================== predict type =============================================================*/
    *out = arctan_backward_fusefn[result_type]((PyArrayObject *)tmp1->data, (PyArrayObject *)grad);
    *null = NULL;
};

void sinh_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    Tensor *tmp1 = (Tensor *)self->x;
    *null = NULL;
    if (TRACK)
    {
        PyObject *cosh = _cosh_internal(tmp1->data, NULL);
        PyObject *result = PyNumber_Multiply(grad, cosh);
        *out = result;
        Py_DECREF(cosh);
        return;
    }
    /*======================================== predict type =============================================================*/
    int result_type = elementwise_result_type(COSH, PyArray_TYPE((PyArrayObject *)tmp1->data));
    result_type = binary_result_type(MUL, PyArray_TYPE((PyArrayObject *)grad), PyArray_DESCR((PyArrayObject *)grad)->elsize,
                                     result_type, type_2_size[result_type]);
    /*======================================== predict type =============================================================*/
    *out = sinh_backward_fusefn[result_type]((PyArrayObject *)tmp1->data, (PyArrayObject *)grad);
    *null = NULL;
};

void cosh_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    Tensor *tmp1 = (Tensor *)self->x;
    *null = NULL;
    if (TRACK)
    {
        PyObject *sinh = _sinh_internal(tmp1->data, NULL);
        PyObject *result = PyNumber_Multiply(grad, sinh);
        *out = result;
        Py_DECREF(sinh);
        return;
    }
    /*======================================== predict type =============================================================*/
    int result_type = elementwise_result_type(SINH, PyArray_TYPE((PyArrayObject *)tmp1->data));
    result_type = binary_result_type(MUL, PyArray_TYPE((PyArrayObject *)grad), PyArray_DESCR((PyArrayObject *)grad)->elsize,
                                     result_type, type_2_size[result_type]);
    /*======================================== predict type =============================================================*/
    *out = cosh_backward_fusefn[result_type]((PyArrayObject *)tmp1->data, (PyArrayObject *)grad);
    *null = NULL;
};

void tanh_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    *null = NULL;
    if (TRACK)
    {
        PyObject *self_val_square = PyNumber_Multiply(self->data, self->data);
        PyObject *one = PyFloat_FromDouble(1.0);
        PyObject *sub = PyNumber_Subtract(one, self_val_square);
        PyObject *result = PyNumber_Multiply(grad, sub);
        *out = result;
        Py_DECREF(self_val_square);
        Py_DECREF(one);
        Py_DECREF(sub);
        return;
    }
    /*======================================== predict type =============================================================*/
    int result_type = binary_result_type(SUB, NPY_FLOAT, type_2_size[NPY_FLOAT], PyArray_TYPE((PyArrayObject *)self->data), PyArray_DESCR((PyArrayObject *)self->data)->elsize);
    result_type = binary_result_type(MUL, PyArray_TYPE((PyArrayObject *)grad), PyArray_DESCR((PyArrayObject *)grad)->elsize,
                                     result_type, type_2_size[result_type]);
    /*======================================== predict type =============================================================*/
    *out = tanh_backward_fusefn[result_type]((PyArrayObject *)self->data, (PyArrayObject *)grad);
};

void arcsinh_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    Tensor *tmp1 = (Tensor *)self->x;
    *null = NULL;
    if (TRACK)
    {
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
        return;
    }
    /*======================================== predict type =============================================================*/
    /*here we directly use add because square operation doesn't change the data type*/
    int result_type = binary_result_type(ADD, NPY_LONG, sizeof(npy_long), PyArray_TYPE((PyArrayObject *)tmp1->data),
                                         PyArray_DESCR((PyArrayObject *)tmp1->data)->elsize);
    result_type = binary_result_type(SQRT, result_type, type_2_size[result_type], NPY_FLOAT, sizeof(npy_float));
    result_type = binary_result_type(DIV, PyArray_TYPE((PyArrayObject *)grad),
                                     PyArray_DESCR((PyArrayObject *)grad)->elsize, result_type, type_2_size[result_type]);
    /*======================================== predict type =============================================================*/
    *out = arcsinh_backward_fusefn[result_type]((PyArrayObject *)tmp1->data, (PyArrayObject *)grad);
};

void arccosh_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    Tensor *tmp1 = (Tensor *)self->x;
    *null = NULL;
    if (TRACK)
    {
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
        return;
    }
    /*======================================== predict type =============================================================*/
    /*here we directly use sub because square operation doesn't change the data type*/
    int result_type = binary_result_type(SUB, PyArray_TYPE((PyArrayObject *)tmp1->data),
                                         PyArray_DESCR((PyArrayObject *)tmp1->data)->elsize, NPY_LONG, sizeof(npy_long));
    result_type = binary_result_type(SQRT, result_type, type_2_size[result_type], NPY_FLOAT, sizeof(npy_float));
    result_type = binary_result_type(DIV, PyArray_TYPE((PyArrayObject *)grad),
                                     PyArray_DESCR((PyArrayObject *)grad)->elsize, result_type, type_2_size[result_type]);
    /*======================================== predict type =============================================================*/
    *out = arccosh_backward_fusefn[result_type]((PyArrayObject *)tmp1->data, (PyArrayObject *)grad);
};

void arctanh_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    Tensor *tmp1 = (Tensor *)self->x;
    *null = NULL;
    if (TRACK)
    {
        PyObject *square = PyNumber_Multiply(tmp1->data, tmp1->data);
        PyObject *one = PyFloat_FromDouble(1.0);
        PyObject *tmp = PyNumber_Subtract(one, square);
        PyObject *grad2 = PyNumber_TrueDivide(grad, tmp);
        *out = grad2;
        Py_DECREF(one);
        Py_DECREF(tmp);
        Py_DECREF(square);
        return;
    }
    /*======================================== predict type =============================================================*/
    /*here we directly use sub because square operation doesn't change the data type*/
    int result_type = binary_result_type(SUB, NPY_LONG, sizeof(npy_long), PyArray_TYPE((PyArrayObject *)tmp1->data),
                                         PyArray_DESCR((PyArrayObject *)tmp1->data)->elsize);
    result_type = binary_result_type(DIV, PyArray_TYPE((PyArrayObject *)grad),
                                     PyArray_DESCR((PyArrayObject *)grad)->elsize, result_type, type_2_size[result_type]);
    /*======================================== predict type =============================================================*/
    *out = arctanh_backward_fusefn[result_type]((PyArrayObject *)tmp1->data, (PyArrayObject *)grad);
};

void exp_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    Tensor *tmp1 = (Tensor *)self->x;
    *null = NULL;
    if (TRACK)
    {
        PyObject *exp = _exp_internal(tmp1->data, NULL);
        PyObject *result = PyNumber_Multiply(grad, exp);
        Py_DECREF(exp);
        *out = result;
        return;
    }
    /*======================================== predict type =============================================================*/
    int result_type = elementwise_result_type(EXP, PyArray_TYPE((PyArrayObject *)tmp1->data));
    result_type = binary_result_type(MUL, PyArray_TYPE((PyArrayObject *)grad), PyArray_DESCR((PyArrayObject *)grad)->elsize,
                                     result_type, type_2_size[result_type]);
    /*======================================== predict type =============================================================*/
    *out = exp_backward_fusefn[result_type]((PyArrayObject *)tmp1->data, (PyArrayObject *)grad);
};

void log_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    Tensor *tmp1 = (Tensor *)self->x;
    *null = NULL;
    if (TRACK)
    {
        PyObject *result = PyNumber_TrueDivide(grad, tmp1->data);
        *out = result;
        return;
    }
    int result_type = binary_result_type(DIV, PyArray_TYPE((PyArrayObject *)grad), PyArray_DESCR((PyArrayObject *)grad)->elsize,
                                         PyArray_TYPE((PyArrayObject *)tmp1->data), PyArray_DESCR((PyArrayObject *)tmp1->data)->elsize);
    *out = log_backward_fusefn[result_type]((PyArrayObject *)grad, (PyArrayObject *)tmp1->data);
    *null = NULL;
};

void log10_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    Tensor *tmp1 = (Tensor *)self->x;
    npy_intp dims[1] = {1};
    double data[1] = {10.0};
    PyObject *ten = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, data);
    PyObject *ln = _log_internal(ten, NULL);
    PyObject *mul = PyNumber_Multiply(tmp1->data, ln);
    PyObject *grad2 = PyNumber_TrueDivide(grad, mul);
    *out = grad2;
    *null = NULL;
    Py_DECREF(ten);
    Py_DECREF(ln);
    Py_DECREF(mul);
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
    npy_intp *shape = get_array_shape(self);
    npy_intp len = get_shape_len(self);
    if (len == -1)
    {
        *out = NULL;
        *null = NULL;
        return;
    }
    PyArray_Dims prev_shape = {shape, (int)len};
    if (TRACK)
    {
        PyObject *tuple = PyTuple_New(len);
        for (int i = 0; i < len; i++)
        {
            PyTuple_SetItem(tuple, i, PyLong_FromLongLong(shape[i]));
        }
        *out = PyObject_CallFunctionObjArgs(JNP_METHOD->reshape, grad, tuple, NULL);
        Py_DECREF(tuple);
        *null = NULL;
        return;
    }
    PyObject *result = PyArray_Newshape((PyArrayObject *)grad, &prev_shape, NPY_CORDER);
    *out = result;
    *null = NULL;
};

void transpose_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    DEBUG_PRINT("transpose_backward_fn start\n")
    npy_intp *axes = get_array_shape(self);
    npy_intp len = get_shape_len(self);
    if (len == -1)
    {
        *out = NULL;
        *null = NULL;
        return;
    }
    DEBUG_PRINT("got axes")
    DEBUG_FOR_LOOP(npy_intp i = 0; i < len; i++)
    {
        DEBUG_PRINT("%lld ", axes[i]);
    }
    DEBUG_PRINT("\n")
    npy_intp *new_axes = (npy_intp *)malloc(sizeof(npy_intp) * len);
    for (npy_intp i = 0; i < len; i++)
    {
        new_axes[i] = search_num(axes, len, i);
    }
    DEBUG_FOR_LOOP(npy_intp i = 0; i < len; i++){
        DEBUG_PRINT("%lld\n", new_axes[i])} PyArray_Dims prev_axes = {new_axes, (int)len};
    PyObject *result = PyArray_Transpose((PyArrayObject *)grad, &prev_axes);
    free(new_axes);
    *out = result;
    *null = NULL;
};

void slice_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null)
{
    npy_intp *origin_shape = NULL;
    PyObject *slice_obj = NULL, *zeros = NULL;
    int nd = 0;
    DEBUG_PRINT("slice_backward_fn start\n")
    get_slice_objs(self, &origin_shape, &slice_obj, &nd, &zeros); // zeros is ndarray
    DEBUG_PRINT("got slice objs\n")
    if (zeros == NULL || slice_obj == NULL || origin_shape == NULL)
    {
        *out = NULL;
        *null = NULL;
        return;
    }
    PyObject *sliced = PyObject_GetItem(zeros, slice_obj);
    DEBUG_PRINT("sliced refcnt: %d\n", sliced->ob_refcnt)
    DEBUG_PRINT("Inplace adding grad to sliced\n")
    PyObject *result = PyNumber_InPlaceAdd(sliced, grad);
    DEBUG_PRINT("sliced refcnt: %d\n", sliced->ob_refcnt)
    DEBUG_PRINT("Inplace adding grad to sliced done\n")
    *out = zeros;
    *null = NULL;
    Py_INCREF(zeros);
    Py_DECREF(sliced);
    Py_DECREF(result);
};