#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "tensor.h"
extern long TRACK;
extern jnp_method *JNP_METHOD;
extern Tensordot_Dict *TENSORDOT_DICT;

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
    if (grad == NULL)
    {
        *out = NULL;
        return;
    }
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

void check_jaxpr_shape(PyArrayObject *predict_data, PyObject *origin_data, PyObject **out, const char *error_msg, PyObject *tracker,
                       jnp_method *jnp_methods)
{

    PyArrayObject_fields *fields = (PyArrayObject_fields *)predict_data;
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
            *out = tracker;
            return;
        }
        else
        {
            uint8_t new_axis = 0;
            for (i = 0; i < NPY_MAXDIMS; i++)
            {
                if (dims1[i] != dims2[i])
                {
                    tracker = PyObject_CallFunctionObjArgs(jnp_methods->sum, tracker, PyLong_FromLong(i), NULL);
                    new_dims[new_axis] = 1;
                }
                else
                {
                    new_dims[new_axis] = dims1[i];
                }
                new_axis++;
            }
            PyObject *tuple = PyTuple_New(nd1);
            for (i = 0; i < nd1; i++)
            {
                PyTuple_SetItem(tuple, i, PyLong_FromLongLong(new_dims[i]));
            }
            tracker = PyObject_CallFunctionObjArgs(jnp_methods->reshape, tracker, tuple, NULL); // reshape to original shape
            Py_DECREF(tuple);
            Py_DECREF(predict_data);
            // if (!vaild_shape((PyArrayObject *)result, (PyArrayObject *)origin_data, error_msg))
            // {
            //     *out = NULL;
            //     return;
            // }
            *out = tracker;
            return;
        }
    }
    else
    {
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
                PyObject *num = PyLong_FromLong(i);
                tracker = PyObject_CallFunctionObjArgs(jnp_methods->sum, tracker, PyLong_FromLong(i), NULL);
                Py_DECREF(num);
            }
        }
        *out = tracker;
        return;
    }
}

void add_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2)
{
    Tensor *tmp1 = (Tensor *)self->x;
    Tensor *tmp2 = (Tensor *)self->y;
    if (TRACK)
    {
        *out1 = PyObject_CallFunctionObjArgs(JNP_METHOD->copy, grad, NULL);
        *out2 = PyObject_CallFunctionObjArgs(JNP_METHOD->copy, grad, NULL);
        return;
    }
    if (!vaild_shape((PyArrayObject *)grad, (PyArrayObject *)self->data, "grad shape not equal to previous output shape"))
    {
        *out1 = NULL;
        *out2 = NULL;
        return;
    }
    PyArrayObject *tmp = (PyArrayObject *)grad;
    check_shape((PyArrayObject *)PyArray_Copy(tmp), tmp1->data, out1, "grad1 shape not equal to previous data shape in addbackward");
    check_shape((PyArrayObject *)PyArray_Copy(tmp), tmp2->data, out2, "grad2 shape not equal to previous data shape in addbackward");
    if (*out1 == NULL || *out2 == NULL)
    {
        *out1 = NULL;
        *out2 = NULL;
    }
};

void sub_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2)
{
    Tensor *tmp1 = (Tensor *)self->x;
    Tensor *tmp2 = (Tensor *)self->y;
    if (TRACK)
    {
        *out1 = PyObject_CallFunctionObjArgs(JNP_METHOD->copy, grad, NULL);
        *out2 = PyObject_CallFunctionObjArgs(JNP_METHOD->negative, grad, NULL);
        return;
    }
    PyArrayObject *tmp = (PyArrayObject *)grad;
    if (!vaild_shape(tmp, (PyArrayObject *)self->data, "grad shape not equal to previous output shape"))
    {
        *out1 = NULL;
        *out2 = NULL;
        return;
    }
    check_shape((PyArrayObject *)PyArray_Copy(tmp), tmp1->data, out1, "grad1 shape not equal to previous data shape in subbackward");
    check_shape((PyArrayObject *)PyNumber_Negative(grad), tmp2->data, out2, "grad2 shape not equal to previous data shape in subbackward");
    if (*out1 == NULL || *out2 == NULL)
    {
        *out1 = NULL;
        *out2 = NULL;
    }
};

void mul_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2)
{
    Tensor *tmp1 = (Tensor *)self->x;
    Tensor *tmp2 = (Tensor *)self->y;
    if (TRACK)
    {
        PyObject *predict1 = PyNumber_Multiply(self->data, tmp2->data);
        PyObject *predict2 = PyNumber_Multiply(self->data, tmp1->data);
        check_jaxpr_shape((PyArrayObject *)predict1, tmp1->data, out1, "grad1 shape not equal to previous data shape in mulbackward",
                          PyNumber_Multiply(grad, tmp2->data), JNP_METHOD);
        check_jaxpr_shape((PyArrayObject *)predict2, tmp2->data, out2, "grad1 shape not equal to previous data shape in mulbackward",
                          PyNumber_Multiply(grad, tmp1->data), JNP_METHOD);
        return;
    }
    if (!vaild_shape((PyArrayObject *)grad, (PyArrayObject *)self->data, "grad shape not equal to previous output shape"))
    {
        *out1 = NULL;
        *out2 = NULL;
        return;
    }
    check_shape((PyArrayObject *)PyNumber_Multiply(grad, tmp2->data), tmp1->data, out1, "grad1 shape not equal to previous data shape in mulbackward");
    check_shape((PyArrayObject *)PyNumber_Multiply(grad, tmp1->data), tmp2->data, out2, "grad2 shape not equal to previous data shape in mulbackward");
    if (*out1 == NULL || *out2 == NULL)
    {
        *out1 = NULL;
        *out2 = NULL;
    }
};

void div_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2)
{
    DEBUG_PRINT("div backward start\n")
    Tensor *tmp1 = (Tensor *)self->x;
    Tensor *tmp2 = (Tensor *)self->y;
    if (TRACK)
    {
        DEBUG_PRINT("track div backward\n");
        *out1 = PyNumber_TrueDivide(grad, tmp2->data);
        PyObject *two = PyLong_FromLong(2);
        PyObject *midle = PyNumber_Power(tmp2->data, two, Py_None);
        PyObject *midle2 = PyNumber_Negative(tmp1->data);
        PyObject *tmp = PyNumber_TrueDivide(midle2, midle);
        *out2 = PyNumber_Multiply(grad, tmp);
        Py_DECREF(tmp);
        Py_DECREF(midle);
        Py_DECREF(midle2);
        Py_DECREF(two);
        return;
    }
    if (!vaild_shape((PyArrayObject *)grad, (PyArrayObject *)self->data, "grad shape not equal to previous output shape"))
    {
        *out1 = NULL;
        *out2 = NULL;
        return;
    }
    DEBUG_PRINT("div backward valid shape done\n")
    PyObject *two = PyLong_FromLong(2);
    PyObject *midle = PyNumber_Power(tmp2->data, two, Py_None);
    PyObject *midle2 = PyNumber_Negative(tmp1->data);
    PyObject *tmp = PyNumber_TrueDivide(midle2, midle);
    DEBUG_PRINT("div backward calculated data\n")
    DEBUG_PRINT("div backward decref done\n")
    PyObject *grad1 = PyNumber_TrueDivide(grad, tmp2->data);
    PyObject *grad2 = PyNumber_Multiply(grad, tmp);
    check_shape((PyArrayObject *)grad1, tmp1->data, out1, "grad1 shape not equal to previous data shape in divbackward");
    check_shape((PyArrayObject *)grad2, tmp2->data, out2, "grad2 shape not equal to previous data shape in divbackward");
    DEBUG_PRINT("div backward check shape done\n")
    Py_DECREF(tmp);
    Py_DECREF(midle);
    Py_DECREF(midle2);
    Py_DECREF(two);
    if (*out1 == NULL || *out2 == NULL)
    {
        *out1 = NULL;
        *out2 = NULL;
    }
};

void matmul_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2)
{
    Tensor *tmp1 = (Tensor *)self->y;
    Tensor *tmp3 = (Tensor *)self->x;
    if (!TRACK && !vaild_shape((PyArrayObject *)grad, (PyArrayObject *)self->data, "grad shape not equal to previous output shape"))
    {
        *out1 = NULL;
        *out2 = NULL;
        return;
    }
    PyObject *transposed1 = NULL;
    PyObject *transposed2 = NULL;
    PyArrayObject *tmp2 = (PyArrayObject *)tmp1->data;
    PyArrayObject_fields *fields = (PyArrayObject_fields *)tmp2;
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
    if (TRACK)
    {
        *out1 = PyNumber_MatrixMultiply(grad, transposed1);
        *out2 = PyNumber_MatrixMultiply(transposed2, grad);
        return;
    }
    free(dims);
    check_shape((PyArrayObject *)PyNumber_MatrixMultiply(grad, transposed1), tmp3->data, out1, "grad1 shape not equal to previous data shape in matmulbackward");
    check_shape((PyArrayObject *)PyNumber_MatrixMultiply(transposed2, grad), tmp1->data, out2, "grad2 shape not equal to previous data shape in matmulbackward");
    if (*out1 == NULL || *out2 == NULL)
    {
        *out1 = NULL;
        *out2 = NULL;
    }
};

void negative_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2)
{
    Tensor *tmp1 = (Tensor *)self->x;
    if (TRACK)
    {
        *out1 = PyNumber_Multiply(grad, self->y);
        *out2 = NULL;
        return;
    }
    check_shape((PyArrayObject *)PyNumber_Multiply(grad, self->y), tmp1->data, out1, "grad1 shape not equal to previous data shape in negativebackward");
    *out2 = NULL;
};

void tensordot_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **out2)
{
    PyObject *reshaped = NULL, *tmp1 = NULL, *tmp2 = NULL, *at_grad = NULL, *bt_grad = NULL,
             *at_grad_reshaped = NULL, *bt_grad_reshaped = NULL, *a_transposed = NULL, *b_transposed = NULL;
    DEBUG_PRINT("tensordot backward start\n");
    Tensordot_Metadata *metadata = get_tensordot_data(self);
    if (metadata == NULL)
    {
        DEBUG_PRINT("metadata is null\n");
        *out = NULL;
        *out2 = NULL;
        return;
    }
    if (TRACK)
    {
        DEBUG_PRINT("tracking tensordot\n");
        int i;
        PyObject *matmul_result_shape = PyTuple_New(metadata->matmul_result_shape.len);
#ifdef DEBUG
        printf("reshaping :");
        printf("(");
        for (i = 0; i < metadata->matmul_result_shape.len; i++)
        {
            printf("%lld ", metadata->matmul_result_shape.ptr[i]);
        }
        printf(")\n");
#endif
        for (i = 0; i < metadata->matmul_result_shape.len; i++)
        {
            PyTuple_SetItem(matmul_result_shape, i, PyLong_FromLongLong(metadata->matmul_result_shape.ptr[i]));
        }
        reshaped = PyObject_CallFunctionObjArgs(JNP_METHOD->reshape, grad, matmul_result_shape, NULL);
        Py_DECREF(matmul_result_shape);
#ifdef DEBUG
        printf("reshaped:\n");
        PyObject_Print(reshaped, stdout, 0);
        printf("\n");
#endif
        if (reshaped == NULL)
        {
            DEBUG_PRINT("reshaped is null\n");
            *out = NULL;
            *out2 = NULL;
            return;
        }
        npy_intp t1_ptr_[] = {1, 0};
        PyArray_Dims t1 = {t1_ptr_, 2};
#ifdef DEBUG
        printf("transposing:\n");
        PyObject_Print(metadata->transposed_reshape_b, stdout, 0);
        printf("\n");
#endif
        tmp1 = PyArray_Transpose((PyArrayObject *)metadata->transposed_reshape_b, &t1);
        tmp2 = PyArray_Transpose((PyArrayObject *)metadata->transposed_reshape_a, &t1);
        if (tmp1 == NULL || tmp2 == NULL)
        {
            DEBUG_PRINT("tmp1 or tmp2 is null\n");
            *out = NULL;
            *out2 = NULL;
            return;
        }
        at_grad = PyNumber_MatrixMultiply(reshaped, tmp1);
        bt_grad = PyNumber_MatrixMultiply(tmp2, reshaped);
        Py_DECREF(reshaped);
        Py_DECREF(tmp1);
        Py_DECREF(tmp2);
        DEBUG_PRINT("matmul done\n");
        PyObject *newshape_a = PyTuple_New(metadata->newshape_a.len);
        for (i = 0; i < metadata->newshape_a.len; i++)
        {
            PyTuple_SetItem(newshape_a, i, PyLong_FromLongLong(metadata->newshape_a.ptr[i]));
        }
        PyObject *newshape_b = PyTuple_New(metadata->newshape_b.len);
        for (i = 0; i < metadata->newshape_b.len; i++)
        {
            PyTuple_SetItem(newshape_b, i, PyLong_FromLongLong(metadata->newshape_b.ptr[i]));
        }
        at_grad_reshaped = PyObject_CallFunctionObjArgs(JNP_METHOD->reshape, at_grad, newshape_a, NULL);
        bt_grad_reshaped = PyObject_CallFunctionObjArgs(JNP_METHOD->reshape, bt_grad, newshape_b, NULL);
        Py_DECREF(at_grad);
        Py_DECREF(bt_grad);
        Py_DECREF(newshape_a);
        Py_DECREF(newshape_b);
        DEBUG_PRINT("at_grad_reshaped && bt_grad_reshaped done\n");

        PyObject *tmp_a_ptr = PyTuple_New(metadata->newaxes_a.len);
        PyObject *tmp_b_ptr = PyTuple_New(metadata->newaxes_b.len);
        for (npy_intp i = 0; i < metadata->newaxes_a.len; i++)
        {
            PyTuple_SetItem(tmp_a_ptr, i, PyLong_FromLongLong(search_num(metadata->newaxes_a.ptr, metadata->newaxes_a.len, i)));
            PyTuple_SetItem(tmp_b_ptr, i, PyLong_FromLongLong(search_num(metadata->newaxes_b.ptr, metadata->newaxes_b.len, i)));
        }
        a_transposed = PyObject_CallFunctionObjArgs(JNP_METHOD->transpose, at_grad_reshaped, tmp_a_ptr, NULL);
        b_transposed = PyObject_CallFunctionObjArgs(JNP_METHOD->transpose, bt_grad_reshaped, tmp_b_ptr, NULL);
        Py_DECREF(tmp_a_ptr);
        Py_DECREF(tmp_b_ptr);
        Py_DECREF(at_grad_reshaped);
        Py_DECREF(bt_grad_reshaped);
        if (a_transposed == NULL || b_transposed == NULL)
        {
            DEBUG_PRINT("a_transposed or b_transposed is null");
            *out = NULL;
            *out2 = NULL;
            return;
        }
        *out = a_transposed;
        *out2 = b_transposed;
        DEBUG_PRINT("tensordot backward tracing done\n");
        return;
    }
    reshaped = PyArray_Newshape((PyArrayObject *)grad, &metadata->matmul_result_shape, NPY_CORDER);
    if (reshaped == NULL)
    {
        DEBUG_PRINT("reshaped is null");
        *out = NULL;
        *out2 = NULL;
        return;
    }
    npy_intp t1_ptr_[] = {1, 0};
    PyArray_Dims t1 = {t1_ptr_, 2};
#ifdef DEBUG
    for (uint8_t i = 0; i < 2; i++)
    {
        DEBUG_PRINT("%d ", t1_ptr_[i]);
    }
    DEBUG_PRINT("\n");
    PyObject_Print((PyObject *)metadata->transposed_reshape_b, stdout, 0);
    DEBUG_PRINT("\n");
    PyObject_Print((PyObject *)metadata->transposed_reshape_a, stdout, 0);
    DEBUG_PRINT("\n");
#endif
    tmp1 = PyArray_Transpose((PyArrayObject *)metadata->transposed_reshape_b, &t1);
    tmp2 = PyArray_Transpose((PyArrayObject *)metadata->transposed_reshape_a, &t1);
    if (tmp1 == NULL || tmp2 == NULL)
    {
        DEBUG_PRINT("tmp1 or tmp2 is null");
        *out = NULL;
        *out2 = NULL;
        return;
    }
    at_grad = PyNumber_MatrixMultiply(reshaped, tmp1);
    bt_grad = PyNumber_MatrixMultiply(tmp2, reshaped);
    if (at_grad == NULL || bt_grad == NULL)
    {
        DEBUG_PRINT("at_grad or bt_grad is null");
        *out = NULL;
        *out2 = NULL;
        return;
    }
#ifdef DEBUG
    for (uint8_t i = 0; i < metadata->newshape_a.len; i++)
    {
        DEBUG_PRINT("%d ", metadata->newshape_a.ptr[i]);
    }
    DEBUG_PRINT("\n");
    PyObject_Print(at_grad, stdout, 0);
    DEBUG_PRINT("\n");
    for (uint8_t i = 0; i < metadata->newshape_b.len; i++)
    {
        DEBUG_PRINT("%d ", metadata->newshape_b.ptr[i]);
    }
    DEBUG_PRINT("\n");
    PyObject_Print(bt_grad, stdout, 0);
    DEBUG_PRINT("\n");
#endif
    at_grad_reshaped = PyArray_Newshape((PyArrayObject *)at_grad, &metadata->newshape_a, NPY_CORDER);
    bt_grad_reshaped = PyArray_Newshape((PyArrayObject *)bt_grad, &metadata->newshape_b, NPY_CORDER);
    if (at_grad_reshaped == NULL || bt_grad_reshaped == NULL)
    {
        DEBUG_PRINT("at_grad_reshaped or bt_grad_reshaped is null");
        *out = NULL;
        *out2 = NULL;
        return;
    }
#ifdef DEBUG
    for (uint8_t i = 0; i < metadata->newaxes_a.len; i++)
    {
        DEBUG_PRINT("%d ", metadata->newaxes_a.ptr[i]);
    }
    DEBUG_PRINT("\n");
    PyObject_Print(at_grad_reshaped, stdout, 0);
    DEBUG_PRINT("\n");
    for (uint8_t i = 0; i < metadata->newaxes_b.len; i++)
    {
        DEBUG_PRINT("%d ", metadata->newaxes_b.ptr[i]);
    }
    DEBUG_PRINT("\n");
    PyObject_Print(bt_grad_reshaped, stdout, 0);
    DEBUG_PRINT("\n");
#endif
    npy_intp *tmp_a_ptr = malloc(sizeof(npy_intp) * metadata->newaxes_a.len);
    PyArray_Dims tmp_a = {tmp_a_ptr, metadata->newaxes_a.len};
    npy_intp *tmp_b_ptr = malloc(sizeof(npy_intp) * metadata->newaxes_b.len);
    PyArray_Dims tmp_b = {tmp_b_ptr, metadata->newaxes_b.len};
    for (npy_intp i = 0; i < metadata->newaxes_a.len; i++)
    {
        tmp_a_ptr[i] = search_num(metadata->newaxes_a.ptr, metadata->newaxes_a.len, i);
        tmp_b_ptr[i] = search_num(metadata->newaxes_b.ptr, metadata->newaxes_b.len, i);
    }
    a_transposed = PyArray_Transpose((PyArrayObject *)at_grad_reshaped, &tmp_a);
    b_transposed = PyArray_Transpose((PyArrayObject *)bt_grad_reshaped, &tmp_b);
    free(tmp_a_ptr);
    free(tmp_b_ptr);
    if (a_transposed == NULL || b_transposed == NULL)
    {
        DEBUG_PRINT("a_transposed or b_transposed is null");
        *out = NULL;
        *out2 = NULL;
        return;
    }
    Py_DECREF(tmp1);
    Py_DECREF(tmp2);
    *out = a_transposed;
    *out2 = b_transposed;
    Py_DECREF(reshaped);
    Py_DECREF(at_grad_reshaped);
    Py_DECREF(bt_grad_reshaped);
    Py_DECREF(at_grad);
    Py_DECREF(bt_grad);
}