#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "tensor.h"
#include <omp.h>
#include "mkl.h"
#include "op.h"
#include "binary_func.h"

void add_uint8(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_uint8 *a_ptr = (npy_uint8 *)PyArray_DATA(a);
    npy_uint8 *b_ptr = (npy_uint8 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), PyArray_TYPE(a), 0);
    npy_uint8 *numpy_ptr = (npy_uint8 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = a_ptr[i] + b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void add_int8(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_int8 *a_ptr = (npy_int8 *)PyArray_DATA(a);
    npy_int8 *b_ptr = (npy_int8 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), PyArray_TYPE(a), 0);
    npy_int8 *numpy_ptr = (npy_int8 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = a_ptr[i] + b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void add_int16(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_int16 *a_ptr = (npy_int16 *)PyArray_DATA(a);
    npy_int16 *b_ptr = (npy_int16 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), PyArray_TYPE(a), 0);
    npy_int16 *numpy_ptr = (npy_int16 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = a_ptr[i] + b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void add_uint16(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_uint16 *a_ptr = (npy_uint16 *)PyArray_DATA(a);
    npy_uint16 *b_ptr = (npy_uint16 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), PyArray_TYPE(a), 0);
    npy_uint16 *numpy_ptr = (npy_uint16 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = a_ptr[i] + b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void add_int32(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_int32 *a_ptr = (npy_int32 *)PyArray_DATA(a);
    npy_int32 *b_ptr = (npy_int32 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), PyArray_TYPE(a), 0);
    npy_int32 *numpy_ptr = (npy_int32 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        numpy_ptr[i] = a_ptr[i] + b_ptr[i];
    }
    *result = (PyObject *)numpy_result;
}

void add_uint32(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_uint32 *a_ptr = (npy_uint32 *)PyArray_DATA(a);
    npy_uint32 *b_ptr = (npy_uint32 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), PyArray_TYPE(a), 0);
    npy_uint32 *numpy_ptr = (npy_uint32 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = a_ptr[i] + b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void add_int64(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_int64 *a_ptr = (npy_int64 *)PyArray_DATA(a);
    npy_int64 *b_ptr = (npy_int64 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), PyArray_TYPE(a), 0);
    npy_int64 *numpy_ptr = (npy_int64 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = (npy_int64)a_ptr[i] + (npy_int64)b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void add_uint64(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_uint64 *a_ptr = (npy_uint64 *)PyArray_DATA(a);
    npy_uint64 *b_ptr = (npy_uint64 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), PyArray_TYPE(a), 0);
    npy_uint64 *numpy_ptr = (npy_uint64 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = a_ptr[i] + b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void add_float32(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_float32 *a_ptr = (npy_float32 *)PyArray_DATA(a);
    npy_float32 *b_ptr = (npy_float32 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), PyArray_TYPE(a), 0);
    npy_float32 *numpy_ptr = (npy_float32 *)PyArray_DATA(numpy_result);
    vsAdd(size, (const npy_float32 *)a_ptr, (const npy_float32 *)b_ptr, numpy_ptr);
    *result = (PyObject *)numpy_result;
}

void add_float64(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_float64 *a_ptr = (npy_float64 *)PyArray_DATA(a);
    npy_float64 *b_ptr = (npy_float64 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), PyArray_TYPE(a), 0);
    npy_float64 *numpy_ptr = (npy_float64 *)PyArray_DATA(numpy_result);
    vsAdd(size, (const npy_float64 *)a_ptr, (const npy_float64 *)b_ptr, numpy_ptr);
    *result = (PyObject *)numpy_result;
}

void add_float128(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_longdouble *a_ptr = (npy_longdouble *)PyArray_DATA(a);
    npy_longdouble *b_ptr = (npy_longdouble *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), PyArray_TYPE(a), 0);
    npy_longdouble *numpy_ptr = (npy_longdouble *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = a_ptr[i] + b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void sub_int8(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_int8 *a_ptr = (npy_int8 *)PyArray_DATA(a);
    npy_int8 *b_ptr = (npy_int8 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), PyArray_TYPE(a), 0);
    npy_int8 *numpy_ptr = (npy_int8 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = a_ptr[i] - b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void sub_uint8(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_uint8 *a_ptr = (npy_uint8 *)PyArray_DATA(a);
    npy_uint8 *b_ptr = (npy_uint8 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), PyArray_TYPE(a), 0);
    npy_uint8 *numpy_ptr = (npy_uint8 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = a_ptr[i] - b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void sub_int16(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_int16 *a_ptr = (npy_int16 *)PyArray_DATA(a);
    npy_int16 *b_ptr = (npy_int16 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), PyArray_TYPE(a), 0);
    npy_int16 *numpy_ptr = (npy_int16 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = a_ptr[i] - b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void sub_uint16(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_uint16 *a_ptr = (npy_uint16 *)PyArray_DATA(a);
    npy_uint16 *b_ptr = (npy_uint16 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), PyArray_TYPE(a), 0);
    npy_uint16 *numpy_ptr = (npy_uint16 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = a_ptr[i] - b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void sub_int32(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_int32 *a_ptr = (npy_int32 *)PyArray_DATA(a);
    npy_int32 *b_ptr = (npy_int32 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), PyArray_TYPE(a), 0);
    npy_int32 *numpy_ptr = (npy_int32 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = a_ptr[i] - b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void sub_uint32(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_uint32 *a_ptr = (npy_uint32 *)PyArray_DATA(a);
    npy_uint32 *b_ptr = (npy_uint32 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), PyArray_TYPE(a), 0);
    npy_uint32 *numpy_ptr = (npy_uint32 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = a_ptr[i] - b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void sub_int64(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_int64 *a_ptr = (npy_int64 *)PyArray_DATA(a);
    npy_int64 *b_ptr = (npy_int64 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), PyArray_TYPE(a), 0);
    npy_int64 *numpy_ptr = (npy_int64 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = a_ptr[i] - b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void sub_uint64(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_uint64 *a_ptr = (npy_uint64 *)PyArray_DATA(a);
    npy_uint64 *b_ptr = (npy_uint64 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), PyArray_TYPE(a), 0);
    npy_uint64 *numpy_ptr = (npy_uint64 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = a_ptr[i] - b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void sub_float16(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_float16 *a_ptr = (npy_float16 *)PyArray_DATA(a);
    npy_float16 *b_ptr = (npy_float16 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), PyArray_TYPE(a), 0);
    npy_float16 *numpy_ptr = (npy_float16 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = a_ptr[i] - b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void sub_float32(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_float32 *a_ptr = (npy_float32 *)PyArray_DATA(a);
    npy_float32 *b_ptr = (npy_float32 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), PyArray_TYPE(a), 0);
    npy_float32 *numpy_ptr = (npy_float32 *)PyArray_DATA(numpy_result);
    vsSub(size, (const npy_float32 *)a_ptr, (const npy_float32 *)b_ptr, numpy_ptr);
    *result = (PyObject *)numpy_result;
}

void sub_float64(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_float64 *a_ptr = (npy_float64 *)PyArray_DATA(a);
    npy_float64 *b_ptr = (npy_float64 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), PyArray_TYPE(a), 0);
    npy_float64 *numpy_ptr = (npy_float64 *)PyArray_DATA(numpy_result);
    vsSub(size, (const npy_float64 *)a_ptr, (const npy_float64 *)b_ptr, numpy_ptr);
    *result = (PyObject *)numpy_result;
}

void sub_float128(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_longdouble *a_ptr = (npy_longdouble *)PyArray_DATA(a);
    npy_longdouble *b_ptr = (npy_longdouble *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), PyArray_TYPE(a), 0);
    npy_longdouble *numpy_ptr = (npy_longdouble *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = a_ptr[i] - b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void div_int8(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_int8 *a_ptr = (npy_int8 *)PyArray_DATA(a);
    npy_int8 *b_ptr = (npy_int8 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), NPY_FLOAT32, 0);
    npy_int8 *numpy_ptr = (npy_int8 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        if (b_ptr[i] == 0)
            numpy_ptr[i] = 0;
        else
            numpy_ptr[i] = (npy_int8)a_ptr[i] / (npy_int8)b_ptr[i];
    }
    *result = (PyObject *)numpy_result;
}

void div_uint8(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_uint8 *a_ptr = (npy_uint8 *)PyArray_DATA(a);
    npy_uint8 *b_ptr = (npy_uint8 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), NPY_FLOAT32, 0);
    npy_uint8 *numpy_ptr = (npy_uint8 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        if (b_ptr[i] == 0)
            numpy_ptr[i] = 0;
        else
            numpy_ptr[i] = a_ptr[i] / b_ptr[i];
    }
    *result = (PyObject *)numpy_result;
}

void div_int16(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_int16 *a_ptr = (npy_int16 *)PyArray_DATA(a);
    npy_int16 *b_ptr = (npy_int16 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), NPY_FLOAT32, 0);
    npy_int16 *numpy_ptr = (npy_int16 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        if (b_ptr[i] == 0)
            numpy_ptr[i] = 0;
        else
            numpy_ptr[i] = (npy_int16)a_ptr[i] / (npy_int16)b_ptr[i];
    }
    *result = (PyObject *)numpy_result;
}

void div_uint16(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_uint16 *a_ptr = (npy_uint16 *)PyArray_DATA(a);
    npy_uint16 *b_ptr = (npy_uint16 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), NPY_FLOAT32, 0);
    npy_uint16 *numpy_ptr = (npy_uint16 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        if (b_ptr[i] == 0)
            numpy_ptr[i] = 0;
        else
            numpy_ptr[i] = a_ptr[i] / b_ptr[i];
    }
    *result = (PyObject *)numpy_result;
}

void div_int32(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_int32 *a_ptr = (npy_int32 *)PyArray_DATA(a);
    npy_int32 *b_ptr = (npy_int32 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), NPY_FLOAT32, 0);
    npy_int32 *numpy_ptr = (npy_int32 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        if (b_ptr[i] == 0)
            numpy_ptr[i] = 0;
        else
            numpy_ptr[i] = (npy_int32)a_ptr[i] / (npy_int32)b_ptr[i];
    }
    *result = (PyObject *)numpy_result;
}

void div_uint32(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_uint32 *a_ptr = (npy_uint32 *)PyArray_DATA(a);
    npy_uint32 *b_ptr = (npy_uint32 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), NPY_FLOAT32, 0);
    npy_uint32 *numpy_ptr = (npy_uint32 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        if (b_ptr[i] == 0)
            numpy_ptr[i] = 0;
        else
            numpy_ptr[i] = a_ptr[i] / b_ptr[i];
    }
    *result = (PyObject *)numpy_result;
}

void div_int64(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_int64 *a_ptr = (npy_int64 *)PyArray_DATA(a);
    npy_int64 *b_ptr = (npy_int64 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), NPY_FLOAT64, 0);
    npy_int64 *numpy_ptr = (npy_int64 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        if (b_ptr[i] == 0)
            numpy_ptr[i] = 0;
        else
            numpy_ptr[i] = (npy_int64)a_ptr[i] / (npy_int64)b_ptr[i];
    }
    *result = (PyObject *)numpy_result;
}

void div_uint64(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_uint64 *a_ptr = (npy_uint64 *)PyArray_DATA(a);
    npy_uint64 *b_ptr = (npy_uint64 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), NPY_FLOAT32, 0);
    npy_uint64 *numpy_ptr = (npy_uint64 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        if (b_ptr[i] == 0)
            numpy_ptr[i] = 0;
        else
            numpy_ptr[i] = a_ptr[i] / b_ptr[i];
    }
    *result = (PyObject *)numpy_result;
}

void div_float32(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_float32 *a_ptr = (npy_float32 *)PyArray_DATA(a);
    npy_float32 *b_ptr = (npy_float32 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), NPY_FLOAT32, 0);
    npy_float32 *numpy_ptr = (npy_float32 *)PyArray_DATA(numpy_result);
    vsDiv(size, (const npy_float32 *)a_ptr, (const npy_float32 *)b_ptr, numpy_ptr);
    *result = (PyObject *)numpy_result;
}

void div_float64(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_float64 *a_ptr = (npy_float64 *)PyArray_DATA(a);
    npy_float64 *b_ptr = (npy_float64 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), NPY_FLOAT64, 0);
    npy_float64 *numpy_ptr = (npy_float64 *)PyArray_DATA(numpy_result);
    vdDiv(size, (const npy_float64 *)a_ptr, (const npy_float64 *)b_ptr, numpy_ptr);
    *result = (PyObject *)numpy_result;
}

void div_float128(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_longdouble *a_ptr = (npy_longdouble *)PyArray_DATA(a);
    npy_longdouble *b_ptr = (npy_longdouble *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), NPY_FLOAT64, 0);
    npy_longdouble *numpy_ptr = (npy_longdouble *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        if (b_ptr[i] == 0)
            numpy_ptr[i] = 0;
        else
            numpy_ptr[i] = (npy_int64)a_ptr[i] / (npy_int64)b_ptr[i];
    }
    *result = (PyObject *)numpy_result;
}

void mul_int8(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_int8 *a_ptr = (npy_int8 *)PyArray_DATA(a);
    npy_int8 *b_ptr = (npy_int8 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), NPY_INT8, 0);
    npy_int8 *numpy_ptr = (npy_int8 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = (npy_int8)a_ptr[i] * (npy_int8)b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void mul_uint8(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_uint8 *a_ptr = (npy_uint8 *)PyArray_DATA(a);
    npy_uint8 *b_ptr = (npy_uint8 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), NPY_INT8, 0);
    npy_uint8 *numpy_ptr = (npy_uint8 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = a_ptr[i] * b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void mul_int16(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_int16 *a_ptr = (npy_int16 *)PyArray_DATA(a);
    npy_int16 *b_ptr = (npy_int16 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), NPY_INT16, 0);
    npy_int16 *numpy_ptr = (npy_int16 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = (npy_int16)a_ptr[i] * (npy_int16)b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void mul_uint16(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_uint16 *a_ptr = (npy_uint16 *)PyArray_DATA(a);
    npy_uint16 *b_ptr = (npy_uint16 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), NPY_INT8, 0);
    npy_uint16 *numpy_ptr = (npy_uint16 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = a_ptr[i] * b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void mul_int32(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_int32 *a_ptr = (npy_int32 *)PyArray_DATA(a);
    npy_int32 *b_ptr = (npy_int32 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), NPY_INT32, 0);
    npy_int32 *numpy_ptr = (npy_int32 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = (npy_int32)a_ptr[i] * (npy_int32)b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void mul_uint32(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_uint32 *a_ptr = (npy_uint32 *)PyArray_DATA(a);
    npy_uint32 *b_ptr = (npy_uint32 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), NPY_INT8, 0);
    npy_uint32 *numpy_ptr = (npy_uint32 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = a_ptr[i] * b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void mul_int64(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_int64 *a_ptr = (npy_int64 *)PyArray_DATA(a);
    npy_int64 *b_ptr = (npy_int64 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), NPY_INT64, 0);
    npy_int64 *numpy_ptr = (npy_int64 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = (npy_int64)a_ptr[i] * (npy_int64)b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void mul_uint64(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_uint64 *a_ptr = (npy_uint64 *)PyArray_DATA(a);
    npy_uint64 *b_ptr = (npy_uint64 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), NPY_INT8, 0);
    npy_uint64 *numpy_ptr = (npy_uint64 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = a_ptr[i] * b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void mul_float32(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_float32 *a_ptr = (npy_float32 *)PyArray_DATA(a);
    npy_float32 *b_ptr = (npy_float32 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), NPY_FLOAT32, 0);
    npy_float32 *numpy_ptr = (npy_float32 *)PyArray_DATA(numpy_result);
    vsMul(size, (const npy_float32 *)a_ptr, (const npy_float32 *)b_ptr, numpy_ptr);
    *result = (PyObject *)numpy_result;
}

void mul_float64(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_float64 *a_ptr = (npy_float64 *)PyArray_DATA(a);
    npy_float64 *b_ptr = (npy_float64 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), NPY_FLOAT64, 0);
    npy_float64 *numpy_ptr = (npy_float64 *)PyArray_DATA(numpy_result);
    vdMul(size, (const npy_float64 *)a_ptr, (const npy_float64 *)b_ptr, numpy_ptr);
    *result = (PyObject *)numpy_result;
}

void mul_longdouble(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_longdouble *a_ptr = (npy_longdouble *)PyArray_DATA(a);
    npy_longdouble *b_ptr = (npy_longdouble *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), NPY_LONGDOUBLE, 0);
    npy_longdouble *numpy_ptr = (npy_longdouble *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = (npy_longdouble)a_ptr[i] * (npy_longdouble)b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void mod_int8(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_int8 *a_ptr = (npy_int8 *)PyArray_DATA(a);
    npy_int8 *b_ptr = (npy_int8 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), NPY_INT8, 0);
    npy_int8 *numpy_ptr = (npy_int8 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = (npy_int8)a_ptr[i] % (npy_int8)b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void mod_uint8(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_uint8 *a_ptr = (npy_uint8 *)PyArray_DATA(a);
    npy_uint8 *b_ptr = (npy_uint8 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), NPY_INT8, 0);
    npy_uint8 *numpy_ptr = (npy_uint8 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = a_ptr[i] % b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void mod_int16(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_int16 *a_ptr = (npy_int16 *)PyArray_DATA(a);
    npy_int16 *b_ptr = (npy_int16 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), NPY_INT16, 0);
    npy_int16 *numpy_ptr = (npy_int16 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = a_ptr[i] % b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void mod_uint16(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_uint16 *a_ptr = (npy_uint16 *)PyArray_DATA(a);
    npy_uint16 *b_ptr = (npy_uint16 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), NPY_INT8, 0);
    npy_uint16 *numpy_ptr = (npy_uint16 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = a_ptr[i] % b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void mod_int32(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_int32 *a_ptr = (npy_int32 *)PyArray_DATA(a);
    npy_int32 *b_ptr = (npy_int32 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), NPY_INT32, 0);
    npy_int32 *numpy_ptr = (npy_int32 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = a_ptr[i] % b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void mod_uint32(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_uint32 *a_ptr = (npy_uint32 *)PyArray_DATA(a);
    npy_uint32 *b_ptr = (npy_uint32 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), NPY_INT8, 0);
    npy_uint32 *numpy_ptr = (npy_uint32 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = a_ptr[i] * b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void mod_int64(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_int64 *a_ptr = (npy_int64 *)PyArray_DATA(a);
    npy_int64 *b_ptr = (npy_int64 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), NPY_INT64, 0);
    npy_int64 *numpy_ptr = (npy_int64 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = (npy_int64)a_ptr[i] % (npy_int64)b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void mod_uint64(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_uint64 *a_ptr = (npy_uint64 *)PyArray_DATA(a);
    npy_uint64 *b_ptr = (npy_uint64 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), NPY_INT8, 0);
    npy_uint64 *numpy_ptr = (npy_uint64 *)PyArray_DATA(numpy_result);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        numpy_ptr[i] = a_ptr[i] % b_ptr[i];
    *result = (PyObject *)numpy_result;
}

void mod_float32(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_float32 *a_ptr = (npy_float32 *)PyArray_DATA(a);
    npy_float32 *b_ptr = (npy_float32 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), NPY_FLOAT32, 0);
    npy_float32 *numpy_ptr = (npy_float32 *)PyArray_DATA(numpy_result);
    vsModf(size, (const npy_float32 *)a_ptr, (const npy_float32 *)b_ptr, numpy_ptr);
    *result = (PyObject *)numpy_result;
}

void mod_float64(PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    npy_float64 *a_ptr = (npy_float64 *)PyArray_DATA(a);
    npy_float64 *b_ptr = (npy_float64 *)PyArray_DATA(b);
    npy_intp size = PyArray_SIZE(a);
    PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), NPY_FLOAT64, 0);
    npy_float64 *numpy_ptr = (npy_float64 *)PyArray_DATA(numpy_result);
    vdModf(size, (const npy_float64 *)a_ptr, (const npy_float64 *)b_ptr, numpy_ptr);
    *result = (PyObject *)numpy_result;
}

BinaryFunc BinaryOp_OperationPicker(int npy_type, int operation)
{
    switch (npy_type)
    {
    case NPY_BOOL:
        return NULL;
    case NPY_BYTE:
        switch (operation)
        {
        case ADD:
            return add_int8;
        case SUB:
            return sub_int8;
        case MUL:
            return mul_int8;
        case DIV:
            return div_int8;
        case MOD:
            return mod_int8;
        }
    case NPY_UBYTE:
        switch (operation)
        {
        case ADD:
            return add_uint8;
        case SUB:
            return NULL;
        case MUL:
            return mul_uint8;
        case DIV:
            return div_uint8;
        case MOD:
            return mod_uint8;
        }
    case NPY_SHORT:
        switch (operation)
        {
        case ADD:
            return add_int16;
        case SUB:

            return sub_int16;
        case MUL:
            return mul_int16;
        case DIV:
            return div_int16;
        case MOD:
            return mod_int16;
        }
    case NPY_USHORT:
        switch (operation)
        {
        case ADD:
            return add_uint16;
        case SUB:
            return NULL;
        case MUL:
            return mul_uint16;
        case DIV:
            return div_uint16;
        case MOD:
            return mod_uint16;
        }
    case NPY_INT:
        switch (operation)
        {
        case ADD:
            return add_int32;
        case SUB:
            return sub_int32;
        case MUL:
            return mul_int32;
        case DIV:
            return div_int32;
        case MOD:
            return mod_int32;
        }
    case NPY_UINT:
        switch (operation)
        {
        case ADD:
            return add_uint32;
        case SUB:
            return NULL;
        case MUL:
            return mul_uint32;
        case DIV:
            return div_uint32;
        case MOD:
            return mod_uint32;
        }
    case NPY_LONG:
        switch (operation)
        {
        case ADD:
            return add_int32;
        case SUB:
            return sub_int32;
        case MUL:
            return mul_int32;
        case DIV:
            return div_int32;
        case MOD:
            return mod_int32;
        }
    case NPY_ULONG:
        switch (operation)
        {
        case ADD:
            return add_uint32;
        case SUB:
            return NULL;
        case MUL:
            return mul_uint32;
        case DIV:
            return div_uint32;
        case MOD:
            return mod_uint32;
        }
    case NPY_LONGLONG:
        switch (operation)
        {
        case ADD:
            return add_int64;
        case SUB:
            return sub_int64;
        case MUL:
            return mul_int64;
        case DIV:
            return div_int64;
        case MOD:
            return mod_int64;
        }
    case NPY_ULONGLONG:
        switch (operation)
        {
        case ADD:
            return add_uint64;
        case SUB:
            return NULL;
        case MUL:
            return mul_uint64;
        case DIV:
            return div_uint64;
        case MOD:
            return mod_uint64;
        }
    case NPY_FLOAT:
        switch (operation)
        {
        case ADD:
            return add_float32;
        case SUB:
            return sub_float32;
        case MUL:
            return mul_float32;
        case DIV:
            return div_float32;
        case MOD:
            return mod_float32;
        }
    case NPY_DOUBLE:
        switch (operation)
        {
        case ADD:
            return add_float64;
        case SUB:
            return sub_float64;
        case MUL:
            return mul_float64;
        case DIV:
            return div_float64;
        }
    case NPY_LONGDOUBLE:
        switch (operation)
        {
        case ADD:
            return add_float64;
        case SUB:
            return sub_float64;
        case MUL:
            return mul_float64;
        case DIV:
            return div_float64;
        }
    }
}