#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "type_convertor.h"

void int8_to_float32(PyArrayObject **array)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT, 0);
    float *array_data = (float *)PyArray_DATA(array_);
    npy_int8 *data = (npy_int8 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (float)data[i];
    }
    Py_DECREF(array);
    *array = array_;
}

void uint8_to_float32(PyArrayObject **array)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT, 0);
    float *array_data = (float *)PyArray_DATA(array_);
    npy_uint8 *data = (npy_uint8 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (float)data[i];
    }
    Py_DECREF(*array);
    *array = array_;
}

void int32_to_float32(PyArrayObject **array)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT, 0);
    float *array_data = (float *)PyArray_DATA(array_);
    npy_int32 *data = (npy_int32 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (float)data[i];
    }
    Py_DECREF(array);
    *array = array_;
}

void uint32_to_float32(PyArrayObject **array)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT, 0);
    float *array_data = (float *)PyArray_DATA(array_);
    npy_uint32 *data = (npy_uint32 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (float)data[i];
    }
    Py_DECREF(array);
    *array = array_;
}

void int64_to_float32(PyArrayObject **array)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT, 0);
    float *array_data = (float *)PyArray_DATA(array_);
    npy_int64 *data = (npy_int64 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (float)data[i];
    }
    Py_DECREF(array);
    *array = array_;
}

void uint64_to_float32(PyArrayObject **array)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT, 0);
    float *array_data = (float *)PyArray_DATA(array_);
    npy_uint64 *data = (npy_uint64 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (float)data[i];
    }
    Py_DECREF(array);
    *array = array_;
}

void int16_to_float32(PyArrayObject **array)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT, 0);
    float *array_data = (float *)PyArray_DATA(array_);
    npy_int16 *data = (npy_int16 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (float)data[i];
    }
    Py_DECREF(array);
    *array = array_;
}

void uint16_to_float32(PyArrayObject **array)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT, 0);
    float *array_data = (float *)PyArray_DATA(array_);
    npy_uint16 *data = (npy_uint16 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (float)data[i];
    }
    Py_DECREF(array);
    *array = array_;
}

void float64_to_float32(PyArrayObject **array)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT, 0);
    float *array_data = (float *)PyArray_DATA(array_);
    npy_float64 *data = (npy_float64 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (float)data[i];
    }
    Py_DECREF(array);
    *array = array_;
}

void float16_to_float32(PyArrayObject **array)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT, 0);
    float *array_data = (float *)PyArray_DATA(array_);
    npy_float16 *data = (npy_float16 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (float)data[i];
    }
    Py_DECREF(array);
    *array = array_;
}

/*to float64*/
void int8_to_float64(PyArrayObject **array)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_DOUBLE, 0);
    npy_float64 *array_data = (npy_float64 *)PyArray_DATA(array_);
    npy_int8 *data = (npy_int8 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_float64)data[i];
    }
    Py_DECREF(array);
    *array = array_;
}

void uint8_to_float64(PyArrayObject **array)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_DOUBLE, 0);
    npy_float64 *array_data = (npy_float64 *)PyArray_DATA(array_);
    npy_uint8 *data = (npy_uint8 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_float64)data[i];
    }
    Py_DECREF(array);
    *array = array_;
}

void int32_to_float64(PyArrayObject **array)
{
    printf("int32_to_float64\n");
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT64, 0);
    npy_float64 *array_data = (npy_float64 *)PyArray_DATA(array_);
    npy_int32 *data = (npy_int32 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_float64)data[i];
    }
    Py_DECREF(array);
    *array = array_;
}

void uint32_to_float64(PyArrayObject **array)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_DOUBLE, 0);
    npy_float64 *array_data = (npy_float64 *)PyArray_DATA(array_);
    npy_uint32 *data = (npy_uint32 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_float64)data[i];
    }
    Py_DECREF(array);
    *array = array_;
}

void int64_to_float64(PyArrayObject **array)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_DOUBLE, 0);
    npy_float64 *array_data = (npy_float64 *)PyArray_DATA(array_);
    npy_int64 *data = (npy_int64 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_float64)data[i];
    }
    Py_DECREF(array);
    *array = array_;
}

void uint64_to_float64(PyArrayObject **array)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_DOUBLE, 0);
    npy_float64 *array_data = (npy_float64 *)PyArray_DATA(array_);
    npy_uint64 *data = (npy_uint64 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_float64)data[i];
    }
    Py_DECREF(array);
    *array = array_;
}

void int16_to_float64(PyArrayObject **array)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_DOUBLE, 0);
    npy_float64 *array_data = (npy_float64 *)PyArray_DATA(array_);
    npy_int16 *data = (npy_int16 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_float64)data[i];
    }
    Py_DECREF(array);
    *array = array_;
}

void uint16_to_float64(PyArrayObject **array)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_DOUBLE, 0);
    npy_float64 *array_data = (npy_float64 *)PyArray_DATA(array_);
    npy_uint16 *data = (npy_uint16 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_float64)data[i];
    }
    Py_DECREF(array);
    *array = array_;
}

void float32_to_float64(PyArrayObject **array)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_DOUBLE, 0);
    npy_float64 *array_data = (npy_float64 *)PyArray_DATA(array_);
    npy_float64 *data = (npy_float64 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_float64)data[i];
    }
    Py_DECREF(array);
    *array = array_;
}

void float16_to_float64(PyArrayObject **array)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_DOUBLE, 0);
    npy_float64 *array_data = (npy_float64 *)PyArray_DATA(array_);
    npy_float16 *data = (npy_float16 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_float64)data[i];
    }
    Py_DECREF(array);
    *array = array_;
}

ConvertFunc *type_converter(int type)
{
    switch (type)
    {
    case NPY_BOOL:
        return uint8_to_float32;
    case NPY_BYTE:
        return int8_to_float32;
    case NPY_UBYTE:
        return uint8_to_float32;
    case NPY_SHORT:
        return int16_to_float32;
    case NPY_USHORT:
        return uint16_to_float32;
    case NPY_INT:
        return int32_to_float32;
    case NPY_UINT:
        return uint32_to_float32;
    case NPY_LONG:
        return int32_to_float32;
    case NPY_ULONG:
        return uint32_to_float32;
    case NPY_LONGLONG:
        return int64_to_float32;
    case NPY_ULONGLONG:
        return uint64_to_float32;
    case NPY_FLOAT:
        return float64_to_float32;
    case NPY_HALF:
        return float16_to_float32;
    default:
        return NULL;
    }
}