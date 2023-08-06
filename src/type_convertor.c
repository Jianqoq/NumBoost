#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "type_convertor.h"

void no_convert(PyArrayObject **array, PyArrayObject **result)
{
    *result = *array;
}

void int8_to_float32(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    float *array_data = (float *)PyArray_DATA(array_);
    npy_int8 *data = (npy_int8 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (float)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void uint8_to_float32(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    float *array_data = (float *)PyArray_DATA(array_);
    npy_uint8 *data = (npy_uint8 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (float)data[i];
    }
    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void int32_to_float32(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    float *array_data = (float *)PyArray_DATA(array_);
    npy_int32 *data = (npy_int32 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (float)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void uint32_to_float32(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    float *array_data = (float *)PyArray_DATA(array_);
    npy_uint32 *data = (npy_uint32 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        array_data[i] = (float)data[i];
    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void int64_to_float32(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    float *array_data = (float *)PyArray_DATA(array_);
    npy_int64 *data = (npy_int64 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (float)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void uint64_to_float32(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    float *array_data = (float *)PyArray_DATA(array_);
    npy_uint64 *data = (npy_uint64 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (float)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void int16_to_float32(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    float *array_data = (float *)PyArray_DATA(array_);
    npy_int16 *data = (npy_int16 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (float)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void uint16_to_float32(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    float *array_data = (float *)PyArray_DATA(array_);
    npy_uint16 *data = (npy_uint16 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (float)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void float64_to_float32(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    float *array_data = (float *)PyArray_DATA(array_);
    npy_float64 *data = (npy_float64 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        array_data[i] = (float)data[i];
    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void float128_to_float32(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    float *array_data = (float *)PyArray_DATA(array_);
    npy_longdouble *data = (npy_longdouble *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_float32)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void float16_to_float32(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    float *array_data = (float *)PyArray_DATA(array_);
    npy_float16 *data = (npy_float16 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = float16_cast_float32(data[i]);
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

/*to float64*/
void int8_to_float64(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_DOUBLE, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_float64 *array_data = (npy_float64 *)PyArray_DATA(array_);
    npy_int8 *data = (npy_int8 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        array_data[i] = (npy_float64)data[i];
    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void uint8_to_float64(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_DOUBLE, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_float64 *array_data = (npy_float64 *)PyArray_DATA(array_);
    npy_uint8 *data = (npy_uint8 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_float64)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void int32_to_float64(PyArrayObject **array, PyArrayObject **result)
{
    printf("int32_to_float64\n");
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT64, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_float64 *array_data = (npy_float64 *)PyArray_DATA(array_);
    npy_int32 *data = (npy_int32 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_float64)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void uint32_to_float64(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_DOUBLE, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_float64 *array_data = (npy_float64 *)PyArray_DATA(array_);
    npy_uint32 *data = (npy_uint32 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_float64)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void int64_to_float64(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_DOUBLE, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_float64 *array_data = (npy_float64 *)PyArray_DATA(array_);
    npy_int64 *data = (npy_int64 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_float64)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void uint64_to_float64(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_DOUBLE, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_float64 *array_data = (npy_float64 *)PyArray_DATA(array_);
    npy_uint64 *data = (npy_uint64 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_float64)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void int16_to_float64(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_DOUBLE, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_float64 *array_data = (npy_float64 *)PyArray_DATA(array_);
    npy_int16 *data = (npy_int16 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_float64)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void uint16_to_float64(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_DOUBLE, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_float64 *array_data = (npy_float64 *)PyArray_DATA(array_);
    npy_uint16 *data = (npy_uint16 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_float64)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void float32_to_float64(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_DOUBLE, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_float64 *array_data = (npy_float64 *)PyArray_DATA(array_);
    npy_float32 *data = (npy_float32 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_float64)data[i];
    }
    Py_INCREF(array_);
    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void float16_to_float64(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_DOUBLE, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_float64 *array_data = (npy_float64 *)PyArray_DATA(array_);
    npy_float16 *data = (npy_float16 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = float16_cast_float64(data[i]);
    }
    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void float128_to_float64(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_DOUBLE, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_longdouble *array_data = (npy_longdouble *)PyArray_DATA(array_);
    npy_float64 *data = (npy_float64 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_float64)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

/*to float16*/
void int8_to_float16(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT16, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_float16 *array_data = (npy_float16 *)PyArray_DATA(array_);
    npy_int8 *data = (npy_int8 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = int8_cast_float16(data[i]);
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void uint8_to_float16(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT16, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_float16 *array_data = (npy_float16 *)PyArray_DATA(array_);
    npy_uint8 *data = (npy_uint8 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = uint8_cast_float16(data[i]);
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void int16_to_float16(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT16, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_float16 *array_data = (npy_float16 *)PyArray_DATA(array_);
    npy_int16 *data = (npy_int16 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = int16_cast_float16(data[i]);
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void uint16_to_float16(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT16, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_float16 *array_data = (npy_float16 *)PyArray_DATA(array_);
    npy_uint16 *data = (npy_uint16 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = uint16_cast_float16(data[i]);
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void int32_to_float16(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT16, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_float16 *array_data = (npy_float16 *)PyArray_DATA(array_);
    npy_int32 *data = (npy_int32 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = int32_cast_float16(data[i]);
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void uint32_to_float16(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT16, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_float16 *array_data = (npy_float16 *)PyArray_DATA(array_);
    npy_uint32 *data = (npy_uint32 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
        array_data[i] = uint32_cast_float16(data[i]);

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void int64_to_float16(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT16, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_float16 *array_data = (npy_float16 *)PyArray_DATA(array_);
    npy_int64 *data = (npy_int64 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = int64_cast_float16(data[i]);
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void uint64_to_float16(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT16, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_float16 *array_data = (npy_float16 *)PyArray_DATA(array_);
    npy_uint64 *data = (npy_uint64 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = uint64_cast_float16(data[i]);
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void float32_to_float16(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT16, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_float16 *array_data = (npy_float16 *)PyArray_DATA(array_);
    npy_float32 *data = (npy_float32 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = float32_cast_float16(data[i]);
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void float64_to_float16(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT16, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_float16 *array_data = (npy_float16 *)PyArray_DATA(array_);
    npy_float64 *data = (npy_float64 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = float64_cast_float16(data[i]);
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void float128_to_float16(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_FLOAT16, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_float16 *array_data = (npy_float16 *)PyArray_DATA(array_);
    npy_longdouble *data = (npy_longdouble *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = float64_cast_float16(data[i]);
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

/*to int64*/
void int8_to_int64(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_INT64, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int64 *array_data = (npy_int64 *)PyArray_DATA(array_);
    npy_int8 *data = (npy_int8 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_int64)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void uint8_to_int64(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_INT64, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int64 *array_data = (npy_int64 *)PyArray_DATA(array_);
    npy_uint8 *data = (npy_uint8 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_int64)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void int16_to_int64(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_INT64, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int64 *array_data = (npy_int64 *)PyArray_DATA(array_);
    npy_int16 *data = (npy_int16 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_int64)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void uint16_to_int64(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_INT64, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int64 *array_data = (npy_int64 *)PyArray_DATA(array_);
    npy_uint16 *data = (npy_uint16 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_int64)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void int32_to_int64(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_INT64, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int64 *array_data = (npy_int64 *)PyArray_DATA(array_);
    npy_int32 *data = (npy_int32 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_int64)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void uint32_to_int64(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_INT64, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int64 *array_data = (npy_int64 *)PyArray_DATA(array_);
    npy_uint32 *data = (npy_uint32 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_int64)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void uint64_to_int64(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_INT64, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int64 *array_data = (npy_int64 *)PyArray_DATA(array_);
    npy_uint64 *data = (npy_uint64 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_int64)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void float32_to_int64(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_INT64, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int64 *array_data = (npy_int64 *)PyArray_DATA(array_);
    npy_float32 *data = (npy_float32 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_int64)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void float64_to_int64(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_INT64, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int64 *array_data = (npy_int64 *)PyArray_DATA(array_);
    npy_float64 *data = (npy_float64 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_int64)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void float16_to_int64(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_INT64, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int64 *array_data = (npy_int64 *)PyArray_DATA(array_);
    npy_float16 *data = (npy_float16 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_int64)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void float128_to_int64(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_INT64, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int64 *array_data = (npy_int64 *)PyArray_DATA(array_);
    npy_longdouble *data = (npy_longdouble *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_int64)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

/*to int32*/
void int8_to_int32(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_INT32, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int32 *array_data = (npy_int32 *)PyArray_DATA(array_);
    npy_int8 *data = (npy_int8 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_int32)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void uint8_to_int32(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_INT32, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int32 *array_data = (npy_int32 *)PyArray_DATA(array_);
    npy_uint8 *data = (npy_uint8 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_int32)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void int16_to_int32(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_INT32, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int32 *array_data = (npy_int32 *)PyArray_DATA(array_);
    npy_int16 *data = (npy_int16 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_int32)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void uint16_to_int32(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_INT32, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int32 *array_data = (npy_int32 *)PyArray_DATA(array_);
    npy_uint16 *data = (npy_uint16 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_int32)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void uint32_to_int32(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_INT32, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int32 *array_data = (npy_int32 *)PyArray_DATA(array_);
    npy_uint32 *data = (npy_uint32 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_int32)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void int64_to_int32(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_INT32, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int32 *array_data = (npy_int32 *)PyArray_DATA(array_);
    npy_int64 *data = (npy_int64 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_int32)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void uint64_to_int32(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_INT32, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int32 *array_data = (npy_int32 *)PyArray_DATA(array_);
    npy_uint64 *data = (npy_uint64 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_int32)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void float16_to_int32(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_INT32, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int32 *array_data = (npy_int32 *)PyArray_DATA(array_);
    npy_float16 *data = (npy_float16 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_int32)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void float32_to_int32(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_INT32, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int32 *array_data = (npy_int32 *)PyArray_DATA(array_);
    npy_float32 *data = (npy_float32 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_int32)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void float64_to_int32(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_INT32, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int32 *array_data = (npy_int32 *)PyArray_DATA(array_);
    npy_float64 *data = (npy_float64 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_int32)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void float128_to_int32(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_INT64, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_longdouble *array_data = (npy_longdouble *)PyArray_DATA(array_);
    npy_int32 *data = (npy_int32 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_int64)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

/*to int16*/
void int8_to_int16(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_INT16, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int16 *array_data = (npy_int16 *)PyArray_DATA(array_);
    npy_int8 *data = (npy_int8 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_int16)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void uint8_to_int16(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_INT16, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int16 *array_data = (npy_int16 *)PyArray_DATA(array_);
    npy_uint8 *data = (npy_uint8 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_int16)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void uint16_to_int16(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_INT16, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int16 *array_data = (npy_int16 *)PyArray_DATA(array_);
    npy_uint16 *data = (npy_uint16 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_int16)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void int32_to_int16(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_INT16, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int16 *array_data = (npy_int16 *)PyArray_DATA(array_);
    npy_int32 *data = (npy_int32 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_int16)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void uint32_to_int16(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_INT16, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int16 *array_data = (npy_int16 *)PyArray_DATA(array_);
    npy_uint32 *data = (npy_uint32 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_int16)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void int64_to_int16(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_INT16, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int16 *array_data = (npy_int16 *)PyArray_DATA(array_);
    npy_int64 *data = (npy_int64 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_int16)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void uint64_to_int16(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_INT16, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int16 *array_data = (npy_int16 *)PyArray_DATA(array_);
    npy_uint64 *data = (npy_uint64 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_int16)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void float16_to_int16(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_INT16, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int16 *array_data = (npy_int16 *)PyArray_DATA(array_);
    npy_float16 *data = (npy_float16 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_int16)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void float32_to_int16(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_INT16, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int16 *array_data = (npy_int16 *)PyArray_DATA(array_);
    npy_float32 *data = (npy_float32 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_int16)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void float64_to_int16(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_INT16, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int16 *array_data = (npy_int16 *)PyArray_DATA(array_);
    npy_float64 *data = (npy_float64 *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_int16)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void float128_to_int16(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    npy_intp size = PyArray_SIZE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_INT16, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int16 *array_data = (npy_int16 *)PyArray_DATA(array_);
    npy_longdouble *data = (npy_longdouble *)PyArray_DATA(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = (npy_int16)data[i];
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

/*to bool*/
void int8_to_bool(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_BOOL, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int8 *data = (npy_int8 *)PyArray_DATA(*array);
    npy_bool *array_data = (npy_bool *)PyArray_DATA(array_);
    npy_intp size = PyArray_SIZE(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = data[i] ? NPY_TRUE : NPY_FALSE;
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void uint8_to_bool(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_BOOL, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_uint8 *data = (npy_uint8 *)PyArray_DATA(*array);
    npy_bool *array_data = (npy_bool *)PyArray_DATA(array_);
    npy_intp size = PyArray_SIZE(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = data[i] ? NPY_TRUE : NPY_FALSE;
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void int16_to_bool(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_BOOL, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int16 *data = (npy_int16 *)PyArray_DATA(*array);
    npy_bool *array_data = (npy_bool *)PyArray_DATA(array_);
    npy_intp size = PyArray_SIZE(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = data[i] ? NPY_TRUE : NPY_FALSE;
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void uint16_to_bool(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_BOOL, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_uint16 *data = (npy_uint16 *)PyArray_DATA(*array);
    npy_bool *array_data = (npy_bool *)PyArray_DATA(array_);
    npy_intp size = PyArray_SIZE(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = data[i] ? NPY_TRUE : NPY_FALSE;
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void int32_to_bool(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_BOOL, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int32 *data = (npy_int32 *)PyArray_DATA(*array);
    npy_bool *array_data = (npy_bool *)PyArray_DATA(array_);
    npy_intp size = PyArray_SIZE(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = data[i] ? NPY_TRUE : NPY_FALSE;
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void uint32_to_bool(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_BOOL, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_uint32 *data = (npy_uint32 *)PyArray_DATA(*array);
    npy_bool *array_data = (npy_bool *)PyArray_DATA(array_);
    npy_intp size = PyArray_SIZE(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = data[i] ? NPY_TRUE : NPY_FALSE;
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void int64_to_bool(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_BOOL, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_int64 *data = (npy_int64 *)PyArray_DATA(*array);
    npy_bool *array_data = (npy_bool *)PyArray_DATA(array_);
    npy_intp size = PyArray_SIZE(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = data[i] ? NPY_TRUE : NPY_FALSE;
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void uint64_to_bool(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_BOOL, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_uint64 *data = (npy_uint64 *)PyArray_DATA(*array);
    npy_bool *array_data = (npy_bool *)PyArray_DATA(array_);
    npy_intp size = PyArray_SIZE(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = data[i] ? NPY_TRUE : NPY_FALSE;
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void float16_to_bool(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_BOOL, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_float16 *data = (npy_float16 *)PyArray_DATA(*array);
    npy_bool *array_data = (npy_bool *)PyArray_DATA(array_);
    npy_intp size = PyArray_SIZE(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = data[i] ? NPY_TRUE : NPY_FALSE;
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void float32_to_bool(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_BOOL, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_float32 *data = (npy_float32 *)PyArray_DATA(*array);
    npy_bool *array_data = (npy_bool *)PyArray_DATA(array_);
    npy_intp size = PyArray_SIZE(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = data[i] ? NPY_TRUE : NPY_FALSE;
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void float64_to_bool(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_BOOL, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_float64 *data = (npy_float64 *)PyArray_DATA(*array);
    npy_bool *array_data = (npy_bool *)PyArray_DATA(array_);
    npy_intp size = PyArray_SIZE(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = data[i] ? NPY_TRUE : NPY_FALSE;
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

void float128_to_bool(PyArrayObject **array, PyArrayObject **result)
{
    npy_intp ndims = PyArray_NDIM(*array);
    npy_intp *shape = PyArray_SHAPE(*array);
    PyArrayObject *array_ = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_BOOL, 0);
    if (array_ == NULL)
    {
        *result = NULL;
        *array = NULL;
        return;
    }
    npy_longdouble *data = (npy_longdouble *)PyArray_DATA(*array);
    npy_bool *array_data = (npy_bool *)PyArray_DATA(array_);
    npy_intp size = PyArray_SIZE(*array);
    npy_intp i;
#pragma omp parallel for
    for (i = 0; i < size; i++)
    {
        array_data[i] = data[i] ? NPY_TRUE : NPY_FALSE;
    }

    if (result != NULL)
    {
        *result = array_;
    }
    else
    {
        Py_DECREF(*array);
        *array = array_;
    }
}

/*to uint8*/
ConvertFunc *Any_to_Float32(int type)
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
        return no_convert;
    case NPY_DOUBLE:
        return float64_to_float32;
    case NPY_LONGDOUBLE:
        return float128_to_float32;
    case NPY_HALF:
        return float16_to_float32;
    default:
        return NULL;
    }
}

ConvertFunc *Any_to_Float64(int type)
{
    switch (type)
    {
    case NPY_BOOL:
        return uint8_to_float64;
    case NPY_BYTE:
        return int8_to_float64;
    case NPY_UBYTE:
        return uint8_to_float64;
    case NPY_SHORT:
        return int16_to_float64;
    case NPY_USHORT:
        return uint16_to_float64;
    case NPY_INT:
        return int32_to_float64;
    case NPY_UINT:
        return uint32_to_float64;
    case NPY_LONG:
        return int32_to_float64;
    case NPY_ULONG:
        return uint32_to_float64;
    case NPY_LONGLONG:
        return int64_to_float64;
    case NPY_ULONGLONG:
        return uint64_to_float64;
    case NPY_FLOAT:
        return float32_to_float64;
    case NPY_DOUBLE:
        return no_convert;
    case NPY_LONGDOUBLE:
        return float128_to_float64;
    case NPY_HALF:
        return float16_to_float64;
    default:
        return NULL;
    }
}

ConvertFunc *Any_to_Float16(int type)
{
    switch (type)
    {
    case NPY_BOOL:
        return uint8_to_float16;
    case NPY_BYTE:
        return int8_to_float16;
    case NPY_UBYTE:
        return uint8_to_float16;
    case NPY_SHORT:
        return int16_to_float16;
    case NPY_USHORT:
        return uint16_to_float16;
    case NPY_INT:
        return int32_to_float16;
    case NPY_UINT:
        return uint32_to_float16;
    case NPY_LONG:
        return int32_to_float16;
    case NPY_ULONG:
        return uint32_to_float16;
    case NPY_LONGLONG:
        return int64_to_float16;
    case NPY_ULONGLONG:
        return uint64_to_float16;
    case NPY_FLOAT:
        return float32_to_float16;
    case NPY_DOUBLE:
        return float64_to_float16;
    case NPY_LONGDOUBLE:
        return float128_to_float16;
    case NPY_HALF:
        return no_convert;
    default:
        return NULL;
    }
}

ConvertFunc *Any_to_Int64(int type)
{
    switch (type)
    {
    case NPY_BOOL:
        return uint8_to_int64;
    case NPY_BYTE:
        return int8_to_int64;
    case NPY_UBYTE:
        return uint8_to_int64;
    case NPY_SHORT:
        return int16_to_int64;
    case NPY_USHORT:
        return uint16_to_int64;
    case NPY_INT:
        return int32_to_int64;
    case NPY_UINT:
        return uint32_to_int64;
    case NPY_LONG:
        return int32_to_int64;
    case NPY_ULONG:
        return uint32_to_int64;
    case NPY_LONGLONG:
        return no_convert;
    case NPY_ULONGLONG:
        return uint64_to_int64;
    case NPY_FLOAT:
        return float32_to_int64;
    case NPY_DOUBLE:
        return float64_to_int64;
    case NPY_HALF:
        return float16_to_int64;
    default:
        return NULL;
    }
}

ConvertFunc *Any_to_Int32(int type)
{
    switch (type)
    {
    case NPY_BOOL:
        return uint8_to_int32;
    case NPY_BYTE:
        return int8_to_int32;
    case NPY_UBYTE:
        return uint8_to_int32;
    case NPY_SHORT:
        return int16_to_int32;
    case NPY_USHORT:
        return uint16_to_int32;
    case NPY_INT:
        return no_convert;
    case NPY_UINT:
        return uint32_to_int32;
    case NPY_LONG:
        return no_convert;
    case NPY_ULONG:
        return uint32_to_int32;
    case NPY_LONGLONG:
        return int64_to_int32;
    case NPY_ULONGLONG:
        return uint64_to_int32;
    case NPY_FLOAT:
        return float32_to_int32;
    case NPY_DOUBLE:
        return float64_to_int32;
    case NPY_HALF:
        return float16_to_int32;
    default:
        return NULL;
    }
}

ConvertFunc *Any_to_Int16(int type)
{
    switch (type)
    {
    case NPY_BOOL:
        return uint8_to_int16;
    case NPY_BYTE:
        return int8_to_int16;
    case NPY_UBYTE:
        return uint8_to_int16;
    case NPY_SHORT:
        return no_convert;
    case NPY_USHORT:
        return uint16_to_int16;
    case NPY_INT:
        return int32_to_int16;
    case NPY_UINT:
        return uint32_to_int16;
    case NPY_LONG:
        return int32_to_int16;
    case NPY_ULONG:
        return uint32_to_int16;
    case NPY_LONGLONG:
        return int64_to_int16;
    case NPY_ULONGLONG:
        return uint64_to_int16;
    case NPY_FLOAT:
        return float32_to_int16;
    case NPY_DOUBLE:
        return float64_to_int16;
    case NPY_HALF:
        return float16_to_int16;
    default:
        return NULL;
    }
}

ConvertFunc *Any_to_Bool(int type)
{
    switch (type)
    {
    case NPY_BOOL:
        return no_convert;
    case NPY_BYTE:
        return int8_to_bool;
    case NPY_UBYTE:
        return uint8_to_bool;
    case NPY_SHORT:
        return int16_to_bool;
    case NPY_USHORT:
        return uint16_to_bool;
    case NPY_INT:
        return int32_to_bool;
    case NPY_UINT:
        return uint32_to_bool;
    case NPY_LONG:
        return int32_to_bool;
    case NPY_ULONG:
        return uint32_to_bool;
    case NPY_LONGLONG:
        return int64_to_bool;
    case NPY_ULONGLONG:
        return uint64_to_bool;
    case NPY_FLOAT:
        return float32_to_bool;
    case NPY_DOUBLE:
        return float64_to_bool;
    case NPY_HALF:
        return float16_to_bool;
    default:
        return NULL;
    }
}

inline void As_Type(PyArrayObject **a, PyArrayObject **result, int self_type, int target_type)
{
    ConvertFunc *convert_func = NULL;
    switch (target_type)
    {
    case NPY_BOOL:
        convert_func = Any_to_Bool(self_type);
        if (convert_func == NULL)
        {
            PyErr_SetString(PyExc_TypeError, "Cannot convert to int64");
            *a = NULL;
        }
        convert_func(a, result);
        break;
    case NPY_BYTE:
        convert_func = Any_to_Bool(self_type);
        if (convert_func == NULL)
        {
            PyErr_SetString(PyExc_TypeError, "Cannot convert to int64");
            *a = NULL;
        }
        convert_func(a, result);
        break;
    case NPY_UBYTE:
        convert_func = Any_to_Bool(self_type);
        if (convert_func == NULL)
        {
            PyErr_SetString(PyExc_TypeError, "Cannot convert to int64");
            *a = NULL;
        }
        convert_func(a, result);
        break;
    case NPY_SHORT:
        convert_func = Any_to_Int16(self_type);
        if (convert_func == NULL)
        {
            PyErr_SetString(PyExc_TypeError, "Cannot convert to int64");
            *a = NULL;
        }
        convert_func(a, result);
        break;
    case NPY_USHORT:
        convert_func = Any_to_Bool(self_type);
        if (convert_func == NULL)
        {
            PyErr_SetString(PyExc_TypeError, "Cannot convert to int64");
            *a = NULL;
        }
        convert_func(a, result);
        break;
    case NPY_INT:
        convert_func = Any_to_Int32(self_type);
        if (convert_func == NULL)
        {
            PyErr_SetString(PyExc_TypeError, "Cannot convert to int64");
            *a = NULL;
        }
        convert_func(a, result);
        break;
    case NPY_UINT:
        convert_func = Any_to_Bool(self_type);
        if (convert_func == NULL)
        {
            PyErr_SetString(PyExc_TypeError, "Cannot convert to int64");
            *a = NULL;
        }
        convert_func(a, result);
        break;
    case NPY_LONG:
        convert_func = Any_to_Int32(self_type);
        if (convert_func == NULL)
        {
            PyErr_SetString(PyExc_TypeError, "Cannot convert to int64");
            *a = NULL;
        }
        convert_func(a, result);
        break;
    case NPY_ULONG:
        convert_func = Any_to_Int32(self_type);
        if (convert_func == NULL)
        {
            PyErr_SetString(PyExc_TypeError, "Cannot convert to int64");
            *a = NULL;
        }
        convert_func(a, result);
        break;
    case NPY_LONGLONG:
        convert_func = Any_to_Bool(self_type);
        if (convert_func == NULL)
        {
            PyErr_SetString(PyExc_TypeError, "Cannot convert to int64");
            *a = NULL;
        }
        convert_func(a, result);
        break;
    case NPY_ULONGLONG:
        convert_func = Any_to_Int64(self_type);
        if (convert_func == NULL)
        {
            PyErr_SetString(PyExc_TypeError, "Cannot convert to int64");
            *a = NULL;
        }
        convert_func(a, result);
        break;
    case NPY_FLOAT:
        convert_func = Any_to_Float32(self_type);
        if (convert_func == NULL)
        {
            PyErr_SetString(PyExc_TypeError, "Cannot convert to float32");
            *a = NULL;
        }
        convert_func(a, result);
        break;
    case NPY_DOUBLE:
        convert_func = Any_to_Float64(self_type);
        if (convert_func == NULL)
        {
            PyErr_SetString(PyExc_TypeError, "Cannot convert to float32");
            *a = NULL;
        }
        convert_func(a, result);
        break;
    case NPY_HALF:
        convert_func = Any_to_Float16(self_type);
        if (convert_func == NULL)
        {
            PyErr_SetString(PyExc_TypeError, "Cannot convert to float32");
            *a = NULL;
        }
        convert_func(a, result);
        break;
    default:
        *a = NULL;
        break;
    }
}

void as_type(PyArrayObject **a, PyArrayObject **result, int target_type)
{
    int a_dtype = PyArray_TYPE(*a);
    As_Type(a, result, a_dtype, target_type);
}