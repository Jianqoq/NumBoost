#ifndef SHAPE_H
#define SHAPE_H
#include "shape.h"
#endif
#include "type_convertor.h"

typedef void (*BinaryFunc)(PyArrayObject *a, PyArrayObject *b, PyObject **result);

void BinaryOp_Picker(int npy_type, int operation, PyArrayObject *a, PyArrayObject *b, PyObject **result);

#define BINARY_LOOP(a_ptr, b_ptr, result_ptr, op) \
    for (i = 0; i < size; i++)                    \
        result_ptr[i] = a_ptr[i] op b_ptr[i];

#define BINARY_OPERATION(a, b, result, op, data_type, npy_enum)                                            \
    {                                                                                                      \
        data_type *a_ptr = (data_type *)PyArray_DATA(a);                                                   \
        data_type *b_ptr = (data_type *)PyArray_DATA(b);                                                   \
        npy_intp size = PyArray_SIZE(a);                                                                   \
        npy_intp *shape = PyArray_SHAPE(a);                                                                \
        PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), shape, npy_enum, 0); \
        data_type *numpy_ptr = (data_type *)PyArray_DATA(numpy_result);                                    \
        npy_intp i;                                                                                        \
        _Pragma("omp parallel for")                                                                        \
            BINARY_LOOP(a_ptr, b_ptr, numpy_ptr, op)                                                       \
                *result = (PyObject *)numpy_result;                                                        \
    }

#define BINARY_OPERATION_VEC(a, b, result, vec_func, data_type, npy_enum)                                             \
    {                                                                                                                 \
        data_type *a_ptr = (data_type *)PyArray_DATA(a);                                                              \
        data_type *b_ptr = (data_type *)PyArray_DATA(b);                                                              \
        npy_intp size = PyArray_SIZE(a);                                                                              \
        PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), npy_enum, 0); \
        data_type *numpy_ptr = (data_type *)PyArray_DATA(numpy_result);                                               \
        vec_func(size, (const data_type *)a_ptr, (const data_type *)b_ptr, numpy_ptr);                                \
        *result = (PyObject *)numpy_result;                                                                           \
    }

#define OPERATION_PICKER(a, b, result, operation, data_type, npy_enum) \
    {                                                                  \
        switch (operation)                                             \
        {                                                              \
        case ADD:                                                      \
            BINARY_OPERATION(a, b, result, +, data_type, npy_enum)     \
            break;                                                     \
        case SUB:                                                      \
            BINARY_OPERATION(a, b, result, -, data_type, npy_enum)     \
            break;                                                     \
        case MUL:                                                      \
            BINARY_OPERATION(a, b, result, *, data_type, npy_enum)     \
            break;                                                     \
        case DIV:                                                      \
            BINARY_OPERATION(a, b, result, /, data_type, npy_enum)     \
            break;                                                     \
        case MOD:                                                      \
            BINARY_OPERATION(a, b, result, %, data_type, npy_enum)     \
            break;                                                     \
        }                                                              \
    }

#define F_OPERATION_PICKER(a, b, result, operation, data_type, npy_enum) \
    {                                                                    \
        switch (operation)                                               \
        {                                                                \
        case ADD:                                                        \
            BINARY_OPERATION(a, b, result, +, data_type, npy_enum)       \
            break;                                                       \
        case SUB:                                                        \
            BINARY_OPERATION(a, b, result, -, data_type, npy_enum)       \
            break;                                                       \
        case MUL:                                                        \
            BINARY_OPERATION(a, b, result, *, data_type, npy_enum)       \
            break;                                                       \
        case DIV:                                                        \
            BINARY_OPERATION(a, b, result, /, data_type, npy_enum)       \
            break;                                                       \
        }                                                                \
    }

#define HALF_OPERATION(a, b, result, op)                                                        \
    {                                                                                           \
        npy_half *a_ptr = (npy_half *)PyArray_DATA(a);                                          \
        npy_half *b_ptr = (npy_half *)PyArray_DATA(b);                                          \
        npy_intp size = PyArray_SIZE(a);                                                        \
        int ndim = PyArray_NDIM(a);                                                             \
        npy_intp *shape = PyArray_SHAPE(a);                                                     \
        PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(ndim, shape, NPY_HALF, 0); \
        npy_half *numpy_ptr = (npy_half *)PyArray_DATA(numpy_result);                           \
        npy_intp i;                                                                             \
        _Pragma("omp parallel for") for (i = 0; i < size; i++)                                  \
        {                                                                                       \
            npy_half a = half_cast_float(a_ptr[i]);                                             \
            npy_half b = half_cast_float(b_ptr[i]);                                             \
            npy_half result = float_cast_half(a op b);                                          \
            numpy_ptr[i] = result;                                                              \
        }                                                                                       \
        *result = (PyObject *)numpy_result;                                                     \
    }