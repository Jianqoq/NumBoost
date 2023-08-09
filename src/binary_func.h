#ifndef SHAPE_H
#define SHAPE_H
#include "shape.h"
#endif
#include "type_convertor.h"
#include <numpy/npy_math.h>

typedef void (*BinaryFunc)(PyArrayObject *a, PyArrayObject *b, PyObject **result);

void BinaryOp_Picker(int operation, PyArrayObject *a, PyArrayObject *b, PyObject **result);

#define BINARY_LOOP(a_ptr, b_ptr, result_ptr, op) \
    for (i = 0; i < size; i++)                    \
    {                                             \
        result_ptr[i] = a_ptr[i] op b_ptr[i];     \
    }

#define FLOAT_DIV_BINARY_LOOP(a_ptr, b_ptr, result_ptr, op, npy_enum, div_by_zero) \
    for (i = 0; i < size; i++)                                                     \
    {                                                                              \
        if (!b_ptr[i])                                                             \
        {                                                                          \
            if (a_ptr[i] > 0)                                                      \
                result_ptr[i] = NPY_INFINITYF;                                     \
            else if (a_ptr[i] < 0)                                                 \
                result_ptr[i] = -NPY_INFINITYF;                                    \
            else                                                                   \
                result_ptr[i] = NPY_NANF;                                          \
            div_by_zero = true;                                                    \
            continue;                                                              \
        }                                                                          \
        else                                                                       \
            result_ptr[i] = ((npy_float)a_ptr[i])op((npy_float)b_ptr[i]);          \
    }

#define DOUBLE_DIV_BINARY_LOOP(a_ptr, b_ptr, result_ptr, op, npy_enum, div_by_zero) \
    for (i = 0; i < size; i++)                                                      \
    {                                                                               \
        if (!b_ptr[i])                                                              \
        {                                                                           \
            if (a_ptr[i] > 0)                                                       \
                result_ptr[i] = NPY_INFINITY;                                       \
            else if (a_ptr[i] < 0)                                                  \
                result_ptr[i] = -NPY_INFINITY;                                      \
            else                                                                    \
                result_ptr[i] = NPY_NAN;                                            \
            div_by_zero = true;                                                     \
            continue;                                                               \
        }                                                                           \
        else                                                                        \
            result_ptr[i] = ((npy_double)a_ptr[i])op((npy_double)b_ptr[i]);         \
    }

#define HALF_DIV_BINARY_LOOP(a_ptr, b_ptr, result_ptr, op, div_by_zero) \
    for (i = 0; i < size; i++)                                          \
    {                                                                   \
        npy_float a = (npy_float)a_ptr[i];                              \
        npy_float b = (npy_float)b_ptr[i];                              \
        if (!b)                                                         \
        {                                                               \
            if (a > 0)                                                  \
                result_ptr[i] = 0x7C00;                                 \
            else if (a < 0)                                             \
                result_ptr[i] = 0xFC00;                                 \
            else                                                        \
                result_ptr[i] = 0x7FFF;                                 \
            continue;                                                   \
            div_by_zero = true;                                         \
        }                                                               \
        npy_float result = a op b;                                      \
        result_ptr[i] = float_cast_half(result);                        \
    }

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

#define DIV_BINARY_OPERATION(a, b, result, op, data_type, npy_enum)                                        \
    {                                                                                                      \
        data_type *a_ptr = (data_type *)PyArray_DATA(a);                                                   \
        data_type *b_ptr = (data_type *)PyArray_DATA(b);                                                   \
        npy_intp size = PyArray_SIZE(a);                                                                   \
        npy_intp *shape = PyArray_SHAPE(a);                                                                \
        PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), shape, npy_enum, 0); \
        npy_intp i;                                                                                        \
        bool div_by_zero = false;                                                                          \
        switch (npy_enum)                                                                                  \
        {                                                                                                  \
        case NPY_HALF:                                                                                     \
            npy_half *numpy_ptr1 = (npy_half *)PyArray_DATA(numpy_result);                                 \
            _Pragma("omp parallel for")                                                                    \
                HALF_DIV_BINARY_LOOP(a_ptr, b_ptr, numpy_ptr1, op, div_by_zero)                            \
                    *result = (PyObject *)numpy_result;                                                    \
            break;                                                                                         \
        case NPY_FLOAT:                                                                                    \
            npy_float *numpy_ptr2 = (npy_float *)PyArray_DATA(numpy_result);                               \
            _Pragma("omp parallel for")                                                                    \
                FLOAT_DIV_BINARY_LOOP(a_ptr, b_ptr, numpy_ptr2, op, NPY_FLOAT, div_by_zero)                \
                    *result = (PyObject *)numpy_result;                                                    \
            break;                                                                                         \
        case NPY_DOUBLE:                                                                                   \
            npy_double *numpy_ptr3 = (npy_double *)PyArray_DATA(numpy_result);                             \
            _Pragma("omp parallel for")                                                                    \
                DOUBLE_DIV_BINARY_LOOP(a_ptr, b_ptr, numpy_ptr3, op, NPY_DOUBLE, div_by_zero)              \
                    *result = (PyObject *)numpy_result;                                                    \
            break;                                                                                         \
        }                                                                                                  \
        if (div_by_zero)                                                                                   \
            fprintf(stderr, "RuntimeWarning: divide by zero encountered in divide\n");                     \
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
            DIV_BINARY_OPERATION(a, b, result, /, data_type,           \
                                 div_result_type_pick(npy_enum))       \
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
            DIV_BINARY_OPERATION(a, b, result, /, data_type, npy_enum)   \
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
            npy_float a = half_cast_float(a_ptr[i]);                                            \
            npy_float b = half_cast_float(b_ptr[i]);                                            \
            npy_half result = float_cast_half(a op b);                                          \
            numpy_ptr[i] = result;                                                              \
        }                                                                                       \
        *result = (PyObject *)numpy_result;                                                     \
    }

#define PICK(TYPE, a, b, result, operation, NPY_TYPE)             \
    case TYPE:                                                    \
        OPERATION_PICKER(a, b, result, operation, NPY_TYPE, TYPE) \
        break;

#define F_PICK(TYPE, a, b, result, operation, NPY_TYPE)             \
    case TYPE:                                                      \
        F_OPERATION_PICKER(a, b, result, operation, NPY_TYPE, TYPE) \
        break;

#define HALF_PICK(a, b, result, operation)  \
    case NPY_HALF:                          \
        switch (operation)                  \
        {                                   \
        case ADD:                           \
            HALF_OPERATION(a, b, result, +) \
            break;                          \
        case SUB:                           \
            HALF_OPERATION(a, b, result, -) \
            break;                          \
        case MUL:                           \
            HALF_OPERATION(a, b, result, *) \
            break;                          \
        case DIV:                           \
            HALF_OPERATION(a, b, result, /) \
            break;                          \
        }                                   \
        break;