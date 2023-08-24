#ifndef BINARY_FUNC_H
#define BINARY_FUNC_H
#include "shape.h"
#include "type_convertor.h"
#include <numpy/npy_math.h>
#include <immintrin.h>
#include "numboost_api.h"
#include "omp.h"

PyArrayObject *numboost_binary(PyArrayObject *a, PyArrayObject *b, int op_enum);
PyArrayObject *numboost_binary_scalar_left(PyObject *a, PyArrayObject *b, int op_enum);
PyArrayObject *numboost_binary_scalar_right(PyArrayObject *a, PyObject *b, int op_enum);

#define Binary_Loop(a_ptr, b_ptr, result_ptr, op, type) \
    for (i = 0; i < size; i++)                          \
    {                                                   \
        result_ptr[i] = op(a_ptr[i], b_ptr[i], type);   \
    }

#define Binary_Loop_Uncontiguous(type, op, inner_loop_size, stride_a, \
                                 stride_b, a_ptr, b_ptr, result_ptr)  \
    for (npy_intp i = 0; i < inner_loop_size; i++)                    \
    {                                                                 \
        type val1 = *((a_ptr + i * stride_a));                        \
        type val2 = *((b_ptr + i * stride_b));                        \
        *(result_ptr + i) = op(val1, val2, type);                     \
    }

#define Mod_Binary_Loop(a_ptr, b_ptr, result_ptr, op, type)      \
    for (i = 0; i < size; i++)                                   \
    {                                                            \
        type val2 = b_ptr[i];                                    \
        if (!val2)                                               \
        {                                                        \
            result_ptr[i] = 0;                                   \
            continue;                                            \
        }                                                        \
        type val1 = a_ptr[i];                                    \
        type ret = op(val1, val2, type);                         \
        ret += ((ret != 0) & ((val1 < 0) != (val2 < 0))) * val2; \
        result_ptr[i] = ret;                                     \
    }

#define Mod_Binary_Loop_Uncontiguous(type, op, inner_loop_size, stride_a, \
                                     stride_b, a_ptr, b_ptr, result_ptr)  \
    for (npy_intp i = 0; i < inner_loop_size; i++)                        \
    {                                                                     \
        type val2 = *((b_ptr + i * stride_b));                            \
        if (!val2)                                                        \
        {                                                                 \
            *(result_ptr + i) = 0;                                        \
            continue;                                                     \
        }                                                                 \
        type val1 = *((a_ptr + i * stride_a));                            \
        type ret = op((val1), (val2), type);                              \
        ret += ((ret != 0) & ((val1 < 0) != (val2 < 0))) * val2;          \
        *(result_ptr + i) = ret;                                          \
    }

#define Modh_Binary_Loop(a_ptr, b_ptr, result_ptr, op, type)                            \
    for (i = 0; i < size; i++)                                                          \
    {                                                                                   \
        float b = half_cast_float(b_ptr[i]);                                            \
        if (!b)                                                                         \
        {                                                                               \
            result_ptr[i] = 0;                                                          \
            continue;                                                                   \
        }                                                                               \
        float a = half_cast_float(a_ptr[i]);                                            \
        float tmp = npy_fmodf(a, b);                                                    \
        result_ptr[i] = float_cast_half(tmp + ((tmp != 0) & ((a < 0) != (b < 0))) * b); \
    }

#define Modh_Binary_Loop_Uncontiguous(type, op, inner_loop_size, stride_a, \
                                      stride_b, a_ptr, b_ptr, result_ptr)  \
    for (npy_intp i = 0; i < inner_loop_size; i++)                         \
    {                                                                      \
        float val2 = half_cast_float(*((b_ptr + i * stride_b)));           \
        if (!val2)                                                         \
        {                                                                  \
            *(result_ptr + i) = 0;                                         \
            continue;                                                      \
        }                                                                  \
        float val1 = half_cast_float(*((a_ptr + i * stride_a)));           \
        float ret = npy_fmodf((val1), (val2), type);                       \
        ret += ((ret != 0) & ((val1 < 0) != (val2 < 0))) * val2;           \
        *(result_ptr + i) = float_cast_half(ret);                          \
    }

#define Binary_Loop_a_Scalar(a, b_ptr, result_ptr, op, type) \
    for (i = 0; i < size; i++)                               \
    {                                                        \
        result_ptr[i] = op(a, b_ptr[i], type);               \
    }

#define Binary_Loop_b_Scalar(a_ptr, b, result_ptr, op, type) \
    for (i = 0; i < size; i++)                               \
    {                                                        \
        result_ptr[i] = op(a_ptr[i], b, type);               \
    }

#define Half_Binary_Loop_A_Scalar(a, b_ptr, result_ptr, op, type) \
    for (i = 0; i < size; i++)                                    \
    {                                                             \
        npy_float b = half_cast_float(b_ptr[i]);                  \
        npy_float result = op(a, b, type);                        \
        result_ptr[i] = float_cast_half(result);                  \
    }

#define Half_Binary_Loop_B_Scalar(a_ptr, b, result_ptr, op, type) \
    for (i = 0; i < size; i++)                                    \
    {                                                             \
        npy_float a = half_cast_float(a_ptr[i]);                  \
        npy_float result = op(a, b, type);                        \
        result_ptr[i] = float_cast_half(result);                  \
    }

#define Float_Div_Binary_Loop(a_ptr, b_ptr, result_ptr, op, type) \
    for (i = 0; i < size; i++)                                    \
    {                                                             \
        if (!b_ptr[i])                                            \
        {                                                         \
            if (a_ptr[i] > 0)                                     \
                result_ptr[i] = NPY_INFINITYF;                    \
            else if (a_ptr[i] < 0)                                \
                result_ptr[i] = -NPY_INFINITYF;                   \
            else                                                  \
                result_ptr[i] = NPY_NANF;                         \
            continue;                                             \
        }                                                         \
        else                                                      \
            result_ptr[i] = op((a_ptr[i]), (b_ptr[i]), type);     \
    }

#define Float_Div_Binary_Loop_Uncontiguous(type, op, inner_loop_size, stride_a, \
                                           stride_b, a_ptr, b_ptr, result_ptr)  \
    for (npy_intp i = 0; i < inner_loop_size; i++)                              \
    {                                                                           \
        npy_float val1 = *((a_ptr + i * stride_a));                             \
        npy_float val2 = *((b_ptr + i * stride_b));                             \
        if (!val2)                                                              \
        {                                                                       \
            if (val1 > 0)                                                       \
                *(result_ptr + i) = NPY_INFINITYF;                              \
            else if (val1 < 0)                                                  \
                *(result_ptr + i) = -NPY_INFINITYF;                             \
            else                                                                \
                *(result_ptr + i) = NPY_NANF;                                   \
            continue;                                                           \
        }                                                                       \
        else                                                                    \
        {                                                                       \
            *(result_ptr + i) = op(val1, val2, npy_float);                      \
        }                                                                       \
    }

#define Double_Div_Binary_Loop(a_ptr, b_ptr, result_ptr, op, type) \
    for (i = 0; i < size; i++)                                     \
    {                                                              \
        if (!b_ptr[i])                                             \
        {                                                          \
            if (a_ptr[i] > 0)                                      \
                result_ptr[i] = NPY_INFINITY;                      \
            else if (a_ptr[i] < 0)                                 \
                result_ptr[i] = -NPY_INFINITY;                     \
            else                                                   \
                result_ptr[i] = NPY_NAN;                           \
            continue;                                              \
        }                                                          \
        else                                                       \
            result_ptr[i] = op((a_ptr[i]), (b_ptr[i]), type);      \
    }

#define Double_Div_Binary_Loop_Uncontiguous(type, op, inner_loop_size, stride_a, \
                                            stride_b, a_ptr, b_ptr, result_ptr)  \
    for (npy_intp i = 0; i < inner_loop_size; i++)                               \
    {                                                                            \
        npy_double val1 = *((a_ptr + i * stride_a));                             \
        npy_double val2 = *((b_ptr + i * stride_b));                             \
        if (!val2)                                                               \
        {                                                                        \
            if (val1 > 0)                                                        \
                *(result_ptr + i) = NPY_INFINITY;                                \
            else if (val1 < 0)                                                   \
                *(result_ptr + i) = -NPY_INFINITY;                               \
            else                                                                 \
                *(result_ptr + i) = NPY_NAN;                                     \
            continue;                                                            \
        }                                                                        \
        else                                                                     \
        {                                                                        \
            *(result_ptr + i) = op(val1, val2, npy_double);                      \
        }                                                                        \
    }

#define LongDouble_Div_Binary_Loop(a_ptr, b_ptr, result_ptr, op, type) \
    for (i = 0; i < size; i++)                                         \
    {                                                                  \
        if (!b_ptr[i])                                                 \
        {                                                              \
            if (a_ptr[i] > 0)                                          \
                result_ptr[i] = NPY_INFINITYL;                         \
            else if (a_ptr[i] < 0)                                     \
                result_ptr[i] = -NPY_INFINITYL;                        \
            else                                                       \
                result_ptr[i] = NPY_NANL;                              \
            continue;                                                  \
        }                                                              \
        else                                                           \
            result_ptr[i] = op((a_ptr[i]), (b_ptr[i]), type);          \
    }

#define LongDouble_Div_Binary_Loop_Uncontiguous(type, op, inner_loop_size, stride_a, \
                                                stride_b, a_ptr, b_ptr, result_ptr)  \
    for (npy_intp i = 0; i < inner_loop_size; i++)                                   \
    {                                                                                \
        npy_longdouble val1 = *((a_ptr + i * stride_a));                             \
        npy_longdouble val2 = *((b_ptr + i * stride_b));                             \
        if (!val2)                                                                   \
        {                                                                            \
            if (val1 > 0)                                                            \
                *(result_ptr + i) = NPY_INFINITYL;                                   \
            else if (val1 < 0)                                                       \
                *(result_ptr + i) = -NPY_INFINITYL;                                  \
            else                                                                     \
                *(result_ptr + i) = NPY_NANL;                                        \
            continue;                                                                \
        }                                                                            \
        else                                                                         \
        {                                                                            \
            *(result_ptr + i) = op(val1, val2, npy_double);                          \
        }                                                                            \
    }

#define Float_Div_Binary_Loop_A_SCALAR(a, b_ptr, result_ptr, op, type) \
    for (i = 0; i < size; i++)                                         \
    {                                                                  \
        if (!b_ptr[i])                                                 \
        {                                                              \
            if (a > 0)                                                 \
                result_ptr[i] = NPY_INFINITYF;                         \
            else if (a < 0)                                            \
                result_ptr[i] = -NPY_INFINITYF;                        \
            else                                                       \
                result_ptr[i] = NPY_NANF;                              \
            continue;                                                  \
        }                                                              \
        else                                                           \
            result_ptr[i] = op(a, b_ptr[i], type);                     \
    }

#define Double_Div_Binary_Loop_A_SCALAR(a, b_ptr, result_ptr, op, type) \
    for (i = 0; i < size; i++)                                          \
    {                                                                   \
        if (!b_ptr[i])                                                  \
        {                                                               \
            if (a > 0)                                                  \
                result_ptr[i] = NPY_INFINITY;                           \
            else if (a < 0)                                             \
                result_ptr[i] = -NPY_INFINITY;                          \
            else                                                        \
                result_ptr[i] = NPY_NAN;                                \
            continue;                                                   \
        }                                                               \
        else                                                            \
            result_ptr[i] = op(a, b_ptr[i], type);                      \
    }

#define LongDouble_Div_Binary_Loop_A_SCALAR(a, b_ptr, result_ptr, op, type) \
    for (i = 0; i < size; i++)                                              \
    {                                                                       \
        if (!b_ptr[i])                                                      \
        {                                                                   \
            if (a > 0)                                                      \
                result_ptr[i] = NPY_INFINITYL;                              \
            else if (a < 0)                                                 \
                result_ptr[i] = -NPY_INFINITYL;                             \
            else                                                            \
                result_ptr[i] = NPY_NANL;                                   \
            continue;                                                       \
        }                                                                   \
        else                                                                \
            result_ptr[i] = op(a, b_ptr[i], type);                          \
    }

#define Float_Div_Binary_Loop_B_SCALAR(a_ptr, b, result_ptr, op, type) \
    for (i = 0; i < size; i++)                                         \
    {                                                                  \
        if (!a_ptr[i])                                                 \
        {                                                              \
            if (a > 0)                                                 \
                result_ptr[i] = NPY_INFINITYF;                         \
            else if (a < 0)                                            \
                result_ptr[i] = -NPY_INFINITYF;                        \
            else                                                       \
                result_ptr[i] = NPY_NANF;                              \
            continue;                                                  \
        }                                                              \
        else                                                           \
            result_ptr[i] = op(a_ptr[i], b, type);                     \
    }

#define Double_Div_Binary_Loop_B_SCALAR(a_ptr, b, result_ptr, op, type) \
    for (i = 0; i < size; i++)                                          \
    {                                                                   \
        if (!a_ptr[i])                                                  \
        {                                                               \
            if (a > 0)                                                  \
                result_ptr[i] = NPY_INFINITY;                           \
            else if (a < 0)                                             \
                result_ptr[i] = -NPY_INFINITY;                          \
            else                                                        \
                result_ptr[i] = NPY_NAN;                                \
            continue;                                                   \
        }                                                               \
        else                                                            \
            result_ptr[i] = op(a_ptr[i], b, type);                      \
    }

#define LongDouble_Div_Binary_Loop_B_SCALAR(a_ptr, b, result_ptr, op, type) \
    for (i = 0; i < size; i++)                                              \
    {                                                                       \
        if (!a_ptr[i])                                                      \
        {                                                                   \
            if (a > 0)                                                      \
                result_ptr[i] = NPY_INFINITYL;                              \
            else if (a < 0)                                                 \
                result_ptr[i] = -NPY_INFINITYL;                             \
            else                                                            \
                result_ptr[i] = NPY_NANL;                                   \
            continue;                                                       \
        }                                                                   \
        else                                                                \
            result_ptr[i] = op(a_ptr[i], b, type);                          \
    }

#define Half_Div_Binary_Loop(a_ptr, b_ptr, result_ptr, op, type) \
    for (i = 0; i < size; i++)                                   \
    {                                                            \
        npy_float a = (npy_float)half_cast_float(a_ptr[i]);      \
        npy_float b = (npy_float)half_cast_float(b_ptr[i]);      \
        if (!b)                                                  \
        {                                                        \
            if (a > 0)                                           \
                result_ptr[i] = 0x7C00;                          \
            else if (a < 0)                                      \
                result_ptr[i] = 0xFC00;                          \
            else                                                 \
                result_ptr[i] = 0x7FFF;                          \
            continue;                                            \
        }                                                        \
        npy_float result = op(a, b, type);                       \
        result_ptr[i] = float_cast_half(result);                 \
    }

#define Half_Div_Binary_Loop_Uncontiguous(type, op, inner_loop_size, stride_a, \
                                          stride_b, a_ptr, b_ptr, result_ptr)  \
    for (npy_intp i = 0; i < inner_loop_size; i++)                             \
    {                                                                          \
        npy_float val2 = half_cast_float(*((b_ptr + i * stride_b)));           \
        npy_float val1 = half_cast_float(*((a_ptr + i * stride_a)));           \
        if (!val2)                                                             \
        {                                                                      \
            if (val1 > 0)                                                      \
                *(result_ptr + i) = 0x7C00;                                    \
            else if (val1 < 0)                                                 \
                *(result_ptr + i) = 0xFC00;                                    \
            else                                                               \
                *(result_ptr + i) = 0x7FFF;                                    \
            continue;                                                          \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            *(result_ptr + i) = float_cast_half((op(val1, val2, type)));       \
        }                                                                      \
    }

// need more investigation
#define Half_Div_Binary_Loop_A_SCALAR(a, b_ptr, result_ptr, op, type) \
    for (i = 0; i < size; i++)                                        \
    {                                                                 \
        npy_float b = half_cast_float(b_ptr[i]);                      \
        if (!b)                                                       \
        {                                                             \
            if (a > 0)                                                \
                result_ptr[i] = 0x7C00;                               \
            else if (a < 0)                                           \
                result_ptr[i] = 0xFC00;                               \
            else                                                      \
                result_ptr[i] = 0x7FFF;                               \
            continue;                                                 \
        }                                                             \
        npy_float result = op(a, b, type);                            \
        result_ptr[i] = float_cast_half(result);                      \
    }

#define Half_Div_Binary_Loop_B_SCALAR(a_ptr, b, result_ptr, op, type) \
    for (i = 0; i < size; i++)                                        \
    {                                                                 \
        npy_float a = half_cast_float(a_ptr[i]);                      \
        if (!b)                                                       \
        {                                                             \
            if (a > 0)                                                \
                result_ptr[i] = 0x7C00;                               \
            else if (a < 0)                                           \
                result_ptr[i] = 0xFC00;                               \
            else                                                      \
                result_ptr[i] = 0x7FFF;                               \
            continue;                                                 \
        }                                                             \
        npy_float result = op(a, b, type);                            \
        result_ptr[i] = float_cast_half(result);                      \
    }

#define Register_Binary_Operation(name, data_type, macro, op, npy_enum, loop_body)     \
    static PyArrayObject *Binary_##name##data_type(PyArrayObject *a, PyArrayObject *b) \
    {                                                                                  \
        macro(a, b, op, npy_##data_type, npy_enum, loop_body)                          \
    }

#define Register_Binary_Operation_A_Scalar(name, data_type, macro, op, npy_enum, loop_body)       \
    static PyArrayObject *Binary_##name##data_type##_a_scalar(Python_Number *a, PyArrayObject *b) \
    {                                                                                             \
        npy_##data_type a_val = a->data.data_type##_;                                             \
        macro(a_val, b, op, npy_##data_type, npy_enum)                                            \
    }

#define Register_Binary_Operation_B_Scalar(name, data_type, macro, op, npy_enum, loop_body)       \
    static PyArrayObject *Binary_##name##data_type##_b_scalar(PyArrayObject *a, Python_Number *b) \
    {                                                                                             \
        npy_##data_type b_val = b->data.data_type##_;                                             \
        macro(a, b_val, op, npy_##data_type, npy_enum)                                            \
    }

#define Register_Binary_Operation_Err(name, data_type, sufix, a_type, b_type)          \
    static PyArrayObject *Binary_##name##data_type##sufix(a_type *a, b_type *b)        \
    {                                                                                  \
        const char *string[] = {"Operation not supported for", #data_type, "type"};    \
        size_t length = strlen(string[0]) + strlen(string[1]) + strlen(string[2]) + 1; \
        char *string_cat = (char *)malloc(length);                                     \
        strcpy(string_cat, string[0]);                                                 \
        strcat(string_cat, string[1]);                                                 \
        strcat(string_cat, string[2]);                                                 \
        PyErr_SetString(PyExc_TypeError, string_cat);                                  \
        free(string_cat);                                                              \
        return NULL;                                                                   \
    }

#define Binary_Operation(a, b, op, data_type, npy_enum, loop_body)                                         \
    {                                                                                                      \
        data_type *a_ptr = (data_type *)PyArray_DATA(a);                                                   \
        data_type *b_ptr = (data_type *)PyArray_DATA(b);                                                   \
        npy_intp size = PyArray_SIZE(a);                                                                   \
        npy_intp *shape = PyArray_SHAPE(a);                                                                \
        PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), shape, npy_enum, 0); \
        data_type *numpy_ptr = (data_type *)PyArray_DATA(numpy_result);                                    \
        if (!PyArray_IS_C_CONTIGUOUS(a) || !PyArray_IS_C_CONTIGUOUS(b))                                    \
        {                                                                                                  \
            BinaryOperation_Uncontiguous(data_type, a, b, numpy_result, op, Binary_Loop_Uncontiguous);     \
            return numpy_result;                                                                           \
        }                                                                                                  \
        else                                                                                               \
        {                                                                                                  \
            npy_intp i;                                                                                    \
            _Pragma("omp parallel for")                                                                    \
                loop_body(a_ptr, b_ptr, numpy_ptr, op, data_type);                                         \
            return numpy_result;                                                                           \
        }                                                                                                  \
    }

#define Binary_Operation_A_Scalar(a, b, op, data_type, npy_enum)                                           \
    {                                                                                                      \
        data_type *b_ptr = (data_type *)PyArray_DATA(b);                                                   \
        npy_intp size = PyArray_SIZE(b);                                                                   \
        npy_intp *shape = PyArray_SHAPE(b);                                                                \
        PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(b), shape, npy_enum, 0); \
        data_type *numpy_ptr = (data_type *)PyArray_DATA(numpy_result);                                    \
        npy_intp i;                                                                                        \
        _Pragma("omp parallel for")                                                                        \
            Binary_Loop_a_Scalar(a, b_ptr, numpy_ptr, op, data_type);                                      \
        return numpy_result;                                                                               \
    }

#define Binary_Operation_B_Scalar(a, b, op, data_type, npy_enum)                                           \
    {                                                                                                      \
        data_type *a_ptr = (data_type *)PyArray_DATA(a);                                                   \
        npy_intp size = PyArray_SIZE(a);                                                                   \
        npy_intp *shape = PyArray_SHAPE(a);                                                                \
        PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), shape, npy_enum, 0); \
        data_type *numpy_ptr = (data_type *)PyArray_DATA(numpy_result);                                    \
        npy_intp i;                                                                                        \
        _Pragma("omp parallel for")                                                                        \
            Binary_Loop_b_Scalar(a_ptr, b, numpy_ptr, op, data_type);                                      \
        return numpy_result;                                                                               \
    }

#define HALF_BINARY_OPERATION_A_SCALAR(a, b, op, data_type, npy_enum)                                      \
    {                                                                                                      \
        data_type *b_ptr = (data_type *)PyArray_DATA(b);                                                   \
        npy_intp size = PyArray_SIZE(b);                                                                   \
        npy_intp *shape = PyArray_SHAPE(b);                                                                \
        PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(b), shape, npy_enum, 0); \
        data_type *numpy_ptr = (data_type *)PyArray_DATA(numpy_result);                                    \
        npy_intp i;                                                                                        \
        _Pragma("omp parallel for")                                                                        \
            Half_Binary_Loop_A_Scalar(a, b_ptr, numpy_ptr, op, data_type);                                 \
        return numpy_result;                                                                               \
    }

#define HALF_BINARY_OPERATION_B_SCALAR(a, b, op, data_type, npy_enum)                                      \
    {                                                                                                      \
        data_type *a_ptr = (data_type *)PyArray_DATA(a);                                                   \
        npy_intp size = PyArray_SIZE(a);                                                                   \
        npy_intp *shape = PyArray_SHAPE(a);                                                                \
        PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), shape, npy_enum, 0); \
        data_type *numpy_ptr = (data_type *)PyArray_DATA(numpy_result);                                    \
        npy_intp i;                                                                                        \
        _Pragma("omp parallel for")                                                                        \
            Half_Binary_Loop_B_Scalar(a_ptr, b, numpy_ptr, op, data_type);                                 \
        return numpy_result;                                                                               \
    }

#define Float_Div_Binary_Operation(a, b, op, data_type, npy_enum, loop_body)                                     \
    {                                                                                                            \
        npy_float *a_ptr = (npy_float *)PyArray_DATA(a);                                                         \
        npy_float *b_ptr = (npy_float *)PyArray_DATA(b);                                                         \
        npy_intp size = PyArray_SIZE(a);                                                                         \
        npy_intp *shape = PyArray_SHAPE(a);                                                                      \
        PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), shape, npy_enum, 0);       \
        npy_intp i;                                                                                              \
        if (!PyArray_IS_C_CONTIGUOUS(a) || !PyArray_IS_C_CONTIGUOUS(b))                                          \
        {                                                                                                        \
            BinaryOperation_Uncontiguous(npy_float, a, b, numpy_result, op, Float_Div_Binary_Loop_Uncontiguous); \
        }                                                                                                        \
        else                                                                                                     \
        {                                                                                                        \
            npy_float *numpy_ptr = (npy_float *)PyArray_DATA(numpy_result);                                      \
            _Pragma("omp parallel for")                                                                          \
                Float_Div_Binary_Loop(a_ptr, b_ptr, numpy_ptr, op, npy_float);                                   \
            return numpy_result;                                                                                 \
        }                                                                                                        \
    }

#define Double_Div_Binary_Operation(a, b, op, data_type, npy_enum, loop_body)                                      \
    {                                                                                                              \
        npy_double *a_ptr = (npy_double *)PyArray_DATA(a);                                                         \
        npy_double *b_ptr = (npy_double *)PyArray_DATA(b);                                                         \
        npy_intp size = PyArray_SIZE(a);                                                                           \
        npy_intp *shape = PyArray_SHAPE(a);                                                                        \
        PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), shape, npy_enum, 0);         \
        npy_intp i;                                                                                                \
        if (!PyArray_IS_C_CONTIGUOUS(a) || !PyArray_IS_C_CONTIGUOUS(b))                                            \
        {                                                                                                          \
            BinaryOperation_Uncontiguous(npy_double, a, b, numpy_result, op, Double_Div_Binary_Loop_Uncontiguous); \
        }                                                                                                          \
        else                                                                                                       \
        {                                                                                                          \
            npy_double *numpy_ptr = (npy_double *)PyArray_DATA(numpy_result);                                      \
            _Pragma("omp parallel for")                                                                            \
                Double_Div_Binary_Loop(a_ptr, b_ptr, numpy_ptr, op, npy_double);                                   \
            return numpy_result;                                                                                   \
        }                                                                                                          \
    }

#define LongDouble_Div_Binary_Operation(a, b, op, data_type, npy_enum, loop_body)                                          \
    {                                                                                                                      \
        npy_longdouble *a_ptr = (npy_longdouble *)PyArray_DATA(a);                                                         \
        npy_longdouble *b_ptr = (npy_longdouble *)PyArray_DATA(b);                                                         \
        npy_intp size = PyArray_SIZE(a);                                                                                   \
        npy_intp *shape = PyArray_SHAPE(a);                                                                                \
        PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), shape, npy_enum, 0);                 \
        npy_intp i;                                                                                                        \
        if (!PyArray_IS_C_CONTIGUOUS(a) || !PyArray_IS_C_CONTIGUOUS(b))                                                    \
        {                                                                                                                  \
            BinaryOperation_Uncontiguous(npy_longdouble, a, b, numpy_result, op, LongDouble_Div_Binary_Loop_Uncontiguous); \
        }                                                                                                                  \
        else                                                                                                               \
        {                                                                                                                  \
            npy_longdouble *numpy_ptr = (npy_longdouble *)PyArray_DATA(numpy_result);                                      \
            _Pragma("omp parallel for")                                                                                    \
                LongDouble_Div_Binary_Loop(a_ptr, b_ptr, numpy_ptr, op, npy_longdouble);                                   \
            return numpy_result;                                                                                           \
        }                                                                                                                  \
    }

#define Half_Div_Binary_Operation(a, b, op, data_type, npy_enum, loop_body)                                    \
    {                                                                                                          \
        npy_half *a_ptr = (npy_half *)PyArray_DATA(a);                                                         \
        npy_half *b_ptr = (npy_half *)PyArray_DATA(b);                                                         \
        npy_intp size = PyArray_SIZE(a);                                                                       \
        npy_intp *shape = PyArray_SHAPE(a);                                                                    \
        PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), shape, npy_enum, 0);     \
        npy_intp i;                                                                                            \
        if (!PyArray_IS_C_CONTIGUOUS(a) || !PyArray_IS_C_CONTIGUOUS(b))                                        \
        {                                                                                                      \
            BinaryOperation_Uncontiguous(npy_half, a, b, numpy_result, op, Half_Div_Binary_Loop_Uncontiguous); \
        }                                                                                                      \
        else                                                                                                   \
        {                                                                                                      \
            npy_half *numpy_ptr = (npy_half *)PyArray_DATA(numpy_result);                                      \
            _Pragma("omp parallel for")                                                                        \
                Half_Div_Binary_Loop(a_ptr, b_ptr, numpy_ptr, op, npy_half);                                   \
            return numpy_result;                                                                               \
        }                                                                                                      \
    }

#define Float_Div_Binary_Operation_A_Scalar(a, b, op, data_type, npy_enum)                                 \
    {                                                                                                      \
        npy_float *b_ptr = (npy_float *)PyArray_DATA(b);                                                   \
        npy_intp size = PyArray_SIZE(b);                                                                   \
        npy_intp *shape = PyArray_SHAPE(b);                                                                \
        PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(b), shape, npy_enum, 0); \
        if (numpy_result == NULL)                                                                          \
        {                                                                                                  \
            return NULL;                                                                                   \
        }                                                                                                  \
        npy_intp i;                                                                                        \
        npy_float *numpy_ptr = (npy_float *)PyArray_DATA(numpy_result);                                    \
        _Pragma("omp parallel for")                                                                        \
            Float_Div_Binary_Loop_A_SCALAR(a, b_ptr, numpy_ptr, op, data_type);                            \
        return numpy_result;                                                                               \
    }

#define Double_Div_Binary_Operation_A_Scalar(a, b, op, data_type, npy_enum)                                \
    {                                                                                                      \
        npy_double *b_ptr = (npy_double *)PyArray_DATA(b);                                                 \
        npy_intp size = PyArray_SIZE(b);                                                                   \
        npy_intp *shape = PyArray_SHAPE(b);                                                                \
        PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(b), shape, npy_enum, 0); \
        if (numpy_result == NULL)                                                                          \
        {                                                                                                  \
            return NULL;                                                                                   \
        }                                                                                                  \
        npy_intp i;                                                                                        \
        npy_double *numpy_ptr = (npy_double *)PyArray_DATA(numpy_result);                                  \
        _Pragma("omp parallel for")                                                                        \
            Double_Div_Binary_Loop_A_SCALAR(a, b_ptr, numpy_ptr, op, data_type);                           \
        return numpy_result;                                                                               \
    }

#define LongDouble_Div_Binary_Operation_A_Scalar(a, b, op, data_type, npy_enum)                            \
    {                                                                                                      \
        npy_longdouble *b_ptr = (npy_longdouble *)PyArray_DATA(b);                                         \
        npy_intp size = PyArray_SIZE(b);                                                                   \
        npy_intp *shape = PyArray_SHAPE(b);                                                                \
        PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(b), shape, npy_enum, 0); \
        if (numpy_result == NULL)                                                                          \
        {                                                                                                  \
            return NULL;                                                                                   \
        }                                                                                                  \
        npy_intp i;                                                                                        \
        npy_longdouble *numpy_ptr = (npy_longdouble *)PyArray_DATA(numpy_result);                          \
        _Pragma("omp parallel for")                                                                        \
            LongDouble_Div_Binary_Loop_A_SCALAR(a, b_ptr, numpy_ptr, op, data_type);                       \
        return numpy_result;                                                                               \
    }

#define Half_Div_Binary_Operation_A_Scalar(a, b, op, data_type, npy_enum)                                  \
    {                                                                                                      \
        npy_half *b_ptr = (npy_half *)PyArray_DATA(b);                                                     \
        npy_intp size = PyArray_SIZE(b);                                                                   \
        npy_intp *shape = PyArray_SHAPE(b);                                                                \
        PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(b), shape, npy_enum, 0); \
        if (numpy_result == NULL)                                                                          \
        {                                                                                                  \
            return NULL;                                                                                   \
        }                                                                                                  \
        npy_intp i;                                                                                        \
        npy_half *numpy_ptr = (npy_half *)PyArray_DATA(numpy_result);                                      \
        _Pragma("omp parallel for")                                                                        \
            Half_Div_Binary_Loop_A_SCALAR(a, b_ptr, numpy_ptr, op, data_type);                             \
        return numpy_result;                                                                               \
    }

#define Float_Div_Binary_Operation_B_Scalar(a, b, op, data_type, npy_enum)                                 \
    {                                                                                                      \
        data_type *a_val = (data_type *)PyArray_DATA(a);                                                   \
        npy_intp size = PyArray_SIZE(a);                                                                   \
        npy_intp *shape = PyArray_SHAPE(a);                                                                \
        PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), shape, npy_enum, 0); \
        npy_intp i;                                                                                        \
        npy_float *numpy_ptr = (npy_float *)PyArray_DATA(numpy_result);                                    \
        _Pragma("omp parallel for")                                                                        \
            Float_Div_Binary_Loop_B_SCALAR(a_val, b, numpy_ptr, op, data_type);                            \
        return numpy_result;                                                                               \
    }

#define Double_Div_Binary_Operation_B_Scalar(a, b, op, data_type, npy_enum)                                \
    {                                                                                                      \
        npy_double *a_val = (npy_double *)PyArray_DATA(a);                                                 \
        npy_intp size = PyArray_SIZE(a);                                                                   \
        npy_intp *shape = PyArray_SHAPE(a);                                                                \
        PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), shape, npy_enum, 0); \
        npy_intp i;                                                                                        \
        npy_double *numpy_ptr = (npy_float *)PyArray_DATA(numpy_result);                                   \
        _Pragma("omp parallel for")                                                                        \
            Double_Div_Binary_Loop_B_SCALAR(a_val, b, numpy_ptr, op, data_type);                           \
        return numpy_result;                                                                               \
    }

#define LongDouble_Div_Binary_Operation_B_Scalar(a, b, op, data_type, npy_enum)                            \
    {                                                                                                      \
        npy_longdouble *a_val = (npy_longdouble *)PyArray_DATA(a);                                         \
        npy_intp size = PyArray_SIZE(a);                                                                   \
        npy_intp *shape = PyArray_SHAPE(a);                                                                \
        PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), shape, npy_enum, 0); \
        npy_intp i;                                                                                        \
        npy_longdouble *numpy_ptr = (npy_longdouble *)PyArray_DATA(numpy_result);                          \
        _Pragma("omp parallel for")                                                                        \
            LongDouble_Div_Binary_Loop_B_SCALAR(a_val, b, numpy_ptr, op, data_type);                       \
        return numpy_result;                                                                               \
    }

#define Half_Div_Binary_Operation_B_Scalar(a, b, op, data_type, npy_enum)                                  \
    {                                                                                                      \
        npy_half *a_val = (npy_half *)PyArray_DATA(a);                                                     \
        npy_intp size = PyArray_SIZE(a);                                                                   \
        npy_intp *shape = PyArray_SHAPE(a);                                                                \
        PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), shape, npy_enum, 0); \
        npy_intp i;                                                                                        \
        npy_half *numpy_ptr = (npy_half *)PyArray_DATA(numpy_result);                                      \
        _Pragma("omp parallel for")                                                                        \
            Half_Div_Binary_Loop_B_SCALAR(a_val, b, numpy_ptr, op, data_type);                             \
        return numpy_result;                                                                               \
    }

#define BINARY_OPERATION_FUSE(a, b, op, data_type, npy_enum)                                               \
    {                                                                                                      \
        data_type *a_ptr = (data_type *)PyArray_DATA(a);                                                   \
        data_type *b_ptr = (data_type *)PyArray_DATA(b);                                                   \
        npy_intp size = PyArray_SIZE(a);                                                                   \
        npy_intp *shape = PyArray_SHAPE(a);                                                                \
        PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), shape, npy_enum, 0); \
        data_type *numpy_ptr = (data_type *)PyArray_DATA(numpy_result);                                    \
        npy_intp i;                                                                                        \
        _Pragma("omp parallel for") for (i = 0; i < size; i++)                                             \
        {                                                                                                  \
            result_ptr[i] = op(a_ptr[i], b_ptr[i]);                                                        \
        }                                                                                                  \
        return numpy_result;                                                                               \
    }

#define BINARY_OPERATION_VEC(a, b, vec_func, data_type, npy_enum)                                                     \
    {                                                                                                                 \
        data_type *a_ptr = (data_type *)PyArray_DATA(a);                                                              \
        data_type *b_ptr = (data_type *)PyArray_DATA(b);                                                              \
        npy_intp size = PyArray_SIZE(a);                                                                              \
        PyArrayObject *numpy_result = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), npy_enum, 0); \
        data_type *numpy_ptr = (data_type *)PyArray_DATA(numpy_result);                                               \
        vec_func(size, (const data_type *)a_ptr, (const data_type *)b_ptr, numpy_ptr);                                \
        return numpy_result;                                                                                          \
    }

#define OPERATION_PICKER_A_SCALAR(a, b, operation, data_type, npy_enum)        \
    {                                                                          \
        switch (operation)                                                     \
        {                                                                      \
        case ADD:                                                              \
            Binary_Operation_A_Scalar(a, b, nb_add, data_type, npy_enum);      \
            break;                                                             \
        case SUB:                                                              \
            Binary_Operation_A_Scalar(a, b, nb_subtract, data_type, npy_enum); \
            break;                                                             \
        case MUL:                                                              \
            Binary_Operation_A_Scalar(a, b, nb_multiply, data_type, npy_enum); \
            break;                                                             \
        case DIV:                                                              \
            Div_Binary_Operation_A_Scalar(a, b, nb_divide, data_type,          \
                                          div_result_type_pick(npy_enum));     \
            break;                                                             \
        default:                                                               \
            return NULL;                                                       \
        }                                                                      \
    }

#define OPERATION_PICKER_B_SCALAR(a, b, operation, data_type, npy_enum)        \
    {                                                                          \
        switch (operation)                                                     \
        {                                                                      \
        case ADD:                                                              \
            Binary_Operation_B_Scalar(a, b, nb_add, data_type, npy_enum);      \
            break;                                                             \
        case SUB:                                                              \
            Binary_Operation_B_Scalar(a, b, nb_subtract, data_type, npy_enum); \
            break;                                                             \
        case MUL:                                                              \
            Binary_Operation_B_Scalar(a, b, nb_multiply, data_type, npy_enum); \
            break;                                                             \
        case DIV:                                                              \
            Div_Binary_Operation_B_Scalar(a, b, nb_divide, data_type,          \
                                          div_result_type_pick(npy_enum));     \
            break;                                                             \
        default:                                                               \
            return NULL;                                                       \
        }                                                                      \
    }

#define F_OPERATION_PICKER_A_SCALAR(a, b, operation, data_type, npy_enum)            \
    {                                                                                \
        switch (operation)                                                           \
        {                                                                            \
        case ADD:                                                                    \
            Binary_Operation_A_Scalar(a, b, nb_add, data_type, npy_enum) break;      \
        case SUB:                                                                    \
            Binary_Operation_A_Scalar(a, b, nb_subtract, data_type, npy_enum) break; \
        case MUL:                                                                    \
            Binary_Operation_A_Scalar(a, b, nb_multiply, data_type, npy_enum) break; \
        case DIV:                                                                    \
            Div_Binary_Operation_A_Scalar(a, b, nb_divide, data_type,                \
                                          div_result_type_pick(npy_enum));           \
            break;                                                                   \
        default:                                                                     \
            return NULL;                                                             \
        }                                                                            \
    }

#define F_OPERATION_PICKER_B_SCALAR(a, b, operation, data_type, npy_enum)            \
    {                                                                                \
        switch (operation)                                                           \
        {                                                                            \
        case ADD:                                                                    \
            Binary_Operation_B_Scalar(a, b, nb_add, data_type, npy_enum) break;      \
        case SUB:                                                                    \
            Binary_Operation_B_Scalar(a, b, nb_subtract, data_type, npy_enum) break; \
        case MUL:                                                                    \
            Binary_Operation_B_Scalar(a, b, nb_multiply, data_type, npy_enum) break; \
        case DIV:                                                                    \
            Div_Binary_Operation_B_Scalar(a, b, nb_divide, data_type,                \
                                          div_result_type_pick(npy_enum));           \
            break;                                                                   \
        default:                                                                     \
            return NULL;                                                             \
        }                                                                            \
    }

#define HALF_OPERATION_PICKER_A_SCALAR(a, b, operation, data_type, npy_enum)       \
    {                                                                              \
        switch (operation)                                                         \
        {                                                                          \
        case ADD:                                                                  \
            HALF_BINARY_OPERATION_A_SCALAR(a, b, nb_add, data_type, npy_enum)      \
            break;                                                                 \
        case SUB:                                                                  \
            HALF_BINARY_OPERATION_A_SCALAR(a, b, nb_subtract, data_type, npy_enum) \
            break;                                                                 \
        case MUL:                                                                  \
            HALF_BINARY_OPERATION_A_SCALAR(a, b, nb_multiply, data_type, npy_enum) \
            break;                                                                 \
        case DIV:                                                                  \
            Div_Binary_Operation_A_Scalar(a, b, nb_divide, data_type,              \
                                          div_result_type_pick(npy_enum));         \
            break;                                                                 \
        default:                                                                   \
            return NULL;                                                           \
        }                                                                          \
    }

#define HALF_OPERATION_PICKER_B_SCALAR(a, b, operation, data_type, npy_enum)       \
    {                                                                              \
        switch (operation)                                                         \
        {                                                                          \
        case ADD:                                                                  \
            HALF_BINARY_OPERATION_B_SCALAR(a, b, nb_add, data_type, npy_enum)      \
            break;                                                                 \
        case SUB:                                                                  \
            HALF_BINARY_OPERATION_B_SCALAR(a, b, nb_subtract, data_type, npy_enum) \
            break;                                                                 \
        case MUL:                                                                  \
            HALF_BINARY_OPERATION_B_SCALAR(a, b, nb_multiply, data_type, npy_enum) \
            break;                                                                 \
        case DIV:                                                                  \
            Div_Binary_Operation_B_Scalar(a, b, nb_divide, data_type,              \
                                          div_result_type_pick(npy_enum));         \
            break;                                                                 \
        default:                                                                   \
            return NULL;                                                           \
        }                                                                          \
    }

#define HALF_OPERATION(a, b, op)                                                                \
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

#define Scalar_B_Retrieve(TYPE, a, b, operation, NPY_TYPE)          \
    switch (TYPE)                                                   \
    {                                                               \
    case NPY_BOOL:                                                  \
        OPERATION_PICKER_B_SCALAR(a,                                \
                                  b->data.bool_, operation,         \
                                  NPY_TYPE, TYPE)                   \
        break;                                                      \
    case NPY_BYTE:                                                  \
        OPERATION_PICKER_B_SCALAR(a,                                \
                                  b->data.byte_, operation,         \
                                  NPY_TYPE, TYPE)                   \
        break;                                                      \
    case NPY_UBYTE:                                                 \
        OPERATION_PICKER_B_SCALAR(a,                                \
                                  b->data.ubyte_, operation,        \
                                  NPY_TYPE, TYPE)                   \
        break;                                                      \
    case NPY_SHORT:                                                 \
        OPERATION_PICKER_B_SCALAR(a,                                \
                                  b->data.short_, operation,        \
                                  NPY_TYPE, TYPE)                   \
        break;                                                      \
    case NPY_USHORT:                                                \
        OPERATION_PICKER_B_SCALAR(a,                                \
                                  b->data.ushort_, operation,       \
                                  NPY_TYPE, TYPE)                   \
        break;                                                      \
    case NPY_INT:                                                   \
        OPERATION_PICKER_B_SCALAR(a,                                \
                                  b->data.int_, operation,          \
                                  NPY_TYPE, TYPE)                   \
        break;                                                      \
    case NPY_UINT:                                                  \
        OPERATION_PICKER_B_SCALAR(a,                                \
                                  b->data.uint_, operation,         \
                                  NPY_TYPE, TYPE)                   \
        break;                                                      \
    case NPY_LONG:                                                  \
        OPERATION_PICKER_B_SCALAR(a,                                \
                                  b->data.long_, operation,         \
                                  NPY_TYPE, TYPE)                   \
        break;                                                      \
    case NPY_ULONG:                                                 \
        OPERATION_PICKER_B_SCALAR(a,                                \
                                  b->data.ulong_, operation,        \
                                  NPY_TYPE, TYPE)                   \
        break;                                                      \
    case NPY_LONGLONG:                                              \
        OPERATION_PICKER_B_SCALAR(a,                                \
                                  b->data.longlong_, operation,     \
                                  NPY_TYPE, TYPE)                   \
        break;                                                      \
    case NPY_ULONGLONG:                                             \
        OPERATION_PICKER_B_SCALAR(a,                                \
                                  b->data.ulonglong_, operation,    \
                                  NPY_TYPE, TYPE)                   \
        break;                                                      \
    case NPY_FLOAT:                                                 \
        F_OPERATION_PICKER_B_SCALAR(a,                              \
                                    b->data.float_, operation,      \
                                    NPY_TYPE, TYPE)                 \
        break;                                                      \
    case NPY_DOUBLE:                                                \
        F_OPERATION_PICKER_B_SCALAR(a,                              \
                                    b->data.double_, operation,     \
                                    NPY_TYPE, TYPE)                 \
        break;                                                      \
    case NPY_LONGDOUBLE:                                            \
        F_OPERATION_PICKER_B_SCALAR(a,                              \
                                    b->data.longdouble_, operation, \
                                    NPY_TYPE, TYPE)                 \
        break;                                                      \
    default:                                                        \
        return NULL;                                                \
    }

#define F_Scalar_B_Retrieve(TYPE, a, b, operation, NPY_TYPE)                      \
    switch (TYPE)                                                                 \
    {                                                                             \
    case NPY_FLOAT:                                                               \
        F_OPERATION_PICKER_B_SCALAR(a,                                            \
                                    b->data.float_, operation,                    \
                                    NPY_TYPE, TYPE)                               \
        break;                                                                    \
    case NPY_DOUBLE:                                                              \
        F_OPERATION_PICKER_B_SCALAR(a,                                            \
                                    b->data.double_, operation,                   \
                                    NPY_TYPE, TYPE)                               \
        break;                                                                    \
    case NPY_LONGDOUBLE:                                                          \
        F_OPERATION_PICKER_B_SCALAR(a,                                            \
                                    b->data.longdouble_, operation,               \
                                    NPY_TYPE, TYPE)                               \
    case NPY_HALF:                                                                \
        HALF_OPERATION_PICKER_B_SCALAR(a,                                         \
                                       half_cast_float(b->data.half_), operation, \
                                       NPY_TYPE, TYPE)                            \
        break;                                                                    \
    default:                                                                      \
        return NULL;                                                              \
    }

#define Scalar_B_PICK(TYPE, a, b, operation, NPY_TYPE) \
    case TYPE:                                         \
        Scalar_B_Retrieve(TYPE, a, b, operation, NPY_TYPE) break;

#define F_Scalar_B_PICK(TYPE, a, b, operation, NPY_TYPE) \
    case TYPE:                                           \
        F_Scalar_B_Retrieve(TYPE, a, b, operation, NPY_TYPE) break;

#define Store_Number(a, a_, Python_Type, npy_type, isfloat)                                                                     \
    switch (npy_type)                                                                                                           \
    {                                                                                                                           \
    case NPY_BOOL:                                                                                                              \
        a_->data.bool_ = Py_IsTrue(a);                                                                                          \
        break;                                                                                                                  \
    case NPY_BYTE:                                                                                                              \
        a_->data.byte_ = isfloat ? (npy_byte)PyFloat_AsDouble(a) : (npy_byte)Py##Python_Type##_AsLong(a);                       \
        break;                                                                                                                  \
    case NPY_UBYTE:                                                                                                             \
        a_->data.ubyte_ = isfloat ? (npy_ubyte)PyFloat_AsDouble(a) : (npy_ubyte)Py##Python_Type##_AsLong(a);                    \
        break;                                                                                                                  \
    case NPY_SHORT:                                                                                                             \
        a_->data.short_ = isfloat ? (npy_short)PyFloat_AsDouble(a) : (npy_short)Py##Python_Type##_AsLong(a);                    \
        break;                                                                                                                  \
    case NPY_USHORT:                                                                                                            \
        a_->data.ushort_ = isfloat ? (npy_ushort)PyFloat_AsDouble(a) : (npy_ushort)Py##Python_Type##_AsLong(a);                 \
        break;                                                                                                                  \
    case NPY_INT:                                                                                                               \
        a_->data.int_ = isfloat ? (npy_int)PyFloat_AsDouble(a) : (npy_int)Py##Python_Type##_AsLong(a);                          \
        break;                                                                                                                  \
    case NPY_UINT:                                                                                                              \
        a_->data.uint_ = isfloat ? (npy_uint)PyFloat_AsDouble(a) : (npy_uint)Py##Python_Type##_AsLong(a);                       \
        break;                                                                                                                  \
    case NPY_LONG:                                                                                                              \
        a_->data.long_ = isfloat ? (npy_long)PyFloat_AsDouble(a) : (npy_long)Py##Python_Type##_AsLong(a);                       \
        break;                                                                                                                  \
    case NPY_ULONG:                                                                                                             \
        a_->data.ulong_ = isfloat ? (npy_ulong)PyFloat_AsDouble(a) : (npy_ulong)Py##Python_Type##_AsLong(a);                    \
        break;                                                                                                                  \
    case NPY_LONGLONG:                                                                                                          \
        a_->data.longlong_ = isfloat ? (npy_longlong)PyFloat_AsDouble(a) : (npy_longlong)Py##Python_Type##_AsLongLong(a);       \
        break;                                                                                                                  \
    case NPY_ULONGLONG:                                                                                                         \
        a_->data.ulonglong_ = isfloat ? (npy_ulonglong)PyFloat_AsDouble(a) : (npy_ulonglong)Py##Python_Type##_AsLongLong(a);    \
        break;                                                                                                                  \
    case NPY_FLOAT:                                                                                                             \
        a_->data.float_ = isfloat ? (npy_float)PyFloat_AsDouble(a) : (npy_float)Py##Python_Type##_AsLong(a);                    \
        break;                                                                                                                  \
    case NPY_DOUBLE:                                                                                                            \
        a_->data.double_ = isfloat ? (npy_double)PyFloat_AsDouble(a) : (npy_double)Py##Python_Type##_AsLongLong(a);             \
        break;                                                                                                                  \
    case NPY_LONGDOUBLE:                                                                                                        \
        a_->data.longdouble_ = isfloat ? (npy_longdouble)PyFloat_AsDouble(a) : (npy_longdouble)Py##Python_Type##_AsLongLong(a); \
        break;                                                                                                                  \
    case NPY_HALF:                                                                                                              \
        a_->data.half_ = isfloat ? double_cast_half(PyFloat_AsDouble(a)) : long_cast_half(Py##Python_Type##_AsLong(a));         \
        break;                                                                                                                  \
    }

#define NotImplement_Err(name, sufix, a_type, b_type)                        \
    Register_Binary_Operation_Err(name, cfloat, sufix, a_type, b_type);      \
    Register_Binary_Operation_Err(name, cdouble, sufix, a_type, b_type);     \
    Register_Binary_Operation_Err(name, clongdouble, sufix, a_type, b_type); \
    Register_Binary_Operation_Err(name, object, sufix, a_type, b_type);      \
    Register_Binary_Operation_Err(name, string, sufix, a_type, b_type);      \
    Register_Binary_Operation_Err(name, unicode, sufix, a_type, b_type);     \
    Register_Binary_Operation_Err(name, void, sufix, a_type, b_type);        \
    Register_Binary_Operation_Err(name, datetime, sufix, a_type, b_type);    \
    Register_Binary_Operation_Err(name, timedelta, sufix, a_type, b_type);

#define Register_Int_Binary_Operations(name, sufix, Operation, nb_method, npy_enum_convert, loop_body)                 \
    Register_Binary_Operation##sufix(name, bool, Operation, nb_method, npy_enum_convert(NPY_BOOL), loop_body);         \
    Register_Binary_Operation##sufix(name, byte, Operation, nb_method, npy_enum_convert(NPY_BYTE), loop_body);         \
    Register_Binary_Operation##sufix(name, ubyte, Operation, nb_method, npy_enum_convert(NPY_UBYTE), loop_body);       \
    Register_Binary_Operation##sufix(name, short, Operation, nb_method, npy_enum_convert(NPY_SHORT), loop_body);       \
    Register_Binary_Operation##sufix(name, ushort, Operation, nb_method, npy_enum_convert(NPY_USHORT), loop_body);     \
    Register_Binary_Operation##sufix(name, int, Operation, nb_method, npy_enum_convert(NPY_INT), loop_body);           \
    Register_Binary_Operation##sufix(name, uint, Operation, nb_method, npy_enum_convert(NPY_UINT), loop_body);         \
    Register_Binary_Operation##sufix(name, long, Operation, nb_method, npy_enum_convert(NPY_LONG), loop_body);         \
    Register_Binary_Operation##sufix(name, ulong, Operation, nb_method, npy_enum_convert(NPY_ULONG), loop_body);       \
    Register_Binary_Operation##sufix(name, longlong, Operation, nb_method, npy_enum_convert(NPY_LONGLONG), loop_body); \
    Register_Binary_Operation##sufix(name, ulonglong, Operation, nb_method, npy_enum_convert(NPY_ULONGLONG), loop_body);

#define Register_Int_Binary_OperationsErr(name, sufix, a_type, b_type)    \
    Register_Binary_Operation_Err(name, bool, sufix, a_type, b_type);     \
    Register_Binary_Operation_Err(name, byte, sufix, a_type, b_type);     \
    Register_Binary_Operation_Err(name, ubyte, sufix, a_type, b_type);    \
    Register_Binary_Operation_Err(name, short, sufix, a_type, b_type);    \
    Register_Binary_Operation_Err(name, ushort, sufix, a_type, b_type);   \
    Register_Binary_Operation_Err(name, int, sufix, a_type, b_type);      \
    Register_Binary_Operation_Err(name, uint, sufix, a_type, b_type);     \
    Register_Binary_Operation_Err(name, long, sufix, a_type, b_type);     \
    Register_Binary_Operation_Err(name, ulong, sufix, a_type, b_type);    \
    Register_Binary_Operation_Err(name, longlong, sufix, a_type, b_type); \
    Register_Binary_Operation_Err(name, ulonglong, sufix, a_type, b_type);

#endif