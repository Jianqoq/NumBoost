
#ifndef _TYPE_CONVERTOR_H
#define _TYPE_CONVERTOR_H
#include <numpy/arrayobject.h>
#include "../allocator/allocator.h"
#include <assert.h>

#ifndef PYTHON_NUMBER_H
#define PYTHON_NUMBER_H
typedef struct
{
    int type;
    union
    {
        npy_bool bool_;
        npy_byte byte_;
        npy_ubyte ubyte_;
        npy_short short_;
        npy_ushort ushort_;
        npy_int int_;
        npy_uint uint_;
        npy_long long_;
        npy_ulong ulong_;
        npy_longlong longlong_;
        npy_ulonglong ulonglong_;
        npy_float float_;
        npy_double double_;
        npy_longdouble longdouble_;
        npy_half half_;
    } data;
} Python_Number;
#endif

#ifndef TYPE_CONVERTOR_H
#define TYPE_CONVERTOR_H

void Any_to_Float(PyArrayObject **array, PyArrayObject **result, int type);

void as_type(PyArrayObject **a, PyArrayObject **result, int target_type);

int div_result_type_pick(int npy_enum);

int binary_result_type(int op, int a_dtype, int a_size, int b_dtype, int b_size);

int elementwise_result_type(int op, int a_dtype);

PyObject *binary_result_type_(PyObject *self, PyObject *const *args, size_t nargsf);

PyObject *set_global_float_type(PyObject *self, PyObject *const *args, size_t nargsf);

extern int type_2_size[];

inline npy_half float_cast_half(npy_float value)
{
    npy_float *p = &value;
    uint32_t b = *((uint32_t *)p);
    uint32_t float32_m = (uint32_t)(b & 0x007fffffu);
    uint32_t float32_sign = (uint32_t)((b & 0x80000000) >> 16);
    uint32_t e = (b & 0x7F800000);
    /* Exponent overflow/NaN converts to signed inf/NaN */
    // float16 overflow binray: 11110(2) = 30(10)
    // 15 = 30 - offset(15)
    // 127 + 15 + 1
    if (e >= 0x47800000u)
    {
        if (e == 0x7f800000u)
        {
            if (float32_m != 0)
            {
                uint16_t ret = (uint16_t)(0x7c00u + (float32_m >> 13));
                /* ...but make sure it stays a NaN */
                if (ret == 0x7c00u)
                    ret++;
                return (uint16_t)(float32_sign + ret);
            }
            else
            {
                return (uint16_t)(float32_sign + 0x7C00u);
            }
        }
        else
        {
            return (uint16_t)(float32_sign + 0x7C00u);
        }
    }

    /* Exponent underflow converts to subnormal half or signed zero */
    // float16 underflow binray: 00001(2) = 1(10)
    // -14 = 1 - offset(15)
    // 127 + (-14) - 1
    if (e <= 0x38000000u)
    {
        if (e < 0x33000000u)
            return float32_sign;
        e >>= 23;
        float32_m += 0x00800000u;
        float32_m >>= (113 - e);
        if (((float32_m & 0x00003fffu) != 0x00001000u) || (b & 0x000007ffu))
        {
            float32_m += 0x00001000u;
        }
        return (uint16_t)float32_sign + (uint16_t)(float32_m >> 13);
    }
    uint16_t h_exp = (uint16_t)((e - 0x38000000u) >> 13);
    if ((float32_m & 0x00003fffu) != 0x00001000u)
        float32_m += 0x00001000u;
    uint16_t h_sig = (uint16_t)(float32_m >> 13);
    h_sig += h_exp;
    return (uint16_t)float32_sign + h_sig;
}

inline npy_half bool_cast_half(npy_bool value)
{
    return value ? (npy_half)0x3c00u : (npy_half)0x0000u;
}
#define DEFINE_HALF_CAST_FUNC(FROM_TYPE, FUNC_NAME) \
    inline npy_half FUNC_NAME(FROM_TYPE value)      \
    {                                               \
        return float_cast_half((npy_float)value);   \
    }

DEFINE_HALF_CAST_FUNC(npy_byte, byte_cast_half)
DEFINE_HALF_CAST_FUNC(npy_ubyte, ubyte_cast_half)
DEFINE_HALF_CAST_FUNC(npy_short, short_cast_half)
DEFINE_HALF_CAST_FUNC(npy_ushort, ushort_cast_half)
DEFINE_HALF_CAST_FUNC(npy_int, int_cast_half)
DEFINE_HALF_CAST_FUNC(npy_uint, uint_cast_half)
DEFINE_HALF_CAST_FUNC(npy_long, long_cast_half)
DEFINE_HALF_CAST_FUNC(npy_ulong, ulong_cast_half)
DEFINE_HALF_CAST_FUNC(npy_longlong, longlong_cast_half)

inline npy_float32 half_cast_float(npy_half value)
{
    uint16_t float16_exp = value & 0x7c00u;
    uint16_t float16_min = value & 0x03ffu;
    uint32_t float16_sign = ((uint32_t)value & 0x8000u) << 16;
    switch (float16_exp)
    {
    case 0x0000u:
    {
        if (float16_min == 0)
        {
            uint32_t *p = &float16_sign;
            npy_float32 P2 = *((npy_float32 *)p);
            return P2;
        }
        else
        {
            while ((float16_min & 0x0400u) == 0)
            {
                float16_min <<= 1;
                float16_exp++;
            }
            float16_exp--;
            uint32_t float32_exp = ((uint32_t)(127 - 15 - float16_exp) << 23);
            uint32_t float32_min = ((((uint32_t)float16_min) & 0x03ffu) << 13);
            uint32_t result = float16_sign + float32_exp + float32_min;
            uint32_t *p = &result;
            npy_float32 P2 = *((npy_float32 *)p);
            return P2;
        }
    }
    case 0x7c00u:
    {
        uint32_t result = float16_sign + 0x7f800000u + (((uint32_t)(float16_min)) << 13);
        uint32_t *p = &result;
        npy_float32 P2 = *((npy_float32 *)p);
        return P2;
    }
    default:
    {
        uint32_t result = (float16_sign + (((uint32_t)(value & 0x7fffu) + 0x1c000u) << 13));
        uint32_t *p = &result;
        npy_float32 P2 = *((npy_float32 *)p);
        return P2;
    }
    }
}

inline npy_uint half_cast_uint(npy_half value)
{
    uint16_t float16_exp = (value & 0x7c00u) >> 10;
    switch (float16_exp)
    {
    case 0x0000u:
        return (value & 0x8000u) ? -0 : 0;
    case 0x7c00u:
        return 32768;
    default:
    {
        int16_t exponent = float16_exp - 15;
        if (exponent < 0)
        {
            npy_uint result = ((((value & 0x03ffu) >> -exponent) >> 10) + (1 >> -exponent)) * ((value & 0x8000u) ? -1 : 1);
            return result;
        }
        else
        {
            npy_uint result = ((((value & 0x03ffu) << exponent) >> 10) + (1 << exponent)) * ((value & 0x8000u) ? -1 : 1);
            return result;
        }
    }
    }
}

inline npy_int half_cast_int(npy_half value)
{
    uint16_t float16_exp = (value & 0x7c00u) >> 10;
    switch (float16_exp)
    {
    case 0x0000u:
        return (value & 0x8000u) ? -0 : 0;
    case 0x7c00u:
        return 32768;
    default:
    {
        int16_t exponent = float16_exp - 15;
        if (exponent < 0)
        {
            npy_int result = ((((value & 0x03ffu) >> -exponent) >> 10) + (1 >> -exponent)) * ((value & 0x8000u) ? -1 : 1);
            return result;
        }
        else
        {
            npy_int result = ((((value & 0x03ffu) << exponent) >> 10) + (1 << exponent)) * ((value & 0x8000u) ? -1 : 1);
            return result;
        }
    }
    }
}

inline npy_byte half_cast_byte(npy_half value)
{
    return ((npy_byte)half_cast_int(value));
}

inline npy_ubyte half_cast_ubyte(npy_half value)
{
    return ((npy_ubyte)half_cast_uint(value));
}

inline npy_short half_cast_short(npy_half value)
{
    uint16_t float16_exp = (value & 0x7c00u) >> 10;
    switch (float16_exp)
    {
    case 0x0000u:
        return (value & 0x8000u) ? -0 : 0;
    case 0x7c00u:
        return 32768;
    default:
    {
        int16_t exponent = float16_exp - 15;
        if (exponent < 0)
        {
            npy_short result = ((((value & 0x03ffu) >> -exponent) >> 10) + (1 >> -exponent)) * ((value & 0x8000u) ? -1 : 1);
            return result;
        }
        else
        {
            npy_short result = ((((value & 0x03ffu) << exponent) >> 10) + (1 << exponent)) * ((value & 0x8000u) ? -1 : 1);
            return result;
        }
    }
    }
}

inline npy_ushort half_cast_ushort(npy_half value)
{
    return ((npy_ushort)half_cast_short(value));
}

inline npy_double half_cast_double(npy_half value)
{
    uint16_t float16_exp = value & 0x7c00u;
    uint16_t float16_min = value & 0x03ffu;
    uint64_t float16_sign = ((uint64_t)value & 0x8000u) << 48;
    switch (float16_exp)
    {
    case 0x0000u:
    {
        if (float16_min == 0)
        {
            uint64_t *p = &float16_sign;
            npy_float64 P2 = *((npy_float64 *)p);
            return P2;
        }
        else
        {
            while ((float16_min & 0x0400u) == 0)
            {
                float16_min <<= 1;
                float16_exp++;
            }
            float16_exp--;
            uint64_t float32_exp = ((uint64_t)(1023 - 15 - float16_exp) << 52);
            uint64_t float32_min = ((uint64_t)(float16_min & 0x03ffu) << 42);
            uint64_t result = float16_sign + float32_exp + float32_min;
            uint64_t *p = &result;
            npy_double P2 = *((npy_double *)p);
            return P2;
        }
    }
    case 0x7c00u:
    {
        uint64_t result = float16_sign + 0x7ff0000000000000ULL + (((uint64_t)(float16_min)) << 42);
        uint64_t *p = &result;
        npy_float64 P2 = *((npy_float64 *)p);
        return P2;
    }
    default:
    {
        uint64_t result = (float16_sign + (((uint64_t)(value & 0x7fffu) + 0xfc000u) << 42));
        uint64_t *p = &result;
        npy_float64 P2 = *((npy_float64 *)p);
        return P2;
    }
    }
}

inline npy_long half_cast_long(npy_half value)
{
    return (npy_long)half_cast_double(value);
}

inline npy_longlong half_cast_longlong(npy_half value)
{
    return ((npy_longlong)half_cast_double(value));
}

inline npy_half double_cast_half(npy_float64 value)
{
    npy_float64 *p = &value;
    uint64_t b = *((uint64_t *)p);
    uint64_t float64_m = (uint64_t)(b & 0x000fffffffffffffULL);
    uint64_t float64_sign = (uint64_t)((b & 0x8000000000000000ULL) >> 48);
    uint64_t e = (b & 0x7ff0000000000000ULL);
    /* Exponent overflow/NaN converts to signed inf/NaN */
    // float16 overflow binray: 11110(2) = 30(10)
    // 15 = 30 - offset(15)
    // 127 + 15 + 1
    if (e >= 0x40f0000000000000ULL)
    {
        if (e == 0x7ff0000000000000ULL)
        {
            if (float64_m != 0)
            {
                uint16_t ret = (uint16_t)(0x7c00u + (float64_m >> 42));
                /* ...but make sure it stays a NaN */
                if (ret == 0x7c00u)
                {
                    ret++;
                }
                return (uint16_t)(float64_sign + ret);
            }
            else
            {
                return (uint16_t)(float64_sign + 0x7C00u);
            }
        }
        else
        {
            return (uint16_t)(float64_sign + 0x7C00u);
        }
    }

    /* Exponent underflow converts to subnormal half or signed zero */
    // float16 underflow binray: 00001(2) = 1(10)
    // -14 = 1 - offset(15)
    // 127 + (-14) - 1
    if (e <= 0x3f00000000000000ULL)
    {
        if (e < 0x3e60000000000000ULL)
            return (uint16_t)float64_sign;
        e >>= 52;
        float64_m += 0x0010000000000000ULL;
        float64_m <<= (e - 998);
        if (((float64_m & 0x003fffffffffffffULL) != 0x0010000000000000ULL))
        {
            float64_m += 0x0010000000000000ULL;
        }
        return (uint16_t)float64_sign + (uint16_t)(float64_m >> 53);
    }
    uint16_t h_exp = (uint16_t)((e - 0x3f00000000000000ULL) >> 42);
    if ((float64_m & 0x000007ffffffffffULL) != 0x0000020000000000ULL)
        float64_m += 0x0000020000000000ULL;
    uint16_t h_sig = (uint16_t)(float64_m >> 42);
    h_sig += h_exp;
    return (uint16_t)float64_sign + h_sig;
}

inline npy_half ulonglong_cast_half(npy_ulonglong value)
{
    return double_cast_half((npy_double)value);
}

inline npy_ulong half_cast_ulong(npy_half value)
{
    return ((npy_ulong)half_cast_long(value));
}

inline npy_ulonglong half_cast_ulonglong(npy_half value)
{
    return ((npy_ulonglong)half_cast_double(value));
}

#endif

#define CAST_ARRAY(source, result, source_type, to_type, npy_enum)                                                             \
    for (i = 0; i < ndims; i++)                                                                                                \
        new_strides[i] = (npy_intp)((strides[i] * sizeof(to_type)) / (npy_intp)sizeof(source_type));                           \
    PyArrayObject *array_##source_type = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(npy_enum), \
                                                                               (int)ndims, (npy_intp const *)shape,            \
                                                                               (npy_intp const *)new_strides, NULL,            \
                                                                               PyArray_FLAGS(*source), NULL);                  \
    if (array_##source_type == NULL)                                                                                           \
    {                                                                                                                          \
        *result = NULL;                                                                                                        \
        *array = NULL;                                                                                                         \
        return;                                                                                                                \
    }                                                                                                                          \
    free(new_strides);                                                                                                         \
    to_type *array_data_##source_type = (to_type *)PyArray_DATA(array_##source_type);                                          \
    source_type *data_##source_type = (source_type *)PyArray_DATA(*source);                                                    \
    _Pragma("omp parallel for") for (i = 0; i < size; i++)                                                                     \
    {                                                                                                                          \
        to_type value = (to_type)data_##source_type[i];                                                                        \
        array_data_##source_type[i] = value;                                                                                   \
    }                                                                                                                          \
    if (result != NULL)                                                                                                        \
        *result = array_##source_type;                                                                                         \
    else                                                                                                                       \
    {                                                                                                                          \
        Py_DECREF(*array);                                                                                                     \
        *source = array_##source_type;                                                                                         \
    }

#define F_CAST_ARRAY(source, result, source_type, to_type, npy_enum, half_cast_func)                                           \
    for (i = 0; i < ndims; i++)                                                                                                \
        new_strides[i] = (npy_intp)((strides[i] * sizeof(to_type)) / (npy_intp)sizeof(source_type));                           \
    PyArrayObject *array_##source_type = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(npy_enum), \
                                                                               (int)ndims, (npy_intp const *)shape,            \
                                                                               (npy_intp const *)new_strides, NULL,            \
                                                                               PyArray_FLAGS(*source), NULL);                  \
    if (array_##source_type == NULL)                                                                                           \
    {                                                                                                                          \
        *result = NULL;                                                                                                        \
        *array = NULL;                                                                                                         \
        return;                                                                                                                \
    }                                                                                                                          \
    free(new_strides);                                                                                                         \
    to_type *array_data_##source_type = (to_type *)PyArray_DATA(array_##source_type);                                          \
    source_type *data_##source_type = (source_type *)PyArray_DATA(*source);                                                    \
    _Pragma("omp parallel for") for (i = 0; i < size; i++)                                                                     \
        array_data_##source_type[i] = (to_type)half_cast_func(data_##source_type[i]);                                          \
    if (result != NULL)                                                                                                        \
        *result = array_##source_type;                                                                                         \
    else                                                                                                                       \
    {                                                                                                                          \
        Py_DECREF(*array);                                                                                                     \
        *source = array_##source_type;                                                                                         \
    }

#define CAST_ARRAY_BOOL(source, result, source_type)                                                     \
    PyArrayObject *array_##source_type = (PyArrayObject *)PyArray_EMPTY((int)ndims, shape, NPY_BOOL, 0); \
    if (array_##source_type == NULL)                                                                     \
    {                                                                                                    \
        *result = NULL;                                                                                  \
        *array = NULL;                                                                                   \
        return;                                                                                          \
    }                                                                                                    \
    npy_bool *array_data_##source_type = (npy_bool *)PyArray_DATA(array_##source_type);                  \
    source_type *data_##source_type = (source_type *)PyArray_DATA(*source);                              \
    _Pragma("omp parallel for") for (i = 0; i < size; i++)                                               \
        array_data_##source_type[i] = data_##source_type[i] ? 1 : 0;                                     \
    if (result != NULL)                                                                                  \
        *result = array_##source_type;                                                                   \
    else                                                                                                 \
    {                                                                                                    \
        Py_DECREF(*array);                                                                               \
        *source = array_##source_type;                                                                   \
    }

#define CAST_ARRAY_CASE(TYPE, FROM, TO, TO_ENUM)     \
    case TYPE:                                       \
        CAST_ARRAY(array, result, FROM, TO, TO_ENUM) \
        break;

#define CAST_ARRAY_CASE_BOOL(TYPE, FROM)     \
    case TYPE:                               \
        CAST_ARRAY_BOOL(array, result, FROM) \
        break;

#define F_CAST_ARRAY_CASE(TYPE, FROM, TO, TO_ENUM, F_CAST_FUNC)     \
    case TYPE:                                                      \
        F_CAST_ARRAY(array, result, FROM, TO, TO_ENUM, F_CAST_FUNC) \
        break;

#define NO_CONVERT_CASE(TYPE) \
    case TYPE:                \
        *result = *array;     \
        break;

#define As_Type_Cases(To, TYPE)            \
    case TYPE:                             \
        Any_to_##To(a, result, self_type); \
        break;

#endif