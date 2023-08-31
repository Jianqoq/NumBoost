#ifndef OP_H
#define OP_H
typedef enum
{
    ADD,
    SUB,
    MUL,
    DIV,
    MOD,
    POW,
    LSHIFT,
    RSHIFT,
    FLOOR_DIV,
    BITWISE_AND,
    BITWISE_OR,
    BITWISE_XOR,
    BITWISE_INVERT,
    POSITIVE,
    NEGATIVE,
    SIN,
    COS,
    TAN,
    ARCSIN,
    ARCCOS,
    ARCTAN,
    SINH,
    COSH,
    TANH,
    ARCSINH,
    ARCCOSH,
    ARCTANH,
} op_type;

#define Get_Size(x, result)               \
    switch (x)                            \
    {                                     \
    case NPY_BOOL:                        \
        result = sizeof(npy_bool);        \
        break;                            \
    case NPY_BYTE:                        \
        result = sizeof(npy_byte);        \
        break;                            \
    case NPY_UBYTE:                       \
        result = sizeof(npy_ubyte);       \
        break;                            \
    case NPY_SHORT:                       \
        result = sizeof(npy_short);       \
        break;                            \
    case NPY_USHORT:                      \
        result = sizeof(npy_ushort);      \
        break;                            \
    case NPY_INT:                         \
        result = sizeof(npy_int);         \
        break;                            \
    case NPY_UINT:                        \
        result = sizeof(npy_uint);        \
        break;                            \
    case NPY_LONG:                        \
        result = sizeof(npy_long);        \
        break;                            \
    case NPY_ULONG:                       \
        result = sizeof(npy_ulong);       \
        break;                            \
    case NPY_LONGLONG:                    \
        result = sizeof(npy_longlong);    \
        break;                            \
    case NPY_ULONGLONG:                   \
        result = sizeof(npy_ulonglong);   \
        break;                            \
    case NPY_FLOAT:                       \
        result = sizeof(npy_float);       \
        break;                            \
    case NPY_DOUBLE:                      \
        result = sizeof(npy_double);      \
        break;                            \
    case NPY_LONGDOUBLE:                  \
        result = sizeof(npy_longdouble);  \
        break;                            \
    case NPY_CFLOAT:                      \
        result = sizeof(npy_cfloat);      \
        break;                            \
    case NPY_CDOUBLE:                     \
        result = sizeof(npy_cdouble);     \
        break;                            \
    case NPY_CLONGDOUBLE:                 \
        result = sizeof(npy_clongdouble); \
        break;                            \
    case NPY_HALF:                        \
        result = sizeof(npy_half);        \
        break;                            \
    case NPY_STRING:                      \
        result = 0;                       \
        break;                            \
    case NPY_UNICODE:                     \
        result = 0;                       \
        break;                            \
    case NPY_VOID:                        \
        result = 0;                       \
        break;                            \
    case NPY_OBJECT:                      \
        result = 0;                       \
        break;                            \
    case NPY_DATETIME:                    \
        result = sizeof(npy_datetime);    \
        break;                            \
    case NPY_TIMEDELTA:                   \
        result = sizeof(npy_timedelta);   \
        break;                            \
    default:                              \
        result = sizeof(npy_float);       \
        break;                            \
    }
#endif