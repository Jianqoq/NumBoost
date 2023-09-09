#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "type_convertor.h"
#include "../op.h"
#include "stdbool.h"
#ifndef max
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
#define min(a, b) (((a) > (b)) ? (b) : (a))
#endif
int gloabal_int_type = NPY_INT;
int gloabal_float_type = NPY_FLOAT;

#define uint_switch_cases(op)                                                  \
  switch (op) {                                                                \
  case ADD:                                                                    \
  case SUB:                                                                    \
  case MUL:                                                                    \
    if (is_float_number(b_dtype))                                              \
      return float_type_based_on_size(max(a_size, b_size));                    \
    else                                                                       \
      return max(a_dtype, b_dtype);                                            \
  case DIV:                                                                    \
    return float_type_based_on_size(max(a_size, b_size));                      \
  case MOD:                                                                    \
    if (is_float_number(b_dtype))                                              \
      return float_type_based_on_size(max(a_size, b_size));                    \
    else                                                                       \
      return max(a_dtype, b_dtype);                                            \
  case POW:                                                                    \
    return gloabal_float_type;                                                 \
  case FLOOR_DIV:                                                              \
    return max(a_dtype, b_dtype);                                              \
  case BITWISE_AND:                                                            \
  case BITWISE_OR:                                                             \
  case BITWISE_XOR:                                                            \
    return NPY_BOOL;                                                           \
  case LSHIFT:                                                                 \
  case RSHIFT:                                                                 \
    if (is_float_number(b_dtype))                                              \
      return -1;                                                               \
    else                                                                       \
      return max(a_dtype, b_dtype);                                            \
  case SQUARE:                                                                 \
    if (is_float_number(b_dtype))                                              \
      return float_type_based_on_size(max(a_size, b_size));                    \
    else                                                                       \
      return max(a_dtype, b_dtype);                                            \
  default:                                                                     \
    return max(a_dtype, b_dtype);                                              \
  }                                                                            \
  break;

#define int_switch_cases(op)                                                   \
  switch (op) {                                                                \
  case ADD:                                                                    \
  case SUB:                                                                    \
  case MUL:                                                                    \
    if (is_float_number(b_dtype))                                              \
      return float_type_based_on_size(max(a_size, b_size));                    \
    else                                                                       \
      return max(a_dtype, b_dtype);                                            \
  case DIV:                                                                    \
    return float_type_based_on_size(max(a_size, b_size));                      \
  case MOD:                                                                    \
    if (is_float_number(b_dtype))                                              \
      return float_type_based_on_size(max(a_size, b_size));                    \
    else                                                                       \
      return max(a_dtype, b_dtype);                                            \
  case POW:                                                                    \
    return gloabal_float_type;                                                 \
  case FLOOR_DIV:                                                              \
    return max(a_dtype, b_dtype);                                              \
  case BITWISE_AND:                                                            \
  case BITWISE_OR:                                                             \
  case BITWISE_XOR:                                                            \
    return NPY_BOOL;                                                           \
  case LSHIFT:                                                                 \
  case RSHIFT:                                                                 \
    if (is_float_number(b_dtype))                                              \
      return -1;                                                               \
    else                                                                       \
      return max(a_dtype, b_dtype);                                            \
  case SQUARE:                                                                 \
    if (is_float_number(b_dtype))                                              \
      return float_type_based_on_size(max(a_size, b_size));                    \
    else                                                                       \
      return max(a_dtype, b_dtype);                                            \
  default:                                                                     \
    return max(a_dtype, b_dtype);                                              \
  }                                                                            \
  break;

PyObject *set_global_float_type(PyObject *self, PyObject *const *args,
                                size_t nargsf) {
  (void)self;
  (void)nargsf;
  int type = (int)PyLong_AsLong(args[0]);
  if (type == NPY_FLOAT || type == NPY_DOUBLE || type == NPY_LONGDOUBLE ||
      type == NPY_HALF)
    gloabal_float_type = type;
  else {
    PyErr_SetString(PyExc_TypeError,
                    "type must be float, double, longdouble or half");
    return NULL;
  }
  Py_RETURN_NONE;
}

/*to float*/
void Any_to_Float(PyArrayObject **array, PyArrayObject **result, int type) {
  npy_intp ndims = PyArray_NDIM(*array);
  npy_intp *shape = PyArray_SHAPE(*array);
  npy_intp size = PyArray_SIZE(*array);
  npy_intp *strides = ((PyArrayObject_fields *)*array)->strides;
  npy_intp *new_strides = (npy_intp *)malloc(sizeof(npy_intp) * ndims);
  npy_intp i;
  switch (type) {
    CAST_ARRAY_CASE(NPY_BOOL, npy_bool, npy_float, NPY_FLOAT)
    CAST_ARRAY_CASE(NPY_BYTE, npy_byte, npy_float, NPY_FLOAT)
    CAST_ARRAY_CASE(NPY_UBYTE, npy_ubyte, npy_float, NPY_FLOAT)
    CAST_ARRAY_CASE(NPY_SHORT, npy_short, npy_float, NPY_FLOAT)
    CAST_ARRAY_CASE(NPY_USHORT, npy_ushort, npy_float, NPY_FLOAT)
    CAST_ARRAY_CASE(NPY_INT, npy_int, npy_float, NPY_FLOAT)
    CAST_ARRAY_CASE(NPY_UINT, npy_uint, npy_float, NPY_FLOAT)
    CAST_ARRAY_CASE(NPY_LONG, npy_long, npy_float, NPY_FLOAT)
    CAST_ARRAY_CASE(NPY_ULONG, npy_ulong, npy_float, NPY_FLOAT)
    CAST_ARRAY_CASE(NPY_LONGLONG, npy_longlong, npy_float, NPY_FLOAT)
    CAST_ARRAY_CASE(NPY_ULONGLONG, npy_ulonglong, npy_float, NPY_FLOAT)
    NO_CONVERT_CASE(NPY_FLOAT)
    CAST_ARRAY_CASE(NPY_DOUBLE, npy_double, npy_float, NPY_FLOAT)
    CAST_ARRAY_CASE(NPY_LONGDOUBLE, npy_longdouble, npy_float, NPY_FLOAT)
    F_CAST_ARRAY_CASE(NPY_HALF, npy_half, npy_float, NPY_FLOAT, half_cast_float)
  }
}

void Any_to_Double(PyArrayObject **array, PyArrayObject **result, int type) {
  npy_intp ndims = PyArray_NDIM(*array);
  npy_intp *shape = PyArray_SHAPE(*array);
  npy_intp size = PyArray_SIZE(*array);
  npy_intp *strides = ((PyArrayObject_fields *)*array)->strides;
  npy_intp *new_strides = (npy_intp *)malloc(sizeof(npy_intp) * ndims);
  npy_intp i;
  switch (type) {
    CAST_ARRAY_CASE(NPY_BOOL, npy_bool, npy_double, NPY_DOUBLE)
    CAST_ARRAY_CASE(NPY_BYTE, npy_byte, npy_double, NPY_DOUBLE)
    CAST_ARRAY_CASE(NPY_UBYTE, npy_ubyte, npy_double, NPY_DOUBLE)
    CAST_ARRAY_CASE(NPY_SHORT, npy_short, npy_double, NPY_DOUBLE)
    CAST_ARRAY_CASE(NPY_USHORT, npy_ushort, npy_double, NPY_DOUBLE)
    CAST_ARRAY_CASE(NPY_INT, npy_int, npy_double, NPY_DOUBLE)
    CAST_ARRAY_CASE(NPY_UINT, npy_uint, npy_double, NPY_DOUBLE)
    CAST_ARRAY_CASE(NPY_LONG, npy_long, npy_double, NPY_DOUBLE)
    CAST_ARRAY_CASE(NPY_ULONG, npy_ulong, npy_double, NPY_DOUBLE)
    CAST_ARRAY_CASE(NPY_LONGLONG, npy_longlong, npy_double, NPY_DOUBLE)
    CAST_ARRAY_CASE(NPY_ULONGLONG, npy_ulonglong, npy_double, NPY_DOUBLE)
    CAST_ARRAY_CASE(NPY_FLOAT, npy_float, npy_double, NPY_DOUBLE)
    NO_CONVERT_CASE(NPY_DOUBLE)
    CAST_ARRAY_CASE(NPY_LONGDOUBLE, npy_longdouble, npy_double, NPY_DOUBLE)
    F_CAST_ARRAY_CASE(NPY_HALF, npy_half, npy_double, NPY_DOUBLE,
                      half_cast_double)
  }
}

void Any_to_Half(PyArrayObject **array, PyArrayObject **result, int type) {
  npy_intp ndims = PyArray_NDIM(*array);
  npy_intp *shape = PyArray_SHAPE(*array);
  npy_intp size = PyArray_SIZE(*array);
  npy_intp *strides = ((PyArrayObject_fields *)*array)->strides;
  npy_intp *new_strides = (npy_intp *)malloc(sizeof(npy_intp) * ndims);
  npy_intp i;
  switch (type) {
    F_CAST_ARRAY_CASE(NPY_BOOL, npy_bool, npy_half, NPY_HALF, bool_cast_half)
    F_CAST_ARRAY_CASE(NPY_BYTE, npy_byte, npy_half, NPY_HALF, byte_cast_half)
    F_CAST_ARRAY_CASE(NPY_UBYTE, npy_ubyte, npy_half, NPY_HALF, ubyte_cast_half)
    F_CAST_ARRAY_CASE(NPY_SHORT, npy_short, npy_half, NPY_HALF, short_cast_half)
    F_CAST_ARRAY_CASE(NPY_USHORT, npy_ushort, npy_half, NPY_HALF,
                      ushort_cast_half)
    F_CAST_ARRAY_CASE(NPY_INT, npy_int, npy_half, NPY_HALF, int_cast_half)
    F_CAST_ARRAY_CASE(NPY_UINT, npy_uint, npy_half, NPY_HALF, uint_cast_half)
    F_CAST_ARRAY_CASE(NPY_LONG, npy_long, npy_half, NPY_HALF, long_cast_half)
    F_CAST_ARRAY_CASE(NPY_ULONG, npy_ulong, npy_half, NPY_HALF, ulong_cast_half)
    F_CAST_ARRAY_CASE(NPY_LONGLONG, npy_longlong, npy_half, NPY_HALF,
                      longlong_cast_half)
    F_CAST_ARRAY_CASE(NPY_ULONGLONG, npy_ulonglong, npy_half, NPY_HALF,
                      ulonglong_cast_half)
    F_CAST_ARRAY_CASE(NPY_FLOAT, npy_float, npy_half, NPY_HALF, float_cast_half)
    F_CAST_ARRAY_CASE(NPY_DOUBLE, npy_double, npy_half, NPY_HALF,
                      double_cast_half)
    F_CAST_ARRAY_CASE(NPY_LONGDOUBLE, npy_longdouble, npy_half, NPY_HALF,
                      double_cast_half)
    NO_CONVERT_CASE(NPY_HALF)
  }
}

void Any_to_LongLong(PyArrayObject **array, PyArrayObject **result, int type) {
  npy_intp ndims = PyArray_NDIM(*array);
  npy_intp *shape = PyArray_SHAPE(*array);
  npy_intp size = PyArray_SIZE(*array);
  npy_intp *strides = ((PyArrayObject_fields *)*array)->strides;
  npy_intp *new_strides = (npy_intp *)malloc(sizeof(npy_intp) * ndims);
  npy_intp i;
  switch (type) {
    CAST_ARRAY_CASE(NPY_BOOL, npy_bool, npy_longlong, NPY_LONGLONG)
    CAST_ARRAY_CASE(NPY_BYTE, npy_byte, npy_longlong, NPY_LONGLONG)
    CAST_ARRAY_CASE(NPY_UBYTE, npy_ubyte, npy_longlong, NPY_LONGLONG)
    CAST_ARRAY_CASE(NPY_SHORT, npy_short, npy_longlong, NPY_LONGLONG)
    CAST_ARRAY_CASE(NPY_USHORT, npy_ushort, npy_longlong, NPY_LONGLONG)
    CAST_ARRAY_CASE(NPY_INT, npy_int, npy_longlong, NPY_LONGLONG)
    CAST_ARRAY_CASE(NPY_UINT, npy_uint, npy_longlong, NPY_LONGLONG)
    CAST_ARRAY_CASE(NPY_LONG, npy_long, npy_longlong, NPY_LONGLONG)
    CAST_ARRAY_CASE(NPY_ULONG, npy_ulong, npy_longlong, NPY_LONGLONG)
    CAST_ARRAY_CASE(NPY_LONGLONG, npy_longlong, npy_longlong, NPY_LONGLONG)
    CAST_ARRAY_CASE(NPY_ULONGLONG, npy_ulonglong, npy_longlong, NPY_LONGLONG)
    CAST_ARRAY_CASE(NPY_FLOAT, npy_float, npy_longlong, NPY_LONGLONG)
    CAST_ARRAY_CASE(NPY_DOUBLE, npy_double, npy_longlong, NPY_LONGLONG)
    CAST_ARRAY_CASE(NPY_LONGDOUBLE, npy_longdouble, npy_longlong, NPY_LONGLONG)
    F_CAST_ARRAY_CASE(NPY_HALF, npy_half, npy_longlong, NPY_LONGLONG,
                      half_cast_longlong)
  }
}

void Any_to_Long(PyArrayObject **array, PyArrayObject **result, int type) {
  npy_intp ndims = PyArray_NDIM(*array);
  npy_intp *shape = PyArray_SHAPE(*array);
  npy_intp size = PyArray_SIZE(*array);
  npy_intp *strides = ((PyArrayObject_fields *)*array)->strides;
  npy_intp *new_strides = (npy_intp *)malloc(sizeof(npy_intp) * ndims);
  npy_intp i;
  switch (type) {
    CAST_ARRAY_CASE(NPY_BOOL, npy_bool, npy_long, NPY_LONG)
    CAST_ARRAY_CASE(NPY_BYTE, npy_byte, npy_long, NPY_LONG)
    CAST_ARRAY_CASE(NPY_UBYTE, npy_ubyte, npy_long, NPY_LONG)
    CAST_ARRAY_CASE(NPY_SHORT, npy_short, npy_long, NPY_LONG)
    CAST_ARRAY_CASE(NPY_USHORT, npy_ushort, npy_long, NPY_LONG)
    CAST_ARRAY_CASE(NPY_INT, npy_int, npy_long, NPY_LONG)
    CAST_ARRAY_CASE(NPY_UINT, npy_uint, npy_long, NPY_LONG)
    NO_CONVERT_CASE(NPY_LONG)
    CAST_ARRAY_CASE(NPY_ULONG, npy_ulong, npy_long, NPY_LONG)
    CAST_ARRAY_CASE(NPY_LONGLONG, npy_longlong, npy_long, NPY_LONG)
    CAST_ARRAY_CASE(NPY_ULONGLONG, npy_ulonglong, npy_long, NPY_LONG)
    CAST_ARRAY_CASE(NPY_FLOAT, npy_float, npy_long, NPY_LONG)
    CAST_ARRAY_CASE(NPY_DOUBLE, npy_double, npy_long, NPY_LONG)
    CAST_ARRAY_CASE(NPY_LONGDOUBLE, npy_longdouble, npy_long, NPY_LONG)
    F_CAST_ARRAY_CASE(NPY_HALF, npy_half, npy_long, NPY_LONG, half_cast_long)
  }
}

void Any_to_Int(PyArrayObject **array, PyArrayObject **result, int type) {
  npy_intp ndims = PyArray_NDIM(*array);
  npy_intp *shape = PyArray_SHAPE(*array);
  npy_intp size = PyArray_SIZE(*array);
  npy_intp *strides = ((PyArrayObject_fields *)*array)->strides;
  npy_intp *new_strides = (npy_intp *)malloc(sizeof(npy_intp) * ndims);
  npy_intp i;
  switch (type) {
    CAST_ARRAY_CASE(NPY_BOOL, npy_bool, npy_int, NPY_INT)
    CAST_ARRAY_CASE(NPY_BYTE, npy_byte, npy_int, NPY_INT)
    CAST_ARRAY_CASE(NPY_UBYTE, npy_ubyte, npy_int, NPY_INT)
    CAST_ARRAY_CASE(NPY_SHORT, npy_short, npy_int, NPY_INT)
    CAST_ARRAY_CASE(NPY_USHORT, npy_ushort, npy_int, NPY_INT)
    NO_CONVERT_CASE(NPY_INT)
    CAST_ARRAY_CASE(NPY_UINT, npy_uint, npy_int, NPY_INT)
    CAST_ARRAY_CASE(NPY_LONG, npy_long, npy_int, NPY_INT)
    CAST_ARRAY_CASE(NPY_ULONG, npy_ulong, npy_int, NPY_INT)
    CAST_ARRAY_CASE(NPY_LONGLONG, npy_longlong, npy_int, NPY_INT)
    CAST_ARRAY_CASE(NPY_ULONGLONG, npy_ulonglong, npy_int, NPY_INT)
    CAST_ARRAY_CASE(NPY_FLOAT, npy_float, npy_int, NPY_INT)
    CAST_ARRAY_CASE(NPY_DOUBLE, npy_double, npy_int, NPY_INT)
    CAST_ARRAY_CASE(NPY_LONGDOUBLE, npy_longdouble, npy_int, NPY_INT)
    F_CAST_ARRAY_CASE(NPY_HALF, npy_half, npy_int, NPY_INT, half_cast_int)
  }
}

void Any_to_Byte_(PyArrayObject **array, PyArrayObject **result, int type) {
  npy_intp ndims = PyArray_NDIM(*array);
  npy_intp *shape = PyArray_SHAPE(*array);
  npy_intp size = PyArray_SIZE(*array);
  npy_intp *strides = ((PyArrayObject_fields *)*array)->strides;
  npy_intp *new_strides = (npy_intp *)malloc(sizeof(npy_intp) * ndims);
  npy_intp i;
  switch (type) {
    CAST_ARRAY_CASE(NPY_BOOL, npy_bool, npy_byte, NPY_BYTE)
    NO_CONVERT_CASE(NPY_BYTE)
    CAST_ARRAY_CASE(NPY_UBYTE, npy_ubyte, npy_byte, NPY_BYTE)
    CAST_ARRAY_CASE(NPY_SHORT, npy_short, npy_byte, NPY_BYTE)
    CAST_ARRAY_CASE(NPY_USHORT, npy_ushort, npy_byte, NPY_BYTE)
    CAST_ARRAY_CASE(NPY_INT, npy_int, npy_byte, NPY_BYTE)
    CAST_ARRAY_CASE(NPY_UINT, npy_uint, npy_byte, NPY_BYTE)
    CAST_ARRAY_CASE(NPY_LONG, npy_long, npy_byte, NPY_BYTE)
    CAST_ARRAY_CASE(NPY_ULONG, npy_ulong, npy_byte, NPY_BYTE)
    CAST_ARRAY_CASE(NPY_LONGLONG, npy_longlong, npy_byte, NPY_BYTE)
    CAST_ARRAY_CASE(NPY_ULONGLONG, npy_ulonglong, npy_byte, NPY_BYTE)
    CAST_ARRAY_CASE(NPY_FLOAT, npy_float, npy_byte, NPY_BYTE)
    CAST_ARRAY_CASE(NPY_DOUBLE, npy_double, npy_byte, NPY_BYTE)
    CAST_ARRAY_CASE(NPY_LONGDOUBLE, npy_longdouble, npy_byte, NPY_BYTE)
    F_CAST_ARRAY_CASE(NPY_HALF, npy_half, npy_byte, NPY_BYTE, half_cast_byte)
  }
}

void Any_to_UByte(PyArrayObject **array, PyArrayObject **result, int type) {
  npy_intp ndims = PyArray_NDIM(*array);
  npy_intp *shape = PyArray_SHAPE(*array);
  npy_intp size = PyArray_SIZE(*array);
  npy_intp *strides = ((PyArrayObject_fields *)*array)->strides;
  npy_intp *new_strides = (npy_intp *)malloc(sizeof(npy_intp) * ndims);
  npy_intp i;
  switch (type) {
    CAST_ARRAY_CASE(NPY_BOOL, npy_bool, npy_ubyte, NPY_UBYTE)
    CAST_ARRAY_CASE(NPY_BYTE, npy_byte, npy_ubyte, NPY_UBYTE)
    NO_CONVERT_CASE(NPY_UBYTE)
    CAST_ARRAY_CASE(NPY_SHORT, npy_short, npy_ubyte, NPY_UBYTE)
    CAST_ARRAY_CASE(NPY_USHORT, npy_ushort, npy_ubyte, NPY_UBYTE)
    CAST_ARRAY_CASE(NPY_INT, npy_int, npy_ubyte, NPY_UBYTE)
    CAST_ARRAY_CASE(NPY_UINT, npy_uint, npy_ubyte, NPY_UBYTE)
    CAST_ARRAY_CASE(NPY_LONG, npy_long, npy_ubyte, NPY_UBYTE)
    CAST_ARRAY_CASE(NPY_ULONG, npy_ulong, npy_ubyte, NPY_UBYTE)
    CAST_ARRAY_CASE(NPY_LONGLONG, npy_longlong, npy_ubyte, NPY_UBYTE)
    CAST_ARRAY_CASE(NPY_ULONGLONG, npy_ulonglong, npy_ubyte, NPY_UBYTE)
    CAST_ARRAY_CASE(NPY_FLOAT, npy_float, npy_ubyte, NPY_UBYTE)
    CAST_ARRAY_CASE(NPY_DOUBLE, npy_double, npy_ubyte, NPY_UBYTE)
    CAST_ARRAY_CASE(NPY_LONGDOUBLE, npy_longdouble, npy_ubyte, NPY_UBYTE)
    F_CAST_ARRAY_CASE(NPY_HALF, npy_half, npy_ubyte, NPY_UBYTE, half_cast_ubyte)
  }
}

void Any_to_Uint(PyArrayObject **array, PyArrayObject **result, int type) {
  npy_intp ndims = PyArray_NDIM(*array);
  npy_intp *shape = PyArray_SHAPE(*array);
  npy_intp size = PyArray_SIZE(*array);
  npy_intp *strides = ((PyArrayObject_fields *)*array)->strides;
  npy_intp *new_strides = (npy_intp *)malloc(sizeof(npy_intp) * ndims);
  npy_intp i;
  switch (type) {
    CAST_ARRAY_CASE(NPY_BOOL, npy_bool, npy_uint, NPY_UINT)
    CAST_ARRAY_CASE(NPY_BYTE, npy_byte, npy_uint, NPY_UINT)
    CAST_ARRAY_CASE(NPY_UBYTE, npy_ubyte, npy_uint, NPY_UINT)
    CAST_ARRAY_CASE(NPY_SHORT, npy_short, npy_uint, NPY_UINT)
    CAST_ARRAY_CASE(NPY_USHORT, npy_ushort, npy_uint, NPY_UINT)
    CAST_ARRAY_CASE(NPY_INT, npy_int, npy_uint, NPY_UINT)
    NO_CONVERT_CASE(NPY_UINT)
    CAST_ARRAY_CASE(NPY_LONG, npy_long, npy_uint, NPY_UINT)
    CAST_ARRAY_CASE(NPY_ULONG, npy_ulong, npy_uint, NPY_UINT)
    CAST_ARRAY_CASE(NPY_LONGLONG, npy_longlong, npy_uint, NPY_UINT)
    CAST_ARRAY_CASE(NPY_ULONGLONG, npy_ulonglong, npy_uint, NPY_UINT)
    CAST_ARRAY_CASE(NPY_FLOAT, npy_float, npy_uint, NPY_UINT)
    CAST_ARRAY_CASE(NPY_DOUBLE, npy_double, npy_uint, NPY_UINT)
    CAST_ARRAY_CASE(NPY_LONGDOUBLE, npy_longdouble, npy_uint, NPY_UINT)
    F_CAST_ARRAY_CASE(NPY_HALF, npy_half, npy_uint, NPY_UINT, half_cast_uint)
  }
}

void Any_to_Ulong(PyArrayObject **array, PyArrayObject **result, int type) {
  npy_intp ndims = PyArray_NDIM(*array);
  npy_intp *shape = PyArray_SHAPE(*array);
  npy_intp size = PyArray_SIZE(*array);
  npy_intp *strides = ((PyArrayObject_fields *)*array)->strides;
  npy_intp *new_strides = (npy_intp *)malloc(sizeof(npy_intp) * ndims);
  npy_intp i;
  switch (type) {
    CAST_ARRAY_CASE(NPY_BOOL, npy_bool, npy_ulong, NPY_ULONG)
    CAST_ARRAY_CASE(NPY_BYTE, npy_byte, npy_ulong, NPY_ULONG)
    CAST_ARRAY_CASE(NPY_UBYTE, npy_ubyte, npy_ulong, NPY_ULONG)
    CAST_ARRAY_CASE(NPY_SHORT, npy_short, npy_ulong, NPY_ULONG)
    CAST_ARRAY_CASE(NPY_USHORT, npy_ushort, npy_ulong, NPY_ULONG)
    CAST_ARRAY_CASE(NPY_INT, npy_int, npy_ulong, NPY_ULONG)
    CAST_ARRAY_CASE(NPY_UINT, npy_uint, npy_ulong, NPY_ULONG)
    CAST_ARRAY_CASE(NPY_LONG, npy_long, npy_ulong, NPY_ULONG)
    NO_CONVERT_CASE(NPY_ULONG)
    CAST_ARRAY_CASE(NPY_LONGLONG, npy_longlong, npy_ulong, NPY_ULONG)
    CAST_ARRAY_CASE(NPY_ULONGLONG, npy_ulonglong, npy_ulong, NPY_ULONG)
    CAST_ARRAY_CASE(NPY_FLOAT, npy_float, npy_ulong, NPY_ULONG)
    CAST_ARRAY_CASE(NPY_DOUBLE, npy_double, npy_ulong, NPY_ULONG)
    CAST_ARRAY_CASE(NPY_LONGDOUBLE, npy_longdouble, npy_ulong, NPY_ULONG)
    F_CAST_ARRAY_CASE(NPY_HALF, npy_half, npy_ulong, NPY_ULONG, half_cast_ulong)
  }
}

void Any_to_ULongLong(PyArrayObject **array, PyArrayObject **result, int type) {
  npy_intp ndims = PyArray_NDIM(*array);
  npy_intp *shape = PyArray_SHAPE(*array);
  npy_intp size = PyArray_SIZE(*array);
  npy_intp *strides = ((PyArrayObject_fields *)*array)->strides;
  npy_intp *new_strides = (npy_intp *)malloc(sizeof(npy_intp) * ndims);
  npy_intp i;
  switch (type) {
    CAST_ARRAY_CASE(NPY_BOOL, npy_bool, npy_ulonglong, NPY_ULONGLONG)
    CAST_ARRAY_CASE(NPY_BYTE, npy_byte, npy_ulonglong, NPY_ULONGLONG)
    CAST_ARRAY_CASE(NPY_UBYTE, npy_ubyte, npy_ulonglong, NPY_ULONGLONG)
    CAST_ARRAY_CASE(NPY_SHORT, npy_short, npy_ulonglong, NPY_ULONGLONG)
    CAST_ARRAY_CASE(NPY_USHORT, npy_ushort, npy_ulonglong, NPY_ULONGLONG)
    CAST_ARRAY_CASE(NPY_INT, npy_int, npy_ulonglong, NPY_ULONGLONG)
    CAST_ARRAY_CASE(NPY_UINT, npy_uint, npy_ulonglong, NPY_ULONGLONG)
    CAST_ARRAY_CASE(NPY_LONG, npy_long, npy_ulonglong, NPY_ULONGLONG)
    CAST_ARRAY_CASE(NPY_ULONG, npy_ulong, npy_ulonglong, NPY_ULONGLONG)
    CAST_ARRAY_CASE(NPY_LONGLONG, npy_longlong, npy_ulonglong, NPY_ULONGLONG)
    NO_CONVERT_CASE(NPY_ULONGLONG)
    CAST_ARRAY_CASE(NPY_FLOAT, npy_float, npy_ulonglong, NPY_ULONGLONG)
    CAST_ARRAY_CASE(NPY_DOUBLE, npy_double, npy_ulonglong, NPY_ULONGLONG)
    CAST_ARRAY_CASE(NPY_LONGDOUBLE, npy_longdouble, npy_ulonglong,
                    NPY_ULONGLONG)
    F_CAST_ARRAY_CASE(NPY_HALF, npy_half, npy_ulonglong, NPY_ULONGLONG,
                      half_cast_ulonglong)
  }
}

void Any_to_Short(PyArrayObject **array, PyArrayObject **result, int type) {
  npy_intp ndims = PyArray_NDIM(*array);
  npy_intp *shape = PyArray_SHAPE(*array);
  npy_intp size = PyArray_SIZE(*array);
  npy_intp *strides = ((PyArrayObject_fields *)*array)->strides;
  npy_intp *new_strides = (npy_intp *)malloc(sizeof(npy_intp) * ndims);
  npy_intp i;
  switch (type) {
    CAST_ARRAY_CASE(NPY_BOOL, npy_bool, npy_short, NPY_SHORT)
    CAST_ARRAY_CASE(NPY_BYTE, npy_byte, npy_short, NPY_SHORT)
    CAST_ARRAY_CASE(NPY_UBYTE, npy_ubyte, npy_short, NPY_SHORT)
    NO_CONVERT_CASE(NPY_SHORT)
    CAST_ARRAY_CASE(NPY_USHORT, npy_ushort, npy_short, NPY_SHORT)
    CAST_ARRAY_CASE(NPY_INT, npy_int, npy_short, NPY_SHORT)
    CAST_ARRAY_CASE(NPY_UINT, npy_uint, npy_short, NPY_SHORT)
    CAST_ARRAY_CASE(NPY_LONG, npy_long, npy_short, NPY_SHORT)
    CAST_ARRAY_CASE(NPY_ULONG, npy_ulong, npy_short, NPY_SHORT)
    CAST_ARRAY_CASE(NPY_LONGLONG, npy_longlong, npy_short, NPY_SHORT)
    CAST_ARRAY_CASE(NPY_ULONGLONG, npy_ulonglong, npy_short, NPY_SHORT)
    CAST_ARRAY_CASE(NPY_FLOAT, npy_float, npy_short, NPY_SHORT)
    CAST_ARRAY_CASE(NPY_DOUBLE, npy_double, npy_short, NPY_SHORT)
    CAST_ARRAY_CASE(NPY_LONGDOUBLE, npy_longdouble, npy_short, NPY_SHORT)
    F_CAST_ARRAY_CASE(NPY_HALF, npy_half, npy_short, NPY_SHORT, half_cast_short)
  }
}

void Any_to_UShort(PyArrayObject **array, PyArrayObject **result, int type) {
  npy_intp ndims = PyArray_NDIM(*array);
  npy_intp *shape = PyArray_SHAPE(*array);
  npy_intp size = PyArray_SIZE(*array);
  npy_intp *strides = ((PyArrayObject_fields *)*array)->strides;
  npy_intp *new_strides = (npy_intp *)malloc(sizeof(npy_intp) * ndims);
  npy_intp i;
  switch (type) {
    CAST_ARRAY_CASE(NPY_BOOL, npy_bool, npy_ushort, NPY_USHORT)
    CAST_ARRAY_CASE(NPY_BYTE, npy_byte, npy_ushort, NPY_USHORT)
    CAST_ARRAY_CASE(NPY_UBYTE, npy_ubyte, npy_ushort, NPY_USHORT)
    CAST_ARRAY_CASE(NPY_SHORT, npy_short, npy_ushort, NPY_USHORT)
    NO_CONVERT_CASE(NPY_USHORT)
    CAST_ARRAY_CASE(NPY_INT, npy_int, npy_ushort, NPY_USHORT)
    CAST_ARRAY_CASE(NPY_UINT, npy_uint, npy_ushort, NPY_USHORT)
    CAST_ARRAY_CASE(NPY_LONG, npy_long, npy_ushort, NPY_USHORT)
    CAST_ARRAY_CASE(NPY_ULONG, npy_ulong, npy_ushort, NPY_USHORT)
    CAST_ARRAY_CASE(NPY_LONGLONG, npy_longlong, npy_ushort, NPY_USHORT)
    CAST_ARRAY_CASE(NPY_ULONGLONG, npy_ulonglong, npy_ushort, NPY_USHORT)
    CAST_ARRAY_CASE(NPY_FLOAT, npy_float, npy_ushort, NPY_USHORT)
    CAST_ARRAY_CASE(NPY_DOUBLE, npy_double, npy_ushort, NPY_USHORT)
    CAST_ARRAY_CASE(NPY_LONGDOUBLE, npy_longdouble, npy_ushort, NPY_USHORT)
    F_CAST_ARRAY_CASE(NPY_HALF, npy_half, npy_ushort, NPY_USHORT,
                      half_cast_ushort)
  }
}

void Any_to_Bool(PyArrayObject **array, PyArrayObject **result, int type) {
  npy_intp ndims = PyArray_NDIM(*array);
  npy_intp *shape = PyArray_SHAPE(*array);
  npy_intp size = PyArray_SIZE(*array);
  npy_intp *strides = ((PyArrayObject_fields *)*array)->strides;
  npy_intp *new_strides = (npy_intp *)malloc(sizeof(npy_intp) * ndims);
  npy_intp i;
  switch (type) {
    NO_CONVERT_CASE(NPY_BOOL)
    CAST_ARRAY_CASE_BOOL(NPY_BYTE, npy_byte)
    CAST_ARRAY_CASE_BOOL(NPY_UBYTE, npy_ubyte)
    CAST_ARRAY_CASE_BOOL(NPY_SHORT, npy_short)
    CAST_ARRAY_CASE_BOOL(NPY_USHORT, npy_ushort)
    CAST_ARRAY_CASE_BOOL(NPY_INT, npy_int)
    CAST_ARRAY_CASE_BOOL(NPY_UINT, npy_uint)
    CAST_ARRAY_CASE_BOOL(NPY_LONG, npy_long)
    CAST_ARRAY_CASE_BOOL(NPY_ULONG, npy_ulong)
    CAST_ARRAY_CASE_BOOL(NPY_LONGLONG, npy_longlong)
    CAST_ARRAY_CASE_BOOL(NPY_ULONGLONG, npy_ulonglong)
    CAST_ARRAY_CASE_BOOL(NPY_FLOAT, npy_float)
    CAST_ARRAY_CASE_BOOL(NPY_DOUBLE, npy_double)
    CAST_ARRAY_CASE_BOOL(NPY_LONGDOUBLE, npy_longdouble)
    F_CAST_ARRAY_CASE(NPY_HALF, npy_half, npy_bool, NPY_BOOL, half_cast_short)
  }
}

inline void As_Type(PyArrayObject **a, PyArrayObject **result, int self_type,
                    int target_type) {
  switch (target_type) {
    As_Type_Cases(Bool, NPY_BOOL);
    As_Type_Cases(Byte_, NPY_BYTE);
    As_Type_Cases(UByte, NPY_UBYTE);
    As_Type_Cases(Short, NPY_SHORT);
    As_Type_Cases(UShort, NPY_USHORT);
    As_Type_Cases(Int, NPY_INT);
    As_Type_Cases(Uint, NPY_UINT);
    As_Type_Cases(Long, NPY_LONG);
    As_Type_Cases(Ulong, NPY_ULONG); // haven't implemented yet
    As_Type_Cases(LongLong, NPY_LONGLONG);
    As_Type_Cases(ULongLong, NPY_ULONGLONG);
    As_Type_Cases(Float, NPY_FLOAT);
    As_Type_Cases(Double, NPY_DOUBLE);
    As_Type_Cases(Half, NPY_HALF);
  default:
    *a = NULL;
    break;
  }
}

void as_type(PyArrayObject **a, PyArrayObject **result, int target_type) {
  int a_dtype = PyArray_TYPE(*a);
  As_Type(a, result, a_dtype, target_type);
}

int div_result_type_pick(int npy_enum) {
  switch (npy_enum) {
  case NPY_BOOL:
  case NPY_BYTE:
  case NPY_UBYTE:
  case NPY_SHORT:
  case NPY_USHORT:
    return NPY_HALF;
  case NPY_INT:
  case NPY_UINT:
  case NPY_LONG:
  case NPY_ULONG:
    return NPY_FLOAT;
  case NPY_LONGLONG:
  case NPY_ULONGLONG:
    return NPY_DOUBLE;
  case NPY_FLOAT:
    return NPY_FLOAT;
  case NPY_DOUBLE:
    return NPY_DOUBLE;
  case NPY_LONGDOUBLE:
    return NPY_LONGDOUBLE;
  case NPY_HALF:
    return NPY_HALF;
  default:
    return -1;
  }
}

bool is_float_number(int dtype) {
  switch (dtype) {
  case NPY_FLOAT:
  case NPY_DOUBLE:
  case NPY_LONGDOUBLE:
  case NPY_HALF:
    return true;
  default:
    return false;
  }
}

bool is_uint_number(int dtype) {
  return (dtype == NPY_UBYTE || dtype == NPY_USHORT || dtype == NPY_UINT ||
          dtype == NPY_ULONG || dtype == NPY_ULONGLONG);
}

int float_type_based_on_size(int size) {
  switch (size) {
  case 2:
    return NPY_HALF;
  case 4:
    return NPY_FLOAT;
  case 8:
    return NPY_DOUBLE;
  case 16:
    return NPY_LONGDOUBLE;
  default:
    return gloabal_float_type;
  }
}

int binary_result_type(int op, int a_dtype, int a_size, int b_dtype,
                       int b_size) {
  switch (a_dtype) {
  case NPY_BOOL:
  case NPY_BYTE:
    int_switch_cases(op);
  case NPY_UBYTE:
    uint_switch_cases(op);
  case NPY_SHORT:
    int_switch_cases(op);
  case NPY_USHORT:
    uint_switch_cases(op);
  case NPY_INT:
    int_switch_cases(op);
  case NPY_UINT:
    uint_switch_cases(op);
  case NPY_LONG:
    int_switch_cases(op);
  case NPY_ULONG:
    uint_switch_cases(op);
  case NPY_LONGLONG:
    int_switch_cases(op);
  case NPY_ULONGLONG:
    uint_switch_cases(op);
  case NPY_FLOAT:
  case NPY_DOUBLE:
  case NPY_LONGDOUBLE:
    switch (op) {
    case ADD:
    case SUB:
    case MUL:
    case DIV:
    case MOD:
      return float_type_based_on_size(max(a_size, b_size));
    case POW:
      return gloabal_float_type;
    case FLOOR_DIV:
      return max(a_dtype, b_dtype);
    case BITWISE_AND:
    case BITWISE_OR:
    case BITWISE_XOR:
      return NPY_BOOL;
    case LSHIFT:
    case RSHIFT:
      PyErr_SetString(
          PyExc_TypeError,
          "unsupported operand type(s) for >> or <<: 'float' and 'float'");
      return -1;
    case SQUARE:
      return float_type_based_on_size(max(a_size, b_size));
    default:
      return max(a_dtype, b_dtype);
    }
    break;
  case NPY_CFLOAT:
  case NPY_CDOUBLE:
  case NPY_CLONGDOUBLE:
  case NPY_OBJECT:
  case NPY_STRING:
  case NPY_UNICODE:
  case NPY_VOID:
  case NPY_DATETIME:
  case NPY_TIMEDELTA:
    PyErr_SetString(PyExc_TypeError, "unsupported data type");
    return -1;
  case NPY_HALF:
    switch (op) {
    case ADD:
    case SUB:
    case MUL:
    case DIV:
    case MOD:
      return float_type_based_on_size(max(a_size, b_size));
    case POW:
      return gloabal_float_type;
    case FLOOR_DIV:
      if (is_float_number(b_dtype))
        return float_type_based_on_size(max(a_size, b_size));
      else
        return gloabal_float_type;
    case BITWISE_AND:
    case BITWISE_OR:
    case BITWISE_XOR:
      return NPY_BOOL;
    case LSHIFT:
    case RSHIFT:
      PyErr_SetString(
          PyExc_TypeError,
          "unsupported operand type(s) for >> or <<: 'float' and 'float'");
      return -1;
    case SQUARE:
      return float_type_based_on_size(max(a_size, b_size));
    default:
      return max(a_dtype, b_dtype);
    }
    break;
  default:
    PyErr_SetString(PyExc_TypeError, "unsupported data type");
    return -1;
  }
}

int elementwise_result_type(int op, int a_dtype) {
  int a_size = type_2_size[a_dtype];
  int gloabal_float_size = type_2_size[gloabal_float_type];
  switch (op) {
  case SIN:
  case COS:
  case TAN:
  case ARCSIN:
  case ARCCOS:
  case ARCTAN:
  case SINH:
  case COSH:
  case TANH:
  case ARCSINH:
  case ARCCOSH:
  case ARCTANH:
  case SQRT:
  case EXP:
  case LOG:
    return float_type_based_on_size(max(a_size, gloabal_float_size));
  case ABS:
    return a_dtype;
  default:
    return a_dtype;
  }
}

PyObject *binary_result_type_(PyObject *self, PyObject *const *args,
                              size_t nargsf) {
  (void)self;
  (void)nargsf;
  long op = PyLong_AsLong(args[0]);
  long a_dtype = PyLong_AsLong(args[1]);
  long a_size = PyLong_AsLong(args[2]);
  long b_dtype = PyLong_AsLong(args[3]);
  long b_size = PyLong_AsLong(args[4]);
  if ((op == -1 || a_dtype == -1 || a_size == -1 || b_dtype == -1 ||
       b_size == -1) &&
      PyErr_Occurred())
    return NULL;
  int result = binary_result_type((int)op, (int)a_dtype, (int)a_size,
                                  (int)b_dtype, (int)b_size);
  return PyLong_FromLong(result);
}

int any_to_type_enum(PyObject *a) {
  if (Py_IS_TYPE(a, Tensor_type)) {
    return ((PyArrayObject_fields *)((Tensor *)a)->data)->descr->type_num;
  }
  else if (PyArray_Check(a))
    return ((PyArrayObject_fields *)a)->descr->type_num;
  else if (PyBool_Check(a))
    return NPY_BOOL;
  else if (PyLong_Check(a))
    return NPY_LONG;
  else if (PyFloat_Check(a))
    return gloabal_float_type;
  else if (PyComplex_Check(a))
    return NPY_CDOUBLE;
  else if (PyBytes_Check(a))
    return NPY_STRING;
  else if (PyUnicode_Check(a))
    return NPY_UNICODE;
  else
    return -1;
}

int type_2_size[] = {
    [NPY_BOOL] = sizeof(npy_bool),
    [NPY_BYTE] = sizeof(npy_byte),
    [NPY_UBYTE] = sizeof(npy_ubyte),
    [NPY_SHORT] = sizeof(npy_short),
    [NPY_USHORT] = sizeof(npy_ushort),
    [NPY_INT] = sizeof(npy_int),
    [NPY_UINT] = sizeof(npy_uint),
    [NPY_LONG] = sizeof(npy_long),
    [NPY_ULONG] = sizeof(npy_ulong),
    [NPY_LONGLONG] = sizeof(npy_longlong),
    [NPY_ULONGLONG] = sizeof(npy_ulonglong),
    [NPY_FLOAT] = sizeof(npy_float),
    [NPY_DOUBLE] = sizeof(npy_double),
    [NPY_LONGDOUBLE] = sizeof(npy_longdouble),
    [NPY_CFLOAT] = sizeof(npy_cfloat),
    [NPY_CDOUBLE] = sizeof(npy_cdouble),
    [NPY_CLONGDOUBLE] = sizeof(npy_clongdouble),
    [NPY_HALF] = sizeof(npy_half),
    [NPY_STRING] = 0,
    [NPY_UNICODE] = 0,
    [NPY_VOID] = 0,
    [NPY_OBJECT] = 0,
    [NPY_DATETIME] = sizeof(npy_datetime),
    [NPY_TIMEDELTA] = sizeof(npy_timedelta),
};