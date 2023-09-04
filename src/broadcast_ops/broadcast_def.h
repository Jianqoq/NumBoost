#ifndef BROADCAST_FUNC_DEF_H
#define BROADCAST_FUNC_DEF_H
#include "numpy/arrayobject.h"
#include "broadcast_impl.h"

#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

#define Register_Broadcast_Operation_All_Err(type)                             \
  Register_Broadcast_Operation_Err(type, add_);                                \
  Register_Broadcast_Operation_Err(type, sub_);                                \
  Register_Broadcast_Operation_Err(type, mul_);                                \
  Register_Broadcast_Operation_Err(type, div_);                                \
  Register_Broadcast_Operation_Err(type, mod_);                                \
  Register_Broadcast_Operation_Err(type, lshift_);                             \
  Register_Broadcast_Operation_Err(type, rshift_);                             \
  Register_Broadcast_Operation_Err(type, pow_);

#define Register_Broadcast_Operation_Err(type, suffix)                         \
  static PyArrayObject *Broadcast_Standard_##type##_##suffix(                  \
      PyArrayObject *a, PyArrayObject *b, int op_enum, int result_type) {      \
    const char *string[] = {"Operation not supported for", #type, "type"};     \
    size_t length =                                                            \
        strlen(string[0]) + strlen(string[1]) + strlen(string[2]) + 1;         \
    char *string_cat = (char *)malloc(length);                                 \
    strcpy(string_cat, string[0]);                                             \
    strcat(string_cat, string[1]);                                             \
    strcat(string_cat, string[2]);                                             \
    PyErr_SetString(PyExc_TypeError, string_cat);                              \
    free(string_cat);                                                          \
    return NULL;                                                               \
  }
  
#define Register_Broadcast_Operation_Array(sufix)                                                                                                         \
    PyArrayObject *(*broadcast_##sufix[])(PyArrayObject *, PyArrayObject *, int, int) = {                                                                 \
        Broadcast_Standard_bool_##sufix, Broadcast_Standard_byte_##sufix, Broadcast_Standard_ubyte_##sufix, Broadcast_Standard_short_##sufix,             \
        Broadcast_Standard_ushort_##sufix, Broadcast_Standard_int_##sufix, Broadcast_Standard_uint_##sufix, Broadcast_Standard_long_##sufix,              \
        Broadcast_Standard_ulong_##sufix, Broadcast_Standard_longlong_##sufix, Broadcast_Standard_ulonglong_##sufix, Broadcast_Standard_float_##sufix,    \
        Broadcast_Standard_double_##sufix, Broadcast_Standard_longdouble_##sufix, Broadcast_Standard_cfloat_##sufix, Broadcast_Standard_cdouble_##sufix,  \
        Broadcast_Standard_clongdouble_##sufix, Broadcast_Standard_object_##sufix, Broadcast_Standard_string_##sufix, Broadcast_Standard_unicode_##sufix, \
        Broadcast_Standard_void_##sufix, Broadcast_Standard_datetime_##sufix, Broadcast_Standard_timedelta_##sufix, Broadcast_Standard_half_##sufix};

extern PyArrayObject *(**broadcast_operations[])(PyArrayObject *, PyArrayObject *, int, int);

#endif