#ifndef BROADCAST_FUNC_DEF_H
#define BROADCAST_FUNC_DEF_H
#include "numpy/arrayobject.h"
#include "broadcast_impl.h"

#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

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