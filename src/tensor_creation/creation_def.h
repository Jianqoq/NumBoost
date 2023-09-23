#ifndef TENSOR_CREATION_CREATION_DEF_H
#define TENSOR_CREATION_CREATION_DEF_H

#include "../set_tensor_properties.h"
#include "../tensor.h"

#define Register_Arange_Array()                                                \
  void (*arange_operations[])(PyArrayObject * array, int start, int step,      \
                              npy_intp size) = {                               \
      arange_bool,        arange_byte,       arange_ubyte,     arange_short,   \
      arange_ushort,      arange_int,        arange_uint,      arange_long,    \
      arange_ulong,       arange_longlong,   arange_ulonglong, arange_float,   \
      arange_double,      arange_longdouble, arange_cfloat,    arange_cdouble, \
      arange_clongdouble, arange_object,     arange_string,    arange_unicode, \
      arange_void,        arange_datetime,   arange_timedelta, arange_half};

#define Register_Arange_Method(type)                                           \
  void arange_##type(PyArrayObject *array, int start, int step,                \
                     npy_intp size) {                                          \
    npy_##type *data = (npy_##type *)PyArray_DATA(array);                      \
    npy_intp i;                                                                \
    _Pragma("omp parallel for") for (i = 0; i < size; i++) {                   \
      Generic(npy_##type) val2 = (Generic(npy_##type))(start + i * step);                             \
      data[i] = Demote(npy_##type, val2);                                      \
    }                                                                          \
  }

#define Register_Arange_Not_Support_Type(type)                                 \
  void arange_##type(PyArrayObject *array, int start, int step,                \
                     npy_intp size) {                                          \
    PyErr_SetString(PyExc_RuntimeError, "Not support type");                   \
    return;                                                                    \
  }

PyObject *tensor_new(Tensor *tensor, PyObject *other, PyObject *data,
                     const char *grad_fn);
PyObject *tensor_empty(PyObject *data);
PyObject *arange(PyObject *self, PyObject *args, PyObject *kwds);

#endif // TENSOR_CREATION_CREATION_DEF_H