#ifndef _ELEMENT_OPS_DEF_H
#define _ELEMENT_OPS_DEF_H
#include "../numboost_api.h"
#include "elementwise_kernels.h"

#define Register_ElementWise_Operation_Array(name, sufix)                      \
  PyObject *(*name##_operations##sufix[])(PyObject *, PyObject **, int) = {    \
      elementwise_##name##_bool##sufix,                                        \
      elementwise_##name##_byte##sufix,                                        \
      elementwise_##name##_ubyte##sufix,                                       \
      elementwise_##name##_short##sufix,                                       \
      elementwise_##name##_ushort##sufix,                                      \
      elementwise_##name##_int##sufix,                                         \
      elementwise_##name##_uint##sufix,                                        \
      elementwise_##name##_long##sufix,                                        \
      elementwise_##name##_ulong##sufix,                                       \
      elementwise_##name##_longlong##sufix,                                    \
      elementwise_##name##_ulonglong##sufix,                                   \
      elementwise_##name##_float##sufix,                                       \
      elementwise_##name##_double##sufix,                                      \
      elementwise_##name##_longdouble##sufix,                                  \
      elementwise_##name##_cfloat##sufix,                                      \
      elementwise_##name##_cdouble##sufix,                                     \
      elementwise_##name##_clongdouble##sufix,                                 \
      elementwise_##name##_object##sufix,                                      \
      elementwise_##name##_string##sufix,                                      \
      elementwise_##name##_unicode##sufix,                                     \
      elementwise_##name##_void##sufix,                                        \
      elementwise_##name##_datetime##sufix,                                    \
      elementwise_##name##_timedelta##sufix,                                   \
      elementwise_##name##_half##sufix};

#define Register_ElementWise_Operation_Err(name, type)                         \
  PyObject *elementwise_##name##_##type(PyObject *a, PyObject **out_arr,       \
                                        int out_arr_len) {                     \
    (void)out_arr;                                                             \
    (void)out_arr_len;                                                         \
    PyErr_SetString(PyExc_TypeError, Str(name not supported for type));        \
    PyObject_Print(a, stderr, 0);                                              \
    return NULL;                                                               \
  }

#define Register_ElementWise_Operation(name, in_type, out_type, result_type,   \
                                       inner_loop_body_universal)              \
  PyObject *elementwise_##name##_##in_type(PyObject *a, PyObject **out_arr,    \
                                           int out_arr_len) {                  \
    PyArrayObject **result =                                                   \
        (PyArrayObject **)malloc(sizeof(PyArrayObject *));                     \
    Perform_Universal_Operation(npy_##in_type, npy_##out_type, result,         \
                                result_type, inner_loop_body_universal,        \
                                out_arr, out_arr_len, (result), a);            \
    if (result == NULL) {                                                      \
      return NULL;                                                             \
    } else {                                                                   \
      PyArrayObject *result_array = result[0];                                 \
      free(result);                                                            \
      return (PyObject *)result_array;                                         \
    }                                                                          \
  }

#define Register_ElementWise_Operations_Floating_Types(name,                   \
                                                       universal_loop_body)    \
  Register_ElementWise_Operation(name, float, float, NPY_FLOAT,                \
                                 universal_loop_body);                         \
  Register_ElementWise_Operation(name, double, double, NPY_DOUBLE,             \
                                 universal_loop_body);                         \
  Register_ElementWise_Operation(name, longdouble, longdouble, NPY_LONGDOUBLE, \
                                 universal_loop_body);                         \
  Register_ElementWise_Operation(name, half, half, NPY_HALF,                   \
                                 universal_loop_body);

#define Register_ElementWise_Operations_Interger_Types(name,                   \
                                                       universal_loop_body)    \
  Register_ElementWise_Operation(name, bool, bool, NPY_BOOL,                   \
                                 universal_loop_body);                         \
  Register_ElementWise_Operation(name, byte, byte, NPY_BYTE,                   \
                                 universal_loop_body);                         \
  Register_ElementWise_Operation(name, ubyte, ubyte, NPY_UBYTE,                \
                                 universal_loop_body);                         \
  Register_ElementWise_Operation(name, short, short, NPY_SHORT,                \
                                 universal_loop_body);                         \
  Register_ElementWise_Operation(name, ushort, ushort, NPY_USHORT,             \
                                 universal_loop_body);                         \
  Register_ElementWise_Operation(name, int, int, NPY_INT,                      \
                                 universal_loop_body);                         \
  Register_ElementWise_Operation(name, uint, uint, NPY_UINT,                   \
                                 universal_loop_body);                         \
  Register_ElementWise_Operation(name, long, long, NPY_LONG,                   \
                                 universal_loop_body);                         \
  Register_ElementWise_Operation(name, ulong, ulong, NPY_ULONG,                \
                                 universal_loop_body);                         \
  Register_ElementWise_Operation(name, longlong, longlong, NPY_LONGLONG,       \
                                 universal_loop_body);                         \
  Register_ElementWise_Operation(name, ulonglong, ulonglong, NPY_ULONGLONG,    \
                                 universal_loop_body);

#define Register_ElementWise_Operation_Err_Interger_Types(name)                \
  Register_ElementWise_Operation_Err(name, bool);                              \
  Register_ElementWise_Operation_Err(name, byte);                              \
  Register_ElementWise_Operation_Err(name, ubyte);                             \
  Register_ElementWise_Operation_Err(name, short);                             \
  Register_ElementWise_Operation_Err(name, ushort);                            \
  Register_ElementWise_Operation_Err(name, int);                               \
  Register_ElementWise_Operation_Err(name, uint);                              \
  Register_ElementWise_Operation_Err(name, long);                              \
  Register_ElementWise_Operation_Err(name, ulong);                             \
  Register_ElementWise_Operation_Err(name, longlong);                          \
  Register_ElementWise_Operation_Err(name, ulonglong);

#define Register_ElementWise_Operation_Err_Not_Support_Types(name)             \
  Register_ElementWise_Operation_Err(name, cfloat);                            \
  Register_ElementWise_Operation_Err(name, cdouble);                           \
  Register_ElementWise_Operation_Err(name, clongdouble);                       \
  Register_ElementWise_Operation_Err(name, object);                            \
  Register_ElementWise_Operation_Err(name, string);                            \
  Register_ElementWise_Operation_Err(name, unicode);                           \
  Register_ElementWise_Operation_Err(name, void);                              \
  Register_ElementWise_Operation_Err(name, datetime);                          \
  Register_ElementWise_Operation_Err(name, timedelta);

#define Register_ElementWise_Operation_Method(name, op_enum)                   \
  PyObject *numboost_##name(PyObject *a, PyObject **out_arr) {                 \
    int input_type = any_to_type_enum(a);                                      \
    int result_type = elementwise_result_type(op_enum, input_type);            \
    if (result_type == -1) {                                                   \
      PyErr_SetString(PyExc_TypeError,                                         \
                      Str(name not supported for type));                       \
      return NULL;                                                             \
    }                                                                          \
    assert(result_type <= NPY_HALF);                                           \
    PyObject *result = name##_operations[result_type](a, out_arr, 1);          \
    if (result == NULL) {                                                      \
      return NULL;                                                             \
    }                                                                          \
    return result;                                                             \
  }

#define Register_mudule_elementwise_methods(name, backward_fn_name)            \
  Tensor *_##name(PyObject *numboost_module, PyObject *args, PyObject *kwds) { \
    (void)numboost_module;                                                     \
    PyObject *a = NULL, *out = NULL;                                           \
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", keyword_list, &a,      \
                                     &out)) {                                  \
      return NULL;                                                             \
    }                                                                          \
    if (!Py_IS_TYPE(a, Tensor_type)) {                                         \
      PyErr_SetString(PyExc_TypeError, "a must be Tensor");                    \
      return NULL;                                                             \
    }                                                                          \
    PyObject *outs;                                                            \
    Tensor *to_replace = NULL;                                                 \
    if (out == Py_None || out == NULL) {                                       \
      outs = NULL;                                                             \
    } else if (Py_IS_TYPE(out, Tensor_type)) {                                 \
      to_replace = (Tensor *)out;                                              \
      outs = to_replace->data;                                                 \
    } else {                                                                   \
      PyErr_SetString(PyExc_TypeError, "out must be None or Tensor");          \
      return NULL;                                                             \
    }                                                                          \
    PyObject *result = numboost_##name(a, &outs);                              \
    Numboost_AssertNULL(result);                                               \
    if (outs) {                                                                \
      if (result != to_replace->data) {                                        \
        Py_DECREF(to_replace->data);                                           \
        to_replace->data = result;                                             \
        Py_INCREF(to_replace);                                                 \
        return to_replace;                                                     \
      } else {                                                                 \
        Py_INCREF(to_replace);                                                 \
        return to_replace;                                                     \
      }                                                                        \
    } else {                                                                   \
      PyObject *to_return =                                                    \
          tensor_new((Tensor *)a, Py_None, result, backward_fn_name);          \
      return (Tensor *)to_return;                                              \
    }                                                                          \
  }

PyObject *numboost_abs(PyObject *a, PyObject **out_arr);
PyObject *numboost_negative(PyObject *a, PyObject **out_arr);
PyObject *numboost_sin(PyObject *a, PyObject **out_arr);
PyObject *numboost_cos(PyObject *a, PyObject **out_arr);
PyObject *numboost_tan(PyObject *a, PyObject **out_arr);
PyObject *numboost_asin(PyObject *a, PyObject **out_arr);
PyObject *numboost_acos(PyObject *a, PyObject **out_arr);
PyObject *numboost_atan(PyObject *a, PyObject **out_arr);
PyObject *numboost_sinh(PyObject *a, PyObject **out_arr);
PyObject *numboost_cosh(PyObject *a, PyObject **out_arr);
PyObject *numboost_tanh(PyObject *a, PyObject **out_arr);
PyObject *numboost_asinh(PyObject *a, PyObject **out_arr);
PyObject *numboost_acosh(PyObject *a, PyObject **out_arr);
PyObject *numboost_atanh(PyObject *a, PyObject **out_arr);
PyObject *numboost_sqrt(PyObject *a, PyObject **out_arr);
PyObject *numboost_log(PyObject *a, PyObject **out_arr);
PyObject *numboost_log10(PyObject *a, PyObject **out_arr);
PyObject *numboost_exp(PyObject *a, PyObject **out_arr);
#endif