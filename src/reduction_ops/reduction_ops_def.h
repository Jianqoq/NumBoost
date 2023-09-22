#ifndef REDUCTION_OPS_DEF_H
#define REDUCTION_OPS_DEF_H
#include "../numboost_api.h"
#include "reduction_kernels.h"
#include "immintrin.h"

#define Register_Reduction_Operation_Array(name, sufix)                        \
  PyObject *(*reduction_##name##_##sufix[])(PyObject *, PyObject **, int *,    \
                                            int, bool) = {                     \
      reduction_##name##_bool##sufix,                                          \
      reduction_##name##_byte##sufix,                                          \
      reduction_##name##_ubyte##sufix,                                         \
      reduction_##name##_short##sufix,                                         \
      reduction_##name##_ushort##sufix,                                        \
      reduction_##name##_int##sufix,                                           \
      reduction_##name##_uint##sufix,                                          \
      reduction_##name##_long##sufix,                                          \
      reduction_##name##_ulong##sufix,                                         \
      reduction_##name##_longlong##sufix,                                      \
      reduction_##name##_ulonglong##sufix,                                     \
      reduction_##name##_float##sufix,                                         \
      reduction_##name##_double##sufix,                                        \
      reduction_##name##_longdouble##sufix,                                    \
      reduction_##name##_cfloat##sufix,                                        \
      reduction_##name##_cdouble##sufix,                                       \
      reduction_##name##_clongdouble##sufix,                                   \
      reduction_##name##_object##sufix,                                        \
      reduction_##name##_string##sufix,                                        \
      reduction_##name##_unicode##sufix,                                       \
      reduction_##name##_void##sufix,                                          \
      reduction_##name##_datetime##sufix,                                      \
      reduction_##name##_timedelta##sufix,                                     \
      reduction_##name##_half##sufix};

#define Register_Reduction_Operation_Err(name, type)                           \
  PyObject *reduction_##name##_##type(PyObject *a, PyObject **out_arr,         \
                                      int *axes, int axis_len,                 \
                                      bool keepdims) {                         \
    (void)out_arr;                                                             \
    (void)axis_len;                                                            \
    (void)keepdims;                                                            \
    (void)axes;                                                                \
    PyErr_SetString(PyExc_TypeError, Str(name not supported for type));        \
    PyObject_Print(a, stderr, 0);                                              \
    return NULL;                                                               \
  }

#define Register_Reduction_Operation(name, type, result_type_enum, init_val,   \
                                     kernel, Kernel_Pre, Kernel_Post)          \
  PyObject *reduction_##name##_##type(PyObject *a, PyObject **out_arr,         \
                                      int *axes, int axis_len,                 \
                                      bool keepdims) {                         \
    PyArrayObject *out = NULL;                                                 \
    if (*out_arr != NULL) {                                                    \
      if (Py_IS_TYPE(*out_arr, Tensor_type)) {                                 \
        Tensor *to_replace = (Tensor *)*out_arr;                               \
        out = (PyArrayObject *)to_replace->data;                               \
      } else if (Py_IS_TYPE(*out_arr, &PyArray_Type)) {                        \
        out = (PyArrayObject *)*out_arr;                                       \
      } else {                                                                 \
        PyErr_SetString(PyExc_TypeError, "out type not supported");            \
        return NULL;                                                           \
      }                                                                        \
    }                                                                          \
    PyArrayObject *a_ = (PyArrayObject *)a;                                    \
    PyArrayObject *result = NULL;                                              \
    Perform_Reduction_Operation(a_, result, axes, axis_len, out, init_val,     \
                                npy_##type, result_type_enum, keepdims,        \
                                kernel, Kernel_Pre, Kernel_Post);              \
    return (PyObject *)result;                                                 \
  }

#define Register_Arg_Reduction_Operation(name, type, result_type_enum,         \
                                         init_val, kernel)                     \
  PyObject *reduction_##name##_##type(PyObject *a, PyObject **out_arr,         \
                                      int *axes, int axis_len,                 \
                                      bool keepdims) {                         \
    PyArrayObject *out = NULL;                                                 \
    if (*out_arr != NULL) {                                                    \
      if (Py_IS_TYPE(*out_arr, Tensor_type)) {                                 \
        Tensor *to_replace = (Tensor *)*out_arr;                               \
        out = (PyArrayObject *)to_replace->data;                               \
      } else if (Py_IS_TYPE(*out_arr, &PyArray_Type)) {                        \
        out = (PyArrayObject *)*out_arr;                                       \
      } else {                                                                 \
        PyErr_SetString(PyExc_TypeError, "out type not supported");            \
        return NULL;                                                           \
      }                                                                        \
    }                                                                          \
    PyArrayObject *a_ = (PyArrayObject *)a;                                    \
    PyArrayObject *result = NULL;                                              \
    Arg_Reduction_Operation(a_, result, axes, axis_len, out, init_val,         \
                            npy_##type, result_type_enum, keepdims, kernel);   \
    return (PyObject *)result;                                                 \
  }

#define Register_Mean_Reduction_Operation(name, type, result_type_enum,        \
                                          init_val)                            \
  PyObject *reduction_##name##_##type(PyObject *a, PyObject **out_arr,         \
                                      int *axes, int axis_len,                 \
                                      bool keepdims) {                         \
    PyArrayObject *out = NULL;                                                 \
    if (*out_arr != NULL) {                                                    \
      if (Py_IS_TYPE(*out_arr, Tensor_type)) {                                 \
        Tensor *to_replace = (Tensor *)*out_arr;                               \
        out = (PyArrayObject *)to_replace->data;                               \
      } else if (Py_IS_TYPE(*out_arr, &PyArray_Type)) {                        \
        out = (PyArrayObject *)*out_arr;                                       \
      } else {                                                                 \
        PyErr_SetString(PyExc_TypeError, "out type not supported");            \
        return NULL;                                                           \
      }                                                                        \
    }                                                                          \
    PyArrayObject *a_ = (PyArrayObject *)a;                                    \
    PyArrayObject *result = NULL;                                              \
    Perform_Mean_Operation(a_, result, axes, axis_len, out, init_val,          \
                           npy_##type, result_type_enum, keepdims);            \
    return (PyObject *)result;                                                 \
  }
#define Register_Reduction_Operations_Floating_Types(name, init_val, kernel,   \
                                                     Kernel_Pre, Kernel_Post)  \
  Register_Reduction_Operation(name, float, NPY_FLOAT, init_val, kernel,       \
                               Kernel_Pre, Kernel_Post);                       \
  Register_Reduction_Operation(name, double, NPY_DOUBLE, init_val, kernel,     \
                               Kernel_Pre, Kernel_Post);                       \
  Register_Reduction_Operation(name, longdouble, NPY_LONGDOUBLE, init_val,     \
                               kernel, Kernel_Pre, Kernel_Post);               \
  Register_Reduction_Operation(name, half, NPY_HALF, init_val, kernel,         \
                               Kernel_Pre, Kernel_Post);

#define Register_Reduction_Operations_Interger_Types(name, init_val, kernel,   \
                                                     Kernel_Pre, Kernel_Post)  \
  Register_Reduction_Operation(name, bool, NPY_BOOL, init_val, kernel,         \
                               Kernel_Pre, Kernel_Post);                       \
  Register_Reduction_Operation(name, byte, NPY_BYTE, init_val, kernel,         \
                               Kernel_Pre, Kernel_Post);                       \
  Register_Reduction_Operation(name, ubyte, NPY_UBYTE, init_val, kernel,       \
                               Kernel_Pre, Kernel_Post);                       \
  Register_Reduction_Operation(name, short, NPY_SHORT, init_val, kernel,       \
                               Kernel_Pre, Kernel_Post);                       \
  Register_Reduction_Operation(name, ushort, NPY_USHORT, init_val, kernel,     \
                               Kernel_Pre, Kernel_Post);                       \
  Register_Reduction_Operation(name, int, NPY_INT, init_val, kernel,           \
                               Kernel_Pre, Kernel_Post);                       \
  Register_Reduction_Operation(name, uint, NPY_UINT, init_val, kernel,         \
                               Kernel_Pre, Kernel_Post);                       \
  Register_Reduction_Operation(name, long, NPY_LONG, init_val, kernel,         \
                               Kernel_Pre, Kernel_Post);                       \
  Register_Reduction_Operation(name, ulong, NPY_ULONG, init_val, kernel,       \
                               Kernel_Pre, Kernel_Post);                       \
  Register_Reduction_Operation(name, longlong, NPY_LONGLONG, init_val, kernel, \
                               Kernel_Pre, Kernel_Post);                       \
  Register_Reduction_Operation(name, ulonglong, NPY_ULONGLONG, init_val,       \
                               kernel, Kernel_Pre, Kernel_Post);

#define Register_Reduction_Operation_Err_Interger_Types(name)                  \
  Register_Reduction_Operation_Err(name, bool);                                \
  Register_Reduction_Operation_Err(name, byte);                                \
  Register_Reduction_Operation_Err(name, ubyte);                               \
  Register_Reduction_Operation_Err(name, short);                               \
  Register_Reduction_Operation_Err(name, ushort);                              \
  Register_Reduction_Operation_Err(name, int);                                 \
  Register_Reduction_Operation_Err(name, uint);                                \
  Register_Reduction_Operation_Err(name, long);                                \
  Register_Reduction_Operation_Err(name, ulong);                               \
  Register_Reduction_Operation_Err(name, longlong);                            \
  Register_Reduction_Operation_Err(name, ulonglong);

#define Register_Reduction_Operation_Err_Not_Support_Types(name)               \
  Register_Reduction_Operation_Err(name, cfloat);                              \
  Register_Reduction_Operation_Err(name, cdouble);                             \
  Register_Reduction_Operation_Err(name, clongdouble);                         \
  Register_Reduction_Operation_Err(name, object);                              \
  Register_Reduction_Operation_Err(name, string);                              \
  Register_Reduction_Operation_Err(name, unicode);                             \
  Register_Reduction_Operation_Err(name, void);                                \
  Register_Reduction_Operation_Err(name, datetime);                            \
  Register_Reduction_Operation_Err(name, timedelta);

#define Register_Reduction_Operation_Method(name, sufix, op_enum)              \
  PyObject *numboost_##name(PyObject *a, PyObject **out_arr, int *axes,        \
                            int axis_len, bool keepdims) {                     \
    int input_type = any_to_type_enum(a);                                      \
    int result_type = reduction_result_type(op_enum, input_type);              \
    if (result_type == -1) {                                                   \
      PyErr_SetString(PyExc_TypeError,                                         \
                      Str(name not supported for type));                       \
      return NULL;                                                             \
    }                                                                          \
    assert(result_type <= NPY_HALF);                                           \
    PyObject *result = reduction_##name##_##sufix[result_type](                \
        a, out_arr, axes, axis_len, keepdims);                                 \
    if (result == NULL) {                                                      \
      return NULL;                                                             \
    }                                                                          \
    return result;                                                             \
  }

#define Register_Arg_Reduction_Operation_Method(name, sufix, op_enum)          \
  PyObject *numboost_##name(PyObject *a, PyObject **out_arr, int *axes,        \
                            int axis_len, bool keepdims) {                     \
    int input_type = any_to_type_enum(a);                                      \
    PyObject *result = reduction_##name##_##sufix[PyArray_TYPE(a)](            \
        a, out_arr, axes, axis_len, keepdims);                                 \
    if (result == NULL) {                                                      \
      return NULL;                                                             \
    }                                                                          \
    return result;                                                             \
  }

#define Register_mudule_reduction_methods(name, backward_fn_name)              \
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
    Tensor *to_replace;                                                        \
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
      Tensor *to_ret = (Tensor *)outs;                                         \
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
          create_tensor((Tensor *)a, Py_None, result, backward_fn_name);       \
      return (Tensor *)to_return;                                              \
    }                                                                          \
  }

PyObject *numboost_sum(PyObject *a, PyObject **out_arr, int *axes, int axis_len,
                       bool keepdims);
PyObject *numboost_min(PyObject *a, PyObject **out_arr, int *axes, int axis_len,
                       bool keepdims);
PyObject *numboost_max(PyObject *a, PyObject **out_arr, int *axes, int axis_len,
                       bool keepdims);
PyObject *numboost_argmax(PyObject *a, PyObject **out_arr, int *axes,
                          int axis_len, bool keepdims);
PyObject *numboost_argmin(PyObject *a, PyObject **out_arr, int *axes,
                          int axis_len, bool keepdims);
PyObject *numboost_mean(PyObject *a, PyObject **out_arr, int *axes,
                        int axis_len, bool keepdims);
#endif