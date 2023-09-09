#ifndef NUMBOOST_BINARY_PYTHON_METHODS_H
#define NUMBOOST_BINARY_PYTHON_METHODS_H
#include <Python.h>

#define Register_mudule_methods(name, backward_fn_name)                        \
  PyObject *nb_module_##name(PyObject *numboost_module, PyObject *args,        \
                             PyObject *kwds) {                                 \
    (void)numboost_module;                                                     \
    PyObject *a = NULL, *b = NULL, *out = NULL;                                \
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|O", keyword_list, &a, &b, \
                                     &out)) {                                  \
      return NULL;                                                             \
    }                                                                          \
    if (!Py_IS_TYPE(a, Tensor_type)) {                                         \
      PyErr_SetString(PyExc_TypeError, "a must be Tensor");                    \
      return NULL;                                                             \
    }                                                                          \
    PyObject *outs;                                                            \
    if (out == Py_None || out == NULL) {                                       \
      outs = NULL;                                                             \
    } else if (Py_IS_TYPE(out, Tensor_type)) {                                 \
      outs = ((Tensor *)out)->data;                                            \
    } else {                                                                   \
      PyErr_SetString(PyExc_TypeError, "out must be None or Tensor");          \
      return NULL;                                                             \
    }                                                                          \
    PyObject *result = numboost_##name##_new(a, b, &outs);                     \
    Numboost_AssertNULL(result);                                               \
    PyObject *to_return =                                                      \
        create_Tensor((Tensor *)a, b, result, backward_fn_name);               \
    return to_return;                                                          \
  }

PyObject *nb_module_add(PyObject *numboost_module, PyObject *args,
                        PyObject *kwds);
PyObject *nb_module_sub(PyObject *numboost_module, PyObject *args,
                        PyObject *kwds);
PyObject *nb_module_mul(PyObject *numboost_module, PyObject *args,
                        PyObject *kwds);
PyObject *nb_module_div(PyObject *numboost_module, PyObject *args,
                        PyObject *kwds);
PyObject *nb_module_pow(PyObject *numboost_module, PyObject *args,
                        PyObject *kwds);
PyObject *nb_module_mod(PyObject *numboost_module, PyObject *args,
                        PyObject *kwds);
PyObject *nb_module_fdiv(PyObject *numboost_module, PyObject *args,
                         PyObject *kwds);
#endif // NUMBOOST_BINARY_PYTHON_METHODS_H