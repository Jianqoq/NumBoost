#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "binary_module_methods.h"
#include "../python_magic/python_math_magic.h"
#include "../tensor.h"
#include "binary_op_def.h"

static char *keyword_list[] = {"a", "b", "out", NULL};

Register_mudule_methods(add, "AddBackward");
Register_mudule_methods(sub, "SubBackward");
Register_mudule_methods(mul, "MulBackward");
Register_mudule_methods(div, "DivBackward");
Register_mudule_methods(mod, "");
Register_mudule_methods(fdiv, "");

PyObject *nb_module_pow(PyObject *numboost_module, PyObject *args,
                        PyObject *kwds) {
  (void)numboost_module;
  PyObject *a = NULL, *power = NULL, *out = NULL;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|O", keyword_list, &a, &power,
                                   &out)) {
    return NULL;
  }
  if (!Py_IS_TYPE(a, Tensor_type)) {
    PyErr_SetString(PyExc_TypeError, "a must be Tensor");
    return NULL;
  }
  PyObject *outs;
  Tensor *to_replace;
  Tensor *a_ = (Tensor *)a;
  if (out == Py_None || out == NULL) {
    outs = NULL;
  } else if (Py_IS_TYPE(out, Tensor_type)) {
    to_replace = (Tensor *)out;
    outs = ((Tensor *)out)->data;
  } else {
    PyErr_SetString(PyExc_TypeError, "out must be None or Tensor");
    return NULL;
  }
  PyObject *result = numboost_pow(a, power, &outs);
  Numboost_AssertNULL(result);
  if (outs) {
    Tensor *to_ret = (Tensor *)outs;
    if (result != to_replace->data) {
      Py_DECREF(to_replace->data);
      to_replace->data = result;
      Py_INCREF(to_replace);
      return (PyObject *)to_replace;
    } else {
      Py_INCREF(to_replace);
      return (PyObject *)to_replace;
    }
  } else {
    PyObject *to_return = create_tensor(a_, Py_None, result, "PowBackward");
    if (a_->require_grad)
      store_power((Tensor*)to_return, power);
    return to_return;
  }
}
