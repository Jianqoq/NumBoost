#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "binary_python_methods.h"
#include "../python_magic/python_math_magic.h"
#include "../tensor.h"
#include "binary_op_def.h"

char *keyword_list[] = {"a", "b", "out", NULL};

Register_mudule_methods(add, "AddBackward");
Register_mudule_methods(sub, "SubBackward");
Register_mudule_methods(mul, "MulBackward");
Register_mudule_methods(div, "DivBackward");
Register_mudule_methods(mod, "");
Register_mudule_methods(fdiv, "");

PyObject *nb_module_pow(PyObject *numboost_module, PyObject *args,
                        PyObject *kwds) {
  (void)numboost_module;
  PyObject *a = NULL, *b = NULL, *out = NULL;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|O", keyword_list, &a, &b,
                                   &out)) {
    return NULL;
  }
  if (!Py_IS_TYPE(a, Tensor_type)) {
    PyErr_SetString(PyExc_TypeError, "a must be Tensor");
    return NULL;
  }
  PyObject *outs;
  if (out == Py_None || out == NULL) {
    outs = NULL;
  } else if (Py_IS_TYPE(out, Tensor_type)) {
    outs = ((Tensor *)out)->data;
  } else {
    PyErr_SetString(PyExc_TypeError, "out must be None or Tensor");
    return NULL;
  }
  PyObject *result = numboost_pow_new(a, b, &outs);
  Numboost_AssertNULL(result);
  PyObject *to_return = create_Tensor((Tensor *)a, b, result, "PowBackward");
  if (((Tensor *)a)->require_grad)
    store_power((Tensor*)to_return, b);
  return to_return;
}
