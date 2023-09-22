#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#define PY_SSIZE_T_CLEAN
#include "tensor_methods.h"
#include "allocator/tensor_alloc.h"
#include "libraries/hash/uthash.h"
#include "numboost_api.h"
#include "python_magic/python_math_magic.h"
#include "set_tensor_properties.h"
#include "type_convertor/type_convertor.h"
#include "tensor_creation/creation_def.h"

extern jnp_method *JNP_METHOD;
extern Tensor_need_grad_Dict *TENSOR_NEED_GRAD_DICT;
extern PyTypeObject TensorIterator_type;

PyObject *astype(Tensor *self, PyObject *const *args, size_t nargsf) {
  (void)nargsf;
  int tp = (int)PyLong_AsLong(args[0]);
  PyArrayObject *arr = NULL;
  PyArrayObject *self_data = (PyArrayObject *)self->data;
  as_type(&self_data, &arr, tp);
  if (self->data == NULL || arr == NULL)
    return NULL;
  PyObject *result = tensor_empty((PyObject *)arr);
  ((Tensor *)result)->require_grad = self->require_grad;
  return (PyObject *)result;
}

PyObject *__str__(Tensor *self) {
  DEBUG_PRINT("Calling __str__\n");
  char *result, *dest, *prefix = "Tensor(", *end = ")\n";
  if (TRACK) {
    prefix = "\n\tTensor(";
    end = ")";
  }
  // array string
  /*=======================================================================================*/
  PyObject *py_str = PyObject_Str(self->data);
  const char *str = PyUnicode_AsUTF8(py_str);

  /*=======================================================================================*/
  char require_grad[6];
  sprintf(require_grad, "%s", self->require_grad ? "true" : "false");
  uint64_t str_len = strlen(str);
  uint64_t count = 0;
  uint64_t length = strlen((const char *)prefix);
  for (uint64_t i = 0; i < str_len; i++)
    if (str[i] == '\n')
      count++;
  uint64_t len = length * count + str_len;
  result = (char *)malloc((len + 1) * sizeof(char));
  count = 0;
  uint64_t index = 0;
  while (index < len) {
    if (str[count] != '\n') {
      result[index++] = str[count];
    } else {
      result[index++] = '\n';
      for (uint64_t i = 0; i < length; i++) {
        result[index++] = ' ';
      }
    }
    count++;
  }
  result[index++] = '\0';
  if (!strcmp(self->grad_fn, "")) {
    char *string_array[7] = {
        prefix, result, ", dtype=", "", ", requires_grad=", require_grad, end};
    if (PyArray_IsAnyScalar(self->data)) {
      string_array[3] =
          (char *)(PyArray_DescrFromScalar(self->data)->typeobj->tp_name);
    } else {
      string_array[3] = (char *)(PyArray_DESCR((PyArrayObject *)self->data)
                                     ->typeobj->tp_name);
    }
    uint64_t string_array_len = sizeof(string_array) / sizeof(string_array[0]);
    uint64_t string_total_len = 1;
    for (uint64_t i = 0; i < string_array_len; i++) {
      string_total_len += strlen(string_array[i]);
    }
    dest = (char *)malloc(string_total_len * sizeof(char));
    dest[0] = '\0';
    for (uint64_t i = 0; i < string_array_len; i++) {
      strcat(dest, string_array[i]);
    }
  } else {
    const char *string_array[] = {
        (const char *)prefix,
        (const char *)result,
        ", dtype=",
        PyArray_DESCR((PyArrayObject *)self->data)->typeobj->tp_name,
        ", requires_grad=",
        (const char *)require_grad,
        ", backward=",
        "<",
        self->grad_fn,
        ">",
        ")\n"};
    uint64_t string_array_len = sizeof(string_array) / sizeof(string_array[0]);
    uint64_t string_total_len = 1;
    for (uint64_t i = 0; i < string_array_len; i++) {
      string_total_len += strlen(string_array[i]);
    }
    dest = (char *)malloc(string_total_len * sizeof(char));
    dest[0] = '\0';
    for (uint64_t i = 0; i < string_array_len; i++) {
      strcat(dest, string_array[i]);
    }
  }
  PyObject *representation = PyUnicode_FromString((const char *)dest);
  free(dest);
  free(result);
  Py_DECREF(py_str);
  DEBUG_PRINT("Finished calling __str__\n");
  return representation;
}

PyObject *__repr__(Tensor *self) {
  DEBUG_PRINT("Calling __repr__ done\n");
  return __str__(self);
}

Py_ssize_t __len__(Tensor *self) {
  if (PyArray_IsAnyScalar(self->data))
    return (Py_ssize_t)1;
  npy_intp *shape = PyArray_SHAPE((PyArrayObject *)self->data);
  if (shape == NULL)
    return 1;
  npy_intp size = shape[0];
  return (Py_ssize_t)size;
}

PyObject *__iter__(Tensor *self) {
  return (PyObject *)iterator_new(&TensorIterator_type, self);
}

PyObject *rich_compare(Tensor *self, PyObject *other, int op) {
  PyArray_Descr *descr = NULL;
  int ndim = 0;
  if (PyArray_IsAnyScalar(self->data)) {
    descr = PyArray_DescrFromScalar(self->data);
    ndim = 0;
  } else {
    descr = ((PyArrayObject_fields *)((PyArrayObject *)self->data))->descr;
    ndim = ((PyArrayObject_fields *)((PyArrayObject *)self->data))->nd;
  }
  if (ndim > 0) {
    PyErr_SetString(PyExc_ValueError,
                    "The truth value of an array with more than one dimension "
                    "is ambiguous. Use a.any() or a.all()");
    return NULL;
  }
  bool result = false;
  switch (descr->type_num) {
    Compare(NPY_BOOL, npy_bool, self, other, result, descr, op);
    Compare(NPY_BYTE, npy_byte, self, other, result, descr, op);
    Compare(NPY_UBYTE, npy_ubyte, self, other, result, descr, op);
    Compare(NPY_SHORT, npy_short, self, other, result, descr, op);
    Compare(NPY_USHORT, npy_ushort, self, other, result, descr, op);
    Compare(NPY_INT, npy_int, self, other, result, descr, op);
    Compare(NPY_UINT, npy_uint, self, other, result, descr, op);
    Compare(NPY_LONG, npy_long, self, other, result, descr, op);
    Compare(NPY_ULONG, npy_ulong, self, other, result, descr, op);
    Compare(NPY_LONGLONG, npy_longlong, self, other, result, descr, op);
    Compare(NPY_ULONGLONG, npy_ulonglong, self, other, result, descr, op);
    Compare(NPY_FLOAT, npy_float, self, other, result, descr, op);
    Compare(NPY_DOUBLE, npy_double, self, other, result, descr, op);
  }
  if (result)
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

PyObject *__min__(Tensor *self) {
  return PyLong_FromLongLong(
      ((PyArrayObject_fields *)((PyArrayObject *)self->data))->dimensions[0]);
}

PyObject *get_item(Tensor *self, PyObject *item) {
  DEBUG_PRINT("Getting item in get_item\n");
  PyObject *subarray = PyObject_GetItem(self->data, item);
  if (subarray == NULL)
    return NULL;
  if (TRACK)
    return subarray;
  Tensor *to_return = (Tensor *)tensor_new(self, Py_None, subarray, "SliceBackward");
  if (self->require_grad) {
    DEBUG_PRINT("refcount of item: %d\n", (int)Py_REFCNT(item));
    PyArrayObject *arr = (PyArrayObject *)self->data;
    store_for_slicebackward(to_return, item, PyArray_SHAPE(arr),
                            PyArray_NDIM(arr), self);
    Py_INCREF(item);
  }
  return (PyObject *)to_return;
}

Tensor *T(Tensor *self) {
  DEBUG_PRINT("Calling T\n");
  if (PyArray_IsAnyScalar(self->data)) {
    Tensor *to_return = (Tensor *)tensor_empty(self->data);
    if (self->require_grad) {
      npy_intp *new_axes = (npy_intp *)malloc(sizeof(npy_intp) * 1);
      new_axes[0] = 0;
      store_array_shape(to_return, new_axes, 0);
    }
    return to_return;
  }
  int ndim = ((PyArrayObject_fields *)self->data)->nd;
  npy_intp *new_axes = (npy_intp *)malloc(sizeof(npy_intp) * ndim);
  for (int i = 0; i < ndim; i++)
    new_axes[i] = ndim - i - 1;
  PyArray_Dims new_dims = {new_axes, ndim};
  PyObject *transposed =
      PyArray_Transpose((PyArrayObject *)self->data, &new_dims);
  if (transposed == NULL)
    return NULL;
  Tensor *to_return =
      (Tensor *)tensor_new(self, Py_None, transposed, "TransposeBackward");
  if (self->require_grad)
    store_array_shape(to_return, new_axes, ndim);
  else
    free(new_axes);
  DEBUG_PRINT("Finished calling T\n");
  return to_return;
}

PyObject *dtype(Tensor *self) {
  PyArrayObject *data = (PyArrayObject *)(self->data);
  long type_num = PyArray_TYPE(data);
  return PyLong_FromLong(type_num);
}

PyObject *__new__(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  static char *kwlist[] = {"data", "requires_grad", NULL};
  PyObject *data = NULL, *cache = NULL;
  bool *require_grad = NULL;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|p", kwlist, &data,
                                   &require_grad))
    return NULL;
  if (data) {
    if (!TRACK) {
      cache = PyArray_FromAny(data, NULL, 0, 0, NPY_ARRAY_DEFAULT, NULL);
    } else {
      PyObject *jaxarray = PyObject_CallOneArg(JNP_METHOD->array, data);
      return jaxarray;
    }
  } else
    return NULL;
  Tensor *self = (Tensor *)type->tp_alloc(type, 0);
  if (require_grad == NULL)
    self->require_grad = false;
  else
    self->require_grad = require_grad;
  if (self != NULL) {
    PyObject *zero = PyLong_FromLong(0);
    if (!cache)
      return NULL;
    self->data = cache;
    self->x = Py_None;
    self->y = Py_None;
    self->grad = zero;
    self->grad_fn = "";
    self->graph = Py_None;
    self->axis = Py_None;
    self->vars = 1;
    self->has_conv = 0;
    self->dim = 1;
    Py_INCREF(Py_None);
    Py_INCREF(Py_None);
    Py_INCREF(Py_None);
    Py_INCREF(Py_None);
  }
  return (PyObject *)self;
}

PyObject *__tensor(PyObject *self, PyObject *args, PyObject *kwds) {
  (void)self;
  return __new__(Tensor_type, args, kwds);
}

Tensor *self_transpose(Tensor *self, PyObject *const *args, size_t nargsf) {
  Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
  PyArrayObject *array = (PyArrayObject *)self->data;
  npy_intp *dims = malloc(sizeof(npy_intp) * nargs);
  for (Py_ssize_t i = 0; i < nargs; i++) {
    dims[i] = PyLong_AsLong(args[i]);
  }
  PyArray_Dims shape = {dims, (int)nargs};
  PyObject *result = PyArray_Transpose(array, &shape);
  if (result != NULL && result != self->data) {
    self->data = result;
    Py_DECREF(array);
  } else {
    free(dims);
    return NULL;
  }
  free(dims);
  return self;
}

Tensor *self_reshape(Tensor *self, PyObject *const *args, size_t nargsf) {
  size_t nargs = PyVectorcall_NARGS(nargsf);
  PyArrayObject *array;
  int order = 0;
  if (PyUnicode_Check(args[nargs - 1])) {
    PyObject *order_obj = args[nargs - 1];
    if (PyUnicode_CompareWithASCIIString(order_obj, "C") == 0) {
      order = NPY_CORDER;
    } else if (PyUnicode_CompareWithASCIIString(order_obj, "F") == 0) {
      order = NPY_FORTRANORDER;
    } else {
      PyErr_SetString(PyExc_ValueError, "order must be 'C' or 'F'");
      return NULL;
    }
    nargs -= 1;
  }
  Tensor *tensor = self;
  int length = (int)nargs;
  npy_intp dims[NPY_MAXDIMS] = {0};
  for (uint8_t i = 0; i < length; i++) {
    dims[i] = PyLong_AsLongLong(args[i]);
  }
  PyArray_Dims shape = {dims, length};
  array = (PyArrayObject *)tensor->data;
  PyObject *result = PyArray_Newshape(array, &shape, order);
  if (result != tensor->data) {
    tensor->data = result;
    Py_DECREF(array);
  }
  if (result == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Error in reshape");
    return NULL;
  }
  return self;
}

PyObject *collect_gradients_and_cleanup() {
  Tensor_need_grad_Dict *gradient_entry, *gradient_tmp;
  PyObject *map = PyDict_New();
  if (HASH_COUNT(TENSOR_NEED_GRAD_DICT) == 1) {
    HASH_ITER(hh, TENSOR_NEED_GRAD_DICT, gradient_entry, gradient_tmp) {
      PyDict_SetItem(map, (PyObject *)gradient_entry->tensor,
                     gradient_entry->tensor->grad);
      gradient_entry->tensor->grad = PyLong_FromLong(0);
      HASH_DEL(TENSOR_NEED_GRAD_DICT, gradient_entry);
    }
  } else {
    HASH_ITER(hh, TENSOR_NEED_GRAD_DICT, gradient_entry, gradient_tmp) {
      PyDict_SetItem(map, (PyObject *)gradient_entry->tensor,
                     gradient_entry->tensor->grad);
      HASH_DEL(TENSOR_NEED_GRAD_DICT, gradient_entry);
      gradient_entry->tensor->grad = PyLong_FromLong(0);
    }
  }
  // Cleanup: decrease reference count of list and also all tensor in POWER_DICT
  return map;
}

static void store_tensor_need_grad(long long index, Tensor *tensor) {
  Tensor_need_grad_Dict *entry = NULL;
  if (TENSOR_NEED_GRAD_DICT != NULL)
    HASH_FIND_PTR(TENSOR_NEED_GRAD_DICT, &tensor, entry);
  if (entry == NULL) {
    Tensor_need_grad_Dict *entry =
        (Tensor_need_grad_Dict *)malloc(sizeof(Tensor_need_grad_Dict));
    entry->tensor = tensor;
    entry->index = index;
    HASH_ADD_PTR(TENSOR_NEED_GRAD_DICT, tensor, entry);
  }
}

Py_hash_t __hash__(Tensor *self) { return (Py_hash_t)self; }

Tensor *copy(Tensor *self) {
  PyArrayObject *ret = nb_copy((PyArrayObject *)self->data);
  if (ret == NULL)
    return NULL;
  return (Tensor *)tensor_new(self, Py_None, (PyObject *)ret, "");
}

PyObject *backward(PyObject *self, PyObject *args) {
  DEBUG_PRINT("Generic_backward start\n");
  // Declare variables
  PyObject *current_grad1 = NULL;
  PyObject *current_grad2 = NULL;
  PyObject *grad = NULL;
  Tensor *tensor = NULL;
  long long index = 0;
  const char *grad_fn = NULL;
  // Parse the Python argument tuple
  if (!PyArg_ParseTuple(args, "O", &grad)) {
    return NULL;
  }
  Py_INCREF(grad); // Avoid grad reference count to be 0, current grad ref == 2
  Tensor *self_tensor = (Tensor *)self;
  // If the tensor doesn't require gradient, return an error
  if (!self_tensor->require_grad) {
    Py_DECREF(grad);
    PyErr_SetString(PyExc_RuntimeError, "Tensor require_grad is False");
    return NULL;
  }
  grad_fn = self_tensor->grad_fn;
  unsigned long long depth = self_tensor->vars;
  Stack *stack = createStack(depth);
  Tuple tuple = {self, grad};
  push(stack, tuple);
  // Start the main backward loop
  DEBUG_PRINT("Generic_backward main loop start\n");
  while (stack->len != 0) {
    tuple = pop(stack);
    DEBUG_PRINT("Popped tuple\n");
    PyObject *grad_fn_name = PyObject_GetAttrString(tuple.node, "grad_fn");
    // Handle grad_fn_name being NULL
    if (grad_fn_name == NULL) {
      if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError, "grad_fn_name is NULL");
        free_all_resources();
        return NULL;
      }
      continue;
    } else {
      Py_DECREF(grad_fn_name);
      tensor = (Tensor *)tuple.node;
      grad_fn = tensor->grad_fn;
      // If grad_fn is empty string, perform gradient addition
      if (!strcmp(grad_fn, "")) {
        if (tensor == NULL) {
          PyErr_SetString(PyExc_RuntimeError,
                          "can't convert object from stack to Tensor");
          free_all_resources();
          return NULL;
        }
        PyObject *new_grad = NULL;
        if (TRACK) {
          new_grad = PyNumber_Add(tensor->grad, tuple.ndarray);
        } else {
          if (tensor->grad == PyLong_FromLong(0)) {
            new_grad = tuple.ndarray;
            Py_INCREF(new_grad);
          } else {
            new_grad = tensor_add(tensor->grad, tuple.ndarray);
          }
        }
        if (new_grad == NULL) {
          return NULL;
        }
        if (TRACK) {
          // Check whether the tensor needs gradient
          Tensor_need_grad_Dict *entry = NULL;
          HASH_FIND_PTR(TENSOR_NEED_GRAD_DICT, &tensor, entry);
          if (entry != NULL) {
            Py_DECREF(tensor->grad);
            tensor->grad = new_grad;
            Py_DECREF(tuple.ndarray);
            continue;
          } else {
            store_tensor_need_grad(index, tensor);
            index++;
          }
        }
        Py_DECREF(tensor->grad);
        tensor->grad = new_grad;
        Py_DECREF(tuple.ndarray);
        continue;
      }
    }
    DEBUG_PRINT("grad_fn: %s\n", grad_fn);
    // Get the gradient function and apply it
    get_method(grad_fn)(tensor, tuple.ndarray, &current_grad1, &current_grad2);
    // If both gradients are NULL, return an error
    DEBUG_PRINT("current_grad1: %p\n", current_grad1);
    DEBUG_PRINT("current_grad2: %p\n", current_grad2);
    if (current_grad1 == NULL && current_grad2 == NULL) {
      free_all_resources();
      DEBUG_PRINT("current_grad1 and current_grad2 are NULL\n");
      return NULL;
    }
    // Handle the tensor x
    Tensor *tensor_x = (Tensor *)tensor->x;
    if (tensor_x != tensor) {
      if (current_grad1 != NULL && tensor_x->require_grad) {
        Tuple tuple2 = {tensor->x, current_grad1};
        push(stack, tuple2);
      } else {
        Py_DECREF(current_grad1);
      }
    }
    // Handle the tensor y
    if (Py_IS_TYPE(tensor->y, Tensor_type)) {
      bool require_grad = ((Tensor *)tensor->y)->require_grad;
      if (current_grad2 != NULL && require_grad) {
        Tuple tuple2 = {tensor->y, current_grad2};
        push(stack, tuple2);
      } else {
        Py_DECREF(current_grad2);
      }
    }
    DEBUG_PRINT("Next iteration\n");
    Py_DECREF(tuple.ndarray);
  }
  // Cleanup
  freeStack(stack);
  // If tracking, return the gradients as a tuple
  if (TRACK) {
    DEBUG_PRINT("collecting gradients and cleaning up\n");
    return collect_gradients_and_cleanup();
  }
  // If not tracking, just cleanup and return None
  DEBUG_PRINT("finished cleaning up\n");
  Py_INCREF(Py_None);
  return Py_None;
}