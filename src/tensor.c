#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define PY_SSIZE_T_CLEAN
#define NPY_TARGET_VERSION NPY_1_25_API_VERSION
#include "tensor.h"
#include "allocator/allocator.h"
#include "allocator/tensor_alloc.h"
#include "binary_ops/binary_module_methods.h"
#include "clinic/tensor_methods.c.h"
#include "numboost_api.h"
#include "numpy/arrayobject.h"
#include "omp.h"
#include "python_magic/python_math_magic.h"
#include "structmember.h"
#include "tensor_methods.h"
#include "type_convertor/type_convertor.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>

Dict *dict = NULL;
XLA_OPS *xla_ops = NULL;
bool TRACK = 0;
np_method *NP_METHOD = NULL;
Array_Shape *ARRAY_SHAPE = NULL;
Power_Dict *POWER_DICT = NULL;
Log_Dict *LOG_DICT = NULL;
jnp_method *JNP_METHOD = NULL;
Tensor_need_grad_Dict *TENSOR_NEED_GRAD_DICT = NULL;
Tensordot_Dict *TENSORDOT_DICT = NULL;
Slice_Dict *SLICE_DICT = NULL;
Zeros_Array_Dict *ZEROS_ARRAY_DICT = NULL;
extern PyTypeObject TensorIterator_type;

void store_for_slicebackward(Tensor *key, PyObject *slice_obj, npy_intp *ptr,
                             int nd, Tensor *parent) {
  Slice_Dict *entry = NULL;
  HASH_FIND_PTR(SLICE_DICT, &key, entry);
  if (entry == NULL) {
    entry = (Slice_Dict *)malloc(sizeof(Slice_Dict));
    entry->key = key;
    entry->slice_obj = slice_obj;
    entry->origin_shape = ptr;
    entry->origin_shape_nd = nd;
    entry->parent = parent;
    HASH_ADD_PTR(SLICE_DICT, key, entry);
  }
  Zeros_Array_Dict *entry2 = NULL;
  HASH_FIND_PTR(ZEROS_ARRAY_DICT, &parent, entry2);
  if (entry2 == NULL) {
    entry2 = (Zeros_Array_Dict *)malloc(sizeof(Zeros_Array_Dict));
    PyObject *zeros = PyArray_Zeros(nd, (npy_intp const *)ptr, NULL, 0);
    entry2->parent = parent;
    entry2->zeros_array = zeros;
    HASH_ADD_PTR(ZEROS_ARRAY_DICT, parent, entry2);
  }
}

void get_slice_objs(Tensor *key, npy_intp **origin_shape, PyObject **slice_obj,
                    int *nd, PyObject **zeros_array) {
  Slice_Dict *entry = NULL;
  Tensor *parent = NULL;
  DEBUG_PRINT("get_slice_objs\n");
  HASH_FIND_PTR(SLICE_DICT, &key, entry);
  if (entry != NULL) {
    *origin_shape = entry->origin_shape;
    *slice_obj = entry->slice_obj;
    *nd = entry->origin_shape_nd;
    parent = entry->parent;
  } else {
    *origin_shape = NULL;
    *slice_obj = NULL;
  }
  DEBUG_PRINT("get_slice_objs done\n");
  DEBUG_PRINT("get ZEROS_ARRAY_DICT\n")
  DEBUG_PyObject_Print(parent) Zeros_Array_Dict *entry2 = NULL;
  if (parent != NULL)
    HASH_FIND_PTR(ZEROS_ARRAY_DICT, &parent, entry2);
  DEBUG_PRINT("Found ZEROS_ARRAY_DICT\n")
  if (entry2 != NULL)
    *zeros_array = entry2->zeros_array;
  else
    *zeros_array = NULL;
  DEBUG_PRINT("get ZEROS_ARRAY_DICT done\n")
}

Tensor *get_tensor(long long index) {
  Tensor_need_grad_Dict *entry = NULL;
  HASH_FIND_INT(TENSOR_NEED_GRAD_DICT, &index, entry);
  if (entry != NULL) {
    return entry->tensor;
  }
  return NULL;
}

PyObject *convert_tensor_dict_to_Py_dict(PyObject *self, PyObject *const *args,
                                         size_t nargsf) {
  (void)nargsf;
  PyObject *py_dict = PyDict_New();
  Tensor_need_grad_Dict *entry = NULL, *tmp = NULL;
  if (HASH_COUNT(TENSOR_NEED_GRAD_DICT) == 1) {
    HASH_ITER(hh, TENSOR_NEED_GRAD_DICT, entry, tmp) {
      PyDict_SetItem(py_dict, (PyObject *)entry->tensor, args[0]);
      HASH_DEL(TENSOR_NEED_GRAD_DICT, entry);
    }
    return py_dict;
  } else
    HASH_ITER(hh, TENSOR_NEED_GRAD_DICT, entry, tmp) {
      PyDict_SetItem(py_dict, (PyObject *)entry->tensor,
                     PyTuple_GetItem(args[0], entry->index));
      HASH_DEL(TENSOR_NEED_GRAD_DICT, entry);
    }
  return py_dict;
}

void INCREF_TENSOR(Tensor *self) {
  Py_INCREF(self->data);
  Py_INCREF(self->x);
  Py_INCREF(self->y);
  Py_INCREF(self->graph);
  Py_INCREF(self->axis);
}

void add_entry(const char *key,
               void (*method)(Tensor *, PyObject *, PyObject **, PyObject **)) {
  Dict *entry = (Dict *)malloc(sizeof(Dict));
  entry->key = key;
  entry->method = method;
  HASH_ADD_STR(dict, key, entry);
}

void (*get_method(const char *key))(Tensor *, PyObject *, PyObject **,
                                    PyObject **) {
  Dict *entry;
  HASH_FIND_STR(dict, key, entry);
  if (entry != NULL) {
    return entry->method;
  }
  return NULL;
}

Dict *get_address(const char *key) {
  Dict *entry;
  HASH_FIND_INT(dict, &key, entry);
  if (entry != NULL) {
    return entry;
  }
  return NULL;
}

void free_dict(void) {
  Dict *entry, *tmp;
  HASH_ITER(hh, dict, entry, tmp) {
    HASH_DEL(dict, entry);
    free(entry);
  }
}

PyObject *set_track(PyObject *self, PyObject *const *args, size_t nargsf) {
  if (PyDataType_ISNUMBER(args[0]))
    TRACK = PyLong_AsLong(args[0]) != 0 ? true : false;
  else
    TRACK = Py_IsTrue(args[0]);
  Py_INCREF(Py_None);
  return Py_None;
}

static int Tensor_traverse(Tensor *self, visitproc visit, void *arg) {
  Py_VISIT(self->data);
  Py_VISIT(self->x);
  Py_VISIT(self->y);
  Py_VISIT(self->axis);
  Py_VISIT(self->graph);
  Py_VISIT(self->grad);
  return 0;
}

PyMemberDef properties[] = {
    {"data", T_OBJECT, offsetof(Tensor, data), 0, "data"},
    {"x", T_OBJECT, offsetof(Tensor, x), 0, "x"},
    {"y", T_OBJECT, offsetof(Tensor, y), 0, "y"},
    {"has_conv", T_INT, offsetof(Tensor, has_conv), 0, "has_conv"},
    {"depth", T_ULONGLONG, offsetof(Tensor, vars), 0, "depth"},
    {"require_grad", T_BOOL, offsetof(Tensor, require_grad), 0, "require_grad"},
    {"grad_fn", T_STRING, offsetof(Tensor, grad_fn), 0, "grad_fn"},
    {"graph", T_OBJECT, offsetof(Tensor, graph), 0, "graph"},
    {"axis", T_OBJECT, offsetof(Tensor, axis), 0, "axis"},
    {"dim", T_INT, offsetof(Tensor, dim), 0, "dim"},
    {"grad", T_OBJECT, offsetof(Tensor, grad), 0, "grad"},
    {NULL}};

static PyGetSetDef Tensor_getsetters[] = {
    {"T", (getter)T, NULL, "T", NULL},
    {"dtype", (getter)dtype, NULL, "dtype", NULL},
    {NULL} /* Sentinel */
};

static PyNumberMethods tensor_operator_methods = {(binaryfunc)tensor_add,
                                                  (binaryfunc)tensor_sub,
                                                  (binaryfunc)tensor_mul,
                                                  (binaryfunc)tensor_remainder,
                                                  (binaryfunc)tensor_divmod,
                                                  (ternaryfunc)tensor_pow,
                                                  (unaryfunc)tensor_negative,
                                                  (unaryfunc)tensor_positive,
                                                  (unaryfunc)tensor_absolute,
                                                  0, // inquiry tensor_bool,
                                                  (unaryfunc)tensor_invert,
                                                  (binaryfunc)tensor_lshift,
                                                  (binaryfunc)tensor_rshift,
                                                  (binaryfunc)tensor_and,
                                                  (binaryfunc)tensor_xor,
                                                  (binaryfunc)tensor_or,
                                                  (unaryfunc)tensor_int,
                                                  0, // void *tensor_reserved;
                                                  (unaryfunc)tensor_float,
                                                  (binaryfunc)tensor_iadd,
                                                  (binaryfunc)tensor_isub,
                                                  (binaryfunc)tensor_imul,
                                                  (binaryfunc)tensor_iremainder,
                                                  (ternaryfunc)tensor_ipow,
                                                  (binaryfunc)tensor_ilshift,
                                                  (binaryfunc)tensor_irshift,
                                                  (binaryfunc)tensor_iand,
                                                  (binaryfunc)tensor_ixor,
                                                  (binaryfunc)tensor_ior,
                                                  (binaryfunc)tensor_floordiv,
                                                  (binaryfunc)tensor_div,
                                                  (binaryfunc)tensor_ifloordiv,
                                                  (binaryfunc)tensor_idiv,
                                                  0,
                                                  (binaryfunc)tensor_matmul,
                                                  (binaryfunc)tensor_imatmul};

static PyMappingMethods Tensor_as_mapping = {
    NULL,
    (binaryfunc)get_item, // binaryfunc; __getitem__
    NULL,
};

static PyMethodDef Tensor_methods[] = {
    {"backward", (PyCFunction)backward, METH_VARARGS, backward__doc__},
    {"reshape", (PyCFunction)self_reshape, METH_FASTCALL, "Method docstring"},
    {"transpose", (PyCFunction)self_transpose, METH_FASTCALL,
     "Method docstring"},
    {"permute", (PyCFunction)self_transpose, METH_FASTCALL, "Method docstring"},
    {"astype", (PyCFunction)astype, METH_FASTCALL, "Method docstring"},
    {"copy", (PyCFunction)copy, METH_NOARGS, "Method docstring"},
    {NULL} /* Sentinel */
};

static PyMethodDef module_methods[] = {
    {"reshape", (PyCFunction)reshape, METH_FASTCALL, "Method docstring"},
    {"transpose", (PyCFunction)transpose, METH_FASTCALL, "Method docstring"},
    {"argmax", (PyCFunction)_argmax_wrapper, METH_FASTCALL, "Method docstring"},
    {"argmin", (PyCFunction)_argmin_wrapper, METH_FASTCALL, "Method docstring"},
    {"sum", (PyCFunction)_sum, METH_FASTCALL, "Method docstring"},
    {"max", (PyCFunction)_max, METH_FASTCALL, "Method docstring"},
    {"min", (PyCFunction)_min, METH_FASTCALL, "Method docstring"},
    {"sin", (PyCFunction)_sin, METH_KEYWORDS | METH_VARARGS,
     "Method docstring"},
    {"cos", (PyCFunction)_cos, METH_KEYWORDS | METH_VARARGS,
     "Method docstring"},
    {"tan", (PyCFunction)_tan, METH_KEYWORDS | METH_VARARGS,
     "Method docstring"},
    {"arcsin", (PyCFunction)_asin, METH_KEYWORDS | METH_VARARGS,
     "Method docstring"},
    {"arccos", (PyCFunction)_acos, METH_KEYWORDS | METH_VARARGS,
     "Method docstring"},
    {"arctan", (PyCFunction)_atan, METH_KEYWORDS | METH_VARARGS,
     "Method docstring"},
    {"arcsinh", (PyCFunction)_asinh, METH_KEYWORDS | METH_VARARGS,
     "Method docstring"},
    {"arccosh", (PyCFunction)_acosh, METH_KEYWORDS | METH_VARARGS,
     "Method docstring"},
    {"arctanh", (PyCFunction)_atanh, METH_KEYWORDS | METH_VARARGS,
     "Method docstring"},
    {"sinh", (PyCFunction)_sinh, METH_KEYWORDS | METH_VARARGS,
     "Method docstring"},
    {"cosh", (PyCFunction)_cosh, METH_KEYWORDS | METH_VARARGS,
     "Method docstring"},
    {"tanh", (PyCFunction)_tanh, METH_KEYWORDS | METH_VARARGS,
     "Method docstring"},
    {"log10", (PyCFunction)_log10, METH_FASTCALL, "Method docstring"},
    {"log", (PyCFunction)_log, METH_FASTCALL, "Method docstring"},
    {"exp", (PyCFunction)_exp, METH_FASTCALL, "Method docstring"},
    {"sqrt", (PyCFunction)_sqrt, METH_FASTCALL, "Method docstring"},
    {"abs", (PyCFunction)_abs, METH_KEYWORDS | METH_VARARGS,
     "Method docstring"},
    {"power", (PyCFunction)nb_module_pow, METH_KEYWORDS | METH_VARARGS,
     "Method docstring"},
    {"mean", (PyCFunction)_mean, METH_FASTCALL, "Method docstring"},
    {"square", (PyCFunction)_mean, METH_FASTCALL, "Method docstring"},
    {"tensordot", (PyCFunction)tensordot, METH_FASTCALL, "Method docstring"},
    {"set_track", (PyCFunction)set_track, METH_FASTCALL, "Method docstring"},
    {"to_dict", (PyCFunction)convert_tensor_dict_to_Py_dict, METH_FASTCALL,
     "Method docstring"},
    {"global_float_type", (PyCFunction)set_global_float_type, METH_FASTCALL,
     "Method docstring"},
    {"result_type", (PyCFunction)binary_result_type_, METH_FASTCALL,
     "Method docstring"},
    {"tensor", (PyCFunction)__tensor, METH_KEYWORDS | METH_VARARGS,
     "Method docstring"},
    {"add", (PyCFunction)nb_module_add, METH_KEYWORDS | METH_VARARGS,
     "Method docstring"},
    {"sub", (PyCFunction)nb_module_sub, METH_KEYWORDS | METH_VARARGS,
     "Method docstring"},
    {"div", (PyCFunction)nb_module_div, METH_KEYWORDS | METH_VARARGS,
     "Method docstring"},
    {"mul", (PyCFunction)nb_module_mul, METH_KEYWORDS | METH_VARARGS,
     "Method docstring"},
    {"mod", (PyCFunction)nb_module_mod, METH_KEYWORDS | METH_VARARGS,
     "Method docstring"},
    {"fdiv", (PyCFunction)nb_module_fdiv, METH_KEYWORDS | METH_VARARGS,
     "Method docstring"},
    {NULL}};

static PyModuleDef custommodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "Numboost",
    .m_doc = "Tensor is a numpy wrapper which supports autograd",
    .m_size = -1,
    .m_methods = module_methods,
    .m_free = (freefunc)free_all_resources,
};

static PySequenceMethods sequence_methods = {
    .sq_length = (lenfunc)__len__,
};

PyTypeObject Tensor_type_ = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "Tensor",
    .tp_doc = "Tensor objects",
    .tp_basicsize = sizeof(Tensor),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC |
                Py_TPFLAGS_HAVE_VECTORCALL,
    .tp_new = (newfunc)__new__,
    .tp_members = properties,
    .tp_dealloc = (destructor)Tensor_dealloc,
    .tp_alloc = (allocfunc)tensor_alloc,
    .tp_clear = (inquiry)Tensor_clear,
    .tp_traverse = (traverseproc)Tensor_traverse,
    .tp_as_number = &tensor_operator_methods,
    .tp_methods = Tensor_methods,
    .tp_str = (reprfunc)__str__,
    .tp_repr = (reprfunc)__repr__,
    .tp_getset = Tensor_getsetters,
    .tp_as_mapping = &Tensor_as_mapping,
    .tp_iter = (getiterfunc)__iter__,
    .tp_richcompare = (richcmpfunc)rich_compare,
    .tp_as_sequence = &sequence_methods,
    .tp_hash = (hashfunc)__hash__,
};

PyTypeObject *Tensor_type = &Tensor_type_;

void init_map() {
  add_entry("AddBackward", add_backward_fn);
  add_entry("MulBackward", mul_backward_fn);
  add_entry("SubBackward", sub_backward_fn);
  add_entry("DivBackward", div_backward_fn);
  add_entry("MatMulBackward", matmul_backward_fn);
  add_entry("NegativeBackward", negative_backward_fn);
  add_entry("PowBackward", power_backward_fn);
  add_entry("TanhBackward", tanh_backward_fn);
  add_entry("SqrtBackward", sqrt_backward_fn);
  add_entry("ExpBackward", exp_backward_fn);
  add_entry("LogBackward", log_backward_fn);
  add_entry("SinBackward", sin_backward_fn);
  add_entry("CosBackward", cos_backward_fn);
  add_entry("TanBackward", tan_backward_fn);
  add_entry("SinhBackward", sinh_backward_fn);
  add_entry("CoshBackward", cosh_backward_fn);
  add_entry("ArcSinBackward", arcsin_backward_fn);
  add_entry("ArcCosBackward", arccos_backward_fn);
  add_entry("ArcTanBackward", arctan_backward_fn);
  add_entry("ArcSinhBackward", arcsinh_backward_fn);
  add_entry("ArcCoshBackward", arccosh_backward_fn);
  add_entry("ArcTanhBackward", arctanh_backward_fn);
  add_entry("Log10Backward", log10_backward_fn);
  add_entry("SqrtBackward", sqrt_backward_fn);
  add_entry("ReshapeBackward", reshape_backward_fn);
  add_entry("Log10Backward", log10_backward_fn);
  add_entry("TensordotBackward", tensordot_backward_fn);
  add_entry("TransposeBackward", transpose_backward_fn);
  add_entry("SliceBackward", slice_backward_fn);
  add_entry("AbsBackward", abs_backward_fn);
}

PyMODINIT_FUNC PyInit_Numboost(void) {
  Py_Initialize();
  int max_threads = omp_get_max_threads();
  omp_set_num_threads(max_threads);
  if (import_xla_ops(&xla_ops) == NULL)
    return NULL;
  if (import_jnp_methods(&JNP_METHOD) == NULL)
    return NULL;
  if (import_np_methods(&NP_METHOD) == NULL)
    return NULL;
  import_array();
  init_map();
  // PyDataMem_SetHandler(PyDataMem_DefaultHandler);

  PyDataMem_SetHandler(PyCapsule_New(&my_handler, "mem_handler", NULL));
  mem_chain = (Mem_Chain *)malloc(sizeof(Mem_Chain));
  cache *cache_struct = (cache *)malloc(sizeof(cache));
  cache_struct->max_mem = Mem_Pool_Size;
  cache_struct->mem_allocated = 0;
  cache_struct->mem_pool = (void **)malloc(sizeof(void *) * Mem_Pool_Size);
  cache_struct->tensor_size = 0;
  cache_struct->next = cache_struct;
  cache_struct->prev = cache_struct;
  mem_chain->head = cache_struct;
  mem_chain->tail = cache_struct;
  mem_chain->max_possible_cache_size = 0;
  cache_struct->mem_pool[0] = malloc(1);

  mem_chain->move_node_to_head = chain_move_node_to_head;
  mem_chain->pop = chain_pop;
  mem_chain->free_partial_mem_blocks = chain_free_partial_mem_blocks;

  HASH_ADD(hh, cache_pool, tensor_size, sizeof(size_t), cache_struct);
  PyObject *m = PyModule_Create(&custommodule);
  if (m == NULL)
    return NULL;
  if (PyType_Ready(&TensorIterator_type) < 0)
    return NULL;
  if (PyModule_AddObject(m, "TensorIterator",
                         (PyObject *)&TensorIterator_type)) {
    Py_DECREF(&TensorIterator_type);
    Py_DECREF(m);
    return NULL;
  };
  if (PyType_Ready(Tensor_type) < 0)
    return NULL;
  if (PyModule_AddObject(m, "Tensor", (PyObject *)Tensor_type)) {
    Py_DECREF(Tensor_type);
    Py_DECREF(m);
    return NULL;
  };
  return m;
}