#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define PY_SSIZE_T_CLEAN
#define NPY_TARGET_VERSION NPY_1_25_API_VERSION
#include "numpy/arrayobject.h"
#include "utils.h"
#include "omp.h"
#include <stdlib.h>
#include <string.h>
#include "structmember.h"
#include "tensor.h"
#include "python_magic/python_math_magic.h"
#include "type_convertor/type_convertor.h"
#include "tensor_methods.h"
#include "clinic/tensor_methods.c.h"
#include "allocator/allocator.h"

static Dict *dict = NULL;
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

void store_for_slicebackward(Tensor *key, PyObject *slice_obj, npy_intp *ptr, int nd, Tensor *parent)
{
    Slice_Dict *entry = NULL;
    HASH_FIND_PTR(SLICE_DICT, &key, entry);
    if (entry == NULL)
    {
        Slice_Dict *entry = (Slice_Dict *)malloc(sizeof(Slice_Dict));
        entry->key = key;
        entry->slice_obj = slice_obj;
        entry->origin_shape = ptr;
        entry->origin_shape_nd = nd;
        entry->parent = parent;
        HASH_ADD_PTR(SLICE_DICT, key, entry);
    }
    Zeros_Array_Dict *entry2 = NULL;
    HASH_FIND_PTR(ZEROS_ARRAY_DICT, &parent, entry2);
    if (entry2 == NULL)
    {
        Zeros_Array_Dict *entry2 = (Zeros_Array_Dict *)malloc(sizeof(Zeros_Array_Dict));
        PyObject *zeros = PyArray_Zeros(nd, (npy_intp const *)ptr, NULL, 0);
        entry2->parent = parent;
        entry2->zeros_array = zeros;
        HASH_ADD_PTR(ZEROS_ARRAY_DICT, parent, entry2);
    }
}

void get_slice_objs(Tensor *key, npy_intp **origin_shape, PyObject **slice_obj, int *nd, PyObject **zeros_array)
{
    Slice_Dict *entry = NULL;
    Tensor *parent = NULL;
    DEBUG_PRINT("get_slice_objs\n");
    HASH_FIND_PTR(SLICE_DICT, &key, entry);
    if (entry != NULL)
    {
        *origin_shape = entry->origin_shape;
        *slice_obj = entry->slice_obj;
        *nd = entry->origin_shape_nd;
        parent = entry->parent;
    }
    else
    {
        *origin_shape = NULL;
        *slice_obj = NULL;
    }
    DEBUG_PRINT("get_slice_objs done\n");
    DEBUG_PRINT("get ZEROS_ARRAY_DICT\n")
    DEBUG_PyObject_Print(parent)
        Zeros_Array_Dict *entry2 = NULL;
    if (parent != NULL)
        HASH_FIND_PTR(ZEROS_ARRAY_DICT, &parent, entry2);
    DEBUG_PRINT("Found ZEROS_ARRAY_DICT\n")
    if (entry2 != NULL)
        *zeros_array = entry2->zeros_array;
    else
        *zeros_array = NULL;
    DEBUG_PRINT("get ZEROS_ARRAY_DICT done\n")
}

void free_slice_objs(Tensor *key)
{
    Slice_Dict *entry = NULL;
    HASH_FIND_PTR(SLICE_DICT, &key, entry);
    if (entry != NULL)
    {
        DEBUG_PRINT("free_slice_objs\n");
        HASH_DEL(SLICE_DICT, entry);
        Py_DECREF(entry->slice_obj);
        free(entry);
        DEBUG_PRINT("free_slice_objs done\n");
    }
    Zeros_Array_Dict *entry2 = NULL;
    HASH_FIND_PTR(ZEROS_ARRAY_DICT, &key, entry2);
    if (entry2 != NULL)
    {
        DEBUG_PRINT("free zero arrays\n");
        HASH_DEL(ZEROS_ARRAY_DICT, entry2);
        DEBUG_PyObject_Print(entry2->zeros_array);
        Py_DECREF(entry2->zeros_array);
        free(entry2);
        DEBUG_PRINT("free zero arrays done\n");
    }
}

inline void free_tensor_need_grad(Tensor *self)
{
    Tensor_need_grad_Dict *entry = NULL;
    HASH_FIND_PTR(TENSOR_NEED_GRAD_DICT, &self, entry);
    if (entry != NULL)
    {
        HASH_DEL(TENSOR_NEED_GRAD_DICT, entry);
        free(entry);
    }
}

Tensor *get_tensor(long long index)
{
    Tensor_need_grad_Dict *entry = NULL;
    HASH_FIND_INT(TENSOR_NEED_GRAD_DICT, &index, entry);
    if (entry != NULL)
    {
        return entry->tensor;
    }
    return NULL;
}

PyObject *convert_tensor_dict_to_Py_dict(PyObject *self, PyObject *const *args, size_t nargsf)
{
    PyObject *dict = PyDict_New();
    Tensor_need_grad_Dict *entry = NULL, *tmp = NULL;
    if (HASH_COUNT(TENSOR_NEED_GRAD_DICT) == 1)
    {
        HASH_ITER(hh, TENSOR_NEED_GRAD_DICT, entry, tmp)
        {
            PyDict_SetItem(dict, (PyObject *)entry->tensor, args[0]);
        }
        return dict;
    }
    else
        HASH_ITER(hh, TENSOR_NEED_GRAD_DICT, entry, tmp)
        {
            PyDict_SetItem(dict, (PyObject *)entry->tensor, PyTuple_GetItem(args[0], entry->index));
        }
    return dict;
}

void free_tensordot_data()
{
    Tensordot_Dict *entry = NULL, *tmp = NULL;
    HASH_ITER(hh, TENSORDOT_DICT, entry, tmp)
    {
        DEBUG_PRINT("Freeing Tensordot data\n");
        HASH_DEL(TENSORDOT_DICT, entry);
        free(entry->metadata->newaxes_a.ptr);
        free(entry->metadata->newaxes_b.ptr);
        Py_DECREF(entry->metadata->matmul_result);
        Py_DECREF(entry->metadata->transposed_reshape_a);
        Py_DECREF(entry->metadata->transposed_reshape_b);
        Py_DECREF(entry->key);
        free(entry->metadata);
        free(entry);
    }
}

static inline void free_tensordot_data_self(Tensor *self)
{
    Tensordot_Dict *entry = NULL;
    DEBUG_PRINT("Going to free Tensordot data\n");
    HASH_FIND_PTR(TENSORDOT_DICT, &self, entry);
    if (entry != NULL)
    {
        DEBUG_PRINT("Freeing Tensordot data\n");
        HASH_DEL(TENSORDOT_DICT, entry);
        free(entry->metadata->newaxes_a.ptr);
        free(entry->metadata->newaxes_b.ptr);
        Py_DECREF(entry->metadata->matmul_result);
        Py_DECREF(entry->metadata->transposed_reshape_a);
        Py_DECREF(entry->metadata->transposed_reshape_b);
        free(entry->metadata);
        free(entry);
    }
}

inline void free_array_shape(Tensor *key)
{
    DEBUG_PRINT("Freeing Array shape\n");
    Array_Shape *s = NULL;
    if (ARRAY_SHAPE != NULL)
        HASH_FIND_PTR(ARRAY_SHAPE, &key, s);
    if (s != NULL)
    {
        HASH_DEL(ARRAY_SHAPE, s);
        free(s->shape);
        free(s);
    }
    DEBUG_PRINT("Freeing Array shape done\n");
}

inline void free_power(Tensor *key)
{
    Power_Dict *s = NULL;
    if (POWER_DICT != NULL)
        HASH_FIND_PTR(POWER_DICT, &key, s);
    if (s != NULL)
    {
        HASH_DEL(POWER_DICT, s);
        free(s);
    }
}

inline void free_base(Tensor *key)
{
    Log_Dict *s = NULL;
    if (LOG_DICT != NULL)
        HASH_FIND_PTR(LOG_DICT, &key, s);
    if (s != NULL)
    {
        HASH_DEL(LOG_DICT, s);
        free(s);
    }
}

void free_all_resources()
{
    cache *s, *tmp;

    HASH_ITER(hh, cache_pool, s, tmp)
    {
        HASH_DEL(cache_pool, s);
        for (int i = 0; i <= s->mem_allocated; i++)
        {
            free(s->mem_pool[i]);
        }
        free(s->mem_pool);
        free(s);
    }
    free(mem_chain);

    Dict *entry, *tmp2;
    HASH_ITER(hh, dict, entry, tmp2)
    {
        HASH_DEL(dict, entry);
        free(entry);
    }
    free_xla_ops(xla_ops);
    free_tensordot_data();
    free_np_methods(NP_METHOD);
    free_jnp_methods(JNP_METHOD);

    Array_Shape *s2, *tmp3;
    HASH_ITER(hh, ARRAY_SHAPE, s2, tmp3)
    {
        HASH_DEL(ARRAY_SHAPE, s2);
        Py_XDECREF(s2->key);
        free(s2->shape);
        free(s2);
    }
    Power_Dict *s3, *tmp4;
    HASH_ITER(hh, POWER_DICT, s3, tmp4)
    {
        HASH_DEL(POWER_DICT, s3);
        Py_XDECREF(s3->key);
        Py_XDECREF(s3->prev_power);
        free(s3);
    }
    Log_Dict *s4, *tmp5;
    HASH_ITER(hh, LOG_DICT, s4, tmp5)
    {
        HASH_DEL(LOG_DICT, s4);
        Py_XDECREF(s4->key);
        Py_XDECREF(s4->base);
        free(s4);
    }
    Tensor_need_grad_Dict *s5, *tmp6;
    HASH_ITER(hh, TENSOR_NEED_GRAD_DICT, s5, tmp6)
    {
        HASH_DEL(TENSOR_NEED_GRAD_DICT, s5);
        Py_XDECREF(s5->tensor);
        free(s5);
    }

    Tensordot_Dict *s6, *tmp7;
    HASH_ITER(hh, TENSORDOT_DICT, s6, tmp7)
    {
        HASH_DEL(TENSORDOT_DICT, s6);
        free(s6);
    }
    HASH_CLEAR(hh, dict);
    HASH_CLEAR(hh, TENSORDOT_DICT);
    HASH_CLEAR(hh, ARRAY_SHAPE);
    HASH_CLEAR(hh, POWER_DICT);
    HASH_CLEAR(hh, LOG_DICT);
    HASH_CLEAR(hh, TENSOR_NEED_GRAD_DICT);
    HASH_CLEAR(hh, SLICE_DICT);
    HASH_CLEAR(hh, ZEROS_ARRAY_DICT);
    Py_CLEAR(Tensor_type);
}

void INCREF_TENSOR(Tensor *self)
{
    Py_INCREF(self->data);
    Py_INCREF(self->x);
    Py_INCREF(self->y);
    Py_INCREF(self->graph);
    Py_INCREF(self->axis);
}

void add_entry(const char *key, void (*method)(Tensor *, PyObject *, PyObject **, PyObject **))
{
    Dict *entry = (Dict *)malloc(sizeof(Dict));
    entry->key = key;
    entry->method = method;
    HASH_ADD_STR(dict, key, entry);
}

void (*get_method(const char *key))(Tensor *, PyObject *, PyObject **, PyObject **)
{
    Dict *entry;
    HASH_FIND_STR(dict, key, entry);
    if (entry != NULL)
    {
        return entry->method;
    }
    return NULL;
}

Dict *get_address(const char *key)
{
    Dict *entry;
    HASH_FIND_INT(dict, &key, entry);
    if (entry != NULL)
    {
        return entry;
    }
    return NULL;
}

void free_dict(void)
{
    Dict *entry, *tmp;
    HASH_ITER(hh, dict, entry, tmp)
    {
        HASH_DEL(dict, entry);
        free(entry);
    }
}

PyObject *set_track(PyObject *self, PyObject *const *args, size_t nargsf)
{
    if (PyDataType_ISNUMBER(args[0]))
        TRACK = PyLong_AsLong(args[0]) != 0 ? true : false;
    else
        TRACK = Py_IsTrue(args[0]);
    Py_INCREF(Py_None);
    return Py_None;
}

static void Tensor_dealloc(Tensor *self)
{
    DEBUG_PRINT("Tensor_dealloc\n");
    PyObject_GC_UnTrack(self);
    Py_CLEAR(self->data); // pretty expensive
    Py_CLEAR(self->x);
    Py_CLEAR(self->y);
    Py_CLEAR(self->axis);
    Py_CLEAR(self->graph);
    Py_CLEAR(self->grad);
    free_tensordot_data_self(self);
    free_array_shape(self);
    free_power(self);
    free_tensor_need_grad(self);
    free_slice_objs(self);
    PyObject_GC_Del(self);
    DEBUG_PRINT("Tensor_dealloc done\n");
}

static int Tensor_clear(Tensor *self)
{
    PyObject_GC_UnTrack(self);
    Py_CLEAR(self->data);
    Py_CLEAR(self->x);
    Py_CLEAR(self->y);
    Py_CLEAR(self->axis);
    Py_CLEAR(self->graph);
    Py_CLEAR(self->grad);
    PyObject_GC_Track(self);
    return 0;
}

static int Tensor_traverse(Tensor *self, visitproc visit, void *arg)
{
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
    {NULL} /* Sentinel */
};

static PyNumberMethods tensor_operator_methods = {
    (binaryfunc)tensor_add,
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
    {"transpose", (PyCFunction)self_transpose, METH_FASTCALL, "Method docstring"},
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
    {"sin", (PyCFunction)_sin, METH_FASTCALL, "Method docstring"},
    {"cos", (PyCFunction)_cos, METH_FASTCALL, "Method docstring"},
    {"tan", (PyCFunction)_tan, METH_FASTCALL, "Method docstring"},
    {"arcsin", (PyCFunction)_asin, METH_FASTCALL, "Method docstring"},
    {"arccos", (PyCFunction)_acos, METH_FASTCALL, "Method docstring"},
    {"arctan", (PyCFunction)_atan, METH_FASTCALL, "Method docstring"},
    {"arcsinh", (PyCFunction)_asinh, METH_FASTCALL, "Method docstring"},
    {"arccosh", (PyCFunction)_acosh, METH_FASTCALL, "Method docstring"},
    {"arctanh", (PyCFunction)_atanh, METH_FASTCALL, "Method docstring"},
    {"sinh", (PyCFunction)_sinh, METH_FASTCALL, "Method docstring"},
    {"cosh", (PyCFunction)_cosh, METH_FASTCALL, "Method docstring"},
    {"tanh", (PyCFunction)_tanh, METH_FASTCALL, "Method docstring"},
    {"log10", (PyCFunction)_log10, METH_FASTCALL, "Method docstring"},
    {"log", (PyCFunction)_log, METH_FASTCALL, "Method docstring"},
    {"exp", (PyCFunction)_exp, METH_FASTCALL, "Method docstring"},
    {"sqrt", (PyCFunction)_sqrt, METH_FASTCALL, "Method docstring"},
    {"abs", (PyCFunction)_abs, METH_FASTCALL, "Method docstring"},
    {"power", (PyCFunction)_pow, METH_FASTCALL, "Method docstring"},
    {"mean", (PyCFunction)_mean, METH_FASTCALL, "Method docstring"},
    {"square", (PyCFunction)_mean, METH_FASTCALL, "Method docstring"},
    {"tensordot", (PyCFunction)tensordot, METH_FASTCALL, "Method docstring"},
    {"set_track", (PyCFunction)set_track, METH_FASTCALL, "Method docstring"},
    {"to_dict", (PyCFunction)convert_tensor_dict_to_Py_dict, METH_FASTCALL, "Method docstring"},
    {"global_float_type", (PyCFunction)set_global_float_type, METH_FASTCALL, "Method docstring"},
    {"result_type", (PyCFunction)binary_result_type_, METH_FASTCALL, "Method docstring"},
    {"tensor", (PyCFunction)__tensor, METH_KEYWORDS | METH_VARARGS, "Method docstring"},
    {NULL}};

static PyModuleDef custommodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "Numboost",
    .m_doc = "Tensor is a numpy wrapper which supports autograd",
    .m_size = -1,
    .m_methods = module_methods,
    .m_free = free_all_resources,
};

static PySequenceMethods sequence_methods = {
    .sq_length = (lenfunc)__len__,
};

PyTypeObject Tensor_type_ = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "Tensor",
    .tp_doc = "Tensor objects",
    .tp_basicsize = sizeof(Tensor),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_HAVE_VECTORCALL,
    .tp_new = (newfunc)__new__,
    .tp_members = properties,
    .tp_dealloc = (destructor)Tensor_dealloc,
    .tp_alloc = PyType_GenericAlloc,
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

void init_map()
{
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
}

PyMODINIT_FUNC PyInit_Numboost(void)
{
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
    mem_chain = (double_linked_list *)malloc(sizeof(double_linked_list));
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
    if (PyModule_AddObject(m, "TensorIterator", (PyObject *)&TensorIterator_type))
    {
        Py_DECREF(&TensorIterator_type);
        Py_DECREF(m);
        return NULL;
    };
    if (PyType_Ready(Tensor_type) < 0)
        return NULL;
    if (PyModule_AddObject(m, "Tensor", (PyObject *)Tensor_type))
    {
        Py_DECREF(Tensor_type);
        Py_DECREF(m);
        return NULL;
    };
    return m;
}