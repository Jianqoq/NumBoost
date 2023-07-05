#define PY_SSIZE_T_CLEAN
#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#include "numpy/arrayobject.h"
#include "tensor.h"

static Dict *dict = NULL;

PyObject *
RunTimeError(PyObject *self, const char *message)
{
    PyErr_SetString(PyExc_RuntimeError, message);
    return NULL;
}

void INCREF_TENSOR(Tensor *self)
{
    Py_INCREF(self->data);
    Py_INCREF(self->x);
    Py_INCREF(self->y);
    Py_INCREF(self->graph);
    Py_INCREF(self->axis);
    Py_INCREF(self->base);
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

Dict *
get_address(const char *key)
{
    Dict *entry;
    HASH_FIND_INT(dict, &key, entry);
    if (entry != NULL)
    {
        return entry;
    }
    return NULL;
}

void
free_dict(void)
{
    Dict *entry, *tmp;
    HASH_ITER(hh, dict, entry, tmp)
    {
        HASH_DEL(dict, entry);
        free(entry);
    }
}

static PyObject *
__new__(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    Tensor *self = (Tensor *)PyObject_GC_New(Tensor, type);
    if (self != NULL)
    {
        Tensor_SetData_without_init_value(self, Py_None);
        Tensor_SetX_without_init_value(self, Py_None);
        Tensor_SetY_without_init_value(self, Py_None);
        Tensor_SetHasConv(self, 0);
        Tensor_SetVars(self, 1);
        Tensor_SetRequireGrad(self, false);
        Tensor_SetGradFn(self, "");
        Tensor_SetGrad_without_init_value(self, Py_None);
        Tensor_SetGraph_without_init_value(self, Py_None);
        Tensor_SetAxis_without_init_value(self, Py_None);
        Tensor_SetBase_without_init_value(self, Py_None);
        Tensor_SetDim(self, 1);
    }
    return (PyObject *)self;
}

static int
__init__(Tensor *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"data", "requires_grad", NULL};
    PyObject *data = NULL, *cache = NULL;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O|p", kwlist, &data, &self->require_grad))
        return -1;

    if (data)
    {
        cache = PyArray_FromAny(data, NULL, 0, 0, NPY_ARRAY_DEFAULT, NULL);
        Tensor_SetData(self, cache);
    }

    PyObject_GC_Track(self);
    return 0;
}

PyObject *
__str__(Tensor *self)
{
    char *result;
    PyObject *py_str = PyObject_Str(self->data);
    char require_grad[6];
    sprintf(require_grad, "%s", self->require_grad ? "true" : "false");
    const char *str = PyUnicode_AsUTF8(py_str);
    uint64_t str_len = strlen(str);
    uint64_t count = 0;
    char *prefix = "Tensor(";
    uint64_t length = strlen((const char *)prefix);
    for (uint64_t i = 0; i < str_len; i++)
        if (str[i] == '\n')
            count++;
    uint64_t len = length * count + str_len;
    result = (char *)malloc((len + 1) * sizeof(char));
    count = 0;
    uint64_t index = 0;
    while (index < len)
    {
        if (str[count] != '\n')
        {
            result[index++] = str[count];
        }
        else
        {
            result[index++] = '\n';
            for (uint64_t i = 0; i < length; i++)
            {
                result[index++] = ' ';
            }
        }
        count++;
    }
    result[index++] = '\0';
    const char *string_array[] = {(const char *)prefix,
                                  (const char *)result,
                                  ", requires_grad=",
                                  (const char *)require_grad,
                                  ", backward=",
                                  "<", self->grad_fn,
                                  ">", ")\n"};
    uint64_t string_array_len = sizeof(string_array) / sizeof(string_array[0]);
    uint64_t string_total_len = 1;
    for (uint64_t i = 0; i < string_array_len; i++)
    {
        string_total_len += strlen(string_array[i]);
    }
    char *dest = (char *)malloc(string_total_len * sizeof(char));
    dest[0] = '\0';
    for (uint64_t i = 0; i < string_array_len; i++)
    {
        strcat(dest, string_array[i]);
    }
    PyObject *representation = PyUnicode_FromString((const char *)dest);
    free(dest);
    free(result);
    Py_DECREF(py_str);
    return representation;
}

PyObject *
__repr__(Tensor *self)
{
    return __str__(self);
}

static void
Tensor_dealloc(Tensor *self)
{
    PyObject_GC_UnTrack(self);
    Py_CLEAR(self->data);
    Py_CLEAR(self->x);
    Py_CLEAR(self->y);
    Py_CLEAR(self->axis);
    Py_CLEAR(self->graph);
    Py_CLEAR(self->base);
    Py_CLEAR(self->grad);
    PyObject_GC_Del(self);
}

static int
Tensor_clear(Tensor *self)
{
    PyObject_GC_UnTrack(self);
    Py_CLEAR(self->data);
    Py_CLEAR(self->x);
    Py_CLEAR(self->y);
    Py_CLEAR(self->axis);
    Py_CLEAR(self->graph);
    Py_CLEAR(self->base);
    Py_CLEAR(self->grad);
    PyObject_GC_Track(self);
    return 0;
}

static int
Tensor_traverse(Tensor *self, visitproc visit, void *arg)
{
    Py_VISIT(self->data);
    Py_VISIT(self->x);
    Py_VISIT(self->y);
    Py_VISIT(self->axis);
    Py_VISIT(self->graph);
    Py_VISIT(self->base);
    Py_VISIT(self->grad);
    return 0;
}

static PyMemberDef
    properties[] = {
        {"data", T_OBJECT, offsetof(Tensor, data), 0, "data"},
        {"x", T_OBJECT, offsetof(Tensor, x), 0, "x"},
        {"y", T_OBJECT, offsetof(Tensor, y), 0, "y"},
        {"has_conv", T_INT, offsetof(Tensor, has_conv), 0, "has_conv"},
        {"depth", T_ULONGLONG, offsetof(Tensor, vars), 0, "depth"},
        {"require_grad", T_BOOL, offsetof(Tensor, require_grad), 0,
         "require_grad"},
        {"grad_fn", T_STRING, offsetof(Tensor, grad_fn), 0, "grad_fn"},
        {"graph", T_OBJECT, offsetof(Tensor, graph), 0, "graph"},
        {"axis", T_OBJECT, offsetof(Tensor, axis), 0, "axis"},
        {"dim", T_INT, offsetof(Tensor, dim), 0, "dim"},
        {"base", T_OBJECT, offsetof(Tensor, base), 0, "base"},
        {"grad", T_OBJECT, offsetof(Tensor, grad), 0, "grad"},
        {NULL}};

static PyNumberMethods
    tensor_operator_methods = {
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

PyObject *
_Generic_backward(PyObject *self, PyObject *args)
{
    PyObject *current_grad1 = NULL;
    PyObject *current_grad2 = NULL;
    PyObject *grad = NULL;
    const char *grad_fn = NULL;
    if (!PyArg_ParseTuple(args, "O", &grad))
    {
        return NULL;
    }
    grad_fn = PyUnicode_AsUTF8(PyObject_GetAttrString(self, "grad_fn"));
    unsigned long depth = PyLong_AsUnsignedLong(PyObject_GetAttrString(self, "depth"));
    Stack *stack = createStack(depth);
    Tuple tuple = {self, grad};
    push(stack, tuple);
    while (stack->len != 0)
    {
        tuple = pop(stack);
        PyObject *grad_fn_name = PyObject_GetAttrString(tuple.node, "grad_fn");
        if (grad_fn_name == NULL)
        {
            if (PyErr_Occurred())
            {
                PyErr_SetString(PyExc_RuntimeError, "grad_fn_name is NULL");
                free_dict();
                return NULL;
            }
            continue;
        }
        else if (grad_fn_name == PyUnicode_FromString(""))
        {
            PyObject *g = PyObject_GetAttrString(tuple.node, "grad");
            if (Py_IsNone(g))
            {
                Py_DECREF(g);
                Py_INCREF(tuple.ndarray);
                PyObject_SetAttrString(tuple.node, "grad", tuple.ndarray);
                continue;
            }
            else
            {
                PyObject *grad = PyObject_GetAttrString(tuple.node, "grad");
                if (grad == NULL)
                {
                    PyErr_SetString(PyExc_RuntimeError, "can't get grad attribute");
                    free_dict();
                    return NULL;
                }
                PyObject *new_grad = PyNumber_Add(grad, tuple.ndarray);
                PyObject *tmp = PyObject_GetAttrString(tuple.node, "grad");
                if (tmp == NULL)
                {
                    PyErr_SetString(PyExc_RuntimeError, "can't get grad attribute");
                    free_dict();
                    return NULL;
                }
                Py_DECREF(tmp);
                Py_INCREF(new_grad);
                PyObject_SetAttrString(tuple.node, "grad", new_grad);
                continue;
            }
            PyObject_SetAttrString(tuple.node, "grad", tuple.ndarray);
        }
        grad_fn = PyUnicode_AsUTF8(grad_fn_name);
        get_method(grad_fn)((Tensor *)tuple.node, tuple.ndarray, &current_grad1, &current_grad2);
        if (current_grad1 == NULL || current_grad2 == NULL)
        {
            free_dict();
            return NULL;
        }
        PyObject *x = PyObject_GetAttrString(tuple.node, "x");
        PyObject *y = PyObject_GetAttrString(tuple.node, "y");
        if (tuple.node != x)
        {
            grad_fn = PyUnicode_AsUTF8(PyObject_GetAttrString(x, "grad_fn"));
            Tuple tuple2 = {x, current_grad1};
            push(stack, tuple2);
        }
        grad_fn_name = PyObject_GetAttrString(y, "grad_fn");
        if (grad_fn_name == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError, "grad_fn_name is NULL");
            free_dict();
            return NULL;
        }
        grad_fn = PyUnicode_AsUTF8(grad_fn_name);
        Tuple tuple2 = {y, current_grad2};
        push(stack, tuple2);
    }
    return PyUnicode_FromString(grad_fn);
};

static PyMethodDef Tensor_methods[] = {
    {"backward", (PyCFunction)_Generic_backward, METH_VARARGS, "Method docstring"},
    {NULL} /* Sentinel */
};

static PyModuleDef
    custommodule = {
        PyModuleDef_HEAD_INIT,
        .m_name = "tensor",
        .m_doc = "Example module that creates an extension type.",
        .m_size = -1,
};

PyTypeObject
    Tensor_type = {
        PyVarObject_HEAD_INIT(NULL, 0).tp_name = "tensor.Tensor",
        .tp_doc = "Tensor objects",
        .tp_basicsize = sizeof(Tensor),
        .tp_itemsize = 0,
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
        .tp_new = __new__,
        .tp_init = (initproc)__init__,
        .tp_members = properties,
        .tp_dealloc = (destructor)Tensor_dealloc,
        .tp_alloc = PyType_GenericAlloc,
        .tp_clear = (inquiry)Tensor_clear,
        .tp_traverse = (traverseproc)Tensor_traverse,
        .tp_as_number = &tensor_operator_methods,
        .tp_methods = Tensor_methods,
        .tp_str = (reprfunc)__str__,
        .tp_repr = (reprfunc)__repr__,
};

void init_map()
{
    add_entry("AddBackward", add_backward_fn);
    add_entry("MulBackward", mul_backward_fn);
    add_entry("SubBackward", sub_backward_fn);
    add_entry("DivBackward", div_backward_fn);
    add_entry("MatMulBackward", matmul_backward_fn);
    add_entry("NegativeBackward", negative_backward_fn);
}

PyMODINIT_FUNC
PyInit_tensor(void)
{
    import_array();
    init_map();
    PyObject *m;
    if (PyType_Ready(&Tensor_type) < 0)
        return NULL;

    m = PyModule_Create(&custommodule);
    if (m == NULL)
        return NULL;
    Py_INCREF(&Tensor_type);
    if (PyModule_AddObject(m, "Tensor", (PyObject *)&Tensor_type))
    {
        Py_DECREF(&Tensor_type);
        Py_DECREF(m);
        return NULL;
    };
    Py_AtExit(free_dict);
    return m;
}