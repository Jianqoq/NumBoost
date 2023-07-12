#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define PY_SSIZE_T_CLEAN
#include "numpy/arrayobject.h"
#include "omp.h"
#include <stdlib.h>
#include <string.h>
#include "structmember.h"
#include "tensor.h"
#include "operators.h"
#include "set_Tensor_properties.h"

static Dict *dict = NULL;

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

void free_dict(void)
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
    self->require_grad = false;
    if (self != NULL)
    {
        static char *kwlist[] = {"data", "requires_grad", NULL};
        PyObject *data = NULL, *cache = NULL;

        if (!PyArg_ParseTupleAndKeywords(
                args, kwds, "O|p", kwlist, &data, &self->require_grad))
            return NULL;
        if (data)
            cache = PyArray_FromAny(data, NULL, 0, 0, NPY_ARRAY_DEFAULT, NULL);
        else
            return NULL;
        PyObject *zero = PyLong_FromLong(0);
        self->data = cache;
        Tensor_SetData_without_init_value(self, cache);
        Tensor_SetX_without_init_value(self, Py_None);
        Tensor_SetY_without_init_value(self, Py_None);
        Tensor_SetHasConv(self, 0);
        Tensor_SetVars(self, 1);
        Tensor_SetGradFn(self, "");
        Tensor_SetGrad_without_init_value(self, zero);
        Tensor_SetGraph_without_init_value(self, Py_None);
        Tensor_SetAxis_without_init_value(self, Py_None);
        Tensor_SetBase_without_init_value(self, Py_None);
        Tensor_SetDim(self, 1);
        Py_DECREF(zero);
    }
    PyObject_GC_Track(self);
    return (PyObject *)self;
}

PyObject *
__str__(Tensor *self)
{
    char *result, *dest, *prefix = "Tensor(";
    PyObject *py_str = PyObject_Str(self->data);
    char require_grad[6];
    sprintf(require_grad, "%s", self->require_grad ? "true" : "false");
    const char *str = PyUnicode_AsUTF8(py_str);
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
    if (!strcmp(self->grad_fn, ""))
    {
        const char *string_array[] = {(const char *)prefix,
                                      (const char *)result,
                                      ", requires_grad=",
                                      (const char *)require_grad, ")\n"};
        uint64_t string_array_len = sizeof(string_array) / sizeof(string_array[0]);
        uint64_t string_total_len = 1;
        for (uint64_t i = 0; i < string_array_len; i++)
        {
            string_total_len += strlen(string_array[i]);
        }
        dest = (char *)malloc(string_total_len * sizeof(char));
        dest[0] = '\0';
        for (uint64_t i = 0; i < string_array_len; i++)
        {
            strcat(dest, string_array[i]);
        }
    }
    else
    {
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
        dest = (char *)malloc(string_total_len * sizeof(char));
        dest[0] = '\0';
        for (uint64_t i = 0; i < string_array_len; i++)
        {
            strcat(dest, string_array[i]);
        }
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

PyMemberDef
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
    Tensor *tensor = NULL;
    const char *grad_fn = NULL;
    if (!PyArg_ParseTuple(args, "O", &grad))
    {
        return NULL;
    }
    Py_INCREF(grad); // Avoid grad reference count to be 0, current grad ref == 2
    Tensor *self_tensor = (Tensor *)self;
    if (!self_tensor->require_grad)
    {
        Py_DECREF(grad);
        PyErr_SetString(PyExc_RuntimeError, "Tensor require_grad is False");
        return NULL;
    }
    grad_fn = self_tensor->grad_fn;
    unsigned long long depth = self_tensor->vars;
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
                return NULL;
            }
            continue;
        }
        else
        {
            tensor = (Tensor *)tuple.node;
            grad_fn = tensor->grad_fn;
            if (!strcmp(grad_fn, ""))
            {
                if (tensor == NULL)
                {
                    PyErr_SetString(PyExc_RuntimeError, "can't convert object from stack to Tensor");
                    free_dict();
                    return NULL;
                }
                PyObject *new_grad = PyNumber_Add(tensor->grad, tuple.ndarray);
                Py_DECREF(tensor->grad);
                tensor->grad = new_grad;
                Py_DECREF(tuple.ndarray);
                continue;
            }
        }
        get_method(grad_fn)(tensor, tuple.ndarray, &current_grad1, &current_grad2);

        if (current_grad1 != NULL)
            if (current_grad2 != NULL)
                if (current_grad1 == NULL && current_grad2 == NULL)
                {
                    free_dict();
                    return NULL;
                }
        Tensor *tensor_x = (Tensor *)tensor->x;
        if (tensor_x != tensor)
        {
            grad_fn = tensor_x->grad_fn;
            Tuple tuple2 = {tensor->x, current_grad1};
            push(stack, tuple2);
        }
        if (Py_IS_TYPE(tensor->y, &Tensor_type))
        {
            bool require_grad = ((Tensor *)tensor->y)->require_grad;
            if (current_grad2 != NULL && require_grad)
            {
                Tuple tuple2 = {tensor->y, current_grad2};
                push(stack, tuple2);
            }
        }
        Py_DECREF(tuple.ndarray);
    }
    freeStack(stack);

    Py_INCREF(Py_None);
    return Py_None;
};

static Tensor *self_reshape(Tensor *self, PyObject *const *args, size_t nargsf, PyObject *kwnames)
{
    size_t nargs = PyVectorcall_NARGS(nargsf);
    PyArrayObject *array;
    int order = 0;
    if (PyUnicode_Check(args[nargs - 1]))
    {
        PyObject *order_obj = args[nargs - 1];
        if (PyUnicode_CompareWithASCIIString(order_obj, "C") == 0)
        {
            order = NPY_CORDER;
        }
        else if (PyUnicode_CompareWithASCIIString(order_obj, "F") == 0)
        {
            order = NPY_FORTRANORDER;
        }
        else
        {
            PyErr_SetString(PyExc_ValueError, "order must be 'C' or 'F'");
            return NULL;
        }
        nargs -= 1;
    }
    Tensor *tensor = self;
    int length = (int)nargs;
    npy_intp dims[NPY_MAXDIMS] = {0};
    for (uint8_t i = 0; i < length; i++)
    {
        dims[i] = PyLong_AsLongLong(args[i]);
    }
    PyArray_Dims shape = {dims, length};
    array = (PyArrayObject *)tensor->data;
    PyObject *result = PyArray_Newshape(array, &shape, order);
    if (result != tensor->data)
    {
        tensor->data = result;
        Py_DECREF(array);
    }
    if (result == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Error in reshape");
        return NULL;
    }
    return self;
}

static Tensor *self_transpose(Tensor *self, PyObject *const *args, size_t nargsf, PyObject *kwnames)
{
    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    PyArrayObject *array = (PyArrayObject *)self->data;
    npy_intp *dims = malloc(sizeof(npy_intp) * nargs);
    for (Py_ssize_t i = 0; i < nargs; i++)
    {
        dims[i] = PyLong_AsLong(args[i]);
    }
    PyArray_Dims shape = {dims, (int)nargs};
    PyObject *result = PyArray_Transpose(array, &shape);
    if (result != NULL && result != self->data)
    {
        self->data = result;
        Py_DECREF(array);
    }
    else
    {
        free(dims);
        return NULL;
    }
    free(dims);
    return self;
}

static PyMethodDef Tensor_methods[] = {
    {"backward", (PyCFunction)_Generic_backward, METH_VARARGS, "Backward method"},
    {"reshape", (PyCFunction)self_reshape, METH_FASTCALL, "Method docstring"},
    {"transpose", (PyCFunction)self_transpose, METH_FASTCALL, "Method docstring"},
    {"permute", (PyCFunction)self_transpose, METH_FASTCALL, "Method docstring"},
    {NULL} /* Sentinel */
};

static PyModuleDef
    custommodule = {
        PyModuleDef_HEAD_INIT,
        .m_name = "tensor",
        .m_doc = "Tensor is a numpy wrapper which supports autograd",
        .m_size = -1,
};

PyTypeObject
    Tensor_type = {
        PyVarObject_HEAD_INIT(NULL, 0).tp_name = "tensor.Tensor",
        .tp_doc = "Tensor objects",
        .tp_basicsize = sizeof(Tensor),
        .tp_itemsize = 0,
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
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
};

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
}

PyMODINIT_FUNC
PyInit_tensor(void)
{
    omp_set_num_threads(16);
    import_array();
    init_map();
    PyObject *m;
    if (PyType_Ready(&Tensor_type) < 0)
        return NULL;

    m = PyModule_Create(&custommodule);
    if (m == NULL)
        return NULL;
    if (PyModule_AddObject(m, "Tensor", (PyObject *)&Tensor_type))
    {
        Py_DECREF(&Tensor_type);
        Py_DECREF(m);
        return NULL;
    };
    return m;
}