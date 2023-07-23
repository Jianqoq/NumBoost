#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define PY_SSIZE_T_CLEAN
#include "numpy/arrayobject.h"
#include "utils.h"
#include "omp.h"
#include <stdlib.h>
#include <string.h>
#include "structmember.h"
#include "tensor.h"
#include "operators.h"
#include "set_Tensor_properties.h"

static Dict *dict = NULL;
XLA_OPS *xla_ops = NULL;
long TRACK = 0;
np_method *NP_METHOD = NULL;
Array_Shape *ARRAY_SHAPE = NULL;
Power_Dict *POWER_DICT = NULL;
Log_Dict *LOG_DICT = NULL;
jnp_method *JNP_METHOD = NULL;
Tensor_need_grad_Dict *TENSOR_NEED_GRAD_DICT = NULL;
Tensordot_Dict *TENSORDOT_DICT = NULL;

static void store_tensor_need_grad(long long index, Tensor *tensor)
{
    Tensor_need_grad_Dict *entry = NULL;
    if (TENSOR_NEED_GRAD_DICT != NULL)
        HASH_FIND_PTR(TENSOR_NEED_GRAD_DICT, &tensor, entry);
    if (entry == NULL)
    {
        Tensor_need_grad_Dict *entry = (Tensor_need_grad_Dict *)malloc(sizeof(Tensor_need_grad_Dict));
        entry->tensor = tensor;
        entry->index = index;
        HASH_ADD_PTR(TENSOR_NEED_GRAD_DICT, tensor, entry);
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

inline void free_tensordot_data_self(Tensor *self)
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
    Array_Shape *s = NULL;
    if (ARRAY_SHAPE != NULL)
        HASH_FIND_PTR(ARRAY_SHAPE, &key, s);
    if (s != NULL)
    {
        HASH_DEL(ARRAY_SHAPE, s);
        free(s);
    }
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

void INCREF_TENSOR(Tensor *self)
{
    Py_INCREF(self->data);
    Py_INCREF(self->x);
    Py_INCREF(self->y);
    Py_INCREF(self->graph);
    Py_INCREF(self->axis);
    Py_INCREF(self->dtype);
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

PyObject *collect_gradients_and_cleanup(PyObject *list)
{
    Tensor_need_grad_Dict *gradient_entry, *gradient_tmp;
    PyObject *to_return = NULL;
    Power_Dict *power_entry = NULL, *power_tmp = NULL;

    DEBUG_PRINT("Collecting gradients and cleaning up, dict length %d.\n", HASH_COUNT(TENSOR_NEED_GRAD_DICT));
    if (HASH_COUNT(TENSOR_NEED_GRAD_DICT) == 1)
    {
        HASH_ITER(hh, TENSOR_NEED_GRAD_DICT, gradient_entry, gradient_tmp)
        {
            to_return = gradient_entry->tensor->grad;
            gradient_entry->tensor->grad = PyLong_FromLong(0);
        }
    }
    else
    {
        HASH_ITER(hh, TENSOR_NEED_GRAD_DICT, gradient_entry, gradient_tmp)
        {
            PyList_Append(list, gradient_entry->tensor->grad);
            gradient_entry->tensor->grad = PyLong_FromLong(0);
        }
        to_return = PyList_AsTuple(list);
    }
    // Cleanup: decrease reference count of list and also all tensor in POWER_DICT
    DEBUG_PRINT("Finished collecting gradients and cleaning up, dict length %d.\n", HASH_COUNT(TENSOR_NEED_GRAD_DICT));
    Py_DECREF(list);
    return to_return;
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
    TRACK = PyLong_AsLong(args[0]);
    Py_INCREF(Py_None);
    return Py_None;
}

Tensor *T(Tensor *self)
{
    DEBUG_PRINT("Transposing tensor in T\n");
    npy_intp ndim = ((PyArrayObject_fields *)self->data)->nd;
    npy_intp *new_shape = (npy_intp *)malloc(sizeof(npy_intp) * ndim);
    for (int i = 0; i < ndim; i++)
        new_shape[i] = ndim - i - 1;
#if DEBUG
    printf("New Shape: ");
    for (int i = 0; i < ndim; i++)
        printf("%ld ", new_shape[i]);
    printf("\n");
#endif
    PyArray_Dims new_dims = {new_shape, ndim};
    PyObject *transposed = PyArray_Transpose((PyArrayObject *)self->data, &new_dims);
    if (transposed == NULL)
        return NULL;
    free(new_shape);
    DEBUG_PRINT("Transposed tensor in T\n");
    Tensor *to_return = (Tensor *)new_Tensor_x(self, transposed, "TransposeBackward");
    if (self->require_grad)
    {
        npy_intp i;
        npy_intp *store_shape = (npy_intp *)malloc(sizeof(npy_intp) * ndim);
        for (i = 0; i < ndim; i++)
            store_shape[i] = ndim - i - 1;

        DEBUG_PRINT("Stored Shape: ");
        DEBUG_FOR_LOOP(i = 0; i < ndim; i++)
        {
            printf("%ld ", store_shape[i]);
        }
        DEBUG_PRINT("\n");
        store_array_shape(to_return, store_shape, ndim);
    }
    return to_return;
}

static PyObject *__new__(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"data", "requires_grad", NULL};
    PyObject *data = NULL, *cache = NULL;
    bool *require_grad = NULL;
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O|p", kwlist, &data, &require_grad))
        return NULL;
    if (data)
    {
        if (!TRACK)
        {
            cache = PyArray_FromAny(data, NULL, 0, 0, NPY_ARRAY_DEFAULT, NULL);
        }
        else
        {
            PyObject *jaxarray = PyObject_CallOneArg(JNP_METHOD->array, data);
            return jaxarray;
        }
    }
    else
        return NULL;
    Tensor *self = (Tensor *)PyObject_GC_New(Tensor, type);
    if (require_grad == NULL)
    {
        self->require_grad = false;
    }
    else
    {
        self->require_grad = require_grad;
    }
    if (self != NULL)
    {
        PyObject *zero = PyLong_FromLong(0);
        self->data = cache;
        Tensor_SetX_without_init_value(self, Py_None);
        Tensor_SetY_without_init_value(self, Py_None);
        Tensor_SetHasConv(self, 0);
        Tensor_SetVars(self, 1);
        Tensor_SetGradFn(self, "");
        Tensor_SetGrad_without_init_value(self, zero);
        Tensor_SetGraph_without_init_value(self, Py_None);
        Tensor_SetAxis_without_init_value(self, Py_None);
        Tensor_SetDtype_without_init_value(self, PyArray_DESCR((PyArrayObject *)cache));
        Tensor_SetDim(self, 1);
        Py_DECREF(zero);
    }
    PyObject_GC_Track(self);
    return (PyObject *)self;
}

PyObject *__str__(Tensor *self)
{
    char *result, *dest, *prefix = "Tensor(", *end = ")\n";
    if (TRACK)
    {
        prefix = "\n\tTensor(";
        end = ")";
    }
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
                                      (const char *)require_grad, end};
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

PyObject *__repr__(Tensor *self)
{
    return __str__(self);
}

static void Tensor_dealloc(Tensor *self)
{
    DEBUG_PRINT("Tensor_dealloc\n");
    PyObject_GC_UnTrack(self);
    Py_CLEAR(self->data);
    Py_CLEAR(self->x);
    Py_CLEAR(self->y);
    Py_CLEAR(self->axis);
    Py_CLEAR(self->graph);
    Py_CLEAR(self->dtype);
    Py_CLEAR(self->grad);
    free_tensordot_data_self(self);
    free_array_shape(self);
    free_power(self);
    free_tensor_need_grad(self);
    PyObject_GC_Del(self);
}

static int Tensor_clear(Tensor *self)
{
    PyObject_GC_UnTrack(self);
    Py_CLEAR(self->data);
    Py_CLEAR(self->x);
    Py_CLEAR(self->y);
    Py_CLEAR(self->axis);
    Py_CLEAR(self->graph);
    Py_CLEAR(self->dtype);
    Py_CLEAR(self->grad);
    PyObject_GC_Track(self);
    if (PyErr_Occurred())
    {
        return -1;
    }
    return 0;
}

static int Tensor_traverse(Tensor *self, visitproc visit, void *arg)
{
    Py_VISIT(self->data);
    Py_VISIT(self->x);
    Py_VISIT(self->y);
    Py_VISIT(self->axis);
    Py_VISIT(self->graph);
    Py_VISIT(self->dtype);
    Py_VISIT(self->grad);
    if (PyErr_Occurred())
    {
        return -1;
    }
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
    {"dtype", T_OBJECT_EX, offsetof(Tensor, dtype), 0, "base"},
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

PyObject *_Generic_backward(PyObject *self, PyObject *args)
{
    DEBUG_PRINT("Generic_backward start\n");
    // Declare variables
    PyObject *current_grad1 = NULL;
    PyObject *current_grad2 = NULL;
    PyObject *grad = NULL;
    Tensor *tensor = NULL;
    PyObject *list = PyList_New(0);
    long long index = 0;
    const char *grad_fn = NULL;
    // Parse the Python argument tuple
    if (!PyArg_ParseTuple(args, "O", &grad))
    {
        return NULL;
    }
    Py_INCREF(grad); // Avoid grad reference count to be 0, current grad ref == 2
    Tensor *self_tensor = (Tensor *)self;
    // If the tensor doesn't require gradient, return an error
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
    // Start the main backward loop
    DEBUG_PRINT("Generic_backward loop start\n");
    while (stack->len != 0)
    {
        tuple = pop(stack);
        PyObject *grad_fn_name = PyObject_GetAttrString(tuple.node, "grad_fn");
        // Handle grad_fn_name being NULL
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
            Py_DECREF(grad_fn_name);
            tensor = (Tensor *)tuple.node;
            grad_fn = tensor->grad_fn;
            // If grad_fn is empty string, perform gradient addition
            if (!strcmp(grad_fn, ""))
            {
                if (tensor == NULL)
                {
                    PyErr_SetString(PyExc_RuntimeError, "can't convert object from stack to Tensor");
                    free_dict();
                    return NULL;
                }
                PyObject *new_grad = PyNumber_Add(tensor->grad, tuple.ndarray);
                if (new_grad == NULL)
                {
                    free_dict();
                    return NULL;
                }
                if (TRACK)
                {
                    // Check whether the tensor needs gradient
                    Tensor_need_grad_Dict *entry = NULL;
                    Tensor_need_grad_Dict *s, *tmp;
                    HASH_FIND_PTR(TENSOR_NEED_GRAD_DICT, &tensor, entry);
                    if (entry != NULL)
                    {
                        Py_DECREF(tensor->grad);
                        tensor->grad = new_grad;
                        Py_DECREF(tuple.ndarray);
                        continue;
                    }
                    else
                    {
                        DEBUG_PRINT("Add tensor to TENSOR_NEED_GRAD_DICT\n");
                        DEBUG_PyObject_Print(tensor);
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
        DEBUG_PRINT("grad_fn_name: %s\n", grad_fn);
        // Get the gradient function and apply it
        get_method(grad_fn)(tensor, tuple.ndarray, &current_grad1, &current_grad2);
        DEBUG_PyObject_Print(current_grad1);
        DEBUG_PyObject_Print(current_grad2);
        // If both gradients are NULL, return an error
        if (current_grad1 == NULL && current_grad2 == NULL)
        {
            free_dict();
            return NULL;
        }
        // Handle the tensor x
        Tensor *tensor_x = (Tensor *)tensor->x;
        if (tensor_x != tensor)
        {
            if (current_grad1 != NULL && tensor_x->require_grad)
            {
                Tuple tuple2 = {tensor->x, current_grad1};
                push(stack, tuple2);
            }
            else
            {
                Py_DECREF(current_grad1);
            }
        }
        // Handle the tensor y
        if (Py_IS_TYPE(tensor->y, &Tensor_type))
        {
            bool require_grad = ((Tensor *)tensor->y)->require_grad;
            if (current_grad2 != NULL && require_grad)
            {
                Tuple tuple2 = {tensor->y, current_grad2};
                push(stack, tuple2);
            }
            else
            {
                Py_DECREF(current_grad2);
            }
        }
        Py_DECREF(tuple.ndarray);
    }
    // Cleanup
    DEBUG_PRINT("Cleaning up\n");
    freeStack(stack);
    Py_DECREF(list);
    DEBUG_PRINT("finished cleaning up on tensordot data\n");
    // If tracking, return the gradients as a tuple
    if (TRACK)
    {
        return collect_gradients_and_cleanup(list);
    }
    // If not tracking, just cleanup and return None
    DEBUG_PRINT("finished cleaning up\n");
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
    {"tensordot", (PyCFunction)tensordot, METH_FASTCALL, "Method docstring"},
    {"set_track", (PyCFunction)set_track, METH_FASTCALL, "Method docstring"},
    {"to_dict", (PyCFunction)convert_tensor_dict_to_Py_dict, METH_FASTCALL, "Method docstring"},
    {NULL}};

static PyModuleDef custommodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "Numboost",
    .m_doc = "Tensor is a numpy wrapper which supports autograd",
    .m_size = -1,
    .m_methods = module_methods,
};

PyTypeObject Tensor_type = {
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
    add_entry("TensordotBackward", tensordot_backward_fn);
}

PyMODINIT_FUNC PyInit_Numboost(void)
{
    Py_Initialize();
    omp_set_num_threads(omp_get_max_threads());
    if (import_xla_ops(&xla_ops) == NULL)
        return NULL;
    if (import_jnp_methods(&JNP_METHOD) == NULL)
        return NULL;
    if (import_np_methods(&NP_METHOD) == NULL)
        return NULL;
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