#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#define PY_SSIZE_T_CLEAN
#include "tensor_methods.h"
#include "operators.h"
#include "type_convertor.h"

PyObject *astype(Tensor *self, PyObject *const *args, size_t nargsf)
{
    int tp = (int)PyLong_AsLong(args[0]);
    PyArrayObject *arr = NULL;
    PyArrayObject *self_data = (PyArrayObject *)self->data;
    as_type(&self_data, &arr, tp);
    if (self->data == NULL || arr == NULL)
        return NULL;
    PyObject *result = Tensor__new__(&Tensor_type, (PyObject *)arr);
    ((Tensor*)result)->require_grad = self->require_grad;
    return (PyObject *)result;
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
                                      ", dtype=",
                                      PyArray_DESCR((PyArrayObject *)self->data)->typeobj->tp_name,
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
                                      ", dtype=",
                                      PyArray_DESCR((PyArrayObject *)self->data)->typeobj->tp_name,
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

PyObject *__len__(Tensor *self)
{
    return PyLong_FromLongLong(((PyArrayObject_fields *)((PyArrayObject *)self->data))->dimensions[0]);
}

PyObject *__iter__(Tensor *self)
{
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *__max__(Tensor *self)
{
    return PyLong_FromLongLong(((PyArrayObject_fields *)((PyArrayObject *)self->data))->dimensions[0]);
}

PyObject *__min__(Tensor *self)
{
    return PyLong_FromLongLong(((PyArrayObject_fields *)((PyArrayObject *)self->data))->dimensions[0]);
}