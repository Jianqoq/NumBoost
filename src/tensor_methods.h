#include "tensor.h"

PyObject *astype(Tensor *self, PyObject *const *args, size_t nargsf);

PyObject *__str__(Tensor *self);

PyObject *__repr__(Tensor *self);

PyObject *__iter__(Tensor *self);

Py_ssize_t __len__(Tensor *self);

PyObject *rich_compare(Tensor *self, PyObject *other, int op);

PyObject *__min__(Tensor *self);

PyObject *get_item(Tensor *self, PyObject *item);

Tensor *T(Tensor *self);

PyObject *__new__(PyTypeObject *type, PyObject *args, PyObject *kwds);

Tensor *self_transpose(Tensor *self, PyObject *const *args, size_t nargsf, PyObject *kwnames);

Tensor *self_reshape(Tensor *self, PyObject *const *args, size_t nargsf, PyObject *kwnames);

PyObject *backward(PyObject *self, PyObject *args);

PyObject *next(Tensor *self);

Py_hash_t __hash__(Tensor *self);

#define General(type, self, other, result, descr, op)                  \
    {                                                                  \
        type a;                                                        \
        type b;                                                        \
        PyArray_CastScalarToCtype(self->data, &a, descr);              \
        PyArray_CastScalarToCtype(((Tensor *)other)->data, &b, descr); \
        if (a > b && op == Py_GT)                                      \
            result = true;                                             \
        else if (a < b && op == Py_LT)                                 \
            result = true;                                             \
        else if (a == b && op == Py_EQ)                                \
            result = true;                                             \
        else                                                           \
            result = false;                                            \
    }

#define Compare(npy_enum, type, self, other, result, descr, op) \
    case npy_enum:                                              \
        General(type, self, other, result, descr, op);

#define F_MAX(npy_enum, type, op, a, size)                        \
    case npy_enum:                                                \
        void *a_ptr = PyArray_DATA(a->data);                      \
        npy_intp i;                                               \
        npy_half max_num = 0;                                     \
        npy_half g_max = 0;                                       \
        _Pragma("omp parallel firstprivate(max_num)")             \
            max_num = 0;                                          \
        {                                                         \
            _Pragma("omp for") for (i = 0; i < size; i++)         \
            {                                                     \
                npy_half a = *((npy_half *)a_ptr);                \
                if (half_cast_float(a) > max_num)                 \
                    max_num = a;                                  \
            }                                                     \
            _Pragma("omp critical")                               \
            {                                                     \
                if (max_num > g_max)                              \
                    g_max = max_num;                              \
            }                                                     \
        }                                                         \
        PyArrayObject *result = PyArray_EMPTY(1, 1, npy_enum, 0); \
        *((type *)PyArray_DATA(result)) = g_max;                  \
        return Tensor_Empty((PyObject *)result);
