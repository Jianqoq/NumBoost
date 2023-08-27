#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "../tensor.h"
#include "structmember.h"
#include "../set_tensor_properties.h"
#include "../python_magic/python_math_magic.h"

static void tensor_iter_dealloc(TensorIteratorObject *self)
{
    PyObject_GC_UnTrack(self);
    Py_CLEAR(self->data_iter);
    PyObject_GC_Del(self);
}

static int tensor_iter_clear(TensorIteratorObject *self)
{
    Py_CLEAR(self->data_iter);
    PyObject_GC_Track(self);
    return 0;
}

static int tensor_iter_traverse(TensorIteratorObject *self, visitproc visit, void *arg)
{
    Py_VISIT(self->data_iter);
    return 0;
}

TensorIteratorObject *iterator_new(PyTypeObject *type, Tensor *self)
{
    TensorIteratorObject *iter = (TensorIteratorObject *)PyObject_GC_New(TensorIteratorObject, type);
    if (iter == NULL)
    {
        return NULL;
    }
    iter->data_iter = PyObject_GetIter(self->data);
    iter->ndim = PyArray_NDIM((PyArrayObject *)self->data);
    if (iter->data_iter == NULL || PyErr_Occurred())
    {
        return NULL;
    }
    return iter;
}

PyObject *iterator_next(TensorIteratorObject *self)
{
    PyObject *next_data = PyIter_Next(self->data_iter);
    if (next_data == NULL)
    {
        PyErr_SetString(PyExc_StopIteration, "No more elements");
        return NULL;
    }
    PyObject *tensor = Tensor_Empty(next_data);
    return tensor;
}

PyTypeObject TensorIterator_type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "TensorIterator",
    .tp_doc = "Tensor objects",
    .tp_basicsize = sizeof(TensorIteratorObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_HAVE_VECTORCALL,
    .tp_new = (newfunc)iterator_new,
    .tp_dealloc = (destructor)tensor_iter_dealloc,
    .tp_alloc = PyType_GenericAlloc,
    .tp_clear = (inquiry)tensor_iter_clear,
    .tp_traverse = (traverseproc)tensor_iter_traverse,
    .tp_iternext = (iternextfunc)iterator_next,
};