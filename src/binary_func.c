#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "tensor.h"
#include <omp.h>
#include "mkl.h"
#include "op.h"
#include "binary_func.h"

void BinaryOp_Picker(int operation, PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    PyArrayObject *a_ = a;
    PyArrayObject *b_ = b;
    PyArray_Descr *descr_a = ((PyArrayObject_fields *)a)->descr;
    PyArray_Descr *descr_b = ((PyArrayObject_fields *)b)->descr;
    int npy_type = PyArray_PromoteTypes(descr_a, descr_b)->type_num;
    if (descr_a->type_num != npy_type)
        as_type(&a, &a_, npy_type);
    if (descr_b->type_num != npy_type)
        as_type(&b, &b_, npy_type);
    switch (npy_type)
    {
        PICK(NPY_BOOL, a_, b_, result, operation, npy_bool)
        PICK(NPY_BYTE, a_, b_, result, operation, npy_byte)
        PICK(NPY_UBYTE, a_, b_, result, operation, npy_ubyte)
        PICK(NPY_SHORT, a_, b_, result, operation, npy_short)
        PICK(NPY_USHORT, a_, b_, result, operation, npy_ushort)
        PICK(NPY_INT, a_, b_, result, operation, npy_int)
        PICK(NPY_UINT, a_, b_, result, operation, npy_uint)
        PICK(NPY_LONG, a_, b_, result, operation, npy_long)
        PICK(NPY_ULONG, a_, b_, result, operation, npy_ulong)
        PICK(NPY_LONGLONG, a_, b_, result, operation, npy_longlong)
        PICK(NPY_ULONGLONG, a_, b_, result, operation, npy_ulonglong)
        F_PICK(NPY_FLOAT, a_, b_, result, operation, npy_float)
        F_PICK(NPY_DOUBLE, a_, b_, result, operation, npy_double)
        F_PICK(NPY_LONGDOUBLE, a_, b_, result, operation, npy_longdouble)
        HALF_PICK(a_, b_, result, operation)
    }
}