#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "tensor.h"
#include <omp.h>
#include "mkl.h"
#include "op.h"
#include "binary_func.h"

void BinaryOp_Picker(int npy_type, int operation, PyArrayObject *a, PyArrayObject *b, PyObject **result)
{
    switch (npy_type)
    {
    case NPY_BOOL:
        break;
    case NPY_BYTE:
        OPERATION_PICKER(a, b, result, operation, npy_byte, NPY_BYTE)
        break;
    case NPY_UBYTE:
        OPERATION_PICKER(a, b, result, operation, npy_ubyte, NPY_UBYTE)
        break;
    case NPY_SHORT:
        OPERATION_PICKER(a, b, result, operation, npy_short, NPY_SHORT)
        break;
    case NPY_USHORT:
        OPERATION_PICKER(a, b, result, operation, npy_ushort, NPY_USHORT)
        break;
    case NPY_INT:
        OPERATION_PICKER(a, b, result, operation, npy_int, NPY_INT)
        break;
    case NPY_UINT:
        OPERATION_PICKER(a, b, result, operation, npy_uint, NPY_UINT)
        break;
    case NPY_LONG:
        OPERATION_PICKER(a, b, result, operation, npy_long, NPY_LONG)
        break;
    case NPY_ULONG:
        OPERATION_PICKER(a, b, result, operation, npy_ulong, NPY_ULONG)
        break;
    case NPY_LONGLONG:
        OPERATION_PICKER(a, b, result, operation, npy_longlong, NPY_LONGLONG)
        break;
    case NPY_ULONGLONG:
        OPERATION_PICKER(a, b, result, operation, npy_ulonglong, NPY_ULONGLONG)
        break;
    case NPY_FLOAT:
        F_OPERATION_PICKER(a, b, result, operation, npy_float, NPY_FLOAT)
        break;
    case NPY_DOUBLE:
        F_OPERATION_PICKER(a, b, result, operation, npy_double, NPY_DOUBLE)
        break;
    case NPY_LONGDOUBLE:
        F_OPERATION_PICKER(a, b, result, operation, npy_longdouble, NPY_LONGDOUBLE)
        break;
    case NPY_HALF:
        switch (operation)
        {
        case ADD:
            HALF_OPERATION(a, b, result, +)
            break;
        case SUB:
            HALF_OPERATION(a, b, result, -)
            break;
        case MUL:
            HALF_OPERATION(a, b, result, *)
            break;
        case DIV:
            HALF_OPERATION(a, b, result, /)
            break;
        }
        break;
    }
}