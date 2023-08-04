#ifndef SHAPE_H
#define SHAPE_H
#include "shape.h"
#endif

typedef void (*BinaryFunc)(PyArrayObject *a, PyArrayObject *b, PyObject **result);

BinaryFunc BinaryOp_OperationPicker(int npy_type, int operation);