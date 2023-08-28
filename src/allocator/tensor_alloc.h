#ifndef TENSOR_ALLOC_H
#define TENSOR_ALLOC_H
#include "../tensor.h"

typedef struct Tensor_Pool
{
    Tensor **tensor_pool;
    int32_t index;
} Tensor_Pool;

Tensor *tensor_alloc(PyTypeObject *type, Py_ssize_t size);
void Tensor_dealloc(Tensor *self);
int Tensor_clear(Tensor *self);
void free_all_resources();

#endif // TENSOR_ALLOC_H