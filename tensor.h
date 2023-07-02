#ifndef TENSOR_H
#define TENSOR_H
#include <Python.h>
#include <stdbool.h>
#include "structmember.h"
#include "numpy/arrayobject.h"
#include "uthash.h"
#include <stdlib.h>
#include <string.h>

typedef struct
{
    PyObject_HEAD
    PyObject *data; /* numpy array */
    PyObject *x;        /* numpy array */
    PyObject *y;        /* numpy array */
    int has_conv;
    unsigned long long vars;
    bool require_grad;
    const char *grad_fn;
    PyObject *graph;
    PyObject *axis;
    PyObject *grad;
    PyObject *shape;
    int dim;
    PyObject *stride;
    PyObject *base;

} Tensor;
void add_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2);
void sub_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2);
void mul_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2);
void div_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2);
void matmul_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2);
PyTypeObject Tensor_type;

PyObject *tensor_add(Tensor *self, PyObject *other);
PyObject *tensor_iadd(Tensor *self, PyObject *other);
PyObject *tensor_mul(Tensor *self, PyObject *other);
PyObject *tensor_imul(Tensor *self, PyObject *other);
PyObject *tensor_div(Tensor *self, PyObject *other);
PyObject *tensor_idiv(Tensor *self, PyObject *other);
PyObject *tensor_negative(Tensor *self);
PyObject *tensor_sub(Tensor *self, PyObject *other);
PyObject *tensor_isub(Tensor *self, PyObject *other);
PyObject *tensor_pow(Tensor *self, PyObject *other);
PyObject *tensor_ipow(Tensor *self, PyObject *other);
PyObject *tensor_matmul(Tensor *self, Tensor *other);
PyObject *tensor_imatmul(Tensor *self, Tensor *other);
void INCREF_TENSOR(Tensor *self);
typedef struct
{
    const char *key;
    void (__cdecl *method)(Tensor *, PyObject *, PyObject **, PyObject **);
    // const char *address;
    UT_hash_handle hh;
} Dict;

void init_map();

void (__cdecl *get_method(const char *key))(Tensor *, PyObject *, PyObject **, PyObject **);

Dict *get_address(const char *key);

typedef struct
{   
    PyObject *node;
    PyObject *ndarray;
} Tuple;

typedef struct
{
    unsigned long len;
    unsigned long max_len;
    long long index;
    Tuple *array;
} Stack;
Stack *createStack(uint64_t capacity);
int isFull(Stack *stack);
int isEmpty(Stack *stack);
PyObject* push(Stack *stack, Tuple item);
Tuple pop(Stack *stack);
PyObject * _Generic_backward(PyObject *self, PyObject *grad);
#endif