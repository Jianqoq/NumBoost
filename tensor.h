
#ifndef TENSOR_IMPORT
#define TENSOR_IMPORT
#include "operators.h"
#endif

#ifndef TENSOR_H
#define TENSOR_H
#include <Python.h>
#include "uthash.h"
#include <stdbool.h>
#include <stdio.h>

#ifndef TENSOR_CLASS
#define TENSOR_CLASS
typedef struct
{
    PyObject_HEAD
        PyObject *data; /* ndarray */
    PyObject *x;        /* Tensor */
    PyObject *y;        /* Tensor|scalar */
    int has_conv;
    unsigned long long vars;
    bool require_grad;
    const char *grad_fn;
    PyObject *graph;
    PyObject *axis;
    PyObject *grad;
    int dim;
    PyObject *base;

} Tensor;
#endif

void free_dict(void);
void init_map();

void (*get_method(const char *key))(Tensor *, PyObject *, PyObject **, PyObject **);
typedef struct
{
    const char *key;
    void (*method)(Tensor *, PyObject *, PyObject **, PyObject **);
    UT_hash_handle hh;
} Dict;

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
Stack *createStack(unsigned long capacity);
int isFull(Stack *stack);
int isEmpty(Stack *stack);
PyObject *push(Stack *stack, Tuple item);
Tuple pop(Stack *stack);
PyObject *_Generic_backward(PyObject *self, PyObject *grad);

void add_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2);
void sub_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2);
void mul_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2);
void div_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2);
void matmul_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2);
void negative_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2);

void INCREF_TENSOR(Tensor *self);
void Tensor_SetData(Tensor *self, PyObject *data);
void Tensor_SetX(Tensor *self, PyObject *x);
void Tensor_SetY(Tensor *self, PyObject *y);
void Tensor_SetGrad(Tensor *self, PyObject *grad);
void Tensor_SetGraph(Tensor *self, PyObject *graph);
void Tensor_SetBase(Tensor *self, PyObject *base);
void Tensor_SetGradFn(Tensor *self, const char *grad_fn);
void Tensor_SetRequireGrad(Tensor *self, bool require_grad);
void Tensor_SetVars(Tensor *self, unsigned long long vars);
void Tensor_SetDim(Tensor *self, int dim);
void Tensor_SetAxis(Tensor *self, PyObject *axis);
void Tensor_SetHasConv(Tensor *self, int has_conv);
void Tensor_SetData_without_init_value(Tensor *self, PyObject *data);
void Tensor_SetX_without_init_value(Tensor *self, PyObject *x);
void Tensor_SetY_without_init_value(Tensor *self, PyObject *y);
void Tensor_SetGrad_without_init_value(Tensor *self, PyObject *grad);
void Tensor_SetGraph_without_init_value(Tensor *self, PyObject *graph);
void Tensor_SetBase_without_init_value(Tensor *self, PyObject *base);
void Tensor_SetAxis_without_init_value(Tensor *self, PyObject *axis);
void Tensor_SetData_startwone(Tensor *self, PyObject *data);
void Tensor_SetX_startwone(Tensor *self, PyObject *x);
void Tensor_SetY_startwone(Tensor *self, PyObject *y);
void Tensor_SetGrad_startwone(Tensor *self, PyObject *grad);
void Tensor_SetGraph_startwone(Tensor *self, PyObject *graph);
void Tensor_SetBase_startwone(Tensor *self, PyObject *base);
void Tensor_SetAxis_startwone(Tensor *self, PyObject *axis);
void Tensor_SetData_startwone_without_init(Tensor *self, PyObject *data);
void Tensor_SetX_startwone_without_init(Tensor *self, PyObject *x);
void Tensor_SetY_startwone_without_init(Tensor *self, PyObject *y);
void Tensor_SetGrad_startwone_without_init(Tensor *self, PyObject *grad);
void Tensor_SetGraph_startwone_without_init(Tensor *self, PyObject *graph);
void Tensor_SetBase_startwone_without_init(Tensor *self, PyObject *base);
void Tensor_SetAxis_startwone_without_init(Tensor *self, PyObject *axis);
#endif