#ifndef TENSOR_H
#define TENSOR_H
#include <Python.h>
#include <stdbool.h>
#include "structmember.h"
#include "uthash.h"
#include <stdlib.h>
#include <string.h>
typedef struct
{
    PyObject_HEAD
    PyObject *data;     /* ndarray */
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

void add_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2);
void sub_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2);
void mul_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2);
void div_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2);
void matmul_backward_fn(Tensor *self, PyObject *grad, PyObject **out1, PyObject **out2);
extern PyTypeObject Tensor_type;

PyObject *tensor_add(Tensor *self, PyObject *other);
PyObject *tensor_iadd(Tensor *self, PyObject *other);
PyObject *tensor_mul(Tensor *self, PyObject *other);
PyObject *tensor_imul(Tensor *self, PyObject *other);
PyObject *tensor_div(Tensor *self, PyObject *other);
PyObject *tensor_idiv(Tensor *self, PyObject *other);
PyObject *tensor_negative(Tensor *self);
PyObject *tensor_inegative(Tensor *self);
PyObject *tensor_sub(Tensor *self, PyObject *other);
PyObject *tensor_isub(Tensor *self, PyObject *other);
PyObject *tensor_pow(Tensor *self, PyObject *other);
PyObject *tensor_ipow(Tensor *self, PyObject *other);
PyObject *tensor_matmul(Tensor *self, Tensor *other);
PyObject *tensor_imatmul(Tensor *self, Tensor *other);
PyObject *tensor_positive(Tensor *self);
PyObject *tensor_absolute(Tensor *self);
PyObject *tensor_invert(Tensor *self);
PyObject *tensor_lshift(Tensor *self, PyObject *other);
PyObject *tensor_rshift(Tensor *self, PyObject *other);
PyObject *tensor_and(Tensor *self, PyObject *other);
PyObject *tensor_xor(Tensor *self, PyObject *other);
PyObject *tensor_or(Tensor *self, PyObject *other);
PyObject *tensor_int(Tensor *self);
PyObject *tensor_float(Tensor *self);
PyObject *tensor_remainder(Tensor *self, PyObject *other);
PyObject *tensor_ior(Tensor *self, PyObject *other);
PyObject *tensor_ixor(Tensor *self, PyObject *other);
PyObject *tensor_iand(Tensor *self, PyObject *other);
PyObject *tensor_ilshift(Tensor *self, PyObject *other);
PyObject *tensor_irshift(Tensor *self, PyObject *other);
PyObject *tensor_divmod(Tensor *self, PyObject *other);
PyObject *tensor_iremainder(Tensor *self, PyObject *other);
PyObject *tensor_floordiv(Tensor *self, PyObject *other);
PyObject *tensor_ifloordiv(Tensor *self, PyObject *other);

void INCREF_TENSOR(Tensor *self);
typedef struct
{
    const char *key;
    void (*method)(Tensor *, PyObject *, PyObject **, PyObject **);
    UT_hash_handle hh;
} Dict;

void init_map();

void (*get_method(const char *key))(Tensor *, PyObject *, PyObject **, PyObject **);

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
#endif