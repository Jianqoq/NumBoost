#ifndef PYTHON_H
#define PYTHON_H
#include <Python.h>
#endif

#include <stdbool.h>
#ifndef HASH_H
#define HASH_H
#include "uthash.h"
#endif

#ifndef tensor_type
#define tensor_type
extern PyTypeObject Tensor_type;
#endif

#ifndef TENSOR_H
#define TENSOR_H
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


#ifndef TENSOR_CORE
#define TENSOR_CORE
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
void power_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null);
void sin_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null);
void cos_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null);
void tan_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null);
void arcsin_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null);
void arccos_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null);
void arctan_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null);
void sinh_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null);
void cosh_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null);
void tanh_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null);
void arcsinh_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null);
void arccosh_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null);
void arctanh_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null);
void exp_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null);
void log_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null);
void log10_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null);
void sqrt_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null);
void abs_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null);
void reshape_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null);
PyObject *_sin_internal(PyObject *args, PyObject *out);
PyObject *_cos_internal(PyObject *args, PyObject *out);
PyObject *_tan_internal(PyObject *args, PyObject *out);
PyObject *_asin_internal(PyObject *args, PyObject *out);
PyObject *_acos_internal(PyObject *args, PyObject *out);
PyObject *_atan_internal(PyObject *args, PyObject *out);
PyObject *_sinh_internal(PyObject *args, PyObject *out);
PyObject *_cosh_internal(PyObject *args, PyObject *out);
PyObject *_tanh_internal(PyObject *args, PyObject *out);
PyObject *_asinh_internal(PyObject *args, PyObject *out);
PyObject *_acosh_internal(PyObject *args, PyObject *out);
PyObject *_atanh_internal(PyObject *args, PyObject *out);
PyObject *_exp_internal(PyObject *args, PyObject *out);
PyObject *_log_internal(PyObject *args, PyObject *out);
PyObject *_log10_internal(PyObject *args, PyObject *out);
PyObject *_sqrt_internal(PyObject *args, PyObject *out);
PyObject *_abs_internal(PyObject *args, PyObject *out);
PyObject *_pow_internal(PyObject *args, PyObject *out);
void INCREF_TENSOR(Tensor *self);
#endif