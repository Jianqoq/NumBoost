#ifndef TENSOR_H
#define TENSOR_H
#include <Python.h>

// #include <jemalloc/jemalloc.h>
// #ifndef JELMALLOC
// #define JELMALLOC
// #ifdef _MSC_VER
// #define malloc(size) je_malloc(size)
// #define free(ptr) je_free(ptr)
// #define realloc(ptr, size) je_realloc(ptr, size)
// #define calloc(count, size) je_calloc(count, size)
// #endif
// #endif

#include <stdbool.h>
#include "uthash.h"
extern bool TRACK;
PyObject *set_track(PyObject *self, PyObject *const *args, size_t nargsf);

#include "import_methods.h"

extern XLA_OPS *xla_ops;

#include "object.h"
extern PyTypeObject *Tensor_type;

#include "numpy/arrayobject.h"

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

} Tensor;

typedef struct
{
    PyObject_HEAD
        PyObject *data_iter;
    int ndim;
} TensorIteratorObject;

TensorIteratorObject *iterator_new(PyTypeObject *type, Tensor *self);
PyObject *iterator_next(TensorIteratorObject *self);

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
    unsigned long long len;
    unsigned long long max_len;
    unsigned long long index;
    Tuple *array;

} Stack;
Stack *createStack(unsigned long long capacity);
void freeStack(Stack *stack);
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
void tensordot_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **out2);
void transpose_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null);
void slice_backward_fn(Tensor *self, PyObject *grad, PyObject **out, PyObject **null);
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

inline npy_intp search_num(npy_intp *arr, npy_intp n, npy_intp x)
{
    npy_intp i;
    for (i = 0; i < n; i++)
        if (arr[i] == x)
            return i;
    return 0;
}

typedef struct
{
    Tensor *key;
    npy_intp *shape;
    int len;
    UT_hash_handle hh;
} Array_Shape;

typedef struct
{
    Tensor *key;
    PyObject *prev_power;
    UT_hash_handle hh;
} Power_Dict;

typedef struct
{
    Tensor *key;
    PyObject *base;
    UT_hash_handle hh;
} Log_Dict;

typedef struct
{
    long long index;
    Tensor *tensor;
    UT_hash_handle hh;
} Tensor_need_grad_Dict;

typedef struct
{
    PyArray_Dims newshape_a;
    PyArray_Dims newshape_b;
    PyArray_Dims newaxes_a;
    PyArray_Dims newaxes_b;
    PyArray_Dims transposed_shape_a;
    PyArray_Dims transposed_shape_b;
    PyArray_Dims matmul_result_shape;
    PyObject *matmul_result;
    PyObject *transposed_reshape_a;
    PyObject *transposed_reshape_b;
} Tensordot_Metadata;

typedef struct
{
    Tensor *key;
    Tensordot_Metadata *metadata;
    UT_hash_handle hh;
} Tensordot_Dict;

typedef struct
{
    Tensor *key;
    PyObject *slice_obj;
    npy_intp *origin_shape;
    Tensor *parent;
    int origin_shape_nd;
    UT_hash_handle hh;
} Slice_Dict;

typedef struct
{
    Tensor *parent;
    PyObject *zeros_array;
    UT_hash_handle hh;
} Zeros_Array_Dict;

void store_array_shape(Tensor *key, npy_intp *shape, int len);
npy_intp *get_array_shape(Tensor *key);
void free_array_shape(Tensor *key);

npy_intp get_shape_len(Tensor *key);

void store_power(Tensor *key, PyObject *power);
PyObject *get_power(Tensor *key);
void free_power(Tensor *key);

void store_base(Tensor *key, PyObject *base);
PyObject *get_base(Tensor *key);
void free_base(Tensor *key);

void store_tensordot_data(Tensor *key, Tensordot_Metadata *metadata);
Tensordot_Metadata *get_tensordot_data(Tensor *key);
void free_tensordot_data();

void store_for_slicebackward(Tensor *key, PyObject *slice_obj, npy_intp *ptr, int nd, Tensor *parent);
void get_slice_objs(Tensor *key, npy_intp **origin_shape, PyObject **slice_obj, int *nd, PyObject **zeros_array);
void free_slice_objs(Tensor *key);

Tensor *reshape(PyObject *self, PyObject *const *args, size_t nargsf, PyObject *kwnames);
PyObject *transpose(PyObject *self, PyObject *const *args, size_t nargsf, PyObject *kwnames);
Tensor *tensordot(PyObject *self, PyObject *const *args, size_t nargsf, PyObject *kwnames);
Tensor *_sin(PyObject *self, PyObject *const *args, size_t nargsf);
Tensor *_cos(PyObject *self, PyObject *const *args, size_t nargsf);
Tensor *_tan(PyObject *self, PyObject *const *args, size_t nargsf);
Tensor *_asin(PyObject *self, PyObject *const *args, size_t nargsf);
Tensor *_acos(PyObject *self, PyObject *const *args, size_t nargsf);
Tensor *_atan(PyObject *self, PyObject *const *args, size_t nargsf);
Tensor *_sinh(PyObject *self, PyObject *const *args, size_t nargsf);
Tensor *_cosh(PyObject *self, PyObject *const *args, size_t nargsf);
Tensor *_tanh(PyObject *self, PyObject *const *args, size_t nargsf);
Tensor *_asinh(PyObject *self, PyObject *const *args, size_t nargsf);
Tensor *_acosh(PyObject *self, PyObject *const *args, size_t nargsf);
Tensor *_atanh(PyObject *self, PyObject *const *args, size_t nargsf);
Tensor *_mean(PyObject *self, PyObject *const *args, size_t nargsf);
Tensor *_sum(PyObject *self, PyObject *const *args, size_t nargsf);
Tensor *_max(PyObject *self, PyObject *const *args, size_t nargsf);
Tensor *_min(PyObject *self, PyObject *const *args, size_t nargsf);
Tensor *_argmax_wrapper(PyObject *self, PyObject *const *args, size_t nargsf);
Tensor *_argmin_wrapper(PyObject *self, PyObject *const *args, size_t nargsf);
Tensor *_exp(PyObject *self, PyObject *const *args, size_t nargsf);
Tensor *_log(PyObject *self, PyObject *const *args, size_t nargsf);
Tensor *_log10(PyObject *self, PyObject *const *args, size_t nargsf);
Tensor *_sqrt(PyObject *self, PyObject *const *args, size_t nargsf);
Tensor *_abs(PyObject *self, PyObject *const *args, size_t nargsf);
Tensor *_pow(PyObject *self, PyObject *const *args, size_t nargsf);

#endif