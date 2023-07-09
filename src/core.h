#ifndef CORE_H
#define CORE_H

typedef struct
{
    PyObject* sin;
    PyObject* cos;
    PyObject* tan;
    PyObject* asin;
    PyObject* acos;
    PyObject* atan;
    PyObject* sinh;
    PyObject* cosh;
    PyObject* tanh;
    PyObject* asinh;
    PyObject* acosh;
    PyObject* atanh;
    PyObject* arcsin;
    PyObject* arccos;
    PyObject* arctan;
    PyObject* arcsinh;
    PyObject* arccosh;
    PyObject* arctanh;
    PyObject* absolute;
    PyObject* exp;
    PyObject* log;
    PyObject* log10;
    PyObject* log1P;
    PyObject* sqrt;
    PyObject* square;
    PyObject* abs;
    PyObject* sign;
    PyObject* ceil;
    PyObject* floor;
    PyObject* round;
    PyObject* trunc;
    PyObject* add;
    PyObject* multiply;
    PyObject* subtract;
    PyObject* divide;
    PyObject* power;
    PyObject* sum;
    PyObject* mean;
    PyObject* max;
    PyObject* min;
    PyObject* argmax;
    PyObject* argmin;
    PyObject* dot;
    PyObject* matmul;
    PyObject* transpose;
    PyObject* reshape;
    PyObject* tensordot;
    PyObject* concatenate;
} np_method;

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

void store_array_shape(Tensor *key, npy_intp *shape, int len);
npy_intp *get_array_shape(Tensor *key);
int *get_shape_len(Tensor *key);
void store_power(Tensor *key, PyObject *power);
PyObject *get_power(Tensor *key);
void store_base(Tensor *key, PyObject *base);
PyObject *get_base(Tensor *key);

Tensor *reshape(PyObject *self, PyObject *const *args, size_t nargsf, PyObject *kwnames);
PyObject *transpose(PyObject *self, PyObject *const *args, size_t nargsf, PyObject *kwnames);
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