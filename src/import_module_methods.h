#ifndef PYTHON_H
#define PYTHON_H
#include <Python.h>
#endif

#ifdef DEBUG
#define DEBUG_PRINT(...) printf("%s: ", __func__); printf(__VA_ARGS__);

#define DEBUG_PRINTLN(...)   \
    {                        \
        printf(__VA_ARGS__); \
        printf("\n");        \
    }

#define DEBUG_PRINT_SHAPE(shape, len)  \
    for (npy_intp j = 0; j < len; j++) \
    {                                  \
        printf("%ld ", i[j]);          \
    }

#define DEBUG_PyObject_Print(obj)   \
    PyObject_Print(obj, stdout, 0); \
    printf("\n");

#define DEBUG_FOR_LOOP_PRINT(arr, len) \
    for (npy_intp j = 0; j < len; j++) \
    {                                  \
        printf("%ld ", arr[j]);        \
    }                                  \
    printf("\n");

#define DEBUG_FOR_LOOP(...) \
    for (__VA_ARGS__)

#else
#define DEBUG_PRINT(...)
#define DEBUG_PyObject_Print(obj)
#define DEBUG_FOR_LOOP(...)
#define DEBUG_PRINTLN(...)
#define DEBUG_PRINT_SHAPE(shape, len)
#define DEBUG_FOR_LOOP_PRINT(arr, len)
#endif

#ifndef XLA_OPS_H
#define XLA_OPS_H
typedef struct
{
    PyObject *abs;
    PyObject *acos;
    PyObject *acosh;
    PyObject *add;
    PyObject *afterall;
    PyObject *allgather;
    PyObject *allreduce;
    PyObject *alltoall;
    PyObject *and_;
    PyObject *approx_topK;
    PyObject *approx_topK_fallback;
    PyObject *approx_TopK_reduction_output_size;
    PyObject *asin;
    PyObject *asinh;
    PyObject *atan;
    PyObject *atan2;
    PyObject *atanh;
    PyObject *besselI0e;
    PyObject *besselI1e;
    PyObject *bit_cast_convert_type;
    PyObject *broadcast;
    PyObject *broadcast_in_dim;
    PyObject *call;
    PyObject *cbrt;
    PyObject *ceil;
    PyObject *cholesky;
    PyObject *clamp;
    PyObject *clz;
    PyObject *collapse;
    PyObject *collective_permute;
    PyObject *Complex;
    PyObject *concat_in_dim;
    PyObject *conditional;
    PyObject *conj;
    PyObject *constant;
    PyObject *constant_literal;
    PyObject *convert_element_type;
    PyObject *conv_general_dilated;
    PyObject *cos;
    PyObject *cosh;
    PyObject *create_token;
    PyObject *cross_replica_sum;
    PyObject *custom_call;
    PyObject *custom_call_with_aliasing;
    PyObject *custom_call_with_computation;
    PyObject *custom_call_with_layout;
    PyObject *digmma;
    PyObject *div;
    PyObject *dot;
    PyObject *dynamic_slice;
    PyObject *dynamic_update_slice;
    PyObject *dotgeneral;
    PyObject *dynamic_reshape;
    PyObject *eigh;
    PyObject *eq;
    PyObject *erf;
    PyObject *erfc;
    PyObject *erf_inv;
    PyObject *exp;
    PyObject *expm1;
    PyObject *fft;
    PyObject *floor;
    PyObject *ge;
    PyObject *get_dimension_size;
    PyObject *get_tuple_element;
    PyObject *gt;
    PyObject *igamma;
    PyObject *igammac;
    PyObject *igamma_grad_a;
    PyObject *imag;
    PyObject *infeed_with_token;
    PyObject *iota;
    PyObject *is_finite;
    PyObject *le;
    PyObject *lgamma;
    PyObject *log;
    PyObject *log1p;
    PyObject *lt;
    PyObject *lu;
    PyObject *map;
    PyObject *max;
    PyObject *min;
    PyObject *mul;
    PyObject *ne;
    PyObject *neg;
    PyObject *next_after;
    PyObject * not_;
    PyObject *optimization_barrier;
    PyObject * or_;
    PyObject *outfeed_with_token;
    PyObject *pad;
    PyObject *parameter;
    PyObject *pow;
    PyObject *ProductOfElementaryHouseholderReflectors;
    PyObject *qr;
    PyObject *qr_decomposition;
    PyObject *random_gamma_grad;
    PyObject *real;
    PyObject *reciprocal;
    PyObject *recv_from_host;
    PyObject *reduce;
    PyObject *reduce_precision;
    PyObject *reduce_scatter;
    PyObject *ReduceWindowWithGeneralPadding;
    PyObject *regularized_incomplete_beta;
    PyObject *rem;
    PyObject *remove_dynamic_dimension;
    PyObject *replica_id;
    PyObject *reshape;
    PyObject *rev;
    PyObject *rng_bit_generator;
    PyObject *rng_normal;
    PyObject *rng_uniform;
    PyObject *round;
    PyObject *rsqrt;
    PyObject *scatter;
    PyObject *select;
    PyObject *select_and_scatter_with_general_padding;
    PyObject *send_to_host;
    PyObject *set_dimension_size;
    PyObject *shift_left;
    PyObject *shift_right_arithmetic;
    PyObject *shift_right_logical;
    PyObject *sign;
    PyObject *sin;
    PyObject *sinh;
    PyObject *slice;
    PyObject *slice_in_dim;
    PyObject *sort;
    PyObject *sqrt;
    PyObject *square;
    PyObject *sub;
    PyObject *svd;
    PyObject *tan;
    PyObject *tanh;
    PyObject *topK;
    PyObject *transpose;
    PyObject *triangular_solve;
    PyObject *tuple;
    PyObject *While;
    PyObject * xor_;
    PyObject *zeta;

} XLA_OPS;
#endif

#ifndef NP_METHOD_H
#define NP_METHOD_H
typedef struct
{
    PyObject *sin;
    PyObject *cos;
    PyObject *tan;
    PyObject *sinh;
    PyObject *cosh;
    PyObject *tanh;
    PyObject *arcsin;
    PyObject *arccos;
    PyObject *arctan;
    PyObject *arcsinh;
    PyObject *arccosh;
    PyObject *arctanh;
    PyObject *absolute;
    PyObject *exp;
    PyObject *log;
    PyObject *log10;
    PyObject *log1P;
    PyObject *sqrt;
    PyObject *square;
    PyObject *abs;
    PyObject *add;
    PyObject *multiply;
    PyObject *subtract;
    PyObject *divide;
    PyObject *power;
    PyObject *mean;
    PyObject *dot;
    PyObject *matmul;
    PyObject *transpose;
    PyObject *reshape;
    PyObject *tensordot;
    PyObject *concatenate;
    PyObject *where;
} np_method;
#endif

#ifndef JNP_METHOD_H
#define JNP_METHOD_H
typedef struct
{
    PyObject *copy;
    PyObject *array;
    PyObject *negative;
    PyObject *sin;
    PyObject *cos;
    PyObject *tan;
    PyObject *sinh;
    PyObject *cosh;
    PyObject *tanh;
    PyObject *arcsin;
    PyObject *arccos;
    PyObject *arctan;
    PyObject *arcsinh;
    PyObject *arccosh;
    PyObject *arctanh;
    PyObject *absolute;
    PyObject *exp;
    PyObject *log;
    PyObject *log10;
    PyObject *log1P;
    PyObject *sqrt;
    PyObject *square;
    PyObject *abs;
    PyObject *add;
    PyObject *multiply;
    PyObject *subtract;
    PyObject *divide;
    PyObject *power;
    PyObject *mean;
    PyObject *dot;
    PyObject *matmul;
    PyObject *transpose;
    PyObject *reshape;
    PyObject *tensordot;
    PyObject *concatenate;
    PyObject *jit;
    PyObject *unspecified_value;
    PyObject *sum;
} jnp_method;
#endif

XLA_OPS *import_xla_ops(XLA_OPS **xla_ops_struct);
np_method *import_np_methods(np_method **np_methods_struct);
jnp_method *import_jnp_methods(jnp_method **JNP_METHOD);
void free_xla_ops(XLA_OPS *xla_ops);
void free_np_methods(np_method *np_method);
void free_jnp_methods(jnp_method *jnp_method);