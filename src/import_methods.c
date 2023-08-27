#define NO_IMPORT_ARRAY
#include "tensor.h"
#include "import_methods.h"

#ifdef _MSC_VER
#define Format_String "%s %lld"
#else
#define Format_String "%s %ld"
#endif

inline static PyObject **_Generic_check_import_if_success(PyObject **struct_with_methods, const char *message, int start_line)
{
    PyObject **begin = struct_with_methods;
    PyObject **end = begin + sizeof(*struct_with_methods) / sizeof(PyObject *);

    for (PyObject **p = begin; p != end; ++p)
    {
        if (*p == NULL)
        {
            char line[100];
            // 23 is the number of ops in the list above, need to change if num ops are changed
            sprintf(line, Format_String, "Failed to import xla ops at import_methods.c line", p - begin + start_line);
            PyErr_SetString(PyExc_RuntimeError, (const char *)line);
            free(struct_with_methods);
            return NULL;
        }
    }
    return struct_with_methods;
}

XLA_OPS *import_xla_ops(XLA_OPS **xla_ops_)
{

    PyObject *xla_client = PyImport_ImportModule("jaxlib.xla_client");
    if (xla_client == NULL)
    {
        return NULL;
    }

    PyObject *ops = PyObject_GetAttrString(xla_client, "ops");
    if (ops == NULL)
    {
        return NULL;
    }
    *xla_ops_ = malloc(sizeof(XLA_OPS));
    XLA_OPS *xla_ops = *xla_ops_;
    xla_ops->abs = PyObject_GetAttrString(ops, "Abs");
    xla_ops->acos = PyObject_GetAttrString(ops, "Acos");
    xla_ops->acosh = PyObject_GetAttrString(ops, "Acosh");
    xla_ops->add = PyObject_GetAttrString(ops, "Add");
    xla_ops->afterall = PyObject_GetAttrString(ops, "AfterAll");
    xla_ops->allgather = PyObject_GetAttrString(ops, "AllGather");
    xla_ops->allreduce = PyObject_GetAttrString(ops, "AllReduce");
    xla_ops->alltoall = PyObject_GetAttrString(ops, "AllToAll");
    xla_ops->and = PyObject_GetAttrString(ops, "And");
    xla_ops->approx_topK = PyObject_GetAttrString(ops, "ApproxTopK");
    xla_ops->approx_topK_fallback = PyObject_GetAttrString(ops, "ApproxTopKFallback");
    xla_ops->approx_TopK_reduction_output_size = PyObject_GetAttrString(ops, "ApproxTopKReductionOutputSize");
    xla_ops->asin = PyObject_GetAttrString(ops, "Asin");
    xla_ops->asinh = PyObject_GetAttrString(ops, "Asinh");
    xla_ops->atan = PyObject_GetAttrString(ops, "Atan");
    xla_ops->atan2 = PyObject_GetAttrString(ops, "Atan2");
    xla_ops->atanh = PyObject_GetAttrString(ops, "Atanh");
    xla_ops->besselI0e = PyObject_GetAttrString(ops, "BesselI0e");
    xla_ops->besselI1e = PyObject_GetAttrString(ops, "BesselI1e");
    xla_ops->bit_cast_convert_type = PyObject_GetAttrString(ops, "BitcastConvertType");
    xla_ops->broadcast = PyObject_GetAttrString(ops, "Broadcast");
    xla_ops->broadcast_in_dim = PyObject_GetAttrString(ops, "BroadcastInDim");
    xla_ops->call = PyObject_GetAttrString(ops, "Call");
    xla_ops->cbrt = PyObject_GetAttrString(ops, "Cbrt");
    xla_ops->ceil = PyObject_GetAttrString(ops, "Ceil");
    xla_ops->cholesky = PyObject_GetAttrString(ops, "Cholesky");
    xla_ops->clamp = PyObject_GetAttrString(ops, "Clamp");
    xla_ops->clz = PyObject_GetAttrString(ops, "Clz");
    xla_ops->collapse = PyObject_GetAttrString(ops, "Collapse");
    xla_ops->collective_permute = PyObject_GetAttrString(ops, "CollectivePermute");
    xla_ops->Complex = PyObject_GetAttrString(ops, "Complex");
    xla_ops->concat_in_dim = PyObject_GetAttrString(ops, "ConcatInDim");
    xla_ops->conditional = PyObject_GetAttrString(ops, "Conditional");
    xla_ops->conj = PyObject_GetAttrString(ops, "Conj");
    xla_ops->constant = PyObject_GetAttrString(ops, "Constant");
    xla_ops->constant_literal = PyObject_GetAttrString(ops, "ConstantLiteral");
    xla_ops->convert_element_type = PyObject_GetAttrString(ops, "ConvertElementType");
    xla_ops->conv_general_dilated = PyObject_GetAttrString(ops, "ConvGeneralDilated");
    xla_ops->cos = PyObject_GetAttrString(ops, "Cos");
    xla_ops->cosh = PyObject_GetAttrString(ops, "Cosh");
    xla_ops->create_token = PyObject_GetAttrString(ops, "CreateToken");
    xla_ops->cross_replica_sum = PyObject_GetAttrString(ops, "CrossReplicaSum");
    xla_ops->custom_call = PyObject_GetAttrString(ops, "CustomCall");
    xla_ops->custom_call_with_aliasing = PyObject_GetAttrString(ops, "CustomCallWithAliasing");
    xla_ops->custom_call_with_computation = PyObject_GetAttrString(ops, "CustomCallWithComputation");
    xla_ops->custom_call_with_layout = PyObject_GetAttrString(ops, "CustomCallWithLayout");
    xla_ops->digmma = PyObject_GetAttrString(ops, "Digamma");
    xla_ops->div = PyObject_GetAttrString(ops, "Div");
    xla_ops->dot = PyObject_GetAttrString(ops, "Dot");
    xla_ops->dynamic_slice = PyObject_GetAttrString(ops, "DynamicSlice");
    xla_ops->dynamic_update_slice = PyObject_GetAttrString(ops, "DynamicUpdateSlice");
    xla_ops->dotgeneral = PyObject_GetAttrString(ops, "DotGeneral");
    xla_ops->dynamic_reshape = PyObject_GetAttrString(ops, "DynamicReshape");
    xla_ops->eigh = PyObject_GetAttrString(ops, "Eigh");
    xla_ops->eq = PyObject_GetAttrString(ops, "Eq");
    xla_ops->erf = PyObject_GetAttrString(ops, "Erf");
    xla_ops->erfc = PyObject_GetAttrString(ops, "Erfc");
    xla_ops->erf_inv = PyObject_GetAttrString(ops, "ErfInv");
    xla_ops->exp = PyObject_GetAttrString(ops, "Exp");
    xla_ops->expm1 = PyObject_GetAttrString(ops, "Expm1");
    xla_ops->fft = PyObject_GetAttrString(ops, "Fft");
    xla_ops->floor = PyObject_GetAttrString(ops, "Floor");
    xla_ops->ge = PyObject_GetAttrString(ops, "Ge");
    xla_ops->get_dimension_size = PyObject_GetAttrString(ops, "GetDimensionSize");
    xla_ops->get_tuple_element = PyObject_GetAttrString(ops, "GetTupleElement");
    xla_ops->gt = PyObject_GetAttrString(ops, "Gt");
    xla_ops->igamma = PyObject_GetAttrString(ops, "Igamma");
    xla_ops->igammac = PyObject_GetAttrString(ops, "Igammac");
    xla_ops->igamma_grad_a = PyObject_GetAttrString(ops, "IgammaGradA");
    xla_ops->imag = PyObject_GetAttrString(ops, "Imag");
    xla_ops->infeed_with_token = PyObject_GetAttrString(ops, "InfeedWithToken");
    xla_ops->iota = PyObject_GetAttrString(ops, "Iota");
    xla_ops->is_finite = PyObject_GetAttrString(ops, "IsFinite");
    xla_ops->le = PyObject_GetAttrString(ops, "Le");
    xla_ops->lgamma = PyObject_GetAttrString(ops, "Lgamma");
    xla_ops->log = PyObject_GetAttrString(ops, "Log");
    xla_ops->log1p = PyObject_GetAttrString(ops, "Log1p");
    xla_ops->lt = PyObject_GetAttrString(ops, "Lt");
    xla_ops->lu = PyObject_GetAttrString(ops, "LU");
    xla_ops->map = PyObject_GetAttrString(ops, "Map");
    xla_ops->max = PyObject_GetAttrString(ops, "Max");
    xla_ops->min = PyObject_GetAttrString(ops, "Min");
    xla_ops->mul = PyObject_GetAttrString(ops, "Mul");
    xla_ops->ne = PyObject_GetAttrString(ops, "Ne");
    xla_ops->neg = PyObject_GetAttrString(ops, "Neg");
    xla_ops->next_after = PyObject_GetAttrString(ops, "NextAfter");
    xla_ops->not = PyObject_GetAttrString(ops, "Not");
    xla_ops->optimization_barrier = PyObject_GetAttrString(ops, "OptimizationBarrier");
    xla_ops->or = PyObject_GetAttrString(ops, "Or");
    xla_ops->outfeed_with_token = PyObject_GetAttrString(ops, "OutfeedWithToken");
    xla_ops->pad = PyObject_GetAttrString(ops, "Pad");
    xla_ops->parameter = PyObject_GetAttrString(ops, "Parameter");
    xla_ops->pow = PyObject_GetAttrString(ops, "Pow");
    xla_ops->ProductOfElementaryHouseholderReflectors = PyObject_GetAttrString(ops, "ProductOfElementaryHouseholderReflectors");
    xla_ops->qr = PyObject_GetAttrString(ops, "QR");
    xla_ops->qr_decomposition = PyObject_GetAttrString(ops, "QrDecomposition");
    xla_ops->random_gamma_grad = PyObject_GetAttrString(ops, "RandomGammaGrad");
    xla_ops->real = PyObject_GetAttrString(ops, "Real");
    xla_ops->reciprocal = PyObject_GetAttrString(ops, "Reciprocal");
    xla_ops->recv_from_host = PyObject_GetAttrString(ops, "RecvFromHost");
    xla_ops->reduce = PyObject_GetAttrString(ops, "Reduce");
    xla_ops->reduce_precision = PyObject_GetAttrString(ops, "ReducePrecision");
    xla_ops->reduce_scatter = PyObject_GetAttrString(ops, "ReduceScatter");
    xla_ops->ReduceWindowWithGeneralPadding = PyObject_GetAttrString(ops, "ReduceWindowWithGeneralPadding");
    xla_ops->regularized_incomplete_beta = PyObject_GetAttrString(ops, "RegularizedIncompleteBeta");
    xla_ops->rem = PyObject_GetAttrString(ops, "Rem");
    xla_ops->remove_dynamic_dimension = PyObject_GetAttrString(ops, "RemoveDynamicDimension");
    xla_ops->replica_id = PyObject_GetAttrString(ops, "ReplicaId");
    xla_ops->reshape = PyObject_GetAttrString(ops, "Reshape");
    xla_ops->rev = PyObject_GetAttrString(ops, "Rev");
    xla_ops->rng_bit_generator = PyObject_GetAttrString(ops, "RngBitGenerator");
    xla_ops->rng_normal = PyObject_GetAttrString(ops, "RngNormal");
    xla_ops->rng_uniform = PyObject_GetAttrString(ops, "RngUniform");
    xla_ops->round = PyObject_GetAttrString(ops, "Round");
    xla_ops->rsqrt = PyObject_GetAttrString(ops, "Rsqrt");
    xla_ops->scatter = PyObject_GetAttrString(ops, "Scatter");
    xla_ops->select = PyObject_GetAttrString(ops, "Select");
    xla_ops->select_and_scatter_with_general_padding = PyObject_GetAttrString(ops, "SelectAndScatterWithGeneralPadding");
    xla_ops->send_to_host = PyObject_GetAttrString(ops, "SendToHost");
    xla_ops->set_dimension_size = PyObject_GetAttrString(ops, "SetDimensionSize");
    xla_ops->shift_left = PyObject_GetAttrString(ops, "ShiftLeft");
    xla_ops->shift_right_arithmetic = PyObject_GetAttrString(ops, "ShiftRightArithmetic");
    xla_ops->shift_right_logical = PyObject_GetAttrString(ops, "ShiftRightLogical");
    xla_ops->sign = PyObject_GetAttrString(ops, "Sign");
    xla_ops->sin = PyObject_GetAttrString(ops, "Sin");
    xla_ops->sinh = PyObject_GetAttrString(ops, "Sinh");
    xla_ops->slice = PyObject_GetAttrString(ops, "Slice");
    xla_ops->slice_in_dim = PyObject_GetAttrString(ops, "SliceInDim");
    xla_ops->sort = PyObject_GetAttrString(ops, "Sort");
    xla_ops->sqrt = PyObject_GetAttrString(ops, "Sqrt");
    xla_ops->square = PyObject_GetAttrString(ops, "Square");
    xla_ops->sub = PyObject_GetAttrString(ops, "Sub");
    xla_ops->svd = PyObject_GetAttrString(ops, "SVD");
    xla_ops->tan = PyObject_GetAttrString(ops, "Tan");
    xla_ops->tanh = PyObject_GetAttrString(ops, "Tanh");
    xla_ops->topK = PyObject_GetAttrString(ops, "TopK");
    xla_ops->transpose = PyObject_GetAttrString(ops, "Transpose");
    xla_ops->triangular_solve = PyObject_GetAttrString(ops, "TriangularSolve");
    xla_ops->tuple = PyObject_GetAttrString(ops, "Tuple");
    xla_ops->While = PyObject_GetAttrString(ops, "While");
    xla_ops->xor = PyObject_GetAttrString(ops, "xor");
    xla_ops->zeta = PyObject_GetAttrString(ops, "Zeta");
    Py_DecRef(ops);
    Py_DecRef(xla_client);
    if (_Generic_check_import_if_success((PyObject **)xla_ops_, "Failed to import xla ops at import_methods.c line", 39) == NULL)
        return NULL;
    else
        return xla_ops;
}

np_method *import_np_methods(np_method **NP_METHOD_)
{
    *NP_METHOD_ = malloc(sizeof(np_method));
    np_method *NP_METHOD = *NP_METHOD_;
    PyObject *module = PyImport_ImportModule("numpy");
    if (module == NULL)
        return NULL;
    NP_METHOD->sin = PyObject_GetAttrString(module, "sin");
    NP_METHOD->cos = PyObject_GetAttrString(module, "cos");
    NP_METHOD->tan = PyObject_GetAttrString(module, "tan");
    NP_METHOD->arcsin = PyObject_GetAttrString(module, "arcsin");
    NP_METHOD->arccos = PyObject_GetAttrString(module, "arccos");
    NP_METHOD->arctan = PyObject_GetAttrString(module, "arctan");
    NP_METHOD->sinh = PyObject_GetAttrString(module, "sinh");
    NP_METHOD->cosh = PyObject_GetAttrString(module, "cosh");
    NP_METHOD->tanh = PyObject_GetAttrString(module, "tanh");
    NP_METHOD->arcsinh = PyObject_GetAttrString(module, "arcsinh");
    NP_METHOD->arccosh = PyObject_GetAttrString(module, "arccosh");
    NP_METHOD->arctanh = PyObject_GetAttrString(module, "arctanh");
    NP_METHOD->absolute = PyObject_GetAttrString(module, "absolute");
    NP_METHOD->exp = PyObject_GetAttrString(module, "exp");
    NP_METHOD->log = PyObject_GetAttrString(module, "log");
    NP_METHOD->log10 = PyObject_GetAttrString(module, "log10");
    NP_METHOD->log1P = PyObject_GetAttrString(module, "log1p");
    NP_METHOD->sqrt = PyObject_GetAttrString(module, "sqrt");
    NP_METHOD->square = PyObject_GetAttrString(module, "square");
    NP_METHOD->abs = PyObject_GetAttrString(module, "abs");
    NP_METHOD->add = PyObject_GetAttrString(module, "add");
    NP_METHOD->multiply = PyObject_GetAttrString(module, "multiply");
    NP_METHOD->subtract = PyObject_GetAttrString(module, "subtract");
    NP_METHOD->divide = PyObject_GetAttrString(module, "divide");
    NP_METHOD->power = PyObject_GetAttrString(module, "power");
    NP_METHOD->mean = PyObject_GetAttrString(module, "mean");
    NP_METHOD->dot = PyObject_GetAttrString(module, "dot");
    NP_METHOD->matmul = PyObject_GetAttrString(module, "matmul");
    NP_METHOD->transpose = PyObject_GetAttrString(module, "transpose");
    NP_METHOD->reshape = PyObject_GetAttrString(module, "reshape");
    NP_METHOD->tensordot = PyObject_GetAttrString(module, "tensordot");
    NP_METHOD->concatenate = PyObject_GetAttrString(module, "concatenate");
    Py_DecRef(module);
    if (_Generic_check_import_if_success((PyObject **)NP_METHOD_, "Failed to import numpy methods at import_methods.c line", 197) == NULL)
        return NULL;
    else
    {
        return NP_METHOD;
    }
}

jnp_method *import_jnp_methods(jnp_method **JNP_METHOD_)
{
    *JNP_METHOD_ = malloc(sizeof(jnp_method));
    jnp_method *JNP_METHOD = *JNP_METHOD_;
    PyObject *module = PyImport_ImportModule("jax.numpy");
    PyObject *module2 = PyImport_ImportModule("jax");
    PyObject *module3 = PyImport_ImportModule("jax._src.sharding_impls");
    if (module == NULL || module2 == NULL || module3 == NULL)
        return NULL;
    JNP_METHOD->unspecified_value = PyObject_GetAttrString(module3, "UnspecifiedValue");
    JNP_METHOD->copy = PyObject_GetAttrString(module, "copy");
    JNP_METHOD->jit = PyObject_GetAttrString(module2, "jit");
    JNP_METHOD->array = PyObject_GetAttrString(module, "array");
    JNP_METHOD->negative = PyObject_GetAttrString(module, "negative");
    JNP_METHOD->sin = PyObject_GetAttrString(module, "sin");
    JNP_METHOD->cos = PyObject_GetAttrString(module, "cos");
    JNP_METHOD->tan = PyObject_GetAttrString(module, "tan");
    JNP_METHOD->arcsin = PyObject_GetAttrString(module, "arcsin");
    JNP_METHOD->arccos = PyObject_GetAttrString(module, "arccos");
    JNP_METHOD->arctan = PyObject_GetAttrString(module, "arctan");
    JNP_METHOD->sinh = PyObject_GetAttrString(module, "sinh");
    JNP_METHOD->cosh = PyObject_GetAttrString(module, "cosh");
    JNP_METHOD->tanh = PyObject_GetAttrString(module, "tanh");
    JNP_METHOD->arcsinh = PyObject_GetAttrString(module, "arcsinh");
    JNP_METHOD->arccosh = PyObject_GetAttrString(module, "arccosh");
    JNP_METHOD->arctanh = PyObject_GetAttrString(module, "arctanh");
    JNP_METHOD->absolute = PyObject_GetAttrString(module, "absolute");
    JNP_METHOD->exp = PyObject_GetAttrString(module, "exp");
    JNP_METHOD->log = PyObject_GetAttrString(module, "log");
    JNP_METHOD->log10 = PyObject_GetAttrString(module, "log10");
    JNP_METHOD->log1P = PyObject_GetAttrString(module, "log1p");
    JNP_METHOD->sqrt = PyObject_GetAttrString(module, "sqrt");
    JNP_METHOD->square = PyObject_GetAttrString(module, "square");
    JNP_METHOD->abs = PyObject_GetAttrString(module, "abs");
    JNP_METHOD->add = PyObject_GetAttrString(module, "add");
    JNP_METHOD->multiply = PyObject_GetAttrString(module, "multiply");
    JNP_METHOD->subtract = PyObject_GetAttrString(module, "subtract");
    JNP_METHOD->divide = PyObject_GetAttrString(module, "divide");
    JNP_METHOD->power = PyObject_GetAttrString(module, "power");
    JNP_METHOD->mean = PyObject_GetAttrString(module, "mean");
    JNP_METHOD->dot = PyObject_GetAttrString(module, "dot");
    JNP_METHOD->matmul = PyObject_GetAttrString(module, "matmul");
    JNP_METHOD->transpose = PyObject_GetAttrString(module, "transpose");
    JNP_METHOD->reshape = PyObject_GetAttrString(module, "reshape");
    JNP_METHOD->tensordot = PyObject_GetAttrString(module, "tensordot");
    JNP_METHOD->concatenate = PyObject_GetAttrString(module, "concatenate");
    JNP_METHOD->sum = PyObject_GetAttrString(module, "sum");
    Py_DecRef(module);
    Py_DecRef(module2);
    Py_DecRef(module3);
    if (_Generic_check_import_if_success((PyObject **)JNP_METHOD_, "Failed to import numpy methods at import_methods.c line", 248) == NULL)
        return NULL;
    else
    {
        return JNP_METHOD;
    }
}

void free_xla_ops(XLA_OPS *xla_ops)
{
    Py_DecRef(xla_ops->abs);
    Py_DecRef(xla_ops->acos);
    Py_DecRef(xla_ops->acosh);
    Py_DecRef(xla_ops->add);
    Py_DecRef(xla_ops->afterall);
    Py_DecRef(xla_ops->allgather);
    Py_DecRef(xla_ops->allreduce);
    Py_DecRef(xla_ops->alltoall);
    Py_DecRef(xla_ops->and);
    Py_DecRef(xla_ops->approx_topK);
    Py_DecRef(xla_ops->approx_topK_fallback);
    Py_DecRef(xla_ops->approx_TopK_reduction_output_size);
    Py_DecRef(xla_ops->asin);
    Py_DecRef(xla_ops->asinh);
    Py_DecRef(xla_ops->atan);
    Py_DecRef(xla_ops->atan2);
    Py_DecRef(xla_ops->atanh);
    Py_DecRef(xla_ops->besselI0e);
    Py_DecRef(xla_ops->besselI1e);
    Py_DecRef(xla_ops->bit_cast_convert_type);
    Py_DecRef(xla_ops->broadcast);
    Py_DecRef(xla_ops->broadcast_in_dim);
    Py_DecRef(xla_ops->call);
    Py_DecRef(xla_ops->cbrt);
    Py_DecRef(xla_ops->ceil);
    Py_DecRef(xla_ops->cholesky);
    Py_DecRef(xla_ops->clamp);
    Py_DecRef(xla_ops->clz);
    Py_DecRef(xla_ops->collapse);
    Py_DecRef(xla_ops->collective_permute);
    Py_DecRef(xla_ops->Complex);
    Py_DecRef(xla_ops->concat_in_dim);
    Py_DecRef(xla_ops->conditional);
    Py_DecRef(xla_ops->conj);
    Py_DecRef(xla_ops->constant);
    Py_DecRef(xla_ops->constant_literal);
    Py_DecRef(xla_ops->convert_element_type);
    Py_DecRef(xla_ops->conv_general_dilated);
    Py_DecRef(xla_ops->cos);
    Py_DecRef(xla_ops->cosh);
    Py_DecRef(xla_ops->create_token);
    Py_DecRef(xla_ops->cross_replica_sum);
    Py_DecRef(xla_ops->custom_call);
    Py_DecRef(xla_ops->custom_call_with_aliasing);
    Py_DecRef(xla_ops->custom_call_with_computation);
    Py_DecRef(xla_ops->custom_call_with_layout);
    Py_DecRef(xla_ops->digmma);
    Py_DecRef(xla_ops->div);
    Py_DecRef(xla_ops->dot);
    Py_DecRef(xla_ops->dynamic_slice);
    Py_DecRef(xla_ops->dynamic_update_slice);
    Py_DecRef(xla_ops->dotgeneral);
    Py_DecRef(xla_ops->dynamic_reshape);
    Py_DecRef(xla_ops->eigh);
    Py_DecRef(xla_ops->eq);
    Py_DecRef(xla_ops->erf);
    Py_DecRef(xla_ops->erfc);
    Py_DecRef(xla_ops->erf_inv);
    Py_DecRef(xla_ops->exp);
    Py_DecRef(xla_ops->expm1);
    Py_DecRef(xla_ops->fft);
    Py_DecRef(xla_ops->floor);
    Py_DecRef(xla_ops->ge);
    Py_DecRef(xla_ops->get_dimension_size);
    Py_DecRef(xla_ops->get_tuple_element);
    Py_DecRef(xla_ops->gt);
    Py_DecRef(xla_ops->igamma);
    Py_DecRef(xla_ops->igammac);
    Py_DecRef(xla_ops->igamma_grad_a);
    Py_DecRef(xla_ops->imag);
    Py_DecRef(xla_ops->infeed_with_token);
    Py_DecRef(xla_ops->iota);
    Py_DecRef(xla_ops->is_finite);
    Py_DecRef(xla_ops->le);
    Py_DecRef(xla_ops->lgamma);
    Py_DecRef(xla_ops->log);
    Py_DecRef(xla_ops->log1p);
    Py_DecRef(xla_ops->lt);
    Py_DecRef(xla_ops->lu);
    Py_DecRef(xla_ops->map);
    Py_DecRef(xla_ops->max);
    Py_DecRef(xla_ops->min);
    Py_DecRef(xla_ops->mul);
    Py_DecRef(xla_ops->ne);
    Py_DecRef(xla_ops->neg);
    Py_DecRef(xla_ops->next_after);
    Py_DecRef(xla_ops->not );
    Py_DecRef(xla_ops->optimization_barrier);
    Py_DecRef(xla_ops->or);
    Py_DecRef(xla_ops->outfeed_with_token);
    Py_DecRef(xla_ops->pad);
    Py_DecRef(xla_ops->parameter);
    Py_DecRef(xla_ops->pow);
    Py_DecRef(xla_ops->ProductOfElementaryHouseholderReflectors);
    Py_DecRef(xla_ops->qr);
    Py_DecRef(xla_ops->qr_decomposition);
    Py_DecRef(xla_ops->random_gamma_grad);
    Py_DecRef(xla_ops->real);
    Py_DecRef(xla_ops->reciprocal);
    Py_DecRef(xla_ops->recv_from_host);
    Py_DecRef(xla_ops->reduce);
    Py_DecRef(xla_ops->reduce_precision);
    Py_DecRef(xla_ops->reduce_scatter);
    Py_DecRef(xla_ops->ReduceWindowWithGeneralPadding);
    Py_DecRef(xla_ops->regularized_incomplete_beta);
    Py_DecRef(xla_ops->rem);
    Py_DecRef(xla_ops->remove_dynamic_dimension);
    Py_DecRef(xla_ops->replica_id);
    Py_DecRef(xla_ops->reshape);
    Py_DecRef(xla_ops->rev);
    Py_DecRef(xla_ops->rng_bit_generator);
    Py_DecRef(xla_ops->rng_normal);
    Py_DecRef(xla_ops->rng_uniform);
    Py_DecRef(xla_ops->round);
    Py_DecRef(xla_ops->rsqrt);
    Py_DecRef(xla_ops->scatter);
    Py_DecRef(xla_ops->select);
    Py_DecRef(xla_ops->select_and_scatter_with_general_padding);
    Py_DecRef(xla_ops->send_to_host);
    Py_DecRef(xla_ops->set_dimension_size);
    Py_DecRef(xla_ops->shift_left);
    Py_DecRef(xla_ops->shift_right_arithmetic);
    Py_DecRef(xla_ops->shift_right_logical);
    Py_DecRef(xla_ops->sign);
    Py_DecRef(xla_ops->sin);
    Py_DecRef(xla_ops->sinh);
    Py_DecRef(xla_ops->slice);
    Py_DecRef(xla_ops->slice_in_dim);
    Py_DecRef(xla_ops->sort);
    Py_DecRef(xla_ops->sqrt);
    Py_DecRef(xla_ops->square);
    Py_DecRef(xla_ops->sub);
    Py_DecRef(xla_ops->svd);
    Py_DecRef(xla_ops->tan);
    Py_DecRef(xla_ops->tanh);
    Py_DecRef(xla_ops->topK);
    Py_DecRef(xla_ops->transpose);
    Py_DecRef(xla_ops->triangular_solve);
    Py_DecRef(xla_ops->tuple);
    Py_DecRef(xla_ops->While);
    Py_DecRef(xla_ops->xor);
    Py_DecRef(xla_ops->zeta);
    free(xla_ops);
}

void free_jnp_methods(jnp_method *jnp_method)
{
    Py_DecRef(jnp_method->unspecified_value);
    Py_DecRef(jnp_method->copy);
    Py_DecRef(jnp_method->jit);
    Py_DecRef(jnp_method->array);
    Py_DecRef(jnp_method->negative);
    Py_DecRef(jnp_method->sin);
    Py_DecRef(jnp_method->cos);
    Py_DecRef(jnp_method->tan);
    Py_DecRef(jnp_method->arcsin);
    Py_DecRef(jnp_method->arccos);
    Py_DecRef(jnp_method->arctan);
    Py_DecRef(jnp_method->sinh);
    Py_DecRef(jnp_method->cosh);
    Py_DecRef(jnp_method->tanh);
    Py_DecRef(jnp_method->arcsinh);
    Py_DecRef(jnp_method->arccosh);
    Py_DecRef(jnp_method->arctanh);
    Py_DecRef(jnp_method->absolute);
    Py_DecRef(jnp_method->exp);
    Py_DecRef(jnp_method->log);
    Py_DecRef(jnp_method->log10);
    Py_DecRef(jnp_method->log1P);
    Py_DecRef(jnp_method->sqrt);
    Py_DecRef(jnp_method->square);
    Py_DecRef(jnp_method->abs);
    Py_DecRef(jnp_method->add);
    Py_DecRef(jnp_method->multiply);
    Py_DecRef(jnp_method->subtract);
    Py_DecRef(jnp_method->divide);
    Py_DecRef(jnp_method->power);
    Py_DecRef(jnp_method->mean);
    Py_DecRef(jnp_method->dot);
    Py_DecRef(jnp_method->matmul);
    Py_DecRef(jnp_method->transpose);
    Py_DecRef(jnp_method->reshape);
    Py_DecRef(jnp_method->tensordot);
    Py_DecRef(jnp_method->concatenate);
    Py_DecRef(jnp_method->sum);
    free(jnp_method);
}

void free_np_methods(np_method *np_method)
{
    Py_DecRef(np_method->sin);
    Py_DecRef(np_method->cos);
    Py_DecRef(np_method->tan);
    Py_DecRef(np_method->arcsin);
    Py_DecRef(np_method->arccos);
    Py_DecRef(np_method->arctan);
    Py_DecRef(np_method->sinh);
    Py_DecRef(np_method->cosh);
    Py_DecRef(np_method->tanh);
    Py_DecRef(np_method->arcsinh);
    Py_DecRef(np_method->arccosh);
    Py_DecRef(np_method->arctanh);
    Py_DecRef(np_method->absolute);
    Py_DecRef(np_method->exp);
    Py_DecRef(np_method->log);
    Py_DecRef(np_method->log10);
    Py_DecRef(np_method->log1P);
    Py_DecRef(np_method->sqrt);
    Py_DecRef(np_method->square);
    Py_DecRef(np_method->abs);
    Py_DecRef(np_method->add);
    Py_DecRef(np_method->multiply);
    Py_DecRef(np_method->subtract);
    Py_DecRef(np_method->divide);
    Py_DecRef(np_method->power);
    Py_DecRef(np_method->mean);
    Py_DecRef(np_method->dot);
    Py_DecRef(np_method->matmul);
    Py_DecRef(np_method->transpose);
    Py_DecRef(np_method->reshape);
    Py_DecRef(np_method->tensordot);
    Py_DecRef(np_method->concatenate);
    free(np_method);
}