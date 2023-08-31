
#define Concat_(a, b) a##b
#define _First_(a, ...) a
#define _First(...) _First_(__VA_ARGS__)
#define Second_(a, b, ...) b
#define Second(...) Second_(__VA_ARGS__)
#define Logic_Not(x) Second(Concat_(Is_, x), 0)
#define Is_True(x) Logic_Not(Logic_Not(x))
#define Is_0 Place_Holder, 1
#define If(x) Concat_(If_, x)
#define Has_Args(...) Is_True(_First(End_Of_Arguments_ __VA_ARGS__)())
#define If_1(...) __VA_ARGS__
#define If_0(...) EMPTY EMPTY()()
#define If2_1(...) __VA_ARGS__
#define If2_0(...) EMPTY EMPTY()()
#define End_Of_Arguments_() 0
#define escape(...) __VA_ARGS__
#define If2(x) Concat_(If2_, x)
#define COMMA() ,

#define MAP0_No_Comma(m, x, ...) \
    m(x)                         \
        If(Has_Args(__VA_ARGS__))(DEFER2(_MAP0_No_Comma)()(m, __VA_ARGS__))

#define MAP0(m, x, ...)                 \
    m(x) If2(Has_Args(__VA_ARGS__))(, ) \
        If(Has_Args(__VA_ARGS__))(DEFER2(_MAP0)()(m, __VA_ARGS__))

#define MAP(m, x, y, ...) \
    m(y, x)               \
        If(Has_Args(__VA_ARGS__))(DEFER2(_MAP)()(m, x, __VA_ARGS__))

#define MAP2(m, x, y, z, ...) \
    m(z, x, y)                \
        If(Has_Args(__VA_ARGS__))(DEFER2(_MAP2)()(m, x, y, __VA_ARGS__))

#define MAP3(m, x, y, z, w, ...) \
    m(y, z, w, x)                \
        If(Has_Args(__VA_ARGS__))(DEFER2(_MAP3)()(m, y, z, w, __VA_ARGS__))

#define _MAP0_No_Comma() MAP0_No_Comma
#define _MAP0() MAP0
#define _MAP() MAP
#define _MAP2() MAP2
#define _MAP3() MAP3
#define Expand(...) Expand1024(__VA_ARGS__)
#define Expand1024(...) Expand512(Expand512(__VA_ARGS__))
#define Expand512(...) Expand256(Expand256(__VA_ARGS__))
#define Expand256(...) Expand128(Expand128(__VA_ARGS__))
#define Expand128(...) Expand64(Expand64(__VA_ARGS__))
#define Expand64(...) Expand32(Expand32(__VA_ARGS__))
#define Expand32(...) Expand16(Expand16(__VA_ARGS__))
#define Expand16(...) Expand8(Expand8(__VA_ARGS__))
#define Expand8(...) Expand4(Expand4(__VA_ARGS__))
#define Expand4(...) Expand2(Expand2(__VA_ARGS__))
#define Expand2(...) Expand1(Expand1(__VA_ARGS__))
#define Expand1(...) __VA_ARGS__

#define EMPTY()
#define _IF_1_ELSE(...)
#define _IF_0_ELSE(...) __VA_ARGS__
#define DEFER1(m) m EMPTY()
#define DEFER2(m) m EMPTY EMPTY()()
#define Ptr_Saved(x) x##_data_ptr_saved
#define Stries_Last(x) stride_##x##_last
#define Parameter_type(x) PyArrayObject *x,
#define Alloc_Copy_Strides_And_Indices(x, type)                                          \
    npy_intp *__strides_##x = PyArray_STRIDES(x);                                        \
    npy_intp *strides_##x = (npy_intp *)malloc(sizeof(npy_intp) * PyArray_NDIM(result)); \
    npy_intp *indice_##x##_cache = (npy_intp *)malloc(sizeof(npy_intp) * ndim);          \
    type *x##_data_ptr_saved = (type *)PyArray_DATA(x);                                  \
    memcpy(strides_##x, __strides_##x, sizeof(npy_intp) * PyArray_NDIM(result));

#define Normalize_Strides_By_Type(x, type) \
    strides_##x[i] /= sizeof(type);

#define Retrieve_Last_Stride(x, max_dim) \
    npy_intp stride_##x##_last = strides_##x[max_dim];

#define Cache_Indice(x, i, shape_cpy) \
    indice_##x##_cache[i] = strides_##x[i] * shape_cpy[i];

#define Adjust_Ptr(x, i, current_process) \
    x##_data_ptr_saved += current_process[i] * strides_##x[i];

#define Adjust_Ptr2(x, i) \
    x##_data_ptr_saved += strides_##x[i];

#define Adjust_Ptr3(x, i) \
    x##_data_ptr_saved -= indice_##x##_cache[i];

#define Free(x)               \
    free(indice_##x##_cache); \
    free(strides_##x);

#define Replicate0_No_Comma(method, ...) \
    Expand(MAP0_No_Comma(method, __VA_ARGS__))

#define Replicate0_With_Comma(method, ...) \
    Expand(MAP0_No_Comma(method, __VA_ARGS__))

#define Replicate0(method, ...) \
    Expand(MAP0(method, __VA_ARGS__))

#define Replicate(method, x, ...) \
    Expand(MAP(method, x, __VA_ARGS__))

#define Replicate2(method, x, y, ...) \
    Expand(MAP2(method, x, y, __VA_ARGS__))

#define Replicate3(method, x, y, ...) \
    Expand(MAP(method, x, y, __VA_ARGS__))

#define INDIRECT_REPLICATE0(func, ...) Replicate0(func, __VA_ARGS__)

#define Str(x) #x
#define Omp_Parallel(num_threads, ...) _Pragma(Str(omp parallel num_threads(num_threads) firstprivate(__VA_ARGS__)))