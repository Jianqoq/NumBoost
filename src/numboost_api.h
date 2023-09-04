#ifndef NUMBOOST_API_H
#define NUMBOOST_API_H
#include "binary_ops/binary_node_fuse.h"
#include "shape.h"
#include "type_convertor/type_convertor.h"
#include <numpy/npy_math.h>

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

/*================================== check half ===================*/
/*if is half, it will apply float_cast_half method*/
#define Is_Half(x) Concat_(Is_, x)
#define Is_npy_half Place_Holder, 1
#define Should_Cast_To(x) Second(Is_Half(x), 0)
#define Cast_Half_If_Is_Half(x) Concat_(Half_, x)
#define Cast_Float_If_Is_Half(x) Concat_(Float_, x)
#define To_Float_If_Is_Half(x) Concat_(To_Float_, x)
#define Should_Change_Type_To_Float(x) Second(Is_Half(x), 0)
#define Half_1 float_cast_half
#define Half_0
#define Float_1 half_cast_float
#define Float_0
#define To_Float_1(x) npy_float
#define To_Float_0(x) x
#define Cast_Half_When_Half(x, args)                                           \
  Cast_Half_If_Is_Half(Should_Cast_To(x))(args)
#define Cast_Float_When_Half(x, args)                                          \
  Cast_Float_If_Is_Half(Should_Cast_To(x))(args)
#define Use_Float_When_Half(x)                                                 \
  To_Float_If_Is_Half(Should_Change_Type_To_Float(x))(x)
/*================================== check half end ===================*/

/*================================== check specific method ===================*/
/*if is npy_float or npy_double, it will apply cosf or cos method*/
#define Should_Use(x) Concat_(Should_Use_, x)
#define Is_Type_npy_half Place_Holder, 1
#define Is_Type_npy_float Place_Holder, 1
#define Is_Type_npy_double Place_Holder, 1
#define Is_Type_npy_longdouble Place_Holder, 1
#define Method_npy_half(method) method##f
#define Method_npy_float(method) method##f
#define Method_npy_double(method) method
#define Method_npy_longdouble(method) method##l
#define Should_Use_1(x) Method_##x
#define Should_Use_0(x) Empty
#define Should_Cast_To_Float(x) Concat_(Should_Cast_To_Float_, x)
#define Should_Cast_To_Float_1(x) half_cast_float(x)
#define Should_Cast_To_Float_0(x) x
#define Empty(x)
#define Is_Type(x) Concat_(Is_Type_, x)
#define Should_Use_Specific_Method(x) Second(Is_Type(x), 0)
#define Use_Method(x, name, ...)                                               \
  Should_Use(Should_Use_Specific_Method(x))(x)(name)(Replicate_With_Comma(     \
      Cast_Float_If_Is_Half, Should_Cast_To, x, __VA_ARGS__))

/*================================== check specific method end
 * ===================*/

/*================================== use sepcific inf/nan ===================*/
#define Should_Use_Specific_Inf_Nan(x) Second(Is_Type(x), 0)
#define Should_Use_Inf_Nan(x) Concat_(Should_Use_Inf_Nan_, x)
#define Inf_npy_half 0x7C00
#define Inf_npy_float NPY_INFINITYF
#define Inf_npy_double NPY_INFINITY
#define Inf_npy_longdouble NPY_INFINITYL
#define Nan_npy_half 0x7E00
#define Nan_npy_float NPY_NANF
#define Nan_npy_double NPY_NAN
#define Nan_npy_longdouble NPY_NANL
#define Should_Use_Inf_1(x) Inf_##x
#define Should_Use_Inf_0(x)                                                    \
  0 /*Should not happen, numboost will always predict div result type as       \
       floating type*/
#define Should_Use_Nan_1(x) Nan_##x
#define Should_Use_Nan_0(x)                                                    \
  0 /*Should not happen, numboost will always predict div result type as       \
       floating type*/
#define Should_Use_Inf(x) Concat_(Should_Use_Inf_, x)
#define Use_Inf(x) Should_Use_Inf(Should_Use_Specific_Inf_Nan(x))(x)
#define Should_Use_Nan(x) Concat_(Should_Use_Nan_, x)
#define Use_Nan(x) Should_Use_Nan(Should_Use_Specific_Inf_Nan(x))(x)

#define Perform_Binary_Operation(a, b, result, operation, data_type, npy_enum) \
  OPERATION_PICKER(a, b, result, operation, data_type, npy_enum)

#define nb_add(x, y) ((x) + (y))
#define nb_add_half(x, y)                                                      \
  (float_cast_half(half_cast_float((x)) + half_cast_float((y))))
#define nb_subtract(x, y) ((x) - (y))
#define nb_subtract_half(x, y)                                                 \
  (float_cast_half(half_cast_float((x)) - half_cast_float((y))))
#define nb_multiply(x, y) ((x) * (y))
#define nb_multiply_half(x, y)                                                 \
  (float_cast_half(half_cast_float((x)) * half_cast_float((y))))
#define nb_divide(x, y) ((x) / (y))
#define nb_divide_half(x, y)                                                   \
  (float_cast_half(half_cast_float((x)) / half_cast_float((y))))
#define nb_power_half(x, y)                                                    \
  (float_cast_half(npy_powf(half_cast_float((x)), half_cast_float((y)))))
#define nb_power_float(x, y) (npy_powf((x), (y)))
#define nb_power_double(x, y) (npy_pow((x), (y)))
#define nb_power_long_double(x, y) (npy_powl((x), (y)))
#define nb_floor_divide_int(x, y)                                              \
  ((y) == 0 ? 0 : ((x) / (y))) // floor division for integer types, e.g. int,
                               // long, ..., ulonglong, etc.
#define nb_floor_divide_float(x, y)                                            \
  ((y) == 0 ? NAN : npy_floorf(((x) / (y)))) // floor division for float
#define nb_floor_divide_double(x, y)                                           \
  ((y) == 0 ? NAN : npy_floor((x) / (y))) // floor division for double
#define nb_floor_divide_long_double(x, y)                                      \
  ((y) == NAN ? 0 : npy_floorl(((x) / (y)))) // floor division for long double
#define nb_floor_divide_half(x, y)                                             \
  ((y) == 0 ? 0x7FFF                                                           \
            : float_cast_half(npy_floorf(                                      \
                  half_cast_float((x)) /                                       \
                  half_cast_float((y))))) // floor division for half
#define nb_remainder(x, y) ((y) == 0 ? 0 : (x) % (y))
#define nb_mod_int(x, y) ((y) == 0 ? 0 : (x) % (y))
#define nb_mod_half(x, y)                                                      \
  ((y) == 0 ? 0                                                                \
            : float_cast_half(                                                 \
                  npy_fmodf(half_cast_float((x)), half_cast_float((y)))))
#define nb_mod_float(x, y) ((y) == 0 ? 0 : npy_fmodf((x), (y)))
#define nb_mod_double(x, y) ((y) == 0 ? 0 : npy_fmod((x), (y)))
#define nb_mod_long_double(x, y) ((y) == 0 ? 0 : npy_fmodl((x), (y)))

#ifdef _MSC_VER
#define nb_lshift(x, y) (((x) << (y)) * ((y) < (sizeof(y) * 8)))
#define nb_rshift(x, y) (((x) >> (y)) * ((y) < (sizeof(y) * 8)))
#elif defined(__GNUC__)
#define nb_lshift(x, y) ((x) << (y))
#define nb_rshift(x, y) ((x) >> (y))
#ifndef min
#define min(x, y) ((x) < (y) ? (x) : (y))
#endif
#endif

#ifdef _WIN32
#include <windows.h>
#define StartTimer(label)                                                      \
  LARGE_INTEGER frequency##label;                                              \
  LARGE_INTEGER start##label, end##label;                                      \
  QueryPerformanceFrequency(&frequency##label);                                \
  QueryPerformanceCounter(&start##label);
#define EndTimer(label)                                                        \
  QueryPerformanceCounter(&end##label);                                        \
  double elapsed##label = (end##label.QuadPart - start##label.QuadPart) *      \
                          1000.0 / frequency##label.QuadPart;
#define PrintTimer(label)                                                      \
  printf("%s: Time taken [%f] ms\n", #label, elapsed##label);
#define GetTime(label) elapsed##label
#elif defined(unix) || defined(__unix__) || defined(__unix)
#include <sys/time.h>
#define StartTimer(label)                                                      \
  struct timeval start, end;                                                   \
  double elapsed##label;                                                       \
  long seconds;                                                                \
  long useconds;                                                               \
  gettimeofday(&start, NULL);
#define EndTimer(label)                                                        \
  gettimeofday(&end, NULL);                                                    \
  seconds = end.tv_sec - start.tv_sec;                                         \
  useconds = end.tv_usec - start.tv_usec;                                      \
  elapsed##label = ((seconds)*1000 + useconds / 1000.0) + 0.5;
#define PrintTimer(label)                                                      \
  printf("%s. Time taken: %lf ms\n", label, elapsed##label);
#define GetTime(label) elapsed##label
#endif

inline PyArrayObject *nb_copy(PyArrayObject *arr) {
  PyArrayObject *result = NULL;
  as_type(&arr, &result, ((PyArrayObject_fields *)arr)->descr->type_num);
  if (!result) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to copy memory for array");
    return NULL;
  }
  return result;
}

/*! @function
  @abstract     Binary operation for uncontiguous array.
    @param  type  C-type [ long, int, float, double, etc.]
    @param  a     the first array [ PyArrayObject* ]
    @param  b     the second array [ PyArrayObject* ]
    @param  result     the result array [ PyArrayObject* ]
    @param  op     Operation [ macro, e.g. nb_add, nb_subtract, etc. ]
    @param  inner_loop_body     Body of the inner loop [ macro ]
 */
#define BinaryOperation_Uncontiguous(type, a, b, result, op, inner_loop_body)  \
  do {                                                                         \
    int ndim = PyArray_NDIM(result);                                           \
    npy_intp max_dim = ndim - 1;                                               \
    npy_intp *__strides_a = PyArray_STRIDES(a);                                \
    npy_intp *strides_a =                                                      \
        (npy_intp *)malloc(sizeof(npy_intp) * PyArray_NDIM(result));           \
    npy_intp *indice_a_cache = (npy_intp *)malloc(sizeof(npy_intp) * ndim);    \
    type *a_data_ptr_saved = (type *)PyArray_DATA(a);                          \
    memcpy(strides_a, __strides_a, sizeof(npy_intp) * PyArray_NDIM(result));   \
                                                                               \
    npy_intp *__strides_b = PyArray_STRIDES(b);                                \
    npy_intp *strides_b =                                                      \
        (npy_intp *)malloc(sizeof(npy_intp) * PyArray_NDIM(result));           \
    npy_intp *indice_b_cache = (npy_intp *)malloc(sizeof(npy_intp) * ndim);    \
    type *b_data_ptr_saved = (type *)PyArray_DATA(b);                          \
    memcpy(strides_b, __strides_b, sizeof(npy_intp) * PyArray_NDIM(result));   \
                                                                               \
    for (int i = 0; i < ndim; i++) {                                           \
      strides_a[i] /= sizeof(type);                                            \
      strides_b[i] /= sizeof(type);                                            \
    }                                                                          \
    npy_intp stride_a_last = strides_a[max_dim];                               \
    npy_intp stride_b_last = strides_b[max_dim];                               \
                                                                               \
    npy_intp _size = PyArray_SIZE(result);                                     \
    npy_intp *shape_cpy =                                                      \
        (npy_intp *)malloc(sizeof(npy_intp) * PyArray_NDIM(result));           \
    npy_intp *__shape = PyArray_SHAPE(result);                                 \
    memcpy(shape_cpy, PyArray_SHAPE(result),                                   \
           sizeof(npy_intp) * PyArray_NDIM(result));                           \
    int axis_sep = ndim - 1;                                                   \
    npy_intp inner_loop_size = PyArray_SHAPE(result)[axis_sep];                \
    npy_intp outter_loop_size = _size / inner_loop_size;                       \
    npy_intp outer_start = max_dim - axis_sep;                                 \
    npy_intp *shape_copy = (npy_intp *)malloc(sizeof(npy_intp) * ndim);        \
    type *result_data_ptr_saved = (type *)PyArray_DATA(result);                \
    type *result_data_ptr_ = (type *)PyArray_DATA(result);                     \
    type *result_data_ptr_cpy = (type *)PyArray_DATA(result);                  \
    for (int i = 0; i < ndim; i++) {                                           \
      shape_cpy[i]--;                                                          \
      shape_copy[i] = 0;                                                       \
      indice_a_cache[i] = strides_a[i] * shape_cpy[i];                         \
      indice_b_cache[i] = strides_b[i] * shape_cpy[i];                         \
    }                                                                          \
    npy_intp k = 0;                                                            \
    npy_intp num_threads = outter_loop_size < omp_get_max_threads()            \
                               ? outter_loop_size                              \
                               : omp_get_max_threads();                        \
    type **result_ptr_ = (type **)malloc(sizeof(type *) * num_threads);        \
    npy_intp **current_shape_process_ =                                        \
        (npy_intp **)malloc(sizeof(npy_intp *) * num_threads);                 \
    for (npy_intp id = 0; id < num_threads; id++) {                            \
      npy_intp start_index = id * (outter_loop_size / num_threads) +           \
                             min(id, outter_loop_size % num_threads);          \
      npy_intp end_index = start_index + outter_loop_size / num_threads +      \
                           (id < outter_loop_size % num_threads);              \
      result_ptr_[id] = result_data_ptr_cpy;                                   \
      result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;      \
      npy_intp prd = result_ptr_[id] - result_data_ptr_saved;                  \
      npy_intp *current_shape_process =                                        \
          (npy_intp *)calloc(ndim, sizeof(npy_intp));                          \
      for (npy_intp j = max_dim; j >= 0; j--) {                                \
        current_shape_process[j] = prd % __shape[j];                           \
        prd /= __shape[j];                                                     \
      }                                                                        \
      current_shape_process_[id] = current_shape_process;                      \
    }                                                                          \
    _Pragma(Str(omp parallel num_threads(num_threads) firstprivate(            \
        result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved))) {              \
      int thread_id = omp_get_thread_num();                                    \
      result_data_ptr_ = result_ptr_[thread_id];                               \
      npy_intp *current_process = current_shape_process_[thread_id];           \
      for (npy_intp j = max_dim; j >= 0; j--) {                                \
        a_data_ptr_saved += current_process[j] * strides_a[j]; /*can clone*/   \
        b_data_ptr_saved += current_process[j] * strides_b[j];                 \
      }                                                                        \
      _Pragma("omp for schedule(static)") for (k = 0; k < outter_loop_size;    \
                                               k++) {                          \
        inner_loop_body(type, op, inner_loop_size, stride_a_last,              \
                        stride_b_last, a_data_ptr_saved, b_data_ptr_saved,     \
                        result_data_ptr_);                                     \
        result_data_ptr_ += inner_loop_size;                                   \
        for (npy_intp j = outer_start; j >= 0; j--) {                          \
          if (current_process[j] < __shape[j]) {                               \
            current_process[j]++;                                              \
            a_data_ptr_saved += strides_a[j];                                  \
            b_data_ptr_saved += strides_b[j];                                  \
            break;                                                             \
          } else {                                                             \
            current_process[j] = 0;                                            \
            a_data_ptr_saved -= indice_a_cache[j];                             \
            b_data_ptr_saved -= indice_b_cache[j];                             \
          }                                                                    \
        }                                                                      \
      }                                                                        \
      free(current_process);                                                   \
    }                                                                          \
    free(current_shape_process_);                                              \
    free(result_ptr_);                                                         \
    free(indice_a_cache);                                                      \
    free(indice_b_cache);                                                      \
    free(strides_a);                                                           \
    free(strides_b);                                                           \
    free(shape_cpy);                                                           \
    free(shape_copy);                                                          \
  } while (0)

#define wrapper(body, type, i, inner_loop_size, result_data_ptr_, ...)         \
  for (i = 0; i < inner_loop_size; ++i) {                                      \
    body(type, i, result_data_ptr_, __VA_ARGS__)                               \
  }

#define Universal_Operation_Sequential(type, result, inner_loop_body, ...)     \
  do {                                                                         \
    Replicate(Ptrs, type, __VA_ARGS__);                                        \
    npy_intp _size = PyArray_SIZE(result);                                     \
    type *result_data_ptr_ = (type *)PyArray_DATA(result);                     \
    npy_intp i;                                                                \
    _Pragma("omp parallel for schedule(static)")                               \
        wrapper(inner_loop_body, type, i, _size, result_data_ptr_,             \
                Replicate0(Ptr_Saved, __VA_ARGS__));                           \
  } while (0)

#define Universal_Operation(type, result, inner_loop_body, new_shapes, ...)    \
  do {                                                                         \
    int ndim = PyArray_NDIM(result);                                           \
    npy_intp max_dim = ndim - 1;                                               \
    Replicate5(Alloc_Copy_Strides_And_Indices, type, new_shapes, __VA_ARGS__); \
    for (int i = 0; i < ndim; i++) {                                           \
      Replicate(Normalize_Strides_By_Type, type, __VA_ARGS__);                 \
    }                                                                          \
    Replicate(Retrieve_Last_Stride, max_dim, __VA_ARGS__);                     \
    npy_intp _size = PyArray_SIZE(result);                                     \
    npy_intp *shape_cpy =                                                      \
        (npy_intp *)malloc(sizeof(npy_intp) * PyArray_NDIM(result));           \
    npy_intp *__shape = PyArray_SHAPE(result);                                 \
    memcpy(shape_cpy, PyArray_SHAPE(result),                                   \
           sizeof(npy_intp) * PyArray_NDIM(result));                           \
    int axis_sep = ndim - 1;                                                   \
    npy_intp inner_loop_size = PyArray_SHAPE(result)[axis_sep];                \
    npy_intp outter_loop_size = _size / inner_loop_size;                       \
    npy_intp outer_start = max_dim - axis_sep;                                 \
    npy_intp *shape_copy = (npy_intp *)malloc(sizeof(npy_intp) * ndim);        \
    type *result_data_ptr_saved = (type *)PyArray_DATA(result);                \
    type *result_data_ptr_ = (type *)PyArray_DATA(result);                     \
    type *result_data_ptr_cpy = (type *)PyArray_DATA(result);                  \
    for (int i = 0; i < ndim; i++) {                                           \
      shape_cpy[i]--;                                                          \
      shape_copy[i] = 0;                                                       \
      Replicate2(Cache_Indice, i, shape_cpy, __VA_ARGS__);                     \
    }                                                                          \
    npy_intp k = 0;                                                            \
    npy_intp num_threads = outter_loop_size < omp_get_max_threads()            \
                               ? outter_loop_size                              \
                               : omp_get_max_threads();                        \
    type **result_ptr_ = (type **)malloc(sizeof(type *) * num_threads);        \
    npy_intp **current_shape_process_ =                                        \
        (npy_intp **)malloc(sizeof(npy_intp *) * num_threads);                 \
    for (npy_intp id = 0; id < num_threads; id++) {                            \
      npy_intp start_index = id * (outter_loop_size / num_threads) +           \
                             min(id, outter_loop_size % num_threads);          \
      npy_intp end_index = start_index + outter_loop_size / num_threads +      \
                           (id < outter_loop_size % num_threads);              \
      result_ptr_[id] = result_data_ptr_cpy;                                   \
      result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;      \
      npy_intp prd = result_ptr_[id] - result_data_ptr_saved;                  \
      npy_intp *current_shape_process =                                        \
          (npy_intp *)calloc(ndim, sizeof(npy_intp));                          \
      for (npy_intp j = max_dim; j >= 0; j--) {                                \
        current_shape_process[j] = prd % __shape[j];                           \
        prd /= __shape[j];                                                     \
      }                                                                        \
      current_shape_process_[id] = current_shape_process;                      \
    }                                                                          \
    Omp_Parallel(num_threads, result_data_ptr_,                                \
                 Replicate0(Ptr_Saved, __VA_ARGS__)) {                         \
      int thread_id = omp_get_thread_num();                                    \
      result_data_ptr_ = result_ptr_[thread_id];                               \
      npy_intp *current_process = current_shape_process_[thread_id];           \
      for (npy_intp j = max_dim; j >= 0; j--) {                                \
        Replicate2(Adjust_Ptr, j, current_process, __VA_ARGS__);               \
      }                                                                        \
      npy_intp i;                                                              \
      _Pragma("omp for schedule(static)") for (k = 0; k < outter_loop_size;    \
                                               k++) {                          \
        wrapper(inner_loop_body, type, i, inner_loop_size, result_data_ptr_,   \
                Replicate0(Stries_Last, __VA_ARGS__),                          \
                Replicate0(Ptr_Saved, __VA_ARGS__));                           \
        result_data_ptr_ += inner_loop_size;                                   \
        for (npy_intp j = outer_start; j >= 0; j--) {                          \
          if (current_process[j] < __shape[j]) {                               \
            current_process[j]++;                                              \
            Replicate(Adjust_Ptr2, j, __VA_ARGS__);                            \
            break;                                                             \
          } else {                                                             \
            current_process[j] = 0;                                            \
            Replicate(Adjust_Ptr3, j, __VA_ARGS__);                            \
          }                                                                    \
        }                                                                      \
      }                                                                        \
      free(current_process);                                                   \
    }                                                                          \
    free(current_shape_process_);                                              \
    free(result_ptr_);                                                         \
    Replicate0_No_Comma(Free, __VA_ARGS__);                                    \
    free(shape_cpy);                                                           \
    free(shape_copy);                                                          \
  } while (0)

#define Perform_Universal_Operation(                                           \
    type, result_type, inner_loop_body_universal, inner_loop_body_seq, ...)    \
  do {                                                                         \
    bool shape_equal = true;                                                   \
    Replicate0_No_Comma(Handlers, __VA_ARGS__);                                \
    Replicate2(Correct_Type, result_type, type, __VA_ARGS__);                  \
    npy_intp *shapes[] = {Replicate0(Shapes, __VA_ARGS__)};                    \
    npy_intp ndims[] = {Replicate0(NDims, __VA_ARGS__)};                       \
    npy_intp *shape_ref = shapes[0];                                           \
    int ndim_ref = ndims[0];                                                   \
    for (int i = 0; i < Args_Num(__VA_ARGS__); i++) {                          \
      if (!shape_isequal(shape_ref, shapes[i], ndim_ref, ndims[i])) {          \
        shape_equal = false;                                                   \
        break;                                                                 \
      }                                                                        \
    }                                                                          \
    PyArrayObject *arr[] = {Replicate0(Arrays, __VA_ARGS__)};                  \
    if (!shape_equal || Replicate_Or(Contiguous_OrNot, __VA_ARGS__)) {         \
      PyArrayObject *biggest_array = NULL;                                     \
      npy_intp biggest_size = 0;                                               \
      for (int i = 0; i < Args_Num(__VA_ARGS__); i++) {                        \
        npy_intp size = 1;                                                     \
        for (int j = 0; j < ndims[i]; j++) {                                   \
          size *= shapes[i][j];                                                \
        }                                                                      \
        if (size > biggest_size) {                                             \
          biggest_size = size;                                                 \
          biggest_array = arr[i];                                              \
        }                                                                      \
      }                                                                        \
      npy_intp *new_shapes[Args_Num(__VA_ARGS__)] = {NULL};                    \
      for (int i = 0; i < Args_Num(__VA_ARGS__); i++) {                        \
        if (!shape_isbroadcastable_to_ex(                                      \
                shapes[i], PyArray_SHAPE(biggest_array), ndims[i],             \
                PyArray_NDIM(biggest_array), &new_shapes[i])) {                \
          PyErr_SetString(PyExc_ValueError, "Cannot broadcast shapes");        \
          return NULL;                                                         \
        }                                                                      \
      }                                                                        \
      npy_intp stride_last =                                                   \
          PyArray_STRIDE((const PyArrayObject *)biggest_array,                 \
                         PyArray_NDIM(biggest_array) - 1);                     \
      npy_intp *broadcast_shape = NULL;                                        \
      for (int j = 0; j < Args_Num(__VA_ARGS__); j++) {                        \
        if (!broadcast_shape) {                                                \
          free(broadcast_shape);                                               \
        }                                                                      \
        predict_broadcast_shape(PyArray_SHAPE(biggest_array), new_shapes[j],   \
                                PyArray_NDIM(biggest_array),                   \
                                &broadcast_shape);                             \
      }                                                                        \
      PyArrayObject *result = (PyArrayObject *)PyArray_EMPTY(                  \
          PyArray_NDIM(biggest_array), broadcast_shape, result_type, 0);       \
      Universal_Operation(type, result, inner_loop_body_universal, new_shapes, \
                          __VA_ARGS__);                                        \
      Replicate0_No_Comma(Free_Array, __VA_ARGS__);                            \
      return (PyObject *)result;                                               \
    } else {                                                                   \
      PyArrayObject *result = (PyArrayObject *)PyArray_EMPTY(                  \
          PyArray_NDIM((PyArrayObject *)_First_(__VA_ARGS__)),                 \
          PyArray_DIMS((PyArrayObject *)_First_(__VA_ARGS__)), result_type,    \
          0);                                                                  \
      Universal_Operation_Sequential(type, result, inner_loop_body_seq,        \
                                     __VA_ARGS__);                             \
      Replicate0_No_Comma(Free_Array, __VA_ARGS__);                            \
      return (PyObject *)result;                                               \
    }                                                                          \
  } while (0)

#endif // NUMBOOST_API_H