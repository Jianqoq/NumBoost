#ifndef NUMBOOST_API_H
#define NUMBOOST_API_H
#include "macro_utils.h"
#include "omp.h"
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
/*================= Assertion =================*/
#define Numboost_AssertNULL(obj)                                               \
  if ((obj) == NULL) {                                                         \
    return NULL;                                                               \
  }

#define Numboost_AssertNULL_Backward(obj, out)                                 \
  if ((obj) == NULL) {                                                         \
    *out = NULL;                                                               \
    return;                                                                    \
  }

#define Numboost_AssertRequireGrad(/*Tensor */ tensor,                         \
                                   /*const char* */ error_message)             \
  if ((tensor)->require_grad) {                                                \
    PyErr_SetString(PyExc_RuntimeError, error_message);                        \
    return NULL;                                                               \
  }
/*================================== check half ===================*/
/*if is half, it will apply float_cast_half method*/
#define Is_Half(x) Concat_(Is_, x)
#define Is_npy_half Place_Holder, 1
#define Should_Cast_To(x) Second(Is_Half(x), 0)
#define Cast_Half_If_Is_Half(x) Concat_(Half_, x)
#define Cast_Float_If_Is_Half(x) Concat_(Float_, x)
#define To_Float_If_Is_Half(x) Concat_(To_Float_, x)
#define Cast_Half_If_Is_Double(x) Concat_(Double_, x)
#define Cast_Half_If_Is_Long(x) Concat_(Long_, x)
#define Cast_Half_If_Is_Bool(x) Concat_(Bool_, x)
#define Should_Change_Type_To_Float(x) Second(Is_Half(x), 0)
#define Half_1 float_cast_half
#define Half_0
#define Float_1 half_cast_float
#define Float_0
#define Double_1 double_cast_half
#define Double_0
#define Long_1 long_cast_half
#define Long_0
#define Bool_1 bool_cast_half
#define Bool_0
#define To_Float_1(x) npy_float
#define To_Float_0(x) x
#define To_Double_1(x) npy_double
#define To_Double_0(x) x
#define Demote(x, args) Cast_Half_If_Is_Half(Should_Cast_To(x))(args)
#define Promote(x, args) Cast_Float_If_Is_Half(Should_Cast_To(x))(args)
#define Generic(x) To_Float_If_Is_Half(Should_Change_Type_To_Float(x))(x)
#define PyDouble_AsHalf(x, args)                                               \
  Cast_Half_If_Is_Double(Should_Change_Type_To_Float(x))(args)
#define PyLong_AsHalf(x, args)                                                 \
  Cast_Half_If_Is_Long(Should_Change_Type_To_Float(x))(args)
#define PyBool_AsHalf(x, args)                                                 \
  Cast_Half_If_Is_Bool(Should_Change_Type_To_Float(x))(args)
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
#define Method_int(method) method##i
#define Should_Use_1(x) Method_##x
#define Should_Use_0(x) Method_int
#define Should_Cast_To_Float(x) Concat_(Should_Cast_To_Float_, x)
#define Should_Cast_To_Float_1(x) half_cast_float(x)
#define Should_Cast_To_Float_0(x) x
#define Is_Type(x) Concat_(Is_Type_, x)
#define Should_Use_Specific_Method(x) Second(Is_Type(x), 0)

/*
@params:
    type: the type of the input parameters
    method_name: the name of the double method, npy_pow, npy_sin, etc.
    ...: the parameters of the method
*/
#define Map_Method(type, method_name, ...)                                     \
  Map_Method_Helper(type, method_name,                                         \
                    Replicate_With_Comma(Cast_Float_If_Is_Half,                \
                                         Should_Cast_To, type, __VA_ARGS__))

#define Map_Method_Helper(type, method_name, ...)                              \
  Should_Use(Should_Use_Specific_Method(type))(type)(method_name)(__VA_ARGS__)
/*================= check specific method end ===================*/

/*================ use sepcific inf/nan ===================*/
#define Should_Use_Specific_Inf_Nan(x) Second(Is_Type(x), 0)
#define Inf_npy_half 0x7C00
#define Inf_npy_float NPY_INFINITYF
#define Inf_npy_double NPY_INFINITY
#define Inf_npy_longdouble NPY_INFINITYL
#define Nan_npy_half 0x7E00
#define Nan_npy_float NPY_NANF
#define Nan_npy_double NPY_NAN
#define Nan_npy_longdouble NPY_NANL
#define Should_Use_Inf_1(x) Inf_##x
#define Should_Use_Inf_0(x) 0
/*Should not happen, numboost will always predict div result type as
     floating type*/
#define Should_Use_Nan_1(x) Nan_##x
#define Should_Use_Nan_0(x) 0
/*Should not happen, numboost will always predict div result type as
       floating type*/
#define Should_Use_Inf(x) Concat_(Should_Use_Inf_, x)
#define Use_Inf(x) Should_Use_Inf(Should_Use_Specific_Inf_Nan(x))(x)
#define Should_Use_Nan(x) Concat_(Should_Use_Nan_, x)
#define Use_Nan(x) Should_Use_Nan(Should_Use_Specific_Inf_Nan(x))(x)

/*================= check is sequential method ===================*/
#define Cat_Seq(x) Concat_(Is_Sequential_, x)
#define If_Seq(x) Second(Cat_Seq(x), 1)
#define Is_Sequential_Seq Place_Holder, 0
#define If_Sequential_1(x, stride) x *stride
#define If_Sequential_0(x, stride) x
#define Should_Use_Stride(x, index, stride)                                    \
  Concat_(If_Sequential_, x)(index, stride)
#define Use_Stride(x, index, stride) Should_Use_Stride(If_Seq(x), index, stride)

#define Replicate6(x, y, ...) Expand(MAP6(x, y, __VA_ARGS__))

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
#define nb_remainder(x, y) ((y) == 0 ? 0 : (x) % (y))
#define nb_mod_int(x, y) ((y) == 0 ? 0 : (x) % (y))
#define nb_mod_half(x, y)                                                      \
  ((y) == 0 ? 0                                                                \
            : float_cast_half(                                                 \
                  npy_fmodf(half_cast_float((x)), half_cast_float((y)))))
#define nb_mod_float(x, y) ((y) == 0 ? 0 : npy_fmodf((x), (y)))
#define nb_mod_double(x, y) ((y) == 0 ? 0 : npy_fmod((x), (y)))
#define nb_mod_long_double(x, y) ((y) == 0 ? 0 : npy_fmodl((x), (y)))

#define Div(val1, val2, result, input_type, output_type)                       \
  do {                                                                         \
    if (!(val2)) {                                                             \
      if (val1 > 0)                                                            \
        result = Use_Inf(output_type);                                         \
      else if (val1 < 0)                                                       \
        result = -Use_Inf(output_type);                                        \
      else                                                                     \
        result = Use_Nan(output_type);                                         \
      continue;                                                                \
    } else                                                                     \
      result = Demote(output_type,                                             \
                      Demote(input_type, val1) / Demote(input_type, val2));    \
  } while (0)

#define Div2(val1, val2, result, input_type, output_type)                      \
  do {                                                                         \
    if (!(val2)) {                                                             \
      if (val1 > 0)                                                            \
        result = Use_Inf(output_type);                                         \
      else if (val1 < 0)                                                       \
        result = -Use_Inf(output_type);                                        \
      else                                                                     \
        result = Use_Nan(output_type);                                         \
    } else                                                                     \
      result = Demote(output_type,                                             \
                      Demote(input_type, val1) / Demote(input_type, val2));    \
  } while (0)

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
  printf("%s. Time taken: %lf ms\n", #label, elapsed##label);
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

#define wrapper(body, generic_type, type, i, inner_loop_size,                  \
                result_data_ptr_, ...)                                         \
  for (i = 0; i < inner_loop_size; i++) {                                      \
    body(generic_type, type, i, result_data_ptr_, __VA_ARGS__)                 \
  }

#define Universal_Operation_Sequential(type, results, first_result,            \
                                       inner_loop_body, ...)                   \
  do {                                                                         \
    Replicate(Ptrs, type, __VA_ARGS__);                                        \
    npy_intp _size = PyArray_SIZE(first_result);                               \
    Replicate(Result_Ptrs, type,                                               \
              Replicate0_With_Comma(Extract, Remove_Parentheses(results)));    \
    npy_intp i;                                                                \
    _Pragma("omp parallel for schedule(static)") wrapper(                      \
        inner_loop_body, Generic(type), type, i, _size,                        \
        Replicate0_With_Comma(Results_Ptr, Remove_Parentheses(results)),       \
        Replicate0(Stries_Last, __VA_ARGS__),                                  \
        Replicate0(Ptr_Saved, __VA_ARGS__));                                   \
  } while (0)

#define Universal_Operation(type, results, first_result, inner_loop_body,      \
                            new_shapes, ...)                                   \
  do {                                                                         \
    int ndim = PyArray_NDIM(first_result);                                     \
    npy_intp max_dim = ndim - 1;                                               \
    Replicate5(Alloc_Copy_Strides_And_Indices, type, new_shapes, first_result, \
               __VA_ARGS__);                                                   \
    for (int i = 0; i < ndim; i++) {                                           \
      Replicate(Normalize_Strides_By_Type, type, __VA_ARGS__);                 \
    }                                                                          \
    Replicate(Retrieve_Last_Stride, max_dim, __VA_ARGS__);                     \
    npy_intp _size = PyArray_SIZE(first_result);                               \
    npy_intp *shape_cpy =                                                      \
        (npy_intp *)malloc(sizeof(npy_intp) * PyArray_NDIM(first_result));     \
    npy_intp *__shape = PyArray_SHAPE(first_result);                           \
    memcpy(shape_cpy, PyArray_SHAPE(first_result),                             \
           sizeof(npy_intp) * PyArray_NDIM(first_result));                     \
    int axis_sep = ndim - 1; /*optimize*/                                      \
    npy_intp inner_loop_size = PyArray_SHAPE(first_result)[axis_sep];          \
    npy_intp outter_loop_size = _size / inner_loop_size;                       \
    npy_intp outer_start = max_dim - 1; /*optimize*/                           \
    npy_intp *shape_copy = (npy_intp *)malloc(sizeof(npy_intp) * ndim);        \
    Replicate(Init_Result_Ptrs, type, Remove_Parentheses(results));            \
    for (int i = 0; i < ndim; i++) {                                           \
      shape_cpy[i]--;                                                          \
      shape_copy[i] = 0;                                                       \
      Replicate2(Cache_Indice, i, shape_cpy, __VA_ARGS__);                     \
    }                                                                          \
    npy_intp num_threads = outter_loop_size < omp_get_max_threads()            \
                               ? outter_loop_size                              \
                               : omp_get_max_threads();                        \
    Replicate2(Init_Result_Ptrs_Arr, num_threads, type,                        \
               Remove_Parentheses(results));                                   \
    npy_intp **current_shape_process_ =                                        \
        (npy_intp **)malloc(sizeof(npy_intp *) * num_threads);                 \
    for (npy_intp id = 0; id < num_threads; id++) {                            \
      npy_intp start_index = id * (outter_loop_size / num_threads) +           \
                             min(id, outter_loop_size % num_threads);          \
      npy_intp end_index = start_index + outter_loop_size / num_threads +      \
                           (id < outter_loop_size % num_threads);              \
      Replicate_Adjust_Result_Ptrs(Adjust_Result_Ptrs, start_index, end_index, \
                                   inner_loop_size,                            \
                                   Remove_Parentheses(results));               \
      Init_Prd(_First(                                                         \
          Replicate0_With_Comma(Extract, Remove_Parentheses(results))));       \
      npy_intp *current_shape_process =                                        \
          (npy_intp *)calloc(ndim, sizeof(npy_intp));                          \
      for (npy_intp j = max_dim; j >= 0; j--) {                                \
        Adjust_Prd(_First(                                                     \
            Replicate0_With_Comma(Extract, Remove_Parentheses(results))))      \
      }                                                                        \
      current_shape_process_[id] = current_shape_process;                      \
    }                                                                          \
    npy_intp k = 0;                                                            \
    Py_BEGIN_ALLOW_THREADS;                                                    \
    Omp_Parallel(num_threads,                                                  \
                 Replicate0(Get_Result_Ptr, Remove_Parentheses(results)),      \
                 Replicate0(Ptr_Saved, __VA_ARGS__)) {                         \
      int thread_id = omp_get_thread_num();                                    \
      Replicate0_No_Comma(Retrieve_Result_Ptrs, Remove_Parentheses(results));  \
      npy_intp *current_process = current_shape_process_[thread_id];           \
      for (npy_intp j = max_dim; j >= 0; j--) {                                \
        Replicate2(Adjust_Ptr, j, current_process, __VA_ARGS__);               \
      }                                                                        \
      _Pragma("omp for schedule(static)") for (k = 0; k < outter_loop_size;    \
                                               k++) {                          \
        npy_intp i;                                                            \
        wrapper(inner_loop_body, Generic(type), type, i, inner_loop_size,      \
                Replicate0(Get_Result_Ptr, Remove_Parentheses(results)),       \
                Replicate0(Stries_Last, __VA_ARGS__),                          \
                Replicate0(Ptr_Saved, __VA_ARGS__));                           \
        Replicate0_No_Comma(Result_Ptr_Jump, Remove_Parentheses(results));     \
        for (npy_intp j = outer_start; j >= 0; j--) {                          \
          if (current_process[j] < shape_cpy[j]) {                             \
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
    Py_END_ALLOW_THREADS;                                                      \
    free(current_shape_process_);                                              \
    Replicate0_No_Comma(Free_Result_Ptr_Arr, Remove_Parentheses(results));     \
    Replicate0_No_Comma(Free, __VA_ARGS__);                                    \
    free(shape_cpy);                                                           \
    free(shape_copy);                                                          \
  } while (0)

#define Perform_Universal_Operation(type, return_arr, result_type,             \
                                    inner_loop_body, outs_array,               \
                                    outs_array_len, results, ...)              \
  do {                                                                         \
    if (outs_array_len > Args_Num(Remove_Parentheses(results))) {              \
      PyErr_SetString(PyExc_ValueError, "Number of outputs must be equal or "  \
                                        "less than the number of results");    \
      return NULL;                                                             \
    }                                                                          \
    bool shape_equal = true;                                                   \
    bool all_scalar = true;                                                    \
    DEBUG_PRINT("%s: Entered\n", __func__);                                    \
    Replicate0_No_Comma(Handlers, __VA_ARGS__);                                \
    DEBUG_PRINT("%s: Handlers done\n", __func__);                              \
    Replicate_Correct_Type(Correct_Type, result_type, type, __VA_ARGS__);      \
    DEBUG_PRINT("%s: Correct_Type done\n", __func__);                          \
    npy_intp *shapes[] = {Replicate0(Shapes, __VA_ARGS__)};                    \
    int ndims[] = {Replicate0(NDims, __VA_ARGS__)};                            \
    npy_intp *shape_ref = shapes[0];                                           \
    int ndim_ref = ndims[0];                                                   \
    for (int i = 0; i < Args_Num(__VA_ARGS__); i++) {                          \
      if (!shape_isequal(shape_ref, shapes[i], ndim_ref, ndims[i])) {          \
        shape_equal = false;                                                   \
        break;                                                                 \
      }                                                                        \
    }                                                                          \
    DEBUG_PRINT("%s: shape_equal done\n", __func__);                           \
    PyArrayObject *arr[] = {Replicate0(Arrays, __VA_ARGS__)};                  \
    npy_intp sizes[Args_Num(__VA_ARGS__)] = {0};                               \
    int biggest_index = 0;                                                     \
    PyArrayObject *biggest_array = NULL;                                       \
    npy_intp biggest_size = 0;                                                 \
    for (int i = 0; i < (Args_Num(__VA_ARGS__)); i++) {                        \
      npy_intp size = 1;                                                       \
      for (int j = 0; j < ndims[i]; j++) {                                     \
        size *= shapes[i][j];                                                  \
      }                                                                        \
      sizes[i] = size;                                                         \
      if (size > biggest_size) {                                               \
        biggest_size = size;                                                   \
        biggest_array = arr[i];                                                \
        biggest_index = i;                                                     \
      }                                                                        \
    }                                                                          \
    DEBUG_PRINT("%s: biggest_size done\n", __func__);                          \
    for (int j = 0; j < (Args_Num(__VA_ARGS__)); j++) {                        \
      if (j != biggest_index && sizes[j] != 1) {                               \
        all_scalar = false;                                                    \
        break;                                                                 \
      }                                                                        \
    }                                                                          \
    PyObject *outs_arr[Args_Num(Remove_Parentheses(results))] = {NULL};        \
    if (outs_array != NULL) {                                                  \
      memcpy(outs_arr, outs_array, sizeof(PyArrayObject *) * outs_array_len);  \
    }                                                                          \
    DEBUG_PRINT("%s: all_scalar done\n", __func__);                            \
    if ((!shape_equal && !all_scalar) ||                                       \
        (Replicate_Or(Contiguous_OrNot, __VA_ARGS__))) {                       \
      DEBUG_PRINT("%s: broadcast start\n", __func__);                          \
      npy_intp *new_shapes[Args_Num(__VA_ARGS__)] = {NULL};                    \
      for (int i = 0; i < (Args_Num(__VA_ARGS__)); i++) {                      \
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
      npy_intp broadcast_size = 1;                                             \
      for (int j = 0; j < (Args_Num(__VA_ARGS__)); j++) {                      \
        npy_intp *broadcast_shape_tmp = NULL;                                  \
        predict_broadcast_shape(PyArray_SHAPE(biggest_array), new_shapes[j],   \
                                PyArray_NDIM(biggest_array),                   \
                                &broadcast_shape_tmp);                         \
        npy_intp tmp = 1;                                                      \
        for (int k = 0; k < PyArray_NDIM(biggest_array); k++) {                \
          tmp *= broadcast_shape_tmp[k];                                       \
        }                                                                      \
        if (tmp < broadcast_size) {                                            \
          free(broadcast_shape_tmp);                                           \
        } else {                                                               \
          broadcast_shape = broadcast_shape_tmp;                               \
          broadcast_size = tmp;                                                \
        }                                                                      \
      }                                                                        \
      Replicate_Alloc_Results(biggest_array, broadcast_shape, result_type,     \
                              Remove_Parentheses(results));                    \
      DEBUG_PRINT("%s: broacast looping start\n", __func__);                   \
      Universal_Operation(                                                     \
          type,                                                                \
          (Replicate0_With_Comma(Get_Results, Remove_Parentheses(results))),   \
          _First(Replicate0_With_Comma(Get_Results,                            \
                                       Remove_Parentheses(results))),          \
          inner_loop_body, new_shapes, __VA_ARGS__);                           \
      DEBUG_PRINT("%s: broacast looping end\n", __func__);                     \
      PyArrayObject *results_arr[] = {                                         \
          Replicate0_With_Comma(Get_Results, Remove_Parentheses(results))};    \
      Replicate0_No_Comma(Free_Array, __VA_ARGS__);                            \
      memcpy(return_arr, results_arr,                                          \
             sizeof(PyArrayObject *) *                                         \
                 (Args_Num(Remove_Parentheses(results))));                     \
      DEBUG_PRINT("%s: Ended\n", __func__);                                    \
    } else {                                                                   \
      Replicate(Get_Strides_Sequential, type, __VA_ARGS__);                    \
      Replicate_Alloc_Results(biggest_array, PyArray_SHAPE(biggest_array),     \
                              result_type, Remove_Parentheses(results));       \
      Universal_Operation_Sequential(                                          \
          type,                                                                \
          (Replicate0_With_Comma(Get_Results, Remove_Parentheses(results))),   \
          _First(Replicate0_With_Comma(Get_Results,                            \
                                       Remove_Parentheses(results))),          \
          inner_loop_body, __VA_ARGS__);                                       \
      Replicate0_No_Comma(Free_Array, __VA_ARGS__);                            \
      PyArrayObject *results_arr[] = {                                         \
          Replicate0_With_Comma(Get_Results, Remove_Parentheses(results))};    \
      memcpy(return_arr, results_arr,                                          \
             sizeof(PyArrayObject *) *                                         \
                 (Args_Num(Remove_Parentheses(results))));                     \
    }                                                                          \
  } while (0)

#endif // NUMBOOST_API_H