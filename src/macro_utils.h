#ifndef BINARY_NODE_FUSE_H
#define BINARY_NODE_FUSE_H
#include <assert.h>

#define Concat_(a, b) a##b
#define CONCAT_INTERNAL(a, b) a##b
#define Concat_Strong(a, b) CONCAT_INTERNAL(a, b)
#define _First_(a, ...) a
#define _First(...) _First_(__VA_ARGS__)
#define Second_(a, b, ...) b
#define Second(...) Second_(__VA_ARGS__)
#define Logic_Not(prefix, x) Second(Concat_(prefix, x), 0)
#define Is_True(x) Logic_Not(Is_, Logic_Not(Is_, x))
#define Is_0 Place_Holder, 1
#define If(x) Concat_(If_, x)
#define Has_Args(...) Is_True(_First(End_Of_Arguments_ __VA_ARGS__)())
#define If_1(...) __VA_ARGS__
#define If_0(...) EMPTY EMPTY()()
#define If2_1(...) __VA_ARGS__
#define If2_0(...) 0
#define End_Of_Arguments_() 0
#define escape(...) __VA_ARGS__
#define If2(x) Concat_(If2_, x)
#define COMMA() ,
#define Remove_Parentheses(x) Remove_Parentheses_Helper x
#define Remove_Parentheses_Helper(...) __VA_ARGS__
#define Extract(x) x
#define Init_zero Place_Holder, 1
#define Should_Init_Zeros(x) Concat_(zeros_, x)
#define Should_Init_Arr(x) Concat_(Init_arr_, x)
#define zeros_0(nd, shape, type_enum, order)                                   \
  PyArray_EMPTY(nd, shape, type_enum, order);
#define zeros_1(nd, shape, type_enum, order)                                   \
  PyArray_ZEROS(nd, shape, type_enum, order);

#define Init_arr_0(size, init_val, result_ptr)                                 \
  npy_intp init_idx;                                                           \
  _Pragma("omp parallel for") for (init_idx = 0; init_idx < size;              \
                                   init_idx++) {                               \
    result_ptr[init_idx] = init_val;                                           \
  }

#define Init_arr_1(size, init_val, result_ptr)

#define MAP0_No_Comma(m, x, ...)                                               \
  m(x) If(Has_Args(__VA_ARGS__))(DEFER2(_MAP0_No_Comma)()(m, __VA_ARGS__))

#define MAP0(m, x, ...)                                                        \
  m(x) If(Has_Args(__VA_ARGS__))(, )                                           \
      If(Has_Args(__VA_ARGS__))(DEFER2(_MAP0)()(m, __VA_ARGS__))

#define MAP1(m0, m, type, x, ...)                                              \
  m0(m(type))(x) If(Has_Args(__VA_ARGS__))(, )                                 \
      If(Has_Args(__VA_ARGS__))(DEFER2(_MAP1)()(m0, m, type, __VA_ARGS__))

#define MAP_Or(m, x, ...)                                                      \
  m(x) If(Has_Args(__VA_ARGS__))(||)                                           \
      If(Has_Args(__VA_ARGS__))(DEFER2(_MAP_Or)()(m, __VA_ARGS__))

#define MAP(m, x, y, ...)                                                      \
  m(y, x) If(Has_Args(__VA_ARGS__))(DEFER2(_MAP)()(m, x, __VA_ARGS__))

#define MAP2(m, x, y, z, ...)                                                  \
  m(z, x, y) If(Has_Args(__VA_ARGS__))(DEFER2(_MAP2)()(m, x, y, __VA_ARGS__))

#define MAP3(m, x, y, z, w, ...)                                               \
  m(w, x, y, z)                                                                \
      If(Has_Args(__VA_ARGS__))(DEFER2(_MAP3)()(m, x, y, z, __VA_ARGS__))

#define Correct_Type_MAP(m, result_type, type, x, ...)                         \
  m(x, result_type, type) If(Has_Args(__VA_ARGS__))(                           \
      DEFER2(_Correct_Type_MAP)()(m, result_type, type, __VA_ARGS__))

#define Alloc_Results_MAP(m, biggest_array, shape, result_type, index, x, ...) \
  m(x, biggest_array, shape, result_type, index)                               \
      If(Has_Args(__VA_ARGS__))(DEFER2(_Alloc_Results_MAP)()(                  \
          m, biggest_array, shape, result_type, index + 1, __VA_ARGS__))

#define Adjust_Result_Ptrs_MAP(m, start_index, end_index, inner_loop_size, x,  \
                               ...)                                            \
  m(start_index, end_index, inner_loop_size, x)                                \
      If(Has_Args(__VA_ARGS__))(DEFER2(_Adjust_Result_Ptrs_MAP)()(             \
          m, start_index, end_index, inner_loop_size, __VA_ARGS__))

#define Init_OutsArr_MAP(m, index, x, ...)                                     \
  m(index, x) If(Has_Args(__VA_ARGS__))(                                       \
      DEFER2(_Init_OutsArr_MAP)()(m, index + 1, __VA_ARGS__))

#define Results_MAP(m, x, ...)                                                 \
  m(x) If(Has_Args(__VA_ARGS__))(DEFER2(_Results_MAP)()(m, __VA_ARGS__))

#define MAP4(x, ...)                                                           \
  x##_ If(Has_Args(__VA_ARGS__))(, )                                           \
      If(Has_Args(__VA_ARGS__))(DEFER2(_MAP4)()(__VA_ARGS__))

#define MAP5(m, type, new_shapes, result, index, w, ...)                       \
  m(w, type, new_shapes, result, index) If(Has_Args(__VA_ARGS__))(             \
      DEFER2(_MAP5)()(m, type, new_shapes, result, index + 1, __VA_ARGS__))

#define MAP6(seq, index, x, ...)                                               \
  Should_Use_Stride(If_Seq(seq), index, x) If(Has_Args(__VA_ARGS__))(, )       \
      If(Has_Args(__VA_ARGS__))(DEFER2(_MAP6)()(seq, index, __VA_ARGS__))

#define NEXT(x, ...)                                                           \
  If(Has_Args(__VA_ARGS__))(+1)                                                \
      If(Has_Args(__VA_ARGS__))(DEFER2(_NEXT)()(__VA_ARGS__))

#define ACCUMULATE(m, x, ...)                                                  \
  m If(Has_Args(__VA_ARGS__))(, )                                              \
      If(Has_Args(__VA_ARGS__))(DEFER2(_ACCUMULATE)()(m + 1, __VA_ARGS__))

#define _Alloc_Results_MAP() Alloc_Results_MAP
#define _Init_OutsArr_MAP() Init_OutsArr_MAP
#define _Adjust_Result_Ptrs_MAP() Adjust_Result_Ptrs_MAP
#define _Results_MAP() Results_MAP
#define _Correct_Type_MAP() Correct_Type_MAP
#define _NEXT() NEXT
#define _ACCUMULATE() ACCUMULATE
#define _MAP0_No_Comma() MAP0_No_Comma
#define _MAP0() MAP0
#define _MAP1() MAP1
#define _MAP_Or() MAP_Or
#define _MAP() MAP
#define _MAP2() MAP2
#define _MAP3() MAP3
#define _MAP4() MAP4
#define _MAP5() MAP5
#define _MAP6() MAP6
#define Expand(...) Expand32(__VA_ARGS__)
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
#define Parameter_type(x) PyObject *x
#define Parameter_type_(x) PyObject *

#define Init_Prd(x) Init_Prd_Helper(x)
#define Init_Prd_Helper(x)                                                     \
  npy_intp prd_##x = result_##x##_ptr_arr[id] - result_##x##_ptr_saved;
#define Adjust_Prd(x) Adjust_Prd_Helper(x)
#define Adjust_Prd_Helper(x)                                                   \
  current_shape_process[j] = prd_##x % __shape[j];                             \
  prd_##x /= __shape[j];

#define Retrieve_Result_Ptrs(x)                                                \
  result_##x##_ptr = result_##x##_ptr_arr[thread_id];

#define Alloc_Results(x, biggest_array, shape, result_type, index)             \
  PyArrayObject *__result_##x = NULL;                                          \
  if (outs_arr[index] == NULL) {                                               \
    __result_##x = (PyArrayObject *)PyArray_EMPTY(PyArray_NDIM(biggest_array), \
                                                  shape, result_type, 0);      \
    if (__result_##x == NULL) {                                                \
      return NULL;                                                             \
    }                                                                          \
  } else {                                                                     \
    if (PyArray_NDIM((PyArrayObject *)outs_arr[index]) !=                      \
        PyArray_NDIM(biggest_array)) {                                         \
      PyErr_SetString(PyExc_RuntimeError, "out ndim not correct");             \
      return NULL;                                                             \
    } else {                                                                   \
      for (int i = 0; i < PyArray_NDIM((PyArrayObject *)outs_arr[index]);      \
           i++) {                                                              \
        if (PyArray_SHAPE((PyArrayObject *)outs_arr[index])[i] != shape[i]) {  \
          PyErr_SetString(PyExc_RuntimeError, "out ndim not correct");         \
          return NULL;                                                         \
        }                                                                      \
      }                                                                        \
      if (PyArray_TYPE((PyArrayObject *)outs_arr[index]) != result_type) {     \
        PyArrayObject *tmp = (PyArrayObject *)outs_arr[index];                 \
        as_type(&tmp, &__result_##x, result_type);                             \
      } else {                                                                 \
        __result_##x = (PyArrayObject *)outs_arr[index];                       \
        Py_INCREF(outs_arr[index]);                                            \
      }                                                                        \
    }                                                                          \
  }

#define Get_Results(x) __result_##x
#define Results_Ptr(x) result_##x##_ptr_saved
#define Get_Result_Ptr(x) result_##x##_ptr
#define Init_Result_Ptrs(x, type)                                              \
  type *result_##x##_ptr = (type *)PyArray_DATA(x);                            \
  type *result_##x##_ptr_saved = (type *)PyArray_DATA(x);                      \
  type *result_##x##_ptr_cpy = (type *)PyArray_DATA(x);

#define Alloc_Copy_Strides_And_Indices(x, type, new_shapes, result, index)     \
  npy_intp *__strides_##x = PyArray_STRIDES(x##_);                             \
  npy_intp *strides_##x = NULL;                                                \
  npy_intp *indice_##x##_cache = (npy_intp *)malloc(sizeof(npy_intp) * ndim);  \
  type *x##_data_ptr_saved = (type *)PyArray_DATA(x##_);                       \
  if (__strides_##x == NULL) {                                                 \
    strides_##x = (npy_intp *)calloc(PyArray_NDIM(result), sizeof(npy_intp));  \
  } else {                                                                     \
    preprocess_strides(new_shapes[index], stride_last, ndim, &strides_##x);    \
  }
#define Result_Ptr_Jump(x) result_##x##_ptr += inner_loop_size;
#define Get_Strides_Sequential(x, type)                                        \
  npy_intp *__strides_##x = PyArray_STRIDES(x##_);                             \
  npy_intp stride_##x##_last = 0;                                              \
  if (__strides_##x == NULL) {                                                 \
    stride_##x##_last = 0;                                                     \
  } else {                                                                     \
    stride_##x##_last =                                                        \
        PyArray_STRIDES(x##_)[PyArray_NDIM((PyArrayObject *)x##_) - 1] /       \
        sizeof(type);                                                          \
  }

#define Correct_Type(x, result_type, type)                                     \
  PyArrayObject *x##_ = NULL;                                                  \
  if (Py_IS_TYPE(x, Tensor_type)) {                                            \
    x = ((Tensor *)x)->data;                                                   \
  }                                                                            \
  if (Py_IS_TYPE(x, &PyArray_Type)) {                                          \
    PyArrayObject *tmp = (PyArrayObject *)x;                                   \
    if (PyArray_TYPE(tmp) != result_type) {                                    \
      as_type(&tmp, &x##_, result_type);                                       \
      handler_##x = x##_;                                                      \
    } else {                                                                   \
      x##_ = (PyArrayObject *)x;                                               \
    }                                                                          \
  } else if (PyArray_IsAnyScalar(x)) {                                         \
    if (PyArray_IsPythonNumber(x)) {                                           \
      npy_intp const dims[] = {1};                                             \
      x##_ = (PyArrayObject *)PyArray_EMPTY(0, dims, result_type, 0);          \
      if (PyFloat_Check(x)) {                                                  \
        *((type *)PyArray_DATA(x##_)) =                                        \
            (type)PyDouble_AsHalf(type, PyFloat_AsDouble((PyObject *)x));      \
      } else if (PyLong_Check(x)) {                                            \
        *((type *)PyArray_DATA(x##_)) =                                        \
            (type)PyLong_AsHalf(type, PyLong_AsLong((PyObject *)x));           \
      } else if (PyBool_Check(x)) {                                            \
        *((type *)PyArray_DATA(x##_)) =                                        \
            (type)PyBool_AsHalf(type, Py_IsTrue((PyObject *)x));               \
      } else {                                                                 \
        PyErr_SetString(PyExc_TypeError, "Scalar type not supported");         \
        return NULL;                                                           \
      }                                                                        \
      handler_##x = x##_;                                                      \
    } else if (Py_IS_TYPE(x, &PyArray_Type)) {                                 \
      PyArrayObject *tmp = (PyArrayObject *)x;                                 \
      as_type(&tmp, &x##_, result_type);                                       \
      handler_##x = x##_;                                                      \
    } else {                                                                   \
      PyErr_SetString(PyExc_TypeError, Str(type is not supported));            \
      return NULL;                                                             \
    }                                                                          \
    assert(PyArray_STRIDES(x##_) == NULL); /*should be NULL*/                  \
  } else {                                                                     \
    PyErr_SetString(PyExc_TypeError, "type not supported");                    \
    return NULL;                                                               \
  }

#define Ptrs(x, type) type *x##_data_ptr_saved = (type *)PyArray_DATA(x##_);
#define Result_Ptrs(x, type)                                                   \
  type *result_##x##_ptr_saved = (type *)PyArray_DATA(x);

#define Handlers(x) PyArrayObject *handler_##x = NULL;

#define Free_Array(x)                                                          \
  if (handler_##x)                                                             \
    Py_DECREF(handler_##x);

#define Normalize_Strides_By_Type(x, type) strides_##x[i] /= sizeof(type);

#define Retrieve_Last_Stride(x, max_dim)                                       \
  npy_intp stride_##x##_last = strides_##x[max_dim];

#define Cache_Indice(x, i, shape_cpy)                                          \
  indice_##x##_cache[i] = strides_##x[i] * shape_cpy[i];

#define Adjust_Ptr(x, i, current_process)                                      \
  x##_data_ptr_saved += current_process[i] * strides_##x[i];

#define Adjust_Ptr2(x, i) x##_data_ptr_saved += strides_##x[i];

#define Adjust_Ptr3(x, i) x##_data_ptr_saved -= indice_##x##_cache[i];

#define Adjust_Result_Ptrs(start_index, end_index, inner_loop_size, x)         \
  result_##x##_ptr_arr[id] = result_##x##_ptr_cpy;                             \
  result_##x##_ptr_cpy += (end_index - start_index) * inner_loop_size;

#define Init_Result_Ptrs_Arr(x, num_threads, type)                             \
  type **result_##x##_ptr_arr = (type **)malloc(sizeof(type *) * num_threads);

#define Contiguous_OrNot(x) !PyArray_ISCONTIGUOUS((PyArrayObject *)x##_)

#define Shapes(x) PyArray_SHAPE(x##_)

#define NDims(x) PyArray_NDIM(x##_)

#define Arrays(x) x##_

#define Free(x)                                                                \
  free(indice_##x##_cache);                                                    \
  free(strides_##x);

#define Free_Result_Ptr_Arr(x) free(result_##x##_ptr_arr);

#define Args_Num(...) If2(Has_Args(__VA_ARGS__))(1) Expand(NEXT(__VA_ARGS__))

#define Args_Index(...) Expand(ACCUMULATE(0, __VA_ARGS__))

#define Replicate0_No_Comma(method, ...)                                       \
  Expand(MAP0_No_Comma(method, __VA_ARGS__))

#define Replicate0_With_Comma(method, ...) Expand(MAP0(method, __VA_ARGS__))

#define Replicate_With_Comma(method0, method, x, ...)                          \
  Expand(MAP1(method0, method, x, __VA_ARGS__))

#define Replicate0(method, ...) Expand(MAP0(method, __VA_ARGS__))

#define Replicate_Or(method, ...) Expand(MAP_Or(method, __VA_ARGS__))

#define Replicate(method, x, ...) Expand(MAP(method, x, __VA_ARGS__))

#define Replicate2(method, x, y, ...) Expand(MAP2(method, x, y, __VA_ARGS__))

#define Replicate_Correct_Type(method, result_type, type, ...)                 \
  Expand(Correct_Type_MAP(method, result_type, type, __VA_ARGS__))

#define Replicate_Alloc_Results(biggest_array, shape, result_type, ...)        \
  Expand(Alloc_Results_MAP(Alloc_Results, biggest_array, shape, result_type,   \
                           0, __VA_ARGS__))

#define Replicate_Adjust_Result_Ptrs(method, start_index, end_index,           \
                                     inner_loop_size, ...)                     \
  Expand(Adjust_Result_Ptrs_MAP(method, start_index, end_index,                \
                                inner_loop_size, __VA_ARGS__))

#define Replicate_Init_OutsArr(method, ...)                                    \
  Expand(Init_OutsArr_MAP(method, 0, __VA_ARGS__))

#define Replicate3(method, x, y, z, ...)                                       \
  Expand(MAP3(method, x, y, z, __VA_ARGS__))

#define Replicate4(...) Expand(MAP4(__VA_ARGS__))

#define Replicate5(method, type, new_shapes, result, ...)                      \
  Expand(MAP5(method, type, new_shapes, result, 0, __VA_ARGS__))

#define INDIRECT_REPLICATE0(func, ...) Replicate0(func, __VA_ARGS__)

#define Str(x) #x
#define Omp_Parallel(num_threads, ...)                                         \
  _Pragma(Str(omp parallel num_threads(num_threads) firstprivate(__VA_ARGS__)))

#endif