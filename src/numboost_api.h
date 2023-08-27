#ifndef NUMBOOST_API_H
#define NUMBOOST_API_H
#include <numpy/npy_math.h>
#include "type_convertor.h"
#include "binary_func.h"
#include "broadcast.h"
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

// PyArrayObject* a, PyArrayObject* b, PyArrayObject* result, int operation, ctype data_type, npy_dtype_enum npy_enum
//
//"result" need to allocate memory before calling this function
//"operation" is enum defined in op.h file
//"data_type" is the C type of the data, e.g. int, float, double, etc.
//"npy_enum" is the numpy enum defined in numpy/ndarraytypes.h
#define Perform_Binary_Operation(a, b, result, operation, data_type, npy_enum) OPERATION_PICKER(a, b, result, operation, data_type, npy_enum)

#define nb_add(x, y) ((x) + (y))
#define nb_add_half(x, y) (float_cast_half(half_cast_float((x)) + half_cast_float((y))))
#define nb_subtract(x, y) ((x) - (y))
#define nb_subtract_half(x, y) (float_cast_half(half_cast_float((x)) - half_cast_float((y))))
#define nb_multiply(x, y) ((x) * (y))
#define nb_multiply_half(x, y) (float_cast_half(half_cast_float((x)) * half_cast_float((y))))
#define nb_divide(x, y) ((x) / (y))
#define nb_divide_half(x, y) (float_cast_half(half_cast_float((x)) / half_cast_float((y))))
#define nb_power_half(x, y) (float_cast_half(npy_powf(half_cast_float((x)), half_cast_float((y)))))
#define nb_power_float(x, y) (npy_powf((x), (y)))
#define nb_power_double(x, y) (npy_pow((x), (y)))
#define nb_power_long_double(x, y) (npy_powl((x), (y)))
#define nb_floor_divide_int(x, y) ((y) == 0 ? 0 : ((x) / (y)))                                                                    // floor division for integer types, e.g. int, long, ..., ulonglong, etc.
#define nb_floor_divide_float(x, y) ((y) == 0 ? NAN : npy_floorf(((x) / (y))))                                                    // floor division for float
#define nb_floor_divide_double(x, y) ((y) == 0 ? NAN : npy_floor((x) / (y)))                                                      // floor division for double
#define nb_floor_divide_long_double(x, y) ((y) == NAN ? 0 : npy_floorl(((x) / (y))))                                              // floor division for long double
#define nb_floor_divide_half(x, y) ((y) == 0 ? 0x7FFF : float_cast_half(npy_floorf(half_cast_float((x)) / half_cast_float((y))))) // floor division for half
#define nb_remainder(x, y) ((y) == 0 ? 0 : (x) % (y))
#define nb_mod_int(x, y) ((y) == 0 ? 0 : (x) % (y))
#define nb_mod_half(x, y) ((y) == 0 ? 0 : float_cast_half(npy_fmodf(half_cast_float((x)), half_cast_float((y)))))
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
#define StartTimer(label)                         \
    LARGE_INTEGER frequency##label;               \
    LARGE_INTEGER start##label, end##label;       \
    QueryPerformanceFrequency(&frequency##label); \
    QueryPerformanceCounter(&start##label);
#define StopTimer(label)                  \
    QueryPerformanceCounter(&end##label); \
    double elapsed##label = (end##label.QuadPart - start##label.QuadPart) * 1000.0 / frequency##label.QuadPart;
#define PrintTimer(label) printf("%s: Time taken [%f] ms\n", #label, elapsed##label);
#define GetElapsed(label) elapsed##label
#elif defined(unix) || defined(__unix__) || defined(__unix)
#include <sys/time.h>
#define StartTimer(label)      \
    struct timeval start, end; \
    double elapsed##label;     \
    long seconds;              \
    long useconds;             \
    gettimeofday(&start, NULL);
#define StopTimer(label)                    \
    gettimeofday(&end, NULL);               \
    seconds = end.tv_sec - start.tv_sec;    \
    useconds = end.tv_usec - start.tv_usec; \
    elapsed##label = ((seconds)*1000 + useconds / 1000.0) + 0.5;
#define PrintTimer(label) printf("%s. Time taken: %lf ms\n", label, elapsed##label);
#define GetElapsed(label) elapsed##label
#endif

inline PyArrayObject *nb_copy(PyArrayObject *arr)
{
    PyArrayObject *result = NULL;
    as_type(&arr, &result, ((PyArrayObject_fields *)arr)->descr->type_num);
    if (!result)
    {
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
#define BinaryOperation_Uncontiguous(type, a, b, result, op, inner_loop_body)                                               \
    do                                                                                                                      \
    {                                                                                                                       \
        npy_intp *__strides_a = PyArray_STRIDES(a);                                                                         \
        npy_intp *__strides_b = PyArray_STRIDES(b);                                                                         \
        npy_intp _size = PyArray_SIZE(a);                                                                                   \
        npy_intp *strides_a = (npy_intp *)malloc(sizeof(npy_intp) * PyArray_NDIM(result));                                  \
        npy_intp *strides_b = (npy_intp *)malloc(sizeof(npy_intp) * PyArray_NDIM(result));                                  \
        memcpy(strides_a, __strides_a, sizeof(npy_intp) * PyArray_NDIM(result));                                            \
        memcpy(strides_b, __strides_b, sizeof(npy_intp) * PyArray_NDIM(result));                                            \
        npy_intp *shape_cpy = (npy_intp *)malloc(sizeof(npy_intp) * PyArray_NDIM(result));                                  \
        npy_intp *__shape = PyArray_SHAPE(result);                                                                          \
        memcpy(shape_cpy, PyArray_SHAPE(result), sizeof(npy_intp) * PyArray_NDIM(result));                                  \
        int ndim = PyArray_NDIM(result);                                                                                    \
        int axis_sep = ndim - 1;                                                                                            \
        npy_intp inner_loop_size = PyArray_SHAPE(result)[axis_sep];                                                         \
        npy_intp outter_loop_size = _size / inner_loop_size;                                                                \
        npy_intp max_dim = ndim - 1;                                                                                        \
        npy_intp outer_start = max_dim - axis_sep;                                                                          \
        npy_intp *shape_copy = (npy_intp *)malloc(sizeof(npy_intp) * ndim);                                                 \
        npy_intp *indice_a_cache = (npy_intp *)malloc(sizeof(npy_intp) * ndim);                                             \
        npy_intp *indice_b_cache = (npy_intp *)malloc(sizeof(npy_intp) * ndim);                                             \
        type *b_data_ptr_saved = (type *)PyArray_DATA(b);                                                                   \
        type *a_data_ptr_saved = (type *)PyArray_DATA(a);                                                                   \
        type *result_data_ptr_saved = (type *)PyArray_DATA(result);                                                         \
        type *result_data_ptr_ = (type *)PyArray_DATA(result);                                                              \
        type *result_data_ptr_cpy = (type *)PyArray_DATA(result);                                                           \
        for (int i = 0; i < ndim; i++)                                                                                      \
        {                                                                                                                   \
            strides_a[i] /= sizeof(type);                                                                                   \
            strides_b[i] /= sizeof(type);                                                                                   \
        }                                                                                                                   \
        for (int i = 0; i < ndim; i++)                                                                                      \
        {                                                                                                                   \
            shape_cpy[i]--;                                                                                                 \
            shape_copy[i] = 0;                                                                                              \
            indice_a_cache[i] = strides_a[i] * shape_cpy[i];                                                                \
            indice_b_cache[i] = strides_b[i] * shape_cpy[i];                                                                \
        }                                                                                                                   \
        npy_intp k = 0;                                                                                                     \
        npy_intp num_threads = outter_loop_size < omp_get_max_threads() ? outter_loop_size : omp_get_max_threads();         \
        type **result_ptr_ = (type **)malloc(sizeof(type *) * num_threads);                                                 \
        npy_intp **current_shape_process_ = (npy_intp **)malloc(sizeof(npy_intp *) * num_threads);                          \
        for (npy_intp id = 0; id < num_threads; id++)                                                                       \
        {                                                                                                                   \
            npy_intp start_index = id * (outter_loop_size / num_threads) + min(id, outter_loop_size % num_threads);         \
            npy_intp end_index = start_index + outter_loop_size / num_threads + (id < outter_loop_size % num_threads);      \
            result_ptr_[id] = result_data_ptr_cpy;                                                                          \
            result_data_ptr_cpy += (end_index - start_index) * inner_loop_size;                                             \
            npy_intp prd = result_ptr_[id] - result_data_ptr_saved;                                                         \
            npy_intp *current_shape_process = (npy_intp *)calloc(ndim, sizeof(npy_intp));                                   \
            for (npy_intp j = max_dim; j >= 0; j--)                                                                         \
            {                                                                                                               \
                current_shape_process[j] = prd % __shape[j];                                                                \
                prd /= __shape[j];                                                                                          \
            }                                                                                                               \
            current_shape_process_[id] = current_shape_process;                                                             \
        }                                                                                                                   \
        npy_intp stride_a_last = strides_a[max_dim];                                                                        \
        npy_intp stride_b_last = strides_b[max_dim];                                                                        \
        _Pragma("omp parallel num_threads(num_threads) firstprivate(result_data_ptr_, a_data_ptr_saved, b_data_ptr_saved)") \
        {                                                                                                                   \
            int thread_id = omp_get_thread_num();                                                                           \
            result_data_ptr_ = result_ptr_[thread_id];                                                                      \
            npy_intp *current_process = current_shape_process_[thread_id];                                                  \
            for (npy_intp j = max_dim; j >= 0; j--)                                                                         \
            {                                                                                                               \
                a_data_ptr_saved += current_process[j] * strides_a[j];                                                      \
                b_data_ptr_saved += current_process[j] * strides_b[j];                                                      \
            }                                                                                                               \
            _Pragma("omp for schedule(static)") for (k = 0; k < outter_loop_size; k++)                                      \
            {                                                                                                               \
                inner_loop_body(type, op, inner_loop_size, stride_a_last,                                                   \
                                stride_b_last, a_data_ptr_saved, b_data_ptr_saved, result_data_ptr_);                       \
                result_data_ptr_ += inner_loop_size;                                                                        \
                for (npy_intp j = outer_start; j >= 0; j--)                                                                 \
                {                                                                                                           \
                    if (current_process[j] < __shape[j])                                                                    \
                    {                                                                                                       \
                        current_process[j]++;                                                                               \
                        a_data_ptr_saved += strides_a[j];                                                                   \
                        b_data_ptr_saved += strides_b[j];                                                                   \
                        break;                                                                                              \
                    }                                                                                                       \
                    else                                                                                                    \
                    {                                                                                                       \
                        current_process[j] = 0;                                                                             \
                        a_data_ptr_saved -= indice_a_cache[j];                                                              \
                        b_data_ptr_saved -= indice_b_cache[j];                                                              \
                    }                                                                                                       \
                }                                                                                                           \
            }                                                                                                               \
            free(current_process);                                                                                          \
        }                                                                                                                   \
        free(current_shape_process_);                                                                                       \
        free(result_ptr_);                                                                                                  \
        free(indice_a_cache);                                                                                               \
        free(indice_b_cache);                                                                                               \
        free(shape_cpy);                                                                                                    \
        free(shape_copy);                                                                                                   \
        free(strides_a);                                                                                                    \
        free(strides_b);                                                                                                    \
    } while (0)

#endif // NUMBOOST_API_H