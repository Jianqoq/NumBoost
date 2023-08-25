#include "omp.h"
#include <Python.h>
#include "numpy/arrayobject.h"
#include "numboost_api.h"

#define str(x) #x

#define normal_loop(a, b, res, op, size) \
    for (int i = 0; i < size; i++)       \
    {                                    \
        res[i] = a[i] op b[i];           \
    }

#define div_loop(a, b, res, op, size) \
    for (int i = 0; i < size; i++)    \
    {                                 \
        if (b[i] == 0)                \
        {                             \
            res[i] = 0;               \
            continue;                 \
        }                             \
        res[i] = a[i] / b[i];         \
    }

#define benchmark_cases(type, op, body)                                                     \
    static npy_intp test_##type##threadshold(float *a, float *b, float *res, npy_intp size, \
                                             double time_, char **string)                   \
    {                                                                                       \
        StartTimer(1);                                                                      \
        body(a, b, res, op, size);                                                          \
        StopTimer(1);                                                                       \
        double result = GetElapsed(1);                                                      \
        npy_intp thread_hold = (npy_intp)((size * time_) / result);                         \
        sprintf(*string, "%s", str(type##threadhold));                                      \
        return thread_hold;                                                                 \
    }

benchmark_cases(add_, +, normal_loop);
benchmark_cases(sub_, -, normal_loop);
benchmark_cases(mul_, *, normal_loop);
benchmark_cases(div_, /, div_loop);

PyObject *test(PyObject *self, PyObject *const *args, size_t nargsf)
{
    int i = 0;
    srand(time(0));
    int random_number = rand() % 11;
    StartTimer(1)
#pragma omp parallel for
        for (i = 0; i < omp_get_max_threads(); i++)
    {
    }
    StopTimer(1);
    PrintTimer(1);
    npy_intp shape[] = {1000000};
    float k = 2.0;
    FILE *file = fopen("numboost_threadshold.h", "w");
    if (file == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }
    npy_intp (*test_func[])(float *, float *, float *, npy_intp, double, char **) = {test_add_threadshold, test_sub_threadshold,
                                                                                     test_mul_threadshold, test_div_threadshold};
    npy_intp thread_hold = 0;
    for (int i = 0; i < 4; i++)
    {
        char *string = malloc(sizeof(char *) * 100);
        for (int k = 0; k < 100; k++)
        {
            PyArrayObject *empty = PyArray_EMPTY(1, shape, NPY_FLOAT32, 0);
            PyArrayObject *empty1 = PyArray_EMPTY(1, shape, NPY_FLOAT32, 0);
            PyArrayObject *empty2 = PyArray_EMPTY(1, shape, NPY_FLOAT32, 0);
            float *empty_data = (float *)PyArray_DATA(empty);
            float *empty_data1 = (float *)PyArray_DATA(empty);
            float *empty_data2 = (float *)PyArray_DATA(empty);
            StartTimer(2);
            thread_hold += test_func[i](empty_data, empty_data1, empty_data2, 1000000, GetElapsed(1), &string);
            StopTimer(2);
            PrintTimer(2);
            Py_DECREF((PyObject *)empty);
            Py_DECREF((PyObject *)empty1);
            Py_DECREF((PyObject *)empty2);
        }
        fprintf(file, "#define %s_Thread_Holds %ld\n", string, thread_hold / 100);
        free(string);
    }
    fclose(file);
    Py_RETURN_NONE;
}

static PyMethodDef module_methods[] = {
    {"test", (PyCFunction)test, METH_FASTCALL, "Method docstring"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static PyModuleDef custommodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "test",
    .m_doc = "Tensor is a numpy wrapper which supports autograd",
    .m_size = -1,
    .m_methods = module_methods,
};

PyMODINIT_FUNC PyInit_test()
{
    Py_Initialize();
    import_array();
    PyObject *m = PyModule_Create(&custommodule);
    if (m == NULL)
        return NULL;
    return m;
}