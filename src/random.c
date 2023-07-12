#define PY_ARRAY_UNIQUE_SYMBOL random_c
#define PY_SSIZE_T_CLEAN
#include "random.h"
#include <time.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "numpy/ndarraytypes.h"
#include "operators.h"
#include "omp.h"
#include <windows.h>

RNG_POOL *rng_pool;

inline double generate_random_double()
{
    double uint32_max = 4294967295.0;
    return (double)pcg32_random_r(&rng_pool[omp_get_thread_num()].rng) / uint32_max;
}

Tensor *random(PyObject *self, PyObject *const *args, size_t nargsf)
{
    npy_intp *shape = malloc(nargsf * sizeof(npy_intp));
    double uint32_max = 4294967295.0;
    for (size_t i = 0; i < nargsf; i++)
    {
        npy_intp arg = PyLong_AsLongLong(args[i]);
        if (arg == -1 && PyErr_Occurred())
        {
            PyErr_SetString(PyExc_TypeError, "expected an integer");
            return NULL;
        }
        shape[i] = arg;
    }
    npy_intp size = 1;
    for (uint8_t i = 0; i < nargsf; i++)
    {
        size *= shape[i];
    }
    PyArrayObject *arr = (PyArrayObject *)PyArray_EMPTY((int)nargsf, shape, NPY_DOUBLE, 0);
    double *data = (double *)PyArray_DATA(arr);
    npy_intp i;
    if (size > 1000000)
    {
#pragma omp parallel for
        for (i = 0; i < size; i++)
        {
            data[i] = pcg32_random_r(&rng_pool[omp_get_thread_num()].rng) / uint32_max;
        }
    }
    else
    {
        pcg32_random_t *rng = &rng_pool[0].rng;
        for (i = 0; i < size; i++)
        {
            data[i] = pcg32_random_r(rng) / uint32_max;
        }
    }
    Tensor *tensor = Tensor__new__(&Tensor_type, (PyObject *)arr);
    Py_DECREF(arr);
    return tensor;
}

void seed(PyObject *num)
{
    long long const se = PyLong_AsLongLong(num);
    uint64_t seed = time(&se);
    for (int i = 0; i < omp_get_max_threads(); i++)
    {
        pcg32_srandom_r(&rng_pool[i].rng, seed, seed ^ (seed >> 16));
        rng_pool[i].thread_id = i;
    }
}

static void create_rng_pool()
{
    rng_pool = (RNG_POOL *)malloc(omp_get_max_threads() * sizeof(RNG_POOL));
    for (int i = 0; i < omp_get_max_threads(); i++)
    {
        uint64_t seed = time(NULL);
        pcg32_srandom_r(&rng_pool[i].rng, seed, seed ^ (seed >> 16));
        rng_pool[i].thread_id = i;
    }
}

static PyMethodDef methods[] = {
    {"random", (PyCFunction)random, METH_FASTCALL, "Method docstring"},
    {NULL}};

static PyModuleDef
    custommodule = {
        PyModuleDef_HEAD_INIT,
        .m_name = "rand",
        .m_doc = "Example module that creates an extension type.",
        .m_size = -1,
        .m_methods = methods,
};

PyMODINIT_FUNC
PyInit_rand(void)
{
    import_array();
    create_rng_pool();
    PyObject *m;
    m = PyModule_Create(&custommodule);
    if (PyType_Ready(&Tensor_type) < 0)
        return NULL;
    Py_INCREF(&Tensor_type);
    if (m == NULL)
        return NULL;
    return m;
}