#define PY_ARRAY_UNIQUE_SYMBOL random_c
#define PY_SSIZE_T_CLEAN
#include "random_.h"
#include <time.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "numpy/ndarraytypes.h"
#include "operators.h"
#include "omp.h"

RNG_POOL *rng_pool;

/*ziggurat source code start*/
/*reference The Ziggurat Method for Generating Random Variables*/
#include <math.h>
static unsigned long jz, jsr = 123456789;

#define SHR3 (jz = jsr, jsr ^= (jsr << 13), jsr ^= (jsr >> 17), jsr ^= (jsr << 5), jz + jsr)
#define UNI (.5 + (signed)SHR3 * .2328306e-9)
#define IUNI SHR3

static long hz;
static unsigned long iz, kn[128], ke[256];
static double wn[128], fn[128], we[256], fe[256];

#define RNOR (hz = SHR3, iz = hz & 127, (fabs(hz) < kn[iz]) ? hz * wn[iz] : nfix())
#define REXP (jz = SHR3, iz = jz & 255, (jz < ke[iz]) ? jz * we[iz] : efix())

inline double generate_random_double()
{
    double uint32_max = 4294967295.0;
    return (double)pcg32_random_r(&rng_pool[omp_get_thread_num()].rng) / uint32_max;
}

double nfix(void)
{
    const double r = 3.442620f; /* The start of the right tail */
    static double x, y;
    for (;;)
    {
        x = hz * wn[iz]; /* iz==0, handles the base strip */
        if (iz == 0)
        {
            do
            {
                x = -log(UNI) * 0.2904764;
                y = -log(UNI);
            } /* .2904764 is 1/r */
            while (y + y < x * x);
            return (hz > 0) ? r + x : -r - x;
        }
        /* iz>0, handle the wedges of other strips */
        if (fn[iz] + UNI * (fn[iz - 1] - fn[iz]) < exp(-.5 * x * x))
            return x;

        /* initiate, try to exit for(;;) for loop*/
        hz = SHR3;
        iz = hz & 127;
        if (fabs(hz) < kn[iz])
            return (hz * wn[iz]);
    }
}

/* efix() generates variates from the residue when rejection in REXP occurs. */
double efix(void)
{
    double x;
    for (;;)
    {
        if (iz == 0)
            return (7.69711 - log(UNI)); /* iz==0 */
        x = jz * we[iz];
        if (fe[iz] + UNI * (fe[iz - 1] - fe[iz]) < exp(-x))
            return (x);

        /* initiate, try to exit for(;;) loop */
        jz = SHR3;
        iz = (jz & 255);
        if (jz < ke[iz])
            return (jz * we[iz]);
    }
}
/*--------This procedure sets the seed and creates the tables------*/

void zigset(unsigned long jsrseed)
{
    const double m1 = 2147483648.0, m2 = 4294967296.;
    double dn = 3.442619855899, tn = dn, vn = 9.91256303526217e-3, q;
    double de = 7.697117470131487, te = de, ve = 3.949659822581572e-3;
    int i;
    jsr ^= jsrseed;

    /* Set up tables for RNOR */
    q = vn / exp(-.5 * dn * dn);
    kn[0] = (dn / q) * m1;
    kn[1] = 0;

    wn[0] = q / m1;
    wn[127] = dn / m1;

    fn[0] = 1.;
    fn[127] = exp(-.5 * dn * dn);

    for (i = 126; i >= 1; i--)
    {
        dn = sqrt(-2. * log(vn / dn + exp(-.5 * dn * dn)));
        kn[i + 1] = (dn / tn) * m1;
        tn = dn;
        fn[i] = exp(-.5 * dn * dn);
        wn[i] = dn / m1;
    }

    /* Set up tables for REXP */
    q = ve / exp(-de);
    ke[0] = (de / q) * m2;
    ke[1] = 0;

    we[0] = q / m2;
    we[255] = de / m2;

    fe[0] = 1.;
    fe[255] = exp(-de);

    for (i = 254; i >= 1; i--)
    {
        de = -log(ve / de + exp(-de));
        ke[i + 1] = (de / te) * m2;
        te = de;
        fe[i] = exp(-de);
        we[i] = de / m2;
    }
}
/*ziggurat source code end*/

Tensor *random_(PyObject *self, PyObject *const *args, size_t nargsf)
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
    Tensor *tensor = Tensor__new__(Tensor_type, (PyObject *)arr);
    Py_DECREF(arr);
    return tensor;
}

Tensor *randn_(PyObject *self, PyObject *const *args, size_t nargsf)
{
    npy_intp *shape = malloc(nargsf * sizeof(npy_intp));
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
    PyArrayObject *arr = (PyArrayObject *)PyArray_EMPTY((int)nargsf, shape, NPY_FLOAT, 0);
    double *data = (double *)PyArray_DATA(arr);
    npy_intp i;
    if (size > 1000000)
    {
        // #pragma omp parallel for
        for (i = 0; i < size; i++)
        {
            data[i] = RNOR;
        }
    }
    Tensor *tensor = (Tensor *)Tensor__new__(Tensor_type, (PyObject *)arr);
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
    {"random", (PyCFunction)random_, METH_FASTCALL, "Method docstring"},
    {"randn", (PyCFunction)randn_, METH_FASTCALL, "Method docstring"},
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
    zigset(5);
    PyObject *m;
    m = PyModule_Create(&custommodule);
    if (PyType_Ready(Tensor_type) < 0)
        return NULL;
    Py_INCREF(Tensor_type);
    if (m == NULL)
        return NULL;
    return m;
}