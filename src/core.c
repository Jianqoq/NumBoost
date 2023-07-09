#define PY_ARRAY_UNIQUE_SYMBOL core_c
#define PY_SSIZE_T_CLEAN
#include "tensor.h"
#include <numpy/arrayobject.h>
#include "core.h"

np_method *NP_METHOD = NULL;
Array_Shape *ARRAY_SHAPE = NULL;
Power_Dict *POWER_DICT = NULL;
Log_Dict *LOG_DICT = NULL;

static PyMethodDef methods[] = {
    {"reshape", (PyCFunction)reshape, METH_FASTCALL, "Method docstring"},
    {"transpose", (PyCFunction)transpose, METH_FASTCALL, "Method docstring"},
    {"argmax", (PyCFunction)_argmax_wrapper, METH_FASTCALL, "Method docstring"},
    {"argmin", (PyCFunction)_argmin_wrapper, METH_FASTCALL, "Method docstring"},
    {"sum", (PyCFunction)_sum, METH_FASTCALL, "Method docstring"},
    {"max", (PyCFunction)_max, METH_FASTCALL, "Method docstring"},
    {"min", (PyCFunction)_min, METH_FASTCALL, "Method docstring"},
    {"sin", (PyCFunction)_sin, METH_FASTCALL, "Method docstring"},
    {"cos", (PyCFunction)_cos, METH_FASTCALL, "Method docstring"},
    {"tan", (PyCFunction)_tan, METH_FASTCALL, "Method docstring"},
    {"arcsin", (PyCFunction)_asin, METH_FASTCALL, "Method docstring"},
    {"arccos", (PyCFunction)_acos, METH_FASTCALL, "Method docstring"},
    {"arctan", (PyCFunction)_atan, METH_FASTCALL, "Method docstring"},
    {"arcsinh", (PyCFunction)_asinh, METH_FASTCALL, "Method docstring"},
    {"arccosh", (PyCFunction)_acosh, METH_FASTCALL, "Method docstring"},
    {"arctanh", (PyCFunction)_atanh, METH_FASTCALL, "Method docstring"},
    {"sinh", (PyCFunction)_sinh, METH_FASTCALL, "Method docstring"},
    {"cosh", (PyCFunction)_cosh, METH_FASTCALL, "Method docstring"},
    {"tanh", (PyCFunction)_tanh, METH_FASTCALL, "Method docstring"},
    {"log10", (PyCFunction)_log10, METH_FASTCALL, "Method docstring"},
    {"log", (PyCFunction)_log, METH_FASTCALL, "Method docstring"},
    {"exp", (PyCFunction)_exp, METH_FASTCALL, "Method docstring"},
    {"sqrt", (PyCFunction)_sqrt, METH_FASTCALL, "Method docstring"},
    {"abs", (PyCFunction)_abs, METH_FASTCALL, "Method docstring"},
    {"power", (PyCFunction)_pow, METH_FASTCALL, "Method docstring"},
    {"mean", (PyCFunction)_mean, METH_FASTCALL, "Method docstring"},
    {NULL}};

static PyModuleDef
    custommodule = {
        PyModuleDef_HEAD_INIT,
        .m_name = "core",
        .m_doc = "Example module that creates an extension type.",
        .m_size = -1,
        .m_methods = methods,
};

PyMODINIT_FUNC
PyInit_core(void)
{
    import_array();
    Py_Initialize();
    init_map();
    NP_METHOD = malloc(sizeof(np_method));
    ARRAY_SHAPE = NULL;
    POWER_DICT = NULL;
    LOG_DICT = NULL;
    PyObject *m;
    PyObject *module = PyImport_ImportModule("numpy");
    if (module == NULL)
    {
        printf("numpy not found\n");
        PyErr_Print();
        return NULL;
    }
    
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
    NP_METHOD->exp = PyObject_GetAttrString(module, "exp");
    NP_METHOD->log = PyObject_GetAttrString(module, "log");
    NP_METHOD->log10 = PyObject_GetAttrString(module, "log10");
    NP_METHOD->sqrt = PyObject_GetAttrString(module, "sqrt");
    NP_METHOD->abs = PyObject_GetAttrString(module, "abs");
    NP_METHOD->power = PyObject_GetAttrString(module, "power");
    // NP_METHOD->add = PyObject_GetAttrString(module, "add");
    // NP_METHOD->subtract = PyObject_GetAttrString(module, "subtract");
    // NP_METHOD->multiply = PyObject_GetAttrString(module, "multiply");
    // NP_METHOD->divide = PyObject_GetAttrString(module, "divide");
    // NP_METHOD->power = PyObject_GetAttrString(module, "power");
    // NP_METHOD->matmul = PyObject_GetAttrString(module, "matmul");
    // NP_METHOD->dot = PyObject_GetAttrString(module, "dot");
    // NP_METHOD->mean = PyObject_GetAttrString(module, "mean");
    NP_METHOD->reshape = PyObject_GetAttrString(module, "reshape");
    // NP_METHOD->tensordot = PyObject_GetAttrString(module, "tensordot");
    // NP_METHOD->concatenate = PyObject_GetAttrString(module, "concatenate");


    if (PyType_Ready(&Tensor_type) < 0)
        return NULL;
    Py_INCREF(&Tensor_type);
    m = PyModule_Create(&custommodule);
    if (m == NULL)
        return NULL;
    Py_DECREF(module);
    return m;
}