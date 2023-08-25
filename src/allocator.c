#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <string.h>
#include "allocator.h"

mem_chain chain[100];
mem_pool *pool = NULL;

typedef struct
{
    void *calloc;
    void *free;
    void *malloc;
    void *realloc;
} PyDataMem_Funcs;

static void *default_malloc(void *ctx, size_t size)
{
    // mem_pool *entry = NULL;
    // if (pool != NULL)
    //     HASH_FIND_PTR(pool, , entry);
    return malloc(size);
}

static void *default_calloc(void *ctx, size_t nelem, size_t elsize)
{
    return calloc(nelem, elsize);
}

static void *default_realloc(void *ctx, void *ptr, size_t new_size)
{
    return realloc(ptr, new_size);
}

static void default_free(void *ctx, void *ptr, size_t size)
{
    free(ptr);
}

void handler_destructor(PyObject *handler)
{
    PyDataMem_Handler *mem_handler = (PyDataMem_Handler *)PyCapsule_GetPointer(handler, "mem_handler");
    if (!mem_handler)
    {
        return;
    }

    Py_XDECREF(((PyDataMem_Funcs *)mem_handler->allocator.ctx)->realloc);

    Py_XDECREF(((PyDataMem_Funcs *)mem_handler->allocator.ctx)->malloc);

    Py_XDECREF(((PyDataMem_Funcs *)mem_handler->allocator.ctx)->free);

    Py_XDECREF(((PyDataMem_Funcs *)mem_handler->allocator.ctx)->calloc);

    free(mem_handler->allocator.ctx);

    free(mem_handler);
}

PyDataMem_Handler my_handler = {
    "mem_handler",
    1,
    .allocator = {
        .ctx = &(PyDataMem_Funcs){
            .malloc = (void *)default_malloc,
            .calloc = (void *)default_calloc,
            .realloc = (void *)default_realloc,
            .free = (void *)default_free,
        }, // 上下文
        default_malloc,
        default_calloc,
        default_realloc,
        default_free,
    }};