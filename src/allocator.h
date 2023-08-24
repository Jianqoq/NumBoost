#ifndef _ALLOCATOR_H
#define _ALLOCATOR_H
#include "tensor.h"
#include "uthash.h"

typedef struct
{
    void *mem_for_small;
    size_t size;
    UT_hash_handle hh;
} mem_chain;

typedef struct
{
    void *mem_for_small;
    void *prev_used_mem;

} mem_pool;

extern mem_pool *pool;
extern PyDataMem_Handler my_handler;
void handler_destructor(PyObject *handler);

#endif