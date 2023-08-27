#ifndef _ALLOCATOR_H
#define _ALLOCATOR_H
#include "../tensor.h"
#include "../hash_lib/uthash.h"

#define Thread_Hold_Value 0.9
#define Mem_Pool_Size 100

typedef struct cache
{
    struct cache *next;
    struct cache *prev;
    void **mem_pool;
    size_t tensor_size;
    int32_t mem_allocated;
    int32_t max_mem;
    UT_hash_handle hh;
} cache;

typedef struct
{
    cache *head;
    cache *tail;
    size_t max_possible_cache_size;
} double_linked_list;

typedef struct
{
    void *calloc;
    void *free;
    void *malloc;
    void *realloc;
} PyDataMem_Funcs;

extern PyDataMem_Handler my_handler;
extern double_linked_list *mem_chain;
extern cache *cache_pool;
void handler_destructor(PyObject *handler);

#endif