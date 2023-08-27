#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <string.h>
#include "allocator.h"

cache *cache_pool = NULL;
double_linked_list *mem_chain = NULL;
uint64_t available_memory = 0;

#ifdef _WIN32
#include <windows.h>
#define Available_Memory(mem)                 \
    MEMORYSTATUSEX mem_status;                \
    mem_status.dwLength = sizeof(mem_status); \
    GlobalMemoryStatusEx(&mem_status);        \
    (mem) = mem_status.ullAvailPhys;
#elif __linux__
#include <sys/sysinfo.h>
#define Available_Memory(mem) \
    struct sysinfo info;      \
    sysinfo(&info);           \
    (mem) = info.freeram;
#endif

#ifdef Allocator_Debug
#define Allocator_Debug_Print(...) printf(__VA_ARGS__);
#else
#define Allocator_Debug_Print(...)
#endif

/*
The default_malloc function serves as a custom memory allocator.
It is designed to manage a hash table called cache_pool and a doubly-linked list named mem_chain.
These structures are used for caching and reusing previously allocated memory blocks.
Numboost try its best to avoid calling malloc and free functions especially free since free() is a very expensive operation.
The default_malloc function will first check if there is any available memory in the cache_pool.
If there is, it will retrieve the memory from the cache_pool.
Otherwise, it will call malloc to allocate memory and create a cache_pool for it.
*/
static void *default_malloc(void *ctx, size_t size)
{
    cache *s = NULL;
    HASH_FIND(hh, cache_pool, &size, sizeof(size_t), s);
    if (!s)
    {
        void *ptr = malloc(sizeof(char) * size);
        cache *cache_struct = (cache *)malloc(sizeof(cache));
        cache_struct->mem_pool = (void **)malloc(sizeof(void *) * Mem_Pool_Size);
        if (!cache_struct || !cache_struct->mem_pool || !ptr)
            goto alloc_err;
        cache_struct->tensor_size = size;
        cache_struct->mem_allocated = -1;
        cache_struct->mem_pool[0] = ptr;
        cache_struct->max_mem = Mem_Pool_Size;
        cache_struct->next = mem_chain->head;
        cache_struct->prev = mem_chain->tail;
        mem_chain->head->prev = cache_struct;
        mem_chain->tail->next = cache_struct;
        mem_chain->head = cache_struct;
        Allocator_Debug_Print("malloc: tail->prev: %p tail->next: %p, tail: %p\n", mem_chain->tail->prev, mem_chain->tail->next, mem_chain->tail);
        Allocator_Debug_Print("malloc: mem_added to %llu: %ld, mem: %p, mem->next: %p, mem->prev: %p\n", cache_struct->tensor_size, cache_struct->mem_allocated + 1, cache_struct,
                              cache_struct->next, cache_struct->prev);
        HASH_ADD(hh, cache_pool, tensor_size, sizeof(size_t), cache_struct);
        Allocator_Debug_Print("=====================================\n");
        return ptr;
    }
    else
    {
        Allocator_Debug_Print("malloc: mem_allocated: %ld in size %llu, tail: %p, tail->prev: %p\n", s->mem_allocated + 1, s->tensor_size, mem_chain->tail, mem_chain->tail->prev);
        Allocator_Debug_Print("before: mem_chain->head: %p, mem_chain->head->next: %p\n", mem_chain->head, mem_chain->head->next);
        if (s != mem_chain->head)
        {
            s->prev->next = s->next;
            s->next->prev = s->prev;
            s->next = mem_chain->head;
            s->prev = mem_chain->tail;
            mem_chain->head->prev = s;
            mem_chain->tail->next = s;
            mem_chain->head = s;
        }
        Allocator_Debug_Print("before: mem_chain->head: %p, mem_chain->head->next: %p\n", mem_chain->head, mem_chain->head->next);
        Allocator_Debug_Print("malloc: already have callocated in size % llu, mem_allocated: %ld in size %llu, tail: %p, tail->prev: %p, tail->next: %p\n", size,
                              s->mem_allocated + 1, s->tensor_size, mem_chain->tail, mem_chain->tail->prev, mem_chain->tail->next);
        if (s->mem_allocated >= 0)
        {
            mem_chain->max_possible_cache_size -= size;
            Allocator_Debug_Print("malloc: retrieved from cache pool, s->mem_allocated: %ld\n", s->mem_allocated);
            Allocator_Debug_Print("=====================================\n");
            void *ptr = s->mem_pool[s->mem_allocated--];
            assert(ptr);
            return ptr;
        }
        else
        {
            Allocator_Debug_Print("malloc: no available mem in size %llu\n", size);
            Allocator_Debug_Print("=====================================\n");
            return malloc(sizeof(char) * size);
        }
    }
alloc_err:
    return NULL;
}

static void *default_calloc(void *ctx, size_t nelem, size_t elsize)
{
    return calloc(nelem, elsize);
}

static void *default_realloc(void *ctx, void *ptr, size_t new_size)
{
    return realloc(ptr, new_size);
}

// static void default_free(void *ctx, void *ptr, size_t size)
// {
//     free(ptr);
// }

static void default_free(void *ctx, void *ptr, size_t size)
{
    cache *s = NULL;
    HASH_FIND(hh, cache_pool, &size, sizeof(size_t), s);
    Allocator_Debug_Print("free: mem_allocated: %ld in size %llu\n", s->mem_allocated + 1, s->tensor_size);
    // when the first time allocation is pretty high, we need to free
    uint64_t mem;
    Available_Memory(mem);
    uint64_t predict_possible_cache_size = mem_chain->max_possible_cache_size + size * s->max_mem * 2;

    if (s->mem_allocated >= 0                                    /*has availabel mem*/
        && predict_possible_cache_size > mem * Thread_Hold_Value /*Lru thread hold value, default 90% available mem*/
    )
    {
        Allocator_Debug_Print("free: thread hold value..., tail->mem_allocated: %d, %p\n", mem_chain->tail->mem_allocated + 1, mem_chain->tail->mem_pool[0]);
        if (size <= mem_chain->tail->tensor_size * (mem_chain->tail->mem_allocated + 1) && mem_chain->tail->mem_allocated >= 0)
        {
            int to_free_num = (size % mem_chain->tail->tensor_size) == 0 ? size / mem_chain->tail->tensor_size : size / mem_chain->tail->tensor_size + 1;
            int end = mem_chain->tail->mem_allocated - to_free_num;
            Allocator_Debug_Print("free: mem in size %llu, can free all mem in one node, allocated: %ld, mem_chain->tail->tensor_size: %llu, chain_len: %lu\n", size,
                                  mem_chain->tail->mem_allocated, mem_chain->tail->tensor_size, to_free_num);
            for (int i = mem_chain->tail->mem_allocated; i > end; i--)
            {
                Allocator_Debug_Print("free: freeing tail mem[%d], tail allocated: %ld, tail: %p, tail->prev: %p\n", i, mem_chain->tail->mem_allocated + 1, mem_chain->tail, mem_chain->tail->prev);
                free(mem_chain->tail->mem_pool[i]);
            }
            mem_chain->tail->mem_allocated -= to_free_num;
        }
        else if (size > mem_chain->tail->tensor_size * (mem_chain->tail->mem_allocated + 1) && mem_chain->tail->mem_allocated >= 0)
        {
            Allocator_Debug_Print("free: mem in size %llu, cannot free all mem in one node, allocated: %ld\n", size,
                                  mem_chain->tail->mem_allocated + 1);
            while (size > mem_chain->tail->tensor_size * (mem_chain->tail->mem_allocated + 1) &&
                   mem_chain->head != mem_chain->tail)
            {
                for (int i = 0; i <= mem_chain->tail->mem_allocated; i++)
                {
                    Allocator_Debug_Print("free: freeing tail mem[%d], tail allocated: %ld, tail: %p, tail->prev: %p\n", i, mem_chain->tail->mem_allocated + 1, mem_chain->tail, mem_chain->tail->prev);
                    free(mem_chain->tail->mem_pool[i]);
                    Allocator_Debug_Print("free: freed mem[%d]\n", i);
                }
                size -= mem_chain->tail->tensor_size * (mem_chain->tail->mem_allocated + 1);
                Allocator_Debug_Print("free: freed all mem in size %llu\n", mem_chain->tail->tensor_size);
                cache *temp = mem_chain->tail;
                Allocator_Debug_Print("free: changed tail to %p\n", temp->prev);
                temp->prev->next = mem_chain->head;
                mem_chain->head->prev = temp->prev;
                mem_chain->tail = temp->prev;
                Allocator_Debug_Print("free: changed tail to %p\n", mem_chain->tail);
                HASH_DEL(cache_pool, temp);
                Allocator_Debug_Print("free: free temp mem_pool: %p\n", temp->mem_pool);
                free(temp->mem_pool);
                Allocator_Debug_Print("free: free temp\n");
                free(temp);
                Allocator_Debug_Print("free: size left: %llu\n", size);
            }
            if (mem_chain->tail == mem_chain->head)
            {
                if (size <= mem_chain->tail->tensor_size * (mem_chain->tail->mem_allocated + 1) && mem_chain->tail->mem_allocated >= 0)
                {
                    int to_free_num = size / mem_chain->tail->tensor_size + 1;
                    int end = mem_chain->tail->mem_allocated - to_free_num;
                    for (int i = mem_chain->tail->mem_allocated; i > end; i--)
                    {
                        free(mem_chain->tail->mem_pool[i]);
                    }
                    mem_chain->tail->mem_allocated -= to_free_num;
                }
                else
                {
                    for (int i = 0; i <= mem_chain->tail->mem_allocated; i++)
                    {
                        free(mem_chain->tail->mem_pool[i]);
                    }
                }
            }
        }
        Allocator_Debug_Print("free: done free for thread hold value...\n");
    }
    else
    {
        Allocator_Debug_Print("free: add to cache pool..., will have %ld in %llu, tail: %p, tail->prev: %p, s->next: %p, s->prev: %p, tail_size: %llu\n", s->mem_allocated + 2, size, mem_chain->tail, mem_chain->tail->prev,
                              s->next, s->prev, mem_chain->tail->tensor_size);
        if (s != mem_chain->head)
        {
            Allocator_Debug_Print("before: mem_chain->head: %p, mem_chain->head->next: %p\n", mem_chain->head, mem_chain->head->next);
            s->prev->next = s->next;
            s->next->prev = s->prev;
            s->next = mem_chain->head;
            mem_chain->head->prev = s;
            s->prev = mem_chain->tail;
            mem_chain->tail->next = s;
            Allocator_Debug_Print("after: mem_chain->head: %p, mem_chain->head->next: %p\n", mem_chain->head, mem_chain->head->next);
            mem_chain->head = s;
        }
        mem_chain->max_possible_cache_size += size;
        Allocator_Debug_Print("free: added to cache pool..., now tail: %p, tail->prev: %p, tail_size: %llu\n", mem_chain->tail, mem_chain->tail->prev, mem_chain->tail->tensor_size);
        if (s->mem_allocated + 1 < s->max_mem)
        {
            s->mem_pool[++s->mem_allocated] = ptr;
        }
        else
        {
            void **mem_pool = realloc(s->mem_pool, sizeof(void *) * (s->max_mem * 2));
            assert(mem_pool);
            s->mem_pool = mem_pool;
            s->mem_pool[++s->mem_allocated] = ptr;
            s->max_mem *= 2;
        }
        Allocator_Debug_Print("free: done add to cache pool..., now mem_allocated %ld in size: %llu, tail: %p\n", s->mem_allocated + 1, s->tensor_size, mem_chain->tail);
    }
    Allocator_Debug_Print("=====================================\n");
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
        },
        default_malloc,
        default_calloc,
        default_realloc,
        default_free,
    }};