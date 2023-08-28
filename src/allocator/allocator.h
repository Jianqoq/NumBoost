#ifndef _ALLOCATOR_H
#define _ALLOCATOR_H
#include "../tensor.h"
#include "../libraries/hash/uthash.h"

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

typedef struct Mem_Chain
{
    cache *head;
    cache *tail;
    size_t max_possible_cache_size;
    void (*move_node_to_head)(struct Mem_Chain *, cache *);
    void (*pop)(struct Mem_Chain *);
    void (*free_partial_mem_blocks)(struct Mem_Chain *, cache *, int32_t);
} Mem_Chain;

typedef struct
{
    void *calloc;
    void *free;
    void *malloc;
    void *realloc;
} PyDataMem_Funcs;

extern PyDataMem_Handler my_handler;
extern Mem_Chain *mem_chain;
extern cache *cache_pool;
void chain_move_node_to_head(Mem_Chain *self, cache *s);
void chain_pop(Mem_Chain *self);
void chain_free_partial_mem_blocks(Mem_Chain *self, cache *s, int32_t to_free_num);
#endif