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

typedef struct double_linked_list
{
    cache *head;
    cache *tail;
    size_t max_possible_cache_size;
    void (*move_node_to_head)(struct double_linked_list *, cache *);
    void (*pop)(struct double_linked_list *);
    void (*free_partial_mem_blocks)(struct double_linked_list *, cache *, int32_t);
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
void chain_move_node_to_head(double_linked_list *self, cache *s);
void chain_pop(double_linked_list *self);
void chain_free_partial_mem_blocks(double_linked_list *self, cache *s, int32_t to_free_num);
#endif