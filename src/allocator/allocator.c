#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <string.h>
#include "allocator.h"
#include "omp.h"

cache *cache_pool = NULL;
Mem_Chain *mem_chain = NULL;
uint64_t available_memory = 0;

#ifdef _WIN32
#include <windows.h>
#define Available_Memory(mem)                 \
    MEMORYSTATUSEX mem_status;                \
    mem_status.dwLength = sizeof(mem_status); \
    GlobalMemoryStatusEx(&mem_status);        \
    (mem) = mem_status.ullAvailPhys;

#define Numboost_Parallel_Memset(ptr, size, value) \
    memset(ptr, value, size);
#elif __linux__
#include <sys/sysinfo.h>
#define Available_Memory(mem) \
    struct sysinfo info;      \
    sysinfo(&info);           \
    (mem) = info.freeram;
#define Numboost_Parallel_Memset(ptr, size, value)         \
    _Pragma("omp parallel for") for (i = 0; i < size; i++) \
        ptr[i] = value;
#endif

#ifdef Allocator_Debug
#define Allocator_Debug_Print(...) printf(__VA_ARGS__);
#else
#define Allocator_Debug_Print(...)
#endif

/*
Being linked to mem_chain free_partial_mem_blocks function in tensor.c PyInit_Numboost function
@param self mem_chain
@param s node to be freed partial memory blocks
@param to_free_num number of memory blocks to be freed
*/
void chain_free_partial_mem_blocks(Mem_Chain *self, cache *s, int32_t to_free_num)
{
    int end = s->mem_allocated - to_free_num;
    void **mem_pool = s->mem_pool;
    for (int i = s->mem_allocated; i > end; i--)
    {
        free(mem_pool[i]);
    }
    s->mem_allocated -= to_free_num;
}

/*
Being linked to mem_chain move_node_to_head function in tensor.c PyInit_Numboost function
@param self mem_chain
@param s node to move to head
*/
void chain_move_node_to_head(Mem_Chain *self, cache *s)
{
    if (s != self->head)
    {
        if (s->next != NULL)
            s->next->prev = s->prev;
        if (s->prev != NULL)
            s->prev->next = s->next;
        s->next = self->head;
        s->prev = self->tail;
        self->head->prev = s;
        self->tail->next = s;
        self->head = s;
    }
}

/*
Being linked to mem_chain pop function in tensor.c PyInit_Numboost function
The chain_pop function is used to free the least recently used memory block.
It will free the memory block in the tail of the mem_chain.
But this node will stay in hash table "cache_pool". This node will points to NULL in next and prev.
@param self mem_chain
 */
void chain_pop(Mem_Chain *self)
{
    int32_t allocated = self->tail->mem_allocated;
    void **mem_pool = self->tail->mem_pool;
    for (int i = 0; i <= allocated; i++)
    {
        free(mem_pool[i]);
    }
    self->tail->mem_allocated = -1;
    cache *temp = self->tail;
    temp->prev->next = self->head;
    self->head->prev = temp->prev;
    self->tail = temp->prev;
    temp->next = NULL;
    temp->prev = NULL;
}

/*
The default_malloc function serves as a custom memory allocator.
It is designed to manage a hash table called cache_pool and a doubly-linked list named mem_chain.
These structures are used for caching and reusing previously allocated memory blocks.
Numboost try its best to avoid calling malloc and free functions especially free since free() is a very expensive operation.
The default_malloc function will first check if there is any available memory in the cache_pool.
If there is, it will retrieve the memory from the cache_pool.
Otherwise, it will call malloc to allocate memory and create a cache_pool for it.
@param ctx unsed, but must be passed to the default_malloc function
@param size size of the memory to allocate. 1 size = 1 char
*/
static void *default_malloc(void *ctx, size_t size)
{
    cache *s = NULL;
    void *ptr = NULL;
    cache *cache_struct = NULL;
    HASH_FIND(hh, cache_pool, &size, sizeof(size_t), s);
    if (!s)
    {
        ptr = malloc(sizeof(char) * size);
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
        HASH_ADD(hh, cache_pool, tensor_size, sizeof(size_t), cache_struct);
        return ptr;
    }
    else
    {
        mem_chain->move_node_to_head(mem_chain, s);
        if (s->mem_allocated >= 0)
        {
            mem_chain->max_possible_cache_size -= size;
            return s->mem_pool[s->mem_allocated--];
        }
        else
        {
            return malloc(sizeof(char) * size);
        }
    }
alloc_err:
    if (cache_struct)
    {
        free(cache_struct);
        if (cache_struct->mem_pool)
            free(cache_struct->mem_pool);
    }
    if (ptr)
        free(ptr);
    return NULL;
}

static void *default_calloc(void *ctx, size_t nelem, size_t elsize)
{
    size_t total_size = nelem * elsize;
    void *ptr = default_malloc(ctx, total_size);
    if (!ptr)
        return NULL;
    char *tmp = (char *)ptr;
    size_t i;
    Numboost_Parallel_Memset(tmp, total_size, 0);
    return ptr;
}

static void *default_realloc(void *ctx, void *ptr, size_t new_size)
{
    cache *s = NULL;
    void *ret_ptr = NULL;
    cache *cache_struct = NULL;
    HASH_FIND(hh, cache_pool, &new_size, sizeof(size_t), s);
    if (!s)
    {
        ret_ptr = realloc(ptr, new_size);
        cache *cache_struct = (cache *)malloc(sizeof(cache));
        cache_struct->mem_pool = (void **)malloc(sizeof(void *) * Mem_Pool_Size);
        if (!cache_struct || !cache_struct->mem_pool || !ret_ptr)
            goto alloc_err;
        cache_struct->tensor_size = new_size;
        cache_struct->mem_allocated = -1;
        cache_struct->mem_pool[0] = ptr;
        cache_struct->max_mem = Mem_Pool_Size;
        cache_struct->next = mem_chain->head;
        cache_struct->prev = mem_chain->tail;
        mem_chain->head->prev = cache_struct;
        mem_chain->tail->next = cache_struct;
        mem_chain->head = cache_struct;
        HASH_ADD(hh, cache_pool, tensor_size, sizeof(size_t), cache_struct);
        return ret_ptr;
    }
    else
    {
        mem_chain->move_node_to_head(mem_chain, s);
        if (s->mem_allocated >= 0)
        {
            mem_chain->max_possible_cache_size -= new_size;
            return s->mem_pool[s->mem_allocated--];
        }
        else
        {
            return realloc(ptr, new_size);
        }
    }
alloc_err:
    if (cache_struct)
    {
        free(cache_struct);
        if (cache_struct->mem_pool)
            free(cache_struct->mem_pool);
    }
    if (ret_ptr)
        free(ret_ptr);
    return NULL;
}

/*
The default_free function serves as a custom memory deallocator.
It is designed to manage a hash table called cache_pool and a doubly-linked list named mem_chain.
These structures are used for caching and reusing previously allocated memory blocks.
Numboost try its best to avoid calling malloc and free functions especially free since free() is a very expensive operation.
The default_free function will first check if there is available memory based on thread hold in the system.
If there is, it will cache the memory in the cache_pool.
Otherwise, it will try to free the least recently used memory block.
Numboost will try to free enough memory which is greater or equal than the size of the memory to be cached.
@param ctx unsed, but must be passed to the default_malloc function
@param ptr pointer to the memory to free
@param size size of the memory to free. 1 size = 1 char
*/
static void default_free(void *ctx, void *ptr, size_t size)
{
    cache *s = NULL;
    HASH_FIND(hh, cache_pool, &size, sizeof(size_t), s);
    uint64_t mem;
    Available_Memory(mem);
    uint64_t predict_possible_cache_size = mem_chain->max_possible_cache_size + size * s->max_mem * 2;

    if (s->mem_allocated >= 0 /*has availabel mem*/
        && predict_possible_cache_size > mem * Thread_Hold_Value /*Lru thread hold value, default 90% available mem*/)
    {
        /*first we need to let to_free ptr points the the node which have available memory to free*/
        cache *to_free = mem_chain->tail;
        if (to_free->mem_allocated < 0)
        {
            while (to_free->mem_allocated < 0 && mem_chain->head != to_free)
            {
                to_free = to_free->prev;
            }
            if (to_free->mem_allocated < 0) /*If this condition == true, to_free will be head of mem_chain and no memory we can free from the mem_chain*/
            {
                free(ptr);
                return;
            }
        }
        /*size to cache is less than the total cached size in tail node, we can free all or partial of the memory in that node*/
        if (size <= to_free->tensor_size * (to_free->mem_allocated + 1))
        {
            int32_t to_free_num = (size % to_free->tensor_size) == 0 ? size / to_free->tensor_size : size / to_free->tensor_size + 1;
            mem_chain->free_partial_mem_blocks(mem_chain, to_free, to_free_num);
            mem_chain->max_possible_cache_size += (size - to_free->tensor_size * (to_free->mem_allocated + 1));
        }
        else /*We need to free memory not only in one node, but multiple*/
        {
            /*keep freeing the memory until we iterated to the head of the mem_chain*/
            while (size > to_free->tensor_size * (to_free->mem_allocated + 1) &&
                   mem_chain->head != to_free)
            {
                if (to_free->mem_allocated < 0)
                {
                    to_free = to_free->prev;
                    continue;
                }
                int32_t to_free_num = to_free->mem_allocated + 1;
                mem_chain->free_partial_mem_blocks(mem_chain, to_free, to_free_num);
                size -= to_free->tensor_size * (to_free->mem_allocated + 1);
                to_free = to_free->prev;
            }
            /*When this condition is true, which means the cached memory in the memory pool is relative small compare with the current memory we want to cache*/
            if (mem_chain->tail == mem_chain->head)
            {
                assert(mem_chain->head == to_free); // to_free should be the head of the mem_chain, otherwise, there is a bug in the code
                if (size <= to_free->tensor_size * (to_free->mem_allocated + 1) && to_free->mem_allocated >= 0)
                {
                    int32_t to_free_num = (size % to_free->tensor_size) == 0 ? size / to_free->tensor_size : size / to_free->tensor_size + 1;
                    mem_chain->free_partial_mem_blocks(mem_chain, to_free, to_free_num);
                }
                else
                {
                    for (int i = 0; i <= to_free->mem_allocated; i++)
                    {
                        free(to_free->mem_pool[i]);
                    }
                    to_free->mem_allocated = -1;
                }
                mem_chain->max_possible_cache_size += (size - to_free->tensor_size * (to_free->mem_allocated + 1));
            }
        }
    }
    else
    {
        mem_chain->move_node_to_head(mem_chain, s);
        mem_chain->max_possible_cache_size += size;
        if (s->mem_allocated + 1 < s->max_mem)
        {
            s->mem_pool[++s->mem_allocated] = ptr;
        }
        else
        {
            void **mem_pool = realloc(s->mem_pool, sizeof(void *) * (s->max_mem * 2));
            if (!mem_pool)
            {
                fprintf(stderr, "Failed to reallocate memory for mem_pool\n");
                free(ptr);
                return;
            }
            s->mem_pool = mem_pool;
            s->mem_pool[++s->mem_allocated] = ptr;
            s->max_mem *= 2;
        }
    }
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