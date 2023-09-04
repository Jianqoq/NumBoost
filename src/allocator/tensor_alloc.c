#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "tensor_alloc.h"
#include "allocator.h"

#define Tensor_Pool_Size 1000

Tensor *tensor_pool[Tensor_Pool_Size];

Tensor_Pool tensor_pool_maintainer = {
    .tensor_pool = tensor_pool,
    .index = -1,
};

Tensor *tensor_alloc(PyTypeObject *type, Py_ssize_t size)
{
    Tensor *tensor = NULL;
    if (tensor_pool_maintainer.index >= 0)
    {
        tensor = tensor_pool[tensor_pool_maintainer.index--];
        return tensor;
    }
    else
    {
        tensor = PyObject_GC_New(Tensor, type);
        if (tensor == NULL)
        {
            PyErr_SetString(PyExc_MemoryError, "Tensor alloc failed");
            return NULL;
        }
        PyObject_GC_Track(tensor);
        return tensor;
    }
}

void free_tensordot_data()
{
    Tensordot_Dict *entry = NULL, *tmp = NULL;
    HASH_ITER(hh, TENSORDOT_DICT, entry, tmp)
    {
        DEBUG_PRINT("Freeing Tensordot data\n");
        HASH_DEL(TENSORDOT_DICT, entry);
        free(entry->metadata->newaxes_a.ptr);
        free(entry->metadata->newaxes_b.ptr);
        Py_DECREF(entry->metadata->matmul_result);
        Py_DECREF(entry->metadata->transposed_reshape_a);
        Py_DECREF(entry->metadata->transposed_reshape_b);
        Py_DECREF(entry->key);
        free(entry->metadata);
        free(entry);
    }
}

inline void free_array_shape(Tensor *key)
{
    DEBUG_PRINT("Freeing Array shape\n");
    Array_Shape *s = NULL;
    if (ARRAY_SHAPE != NULL)
        HASH_FIND_PTR(ARRAY_SHAPE, &key, s);
    if (s != NULL)
    {
        HASH_DEL(ARRAY_SHAPE, s);
        free(s->shape);
        free(s);
    }
    DEBUG_PRINT("Freeing Array shape done\n");
}

static inline void free_tensordot_data_self(Tensor *self)
{
    Tensordot_Dict *entry = NULL;
    DEBUG_PRINT("Going to free Tensordot data\n");
    HASH_FIND_PTR(TENSORDOT_DICT, &self, entry);
    if (entry != NULL)
    {
        DEBUG_PRINT("Freeing Tensordot data\n");
        HASH_DEL(TENSORDOT_DICT, entry);
        free(entry->metadata->newaxes_a.ptr);
        free(entry->metadata->newaxes_b.ptr);
        Py_DECREF(entry->metadata->matmul_result);
        Py_DECREF(entry->metadata->transposed_reshape_a);
        Py_DECREF(entry->metadata->transposed_reshape_b);
        free(entry->metadata);
        free(entry);
    }
}

inline void free_tensor_need_grad(Tensor *self)
{
    Tensor_need_grad_Dict *entry = NULL;
    HASH_FIND_PTR(TENSOR_NEED_GRAD_DICT, &self, entry);
    if (entry != NULL)
    {
        HASH_DEL(TENSOR_NEED_GRAD_DICT, entry);
        free(entry);
    }
}

void free_slice_objs(Tensor *key)
{
    Slice_Dict *entry = NULL;
    HASH_FIND_PTR(SLICE_DICT, &key, entry);
    if (entry != NULL)
    {
        DEBUG_PRINT("free_slice_objs\n");
        HASH_DEL(SLICE_DICT, entry);
        Py_DECREF(entry->slice_obj);
        free(entry);
        DEBUG_PRINT("free_slice_objs done\n");
    }
    Zeros_Array_Dict *entry2 = NULL;
    HASH_FIND_PTR(ZEROS_ARRAY_DICT, &key, entry2);
    if (entry2 != NULL)
    {
        DEBUG_PRINT("free zero arrays\n");
        HASH_DEL(ZEROS_ARRAY_DICT, entry2);
        DEBUG_PyObject_Print(entry2->zeros_array);
        Py_DECREF(entry2->zeros_array);
        free(entry2);
        DEBUG_PRINT("free zero arrays done\n");
    }
}

void Tensor_dealloc(Tensor *self)
{
    Py_CLEAR(self->data); // pretty expensive
    Py_CLEAR(self->x);
    Py_CLEAR(self->y);
    Py_CLEAR(self->axis);
    Py_CLEAR(self->graph);
    Py_CLEAR(self->grad);
    free_tensordot_data_self(self);
    free_array_shape(self);
    free_power(self);
    free_tensor_need_grad(self);
    free_slice_objs(self);
    if (tensor_pool_maintainer.index < Tensor_Pool_Size - 1)
    {
        Py_INCREF(self);
        tensor_pool_maintainer.tensor_pool[++tensor_pool_maintainer.index] = self;
    }
    else
    {
        PyObject_GC_Del(self);
    }
}

int Tensor_clear(Tensor *self)
{
    PyObject_GC_UnTrack(self);
    Py_CLEAR(self->data);
    Py_CLEAR(self->x);
    Py_CLEAR(self->y);
    Py_CLEAR(self->axis);
    Py_CLEAR(self->graph);
    Py_CLEAR(self->grad);
    PyObject_GC_Track(self);
    return 0;
}

inline void free_power(Tensor *key)
{
    Power_Dict *s = NULL;
    if (POWER_DICT != NULL)
        HASH_FIND_PTR(POWER_DICT, &key, s);
    if (s != NULL)
    {
        HASH_DEL(POWER_DICT, s);
        free(s);
    }
}

inline void free_base(Tensor *key)
{
    Log_Dict *s = NULL;
    if (LOG_DICT != NULL)
        HASH_FIND_PTR(LOG_DICT, &key, s);
    if (s != NULL)
    {
        HASH_DEL(LOG_DICT, s);
        free(s);
    }
}

void free_all_resources()
{
    cache *s, *tmp;

    HASH_ITER(hh, cache_pool, s, tmp)
    {
        HASH_DEL(cache_pool, s);
        for (int i = 0; i <= s->mem_allocated; i++)
        {
            free(s->mem_pool[i]);
        }
        free(s->mem_pool);
        free(s);
    }
    free(mem_chain);

    Dict *entry, *tmp2;
    HASH_ITER(hh, dict, entry, tmp2)
    {
        HASH_DEL(dict, entry);
        free(entry);
    }
    free_xla_ops(xla_ops);
    free_tensordot_data();
    free_np_methods(NP_METHOD);
    free_jnp_methods(JNP_METHOD);

    Array_Shape *s2, *tmp3;
    HASH_ITER(hh, ARRAY_SHAPE, s2, tmp3)
    {
        HASH_DEL(ARRAY_SHAPE, s2);
        Py_XDECREF(s2->key);
        free(s2->shape);
        free(s2);
    }
    Power_Dict *s3, *tmp4;
    HASH_ITER(hh, POWER_DICT, s3, tmp4)
    {
        HASH_DEL(POWER_DICT, s3);
        Py_XDECREF(s3->key);
        Py_XDECREF(s3->prev_power);
        free(s3);
    }
    Log_Dict *s4, *tmp5;
    HASH_ITER(hh, LOG_DICT, s4, tmp5)
    {
        HASH_DEL(LOG_DICT, s4);
        Py_XDECREF(s4->key);
        Py_XDECREF(s4->base);
        free(s4);
    }
    Tensor_need_grad_Dict *s5, *tmp6;
    HASH_ITER(hh, TENSOR_NEED_GRAD_DICT, s5, tmp6)
    {
        HASH_DEL(TENSOR_NEED_GRAD_DICT, s5);
        Py_XDECREF(s5->tensor);
        free(s5);
    }

    Tensordot_Dict *s6, *tmp7;
    HASH_ITER(hh, TENSORDOT_DICT, s6, tmp7)
    {
        HASH_DEL(TENSORDOT_DICT, s6);
        free(s6);
    }

    int32_t index = tensor_pool_maintainer.index;
    for (int32_t i = 0; i <= index; i++)
    {
        PyObject_GC_Del(tensor_pool_maintainer.tensor_pool[i]);
    }
    HASH_CLEAR(hh, dict);
    HASH_CLEAR(hh, TENSORDOT_DICT);
    HASH_CLEAR(hh, ARRAY_SHAPE);
    HASH_CLEAR(hh, POWER_DICT);
    HASH_CLEAR(hh, LOG_DICT);
    HASH_CLEAR(hh, TENSOR_NEED_GRAD_DICT);
    HASH_CLEAR(hh, SLICE_DICT);
    HASH_CLEAR(hh, ZEROS_ARRAY_DICT);
    Py_CLEAR(Tensor_type);
}
