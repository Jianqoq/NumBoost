#define PY_ARRAY_UNIQUE_SYMBOL tensor_c
#define NO_IMPORT_ARRAY
#include "import_methods.h"
#include "utils.h"
#include <omp.h>
#include "mkl_vml_functions.h"
#include <numpy/npy_math.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include "numpy/ndarraytypes.h"
#include "set_Tensor_properties.h"
#include "tensor.h"
#include "operators.h"

bool shape_isequal(npy_intp *a_shape, npy_intp *b_shape, int a_ndim, int b_ndim)
{
    if (a_ndim != b_ndim)
        return false;
    for (int i = 0; i < a_ndim; i++)
        if (a_shape[i] != b_shape[i])
            return false;
    return true;
}

inline bool isbroadcastable_same_shape(npy_intp *a_shape, npy_intp *b_shape, int ndim, npy_intp **a_new_shape, npy_intp **b_new_shape)
{
    int i;
    for (i = 0; i < ndim; i++)
        if (a_shape[i] != b_shape[i])
        {
            if (a_shape[i] == 1 || b_shape[i] == 1)
                continue;
            else
                return false;
        }
    if (a_new_shape != NULL && b_new_shape != NULL)
    {
        *a_new_shape = a_shape;
        *b_new_shape = b_shape;
    }
    return true;
}

// this method does not check for a specific Tensor, it only checks if the shape is broadcastable
// use shape_isbroadcastable_to if you want to check if a Tensor is broadcastable to another Tensor
// it assumes the shape is not equal
bool shape_isbroadcastable(npy_intp *a_shape, npy_intp *b_shape, int a_ndim, int b_ndim)
{
    int i;
    if (a_ndim == b_ndim)
    {
        bool isbroad = isbroadcastable_same_shape(a_shape, b_shape, a_ndim, NULL, NULL);
        return isbroad;
    }
    else
    {
        npy_intp *new_shape = NULL;
        int ndim_diff = a_ndim - b_ndim;
        if (ndim_diff > 0)
        {
            new_shape = (npy_intp *)malloc(a_ndim * sizeof(npy_intp));
            for (i = 0; i < a_ndim; i++)
            {
                if (i < ndim_diff)
                    new_shape[i] = 1;
                else
                    new_shape[i] = b_shape[i - ndim_diff];
            }
            b_ndim = a_ndim;
        }
        else
        {
            ndim_diff = -ndim_diff;
            new_shape = (npy_intp *)malloc(b_ndim * sizeof(npy_intp));
            for (i = 0; i < b_ndim; i++)
            {
                if (i < ndim_diff)
                    new_shape[i] = 1;
                else
                    new_shape[i] = a_shape[i + ndim_diff];
            }
            a_ndim = b_ndim;
        }
        bool isbroad = isbroadcastable_same_shape(a_shape, new_shape, a_ndim, NULL, NULL);
        free(new_shape);
        return isbroad;
    }
}

// ex support return new shape
bool shape_isbroadcastable_ex(npy_intp *a_shape, npy_intp *b_shape, int a_ndim, int b_ndim, npy_intp **a_new_shape, npy_intp **b_new_shape)
{
    int i;
    if (a_ndim == b_ndim)
    {
        bool isbroad = isbroadcastable_same_shape(a_shape, b_shape, a_ndim, a_new_shape, b_new_shape);
        return isbroad;
    }
    else
    {
        npy_intp *new_shape = NULL;
        int ndim_diff = a_ndim - b_ndim;
        if (ndim_diff > 0)
        {
            new_shape = (npy_intp *)malloc(a_ndim * sizeof(npy_intp));
            for (i = 0; i < a_ndim; i++)
            {
                if (i < ndim_diff)
                    new_shape[i] = 1;
                else
                    new_shape[i] = b_shape[i - ndim_diff];
            }
            b_ndim = a_ndim;
        }
        else
        {
            ndim_diff = -ndim_diff;
            new_shape = (npy_intp *)malloc(b_ndim * sizeof(npy_intp));
            for (i = 0; i < b_ndim; i++)
            {
                if (i < ndim_diff)
                    new_shape[i] = 1;
                else
                    new_shape[i] = a_shape[i - ndim_diff];
            }
            a_ndim = b_ndim;
        }
        bool isbroad = isbroadcastable_same_shape(ndim_diff > 0 ? a_shape : b_shape, new_shape, a_ndim, a_new_shape, b_new_shape);
        if (!isbroad)
            free(new_shape);
        return isbroad;
    }
}

// this method check for a specific Tensor, it checks if the shape is broadcastable to another Tensor
// it assumes the shape is not equal and a_ndim <= b_ndim
bool shape_isbroadcastable_to(npy_intp *a_shape, npy_intp *b_shape, int a_ndim, int b_ndim)
{
    int i;
    if (a_ndim == b_ndim)
    {
        bool isbroad = isbroadcastable_same_shape(a_shape, b_shape, a_ndim, NULL, NULL);
        return isbroad;
    }
    else
    {
        npy_intp *new_shape = NULL;
        int ndim_diff = a_ndim - b_ndim;
        if (ndim_diff > 0)
        {
            return false;
        }
        else
        {
            ndim_diff = -ndim_diff;
            new_shape = (npy_intp *)malloc(b_ndim * sizeof(npy_intp));
            for (i = 0; i < b_ndim; i++)
            {
                if (i < ndim_diff)
                    new_shape[i] = 1;
                else
                    new_shape[i] = a_shape[i - ndim_diff];
            }
            a_ndim = b_ndim;
            bool isbroad = isbroadcastable_same_shape(b_shape, new_shape, a_ndim, NULL, NULL);
            if (!isbroad)
                free(new_shape);
            return isbroad;
        }
    }
}

// ex support return new shape
bool shape_isbroadcastable_to_ex(npy_intp *a_shape, npy_intp *b_shape, int a_ndim, int b_ndim, npy_intp **a_new_shape)
{
    int i;
    if (a_ndim == b_ndim)
    {
        bool isbroad = isbroadcastable_same_shape(a_shape, b_shape, a_ndim, a_new_shape, NULL);
        return isbroad;
    }
    else
    {
        npy_intp *new_shape = NULL;
        int ndim_diff = a_ndim - b_ndim;
        if (ndim_diff > 0)
        {
            return false;
        }
        else
        {
            ndim_diff = -ndim_diff;
            new_shape = (npy_intp *)malloc(b_ndim * sizeof(npy_intp));
            for (i = 0; i < b_ndim; i++)
            {
                if (i < ndim_diff)
                    new_shape[i] = 1;
                else
                    new_shape[i] = a_shape[i - ndim_diff];
            }
            a_ndim = b_ndim;
            bool isbroad = isbroadcastable_same_shape(b_shape, new_shape, a_ndim, a_new_shape, NULL);
            if (!isbroad)
                free(new_shape);
            return isbroad;
        }
    }
}

int newshape_except_num(npy_intp except, npy_intp *shape, int ndim, npy_intp **new_shape)
{
    int i;
    int cnt = 0;
    for (i = 0; i < ndim; i++)
        if (shape[i] == 1)
            cnt++;
    int new_ndim = ndim - cnt;
    npy_intp *_new_shape = (npy_intp *)malloc(new_ndim * sizeof(npy_intp));
    int track = 0;
    for (i = 0; i < ndim; i++)
    {
        if (i != except)
        {
            _new_shape[track] = shape[i];
            track++;
        }
    }
    *new_shape = _new_shape;
    return new_ndim;
}

