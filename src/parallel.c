#include "numpy/arrayobject.h"
#include "parallel.h"
#include <time.h>

BroadcastThreadpool *CreateBroadcastThreadpool(DWORD numThreads)
{
    BroadcastThreadpool *threadpool = malloc(sizeof(BroadcastThreadpool));
    if (threadpool == NULL)
        return NULL;

    InitializeThreadpoolEnvironment(&threadpool->environment);
    threadpool->pool = CreateThreadpool(NULL);
    SetThreadpoolThreadMaximum(threadpool->pool, numThreads);
    BOOL success = SetThreadpoolThreadMinimum(threadpool->pool, numThreads);
    if (!success)
    {
        CloseThreadpool(threadpool->pool);
        DestroyThreadpoolEnvironment(&threadpool->environment);
        free(threadpool);
        return NULL;
    }
    SetThreadpoolCallbackPool(&threadpool->environment, threadpool->pool);
    threadpool->NumThreads = numThreads;
    return threadpool;
}

void DestroyBroadcastThreadpool(BroadcastThreadpool *threadpool)
{
    CloseThreadpool(threadpool->pool);
    DestroyThreadpoolEnvironment(&threadpool->environment);
    free(threadpool);
}

VOID CALLBACK ComputeBroadcast(PTP_CALLBACK_INSTANCE Instance, PVOID Context)
{
    BroadcastDATA *data = (BroadcastDATA *)Context;
    switch (data->type)
    {
    case NPY_BOOL:
        ComputeBroadcastBranch(npy_bool, data->op);
        break;
    case NPY_BYTE:
        break;
    case NPY_UBYTE:
        break;
    case NPY_SHORT:
        ComputeBroadcastBranch(npy_short, data->op);
        break;
    case NPY_USHORT:
        break;
    case NPY_INT:
        ComputeBroadcastBranch(npy_int, data->op);
        break;
    case NPY_UINT:
        break;
    case NPY_LONG:
        ComputeBroadcastBranch(npy_long, data->op);
        break;
    case NPY_ULONG:
        break;
    case NPY_LONGLONG:
        ComputeBroadcastBranch(npy_longlong, data->op);
        break;
    case NPY_ULONGLONG:
        ComputeBroadcastBranch(npy_ulonglong, data->op);
        break;
    case NPY_FLOAT:
        ComputeBroadcastBranch(npy_float, data->op);
        break;
    case NPY_DOUBLE:
        ComputeBroadcastBranch(npy_double, data->op);
        break;
    case NPY_LONGDOUBLE:
        break;
    case NPY_CFLOAT:
        break;
    case NPY_CDOUBLE:
        break;
    case NPY_CLONGDOUBLE:
        break;
    case NPY_OBJECT:
        break;
    case NPY_STRING:
        break;
    case NPY_UNICODE:
        break;
    case NPY_VOID:
        break;
    case NPY_DATETIME:
        break;
    case NPY_TIMEDELTA:
        break;
    case NPY_HALF:
        break;
    default:
        break;
    }
}

void BroadcastParallel(BroadcastThreadpool *threadpool, char **DataPtrA, char **DataPtrB, char **ResultDataPtr, int type, int op,
                       npy_intp StridesA, npy_intp StridesB, npy_intp TotalLoopSize, clock_t *start)
{
    char *_DataPtrA = *DataPtrA;
    char *_DataPtrB = *DataPtrB;
    float *_ResultDataPtr = (float *)(*ResultDataPtr);
    clock_t start_ = clock();
    BroadcastDATA *data = malloc(sizeof(BroadcastDATA) * threadpool->NumThreads);
    PTP_WORK *WorkItems = malloc(sizeof(PTP_WORK) * threadpool->NumThreads);
    *start += clock() - start_;
    for (DWORD i = 0; i < threadpool->NumThreads; i++)
    {
        data[i].DataPtrA = _DataPtrA;
        data[i].DataPtrB = _DataPtrB;
        data[i].ResultDataPtr = _ResultDataPtr;
        data[i].StridesA = StridesA;
        data[i].StridesB = StridesB;
        data[i].LoopSize = TotalLoopSize / threadpool->NumThreads;
        data[i].ThreadID = i;
        data[i].type = type;
        data[i].op = op;
        _DataPtrA += data[i].LoopSize * StridesA;
        _DataPtrB += data[i].LoopSize * StridesB;
        _ResultDataPtr += data[i].LoopSize * StridesA;
        clock_t start_ = clock();
        WorkItems[i] = CreateThreadpoolWork(ComputeBroadcast, &data[i], NULL);
        SubmitThreadpoolWork(WorkItems[i]);
        *start += clock() - start_;
    }
    DEBUG_PRINTLN("Wait for callbacks")
    for (DWORD i = 0; i < threadpool->NumThreads; i++)
    {
        DEBUG_PRINTLN("Wait for callbacks done[%d]", i)
        WaitForThreadpoolWorkCallbacks(WorkItems[i], FALSE);
    }
    clock_t start_1 = clock();
    for (int i = 0; i < threadpool->NumThreads; i++)
    {
        CloseThreadpoolWork(WorkItems[i]);
    }
    *start += clock() - start_1;
    npy_intp remain = TotalLoopSize % threadpool->NumThreads;
    if (remain)
    {
        data[0].LoopSize = remain;
        data[0].DataPtrA = DataPtrA;
        data[0].DataPtrB = DataPtrB;
        data[0].ResultDataPtr = ResultDataPtr;
        ComputeBroadcast(0, &data[0]);
    }
    free(data);
}