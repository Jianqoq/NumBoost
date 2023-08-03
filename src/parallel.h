#include "import_methods.h"
#include <windows.h>

typedef enum
{
    ADD,
    SUB,
    MUL,
    DIV,
    MOD,
    POW,
} op_type;

typedef struct
{
    char *DataPtrA;
    char *DataPtrB;
    char *ResultDataPtr;
    npy_intp StridesA;
    npy_intp StridesB;
    npy_intp LoopSize;
    DWORD ThreadID;
    int type;
    int op;
} BroadcastDATA;

typedef struct
{
    PTP_POOL pool;
    TP_CALLBACK_ENVIRON_V3 environment;
    DWORD NumThreads;
} BroadcastThreadpool;

#define ComputeBroadcastBranch(type, op_enum) \
    switch (op_enum)                          \
    {                                         \
    case ADD:                                 \
        Calculate(type, +);                   \
        break;                                \
    case SUB:                                 \
        Calculate(type, -);                   \
        break;                                \
    case MUL:                                 \
        Calculate(type, *);                   \
        break;                                \
    case DIV:                                 \
        Calculate(type, /);                   \
        break;                                \
    default:                                  \
        break;                                \
    }

#define Calculate(type, op)                                                            \
    {                                                                                  \
        type *ResultDataPtr = (type *)data->ResultDataPtr;                             \
        for (npy_intp i = 0; i < data->LoopSize; i++)                                  \
        {                                                                              \
            *ResultDataPtr = (*((type *)data->DataPtrB))op(*((type *)data->DataPtrA)); \
            ResultDataPtr++;                                                           \
            data->DataPtrA += data->StridesA;                                          \
            data->DataPtrB += data->StridesB;                                          \
        }                                                                              \
    }

BroadcastThreadpool *CreateBroadcastThreadpool(DWORD numThreads);

void DestroyBroadcastThreadpool(BroadcastThreadpool *threadpool);

void BroadcastParallel(BroadcastThreadpool *threadpool, char **DataPtrA, char **DataPtrB, char **ResultDataPtr, int type, int op,
                       npy_intp StridesA, npy_intp StridesB, npy_intp TotalLoopSize, clock_t *start);