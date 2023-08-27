
#ifndef utils_h
#define utils_h
#include "stdbool.h"
#include "import_module_methods.h"
#include "numboost_api.h"
#define NO_IMPORT_ARRAY

inline bool notin(long i, long *list, long range)
{
    int count = 0;
    if (list == NULL)
        return true;
    for (int j = 0; j < range; j++)
    {
        if (i == list[j])
        {
            count++;
        }
    }
    DEBUG_PRINT("count: %d\n", count);
    if (count == 0)
        return true;
    else
        return false;
}

inline long *range_excluding_list(long start, int end, long *list, long pad, long list_len, long *real_len)
{
    long len = abs(end - start);
    long *__notin = (long *)malloc(sizeof(long) * len);
    for (long i = 0; i < len; i++)
    {
        if (notin(i, list, list_len))
        {
            DEBUG_PRINT("__notin[%ld] = %ld\n", i, i);
            __notin[i] = i;
            (*real_len)++;
        }
        else
        {
            DEBUG_PRINT("__notin[%ld] = %ld\n", i, pad);
            __notin[i] = pad;
        }
    }
    return __notin;
}

#define Numboost_CheckAlloc(ptr) \
    do                                       \
    {                                        \
        if (ptr == NULL)                     \
        {                                    \
            PyErr_NoMemory();                \
            return NULL;                     \
        }                                    \
    } while (0)

#endif