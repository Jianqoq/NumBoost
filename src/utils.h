
#ifndef UTILS_H
#define UTILS_H
#include "stdbool.h"
#include "import_methods.h"
#endif

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
    long *__notin = malloc(sizeof(long) * len);
#ifdef DEBUG
    DEBUG_PRINT("list len: %ld\n", list_len);
    DEBUG_PRINT("list= [");
    for (int i = 0; i < list_len; i++)
    {
        DEBUG_PRINT("%ld ", list[i]);
    }
    DEBUG_PRINT("]\n");
#endif
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
#ifdef DEBUG
    DEBUG_PRINT("excluding_list= [");
    for (int i = 0; i < len; i++)
    {
        DEBUG_PRINT("%ld ", __notin[i]);
    }
    DEBUG_PRINT("]\n");
#endif
    return __notin;
}