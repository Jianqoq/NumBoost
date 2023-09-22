#ifndef NUMBOOST_SORT_UTILS_H
#define NUMBOOST_SORT_UTILS_H

inline int compare(const void *a, const void *b) {
  int int_a = *((int *)a);
  int int_b = *((int *)b);

  if (int_a == int_b)
    return 0;
  else if (int_a < int_b)
    return -1;
  else
    return 1;
}

#endif // NUMBOOST_SORT_UTILS_H