#include <numpy/arrayobject.h>

typedef void (ConvertFunc)(PyArrayObject **array);

ConvertFunc *type_converter(int type);