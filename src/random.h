#ifndef RTENSOR_H
#define RTENSOR_H
#include "tensor.h"
#include "pcg_basic.h"

Tensor *random(PyObject *self, PyObject *const *args, size_t nargsf);
// Tensor *randn(Tensor *t, float mean, float std);
// Tensor *random_bernoulli(Tensor *t, float p);
// Tensor *randint(Tensor *t, float median, float sigma);    
// Tensor *rand(Tensor *t, float lambda);
// Tensor *random_geometric(Tensor *t, float p);
// Tensor *random_logistic(Tensor *t, float median, float scale);
void seed(PyObject *num);
void init();

typedef struct
{
    int thread_id;
    pcg32_random_t rng;

} RNG_POOL;

#endif