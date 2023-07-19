#ifndef SET_TENSOR_PROPERTIES_H
#define SET_TENSOR_PROPERTIES_H
#include "tensor.h"
#endif

void Tensor_SetData(Tensor *self, PyObject *data);
void Tensor_SetX(Tensor *self, PyObject *x);
void Tensor_SetY(Tensor *self, PyObject *y);
void Tensor_SetGrad(Tensor *self, PyObject *grad);
void Tensor_SetGraph(Tensor *self, PyObject *graph);
void Tensor_SetDtype(Tensor *self, PyArray_Descr *dtype);
void Tensor_SetGradFn(Tensor *self, const char *grad_fn);
void Tensor_SetRequireGrad(Tensor *self, bool require_grad);
void Tensor_SetVars(Tensor *self, unsigned long long vars);
void Tensor_SetDim(Tensor *self, int dim);
void Tensor_SetAxis(Tensor *self, PyObject *axis);
void Tensor_SetHasConv(Tensor *self, int has_conv);
void Tensor_SetData_without_init_value(Tensor *self, PyObject *data);
void Tensor_SetX_without_init_value(Tensor *self, PyObject *x);
void Tensor_SetY_without_init_value(Tensor *self, PyObject *y);
void Tensor_SetGrad_without_init_value(Tensor *self, PyObject *grad);
void Tensor_SetGraph_without_init_value(Tensor *self, PyObject *graph);
void Tensor_SetDtype_without_init_value(Tensor *self, PyArray_Descr *dtype);
void Tensor_SetAxis_without_init_value(Tensor *self, PyObject *axis);
void Tensor_SetData_startwone(Tensor *self, PyObject *data);
void Tensor_SetX_startwone(Tensor *self, PyObject *x);
void Tensor_SetY_startwone(Tensor *self, PyObject *y);
void Tensor_SetGrad_startwone(Tensor *self, PyObject *grad);
void Tensor_SetGraph_startwone(Tensor *self, PyObject *graph);
void Tensor_SetDtype_startwone(Tensor *self, PyArray_Descr *dtype);
void Tensor_SetAxis_startwone(Tensor *self, PyObject *axis);
void Tensor_SetData_startwone_without_init(Tensor *self, PyObject *data);
void Tensor_SetX_startwone_without_init(Tensor *self, PyObject *x);
void Tensor_SetY_startwone_without_init(Tensor *self, PyObject *y);
void Tensor_SetGrad_startwone_without_init(Tensor *self, PyObject *grad);
void Tensor_SetGraph_startwone_without_init(Tensor *self, PyObject *graph);
void Tensor_SetDtype_startwone_without_init(Tensor *self, PyArray_Descr *dtype);
void Tensor_SetAxis_startwone_without_init(Tensor *self, PyObject *axis);