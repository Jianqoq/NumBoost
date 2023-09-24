#ifndef UFUNC_KERNELS_H
#define UFUNC_KERNELS_H

#define Where_LoopBody(generic_type, in_type, out_type, i, result_ptr,         \
                       mask_arr_stride, x_stride, y_stride, mask_arr_ptr,      \
                       x_ptr, y_ptr)                                           \
  generic_type mask_val = Promote(in_type, mask_arr_ptr[i * mask_arr_stride]); \
  generic_type x_val = Promote(in_type, x_ptr[i * x_stride]);                  \
  generic_type y_val = Promote(in_type, y_ptr[i * y_stride]);                  \
  if (mask_val)                                                                \
    result_ptr[i] = Demote(in_type, x_val);                                    \
  else                                                                         \
    result_ptr[i] = Demote(in_type, y_val);

#endif // UFUNC_KERNELS_H