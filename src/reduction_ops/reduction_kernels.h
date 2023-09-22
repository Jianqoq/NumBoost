#ifndef _REDUCTION_KERNELS_H
#define _REDUCTION_KERNELS_H

#define Empty_Post(...)
#define Empty_Pre(...)

#define Mean_Main(generic_type, type, result_data_ptr, a_data_ptr, result_idx, \
                  a_idx, a_stride)                                             \
  generic_type a_val = Promote(type, a_data_ptr[a_idx * a_stride]);            \
  result_data_ptr[result_idx] +=                                               \
      Demote(type, (a_val / (inner_loop_size * inner_loop_size_2)));

#define Sum_Main(generic_type, type, result_data_ptr, a_data_ptr, result_idx,  \
                 a_idx, a_stride)                                              \
  generic_type a_val = Promote(type, a_data_ptr[a_idx * a_stride]);            \
  result_data_ptr[result_idx] += Demote(type, a_val);

#define Min_Main(generic_type, type, result_data_ptr, a_data_ptr, result_idx,  \
                 a_idx, a_stride)                                              \
  generic_type a_val = Promote(type, a_data_ptr[a_idx * a_stride]);            \
  generic_type result_val = Promote(type, result_data_ptr[result_idx]);        \
  if (a_val < result_val)                                                      \
    result_data_ptr[result_idx] = Demote(type, a_val);

#define Max_Main(generic_type, type, result_data_ptr, a_data_ptr, result_idx,  \
                 a_idx, a_stride)                                              \
  generic_type a_val = Promote(type, a_data_ptr[a_idx * a_stride]);            \
  generic_type result_val = Promote(type, result_data_ptr[result_idx]);        \
  if (a_val > result_val)                                                      \
    result_data_ptr[result_idx] = Demote(type, a_val);

#define ArgMax_Main(generic_type, type, index, val, result_data_ptr,           \
                    a_data_ptr, result_idx, a_idx, a_stride)                   \
  generic_type a_val = Promote(type, a_data_ptr[a_idx * a_stride]);            \
  if (a_val > val) {                                                           \
    val = a_val;                                                               \
    *result_data_ptr = a_idx;                                                  \
  }

#define ArgMin_Main(generic_type, type, index, val, result_data_ptr,           \
                    a_data_ptr, result_idx, a_idx, a_stride)                   \
  generic_type a_val = Promote(type, a_data_ptr[a_idx * a_stride]);            \
  if (a_val < val) {                                                           \
    val = a_val;                                                               \
    *result_data_ptr = a_idx;                                                  \
  }

#endif
