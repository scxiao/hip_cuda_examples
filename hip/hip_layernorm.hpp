#ifndef __HIP_LAYER_NORM_HPP__
#define __HIP_LAYER_NORM_HPP__

#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

void layernorm_half2_wrapper(const std::vector<__half>& in, 
                                    std::vector<__half>& out,
                                    int batch_size);

void layernorm_half_wrapper(const std::vector<__half>& in, 
                                    std::vector<__half>& out,
                                    int batch_size);

#endif
