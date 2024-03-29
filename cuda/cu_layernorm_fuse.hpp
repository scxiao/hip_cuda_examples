#ifndef __CUDA_LAYER_NORM_FUSE_HPP__
#define __CUDA_LAYER_NORM_FUSE_HPP__

#include <vector>
#include <cuda_fp16.h>


float layernorm_fuse_half2_wrapper(const std::vector<__half>& in,
                                    const std::vector<__half>& w,
                                    const std::vector<__half>& bias,
                                    std::vector<float>& mean,
                                    std::vector<float>& var,
                                    std::vector<__half>& out,
                                    int batch_size,
                                    int repeat_num);

float layernorm_fuse_half_wrapper(const std::vector<__half>& in, 
                                    const std::vector<__half>& w,
                                    const std::vector<__half>& bias,
                                    std::vector<float>& mean,
                                    std::vector<float>& var,
                                    std::vector<__half>& out,
                                    int batch_size,
                                    int repeat_num);

#endif
