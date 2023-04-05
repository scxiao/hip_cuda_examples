#ifndef __COMMON_LAYERNORM_FUSE_HPP__
#define __COMMON_LAYERNORM_FUSE_HPP__

#include <vector>
#if (defined(__NVCC__))
#include <cuda_fp16.h>
#else
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

void layernorm_fuse_baseline(const std::vector<__half>& in,
                        const std::vector<__half>& w,
                        const std::vector<__half>& bias,
                        std::vector<float>& mean,
                        std::vector<float>& var,
                        std::vector<__half>& out,
                        int batch_size);

#endif
