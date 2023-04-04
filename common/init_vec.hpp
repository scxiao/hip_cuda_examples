#ifndef __INIT_VEC_HPP__
#define __INIT_VEC_HPP__

#include <ctime>
#include <cmath>
#include <climits>
#include <vector>

#if (defined(__NVCC__))
#include <cuda_fp16.h>
#else
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

template<class T>
void init_vec(std::vector<T>& vec, std::size_t num)
{
    vec.resize(num);
    //srand(time(nullptr));
    srand(1);
    for (size_t i = 0; i < num; ++i)
    {
        float v = 1.0 * rand() / (1.0 * RAND_MAX);
        vec[i] = v;
    }
}

#endif

