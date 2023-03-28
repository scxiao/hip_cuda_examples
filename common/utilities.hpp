#ifndef __UTILITIES_HPP__
#define __UTILITIES_HPP__

#include <vector>
#if (defined(__NVCC__))
#include <cuda_fp16.h>
#else
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

template<class T>
bool compare(const std::vector<T>& vec1, std::vector<T>& vec2)
{
    if (vec1.size() != vec2.size())
    {
        std::cout << "vec1_size " << vec1.size() << " != vec2_size " << vec2.size() << std::endl;
        return false;
    }

    auto size = vec1.size();
    float eps = 0.001f;
    bool ret = true;
    for (int i = 0; i < size; ++i)
    {
        float v1 = static_cast<float>(vec1[i]);
        float v2 = static_cast<float>(vec2[i]);
        if (v1 - v2 > eps or v1 - v2 < -eps)
        {
            std::cout << "(x[" << i << "] = " << v1 << ") != (y[" << i << "] = " << v2 << ")" << std::endl;
            ret = false;
        }
    }

    return ret;
}

#endif

