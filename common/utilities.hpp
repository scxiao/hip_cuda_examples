#ifndef __UTILITIES_HPP__
#define __UTILITIES_HPP__

#include <iostream>
#include <vector>
#if (defined(__NVCC__))
#include <cuda_fp16.h>
#else
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

template<class T>
bool compare(const std::vector<T>& vec1, std::vector<T>& vec2, float eps = 0.001f)
{
    if (vec1.size() != vec2.size())
    {
        std::cout << "vec1_size " << vec1.size() << " != vec2_size " << vec2.size() << std::endl;
        return false;
    }

    auto size = vec1.size();
    bool ret = true;
    for (int i = 0; i < size; ++i)
    {
        float v1 = static_cast<float>(vec1[i]);
        float v2 = static_cast<float>(vec2[i]);
        float diff = v2 - v1;
        if (std::abs(v1) > 1.0f) {
            diff = std::fabs(diff/v1);
        }
        if (diff > eps)
        {
            std::cout << "(x[" << i << "] = " << v1 << ") != (y[" << i << "] = " << v2 << ")" << std::endl;
            ret = false;
        }
    }

    return ret;
}

template<class T>
void print(std::ostream& os, const std::vector<T>& vec) {
    char c = '{';
    int i = 0;
    for (auto &v : vec) {
        os << c;
        if ((i > 0) and (i % 16 == 0)) os << '\n';
        i++;
        os << v;
        if (c == '{') c = ',';
    }
    os << '}';
}

template<class T>
std::ostream& operator<< (std::ostream& os, const std::vector<T>& vec) {
    print(os, vec);
    return os;
}

#endif

