#ifndef __INIT_VEC_HPP__
#define __INIT_VEC_HPP__

#include <ctime>
#include <cmath>
#include <climits>

template<class T>
void init_vec(std::vector<T>& vec, std::size_t num)
{
    vec.resize(num);
    srand(time(nullptr));
    for (size_t i = 0; i < num; ++i)
    {
        float v = 1.0f * rand() / INT_MAX;
        vec[i] = v;
    }
}

#endif

