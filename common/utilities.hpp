#ifndef __UTILITIES_HPP__
#define __UTILITIES_HPP__

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
        if (vec1[i] - vec2[i] > eps or vec1[i] - vec2[i] < -eps)
        {
            std::cout << "(x[" << i << "] = " << vec1[i] << ") != (y[" << i << "] = " << vec2[i] << ")" << std::endl;
            ret = false;
        }
    }

    return ret;
}

#endif

