#ifndef __HIP_SQRT_HPP__
#define __HIP_SQRT_HPP__

#include <vector>
#include <hip/hip_fp16.h>

using namespace std;

bool hip_sqrt(const std::vector<__half>& in, std::vector<__half>& res);
bool hip_sqrth2(const std::vector<__half>& in, std::vector<__half>& res);
bool hip_sqrt(const std::vector<float>& in, std::vector<float>& res);
bool hip_sqrt(const std::vector<double>& in, std::vector<double>& res);

#endif

