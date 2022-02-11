#ifndef __HIP_VEC_ADD_HPP__
#define __HIP_VEC_ADD_HPP__

#include <vector>
#include <hip/hip_fp16.h>

using namespace std;

bool hip_vec_add(const std::vector<__half>& in1, const std::vector<__half>& in2, std::vector<__half>& res);
bool hip_vec_addH(const std::vector<__half>& in1, const std::vector<__half>& in2, std::vector<__half>& res);
bool hip_vec_add(const std::vector<float>& in1, const std::vector<float>& in2, std::vector<float>& res);
bool hip_vec_add(const std::vector<double>& in1, const std::vector<double>& in2, std::vector<double>& res);

#endif

