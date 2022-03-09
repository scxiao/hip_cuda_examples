#ifndef __HIP_REDUCTION_HPP__
#define __HIP_REDUCTION_HPP__
#include <string>
#include <vector>
#include <hip/hip_fp16.h>

void calc_mem_throughput(const std::string& prefix, int in_size, int out_size, double us_num);

bool reduction0(const std::vector<float>& in, std::vector<float>& out);
bool reduction1(const std::vector<float>& in, std::vector<float>& out);
bool reduction2(const std::vector<float>& in, std::vector<float>& out);
bool reduction3(const std::vector<float>& in, std::vector<float>& out);
bool reduction4(const std::vector<float>& in, std::vector<float>& out);

bool reduction0(const std::vector<__half>& in, std::vector<__half>& out);
bool reduction1(const std::vector<__half>& in, std::vector<__half>& out);
bool reduction2(const std::vector<__half>& in, std::vector<__half>& out);
bool reduction3(const std::vector<__half>& in, std::vector<__half>& out);
bool reduction4(const std::vector<__half>& in, std::vector<__half>& out);


#endif

