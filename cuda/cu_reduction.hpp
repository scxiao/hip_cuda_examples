#ifndef __CU_REDUCTION_HPP__
#define __CU_REDUCTION_HPP__
#include <string>
#include <vector>

void calc_mem_throughput(const std::string& prefix, int in_size, int out_size, double us_num);

bool reduction0(const std::vector<float>& in, std::vector<float>& out);
bool reduction1(const std::vector<float>& in, std::vector<float>& out);
bool reduction2(const std::vector<float>& in, std::vector<float>& out);
bool reduction3(const std::vector<float>& in, std::vector<float>& out);
bool reduction4(const std::vector<float>& in, std::vector<float>& out);

#endif

