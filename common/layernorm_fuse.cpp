#include <iostream>
#include <numeric>
#include <algorithm>
#include <cmath>

#include "layernorm_fuse.hpp"

static void layernorm_one_block(int bid, int elem_num,
                           const std::vector<__half>& in,
                           const std::vector<__half>& w,
                           const std::vector<__half>& bias,
                           float& m,
                           float& var,
                           std::vector<__half>& out) 
{
    int offset = bid * elem_num;
    std::vector<float> fin(in.begin() + offset, in.begin() + offset + elem_num);
    float sum = std::accumulate(fin.begin(), fin.end(), 0.0f);
    m = sum / elem_num;
    std::transform(fin.begin(), fin.end(), fin.begin(), [&](auto v) {
        return v - m;
    });
    std::vector<float> vv(elem_num);
    std::transform(fin.begin(), fin.end(), vv.begin(), [](auto v) {
        return v * v;
    });
    var = std::accumulate(vv.begin(), vv.end(), 0.0f);
    var /= elem_num;
    var += 1.0e-12;
    var = 1.0f / std::sqrt(var);

    for (size_t i = 0; i < elem_num; ++i) {
        out[offset + i] = (fin[i] * var) * __half2float(w[i]) + __half2float(bias[i]);
    }
}

// baseline implementation of the layernorm
void layernorm_fuse_baseline(const std::vector<__half>& in,
                        const std::vector<__half>& w,
                        const std::vector<__half>& bias,
                        std::vector<float>& mean,
                        std::vector<float>& var,
                        std::vector<__half>& out,
                        int batch_size) {
    int block_num = in.size() / batch_size;
    mean.resize(block_num);
    var.resize(block_num);
    out.resize(in.size());
    for (int bid = 0; bid < block_num; ++bid) {
        layernorm_one_block(bid, batch_size,
                           in,
                           w,
                           bias,
                           mean[bid],
                           var[bid],
                           out);
    }
}
