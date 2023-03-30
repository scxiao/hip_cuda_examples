#include <iostream>
#include <init_vec.hpp>
#include <utilities.hpp>

#include "hip_layernorm_fuse.hpp"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " batchs batch_size" << std::endl;
        return 0;
    }

    size_t batches = std::atoi(argv[1]);
    size_t batch_size = std::atoi(argv[2]);

    size_t elem_num = batches * batch_size;

    std::vector<__half> in, w, bias;
    init_vec(in, elem_num);
    init_vec(w, batches);
    init_vec(bias, batches);

    std::vector<__half> out_half, out_half2;
    std::vector<float> mean_half, mean_half2;
    std::vector<float> var_half, var_half2;

    layernorm_fuse_half2_wrapper(in, w, bias, mean_half2, var_half2, out_half2, batch_size);
    layernorm_fuse_half_wrapper(in, w, bias, mean_half, var_half, out_half, batch_size);

    bool ret = true;
    if (not compare(mean_half, mean_half2)) {
        std::cout << "MEAN output failed!" << std::endl;
        ret = false;
    }

    if (not compare(var_half, var_half2)) {
        std::cout << "VAR output failed!" << std::endl;
        ret = false;
    }

    if (not compare(out_half, out_half2, 0.001f)) {
        std::cout << "OUTPUT failed!" << std::endl;
        ret = false;
    }

    std::cout << (ret ? "PASSED" : "FAILED") << std::endl;

    return 0;
}
