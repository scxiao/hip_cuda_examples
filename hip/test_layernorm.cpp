#include <iostream>
#include <init_vec.hpp>
#include <utilities.hpp>

#include "hip_layernorm_fuse.hpp"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " batchs batch_size" << std::endl;
    }

    size_t batches = std::atoi(argv[1]);
    size_t batch_size = std::atoi(argv[2]);

    size_t elem_num = batches * batch_size;

    std::vector<__half> in, out_half2, out_half;
    init_vec(in, elem_num);

    triadd_layernorm_half2_wrapper(in, out_half2, batch_size);
    triadd_layernorm_half_wrapper(in, out_half, batch_size);

    if (compare(out_half, out_half2)) {
        std::cout << "PASSED!" << std::endl;
    }
    else {
        std::cout << "FAILED!" << std::endl;
    }

    return 0;
}
