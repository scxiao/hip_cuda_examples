#include <iostream>
#include <fstream>
#include <numeric>
#include <init_vec.hpp>
#include <utilities.hpp>

#include "cu_layernorm_fuse.hpp"
#include "layernorm_fuse.hpp"

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
    init_vec(w, batch_size);
    init_vec(bias, batch_size);

    std::vector<__half> out_half, out_half2, outb;
    std::vector<float> mean_half, mean_half2, meanb;
    std::vector<float> var_half, var_half2, varb;

    layernorm_fuse_baseline(in, w, bias, meanb, varb, outb, batch_size);

    float thrpt2 = layernorm_fuse_half2_wrapper(in, w, bias, mean_half2, var_half2, out_half2, batch_size, 50);
    std::cout << "Half2 throughput = \t" << thrpt2 << "\t(GB/s)" << std::endl;

#define PRINT_TO_FILE
#ifdef PRINT_TO_FILE
    std::ofstream ofs("../thrpt_half2.txt", std::ofstream::app);
    ofs << batch_size << ", " << thrpt2 << std::endl;
    ofs.close();
#endif

    float thrpt = layernorm_fuse_half_wrapper(in, w, bias, mean_half, var_half, out_half, batch_size, 50);
    std::cout << "Half throughput = \t" << thrpt << "\t(GB/s)" << std::endl;

    if (not (compare(meanb, mean_half) and compare(meanb, mean_half2))) {
        std::cout << "MEAN output failed!" << std::endl;
        return 1;
    }
    else {
        std::cout << "MEAN output correct!" << std::endl;
    }

    if (not (compare(varb, var_half, 0.01f) and compare(varb, var_half2, 0.01f))) {
        std::cout << "VAR output failed!" << std::endl;
        return 1;
    }
    else {
        std::cout << "VAR output correct!" << std::endl;
    }

    if (not (compare(outb, out_half, 0.01f) and compare(outb, out_half2, 0.01f))) {
        std::cout << "OUTPUT failed!" << std::endl;
    }
    else {
        std::cout << "OUTPUT correct!" << std::endl;
    }

    std::cout << "PASSED" << std::endl;

    return 0;
}
