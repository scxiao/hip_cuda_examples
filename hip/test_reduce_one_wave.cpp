#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <init_vec.hpp>

__global__ void reduce_lds(__half *g_idata, __half *g_odata) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float sdata[];
    sdata[tid] = g_idata[i];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


void test_reduce_lds(const std::vector<__half> &in, std::vector<__half> &out) {
    auto in_size = in.size() * sizeof(__half);
    int block_size = 64;
    auto out_size = in_size / block_size;
    out.resize(out_size / sizeof(__half));

    __half *in_d, *out_d;
    hipMalloc((void**)&in_d, in_size);
    hipMalloc((void**)&out_d, out_size);

    hipMemcpy(in_d, in.data(), in_size, hipMemcpyHostToDevice);
    int block_num = in.size() / block_size;
    reduce_lds<<<block_num, block_size, block_size * sizeof(float)>>>(in_d, out_d);
    hipMemcpy((void*)out.data(), out_d, out_size, hipMemcpyDeviceToHost);   
}

__global__ void reduce_bpermute(__half *g_idata, __half *g_odata) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float sdata[];
    sdata[tid] = g_idata[i];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

void golden(const std::vector<__half> &in, std::vector<__half> &out) {
    float sum = 0;
    for (auto v : in) {
        sum += (float)v;
    }
    out.push_back(sum);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " n" << std::endl;
        return 0;
    }

    int sn = atoi(argv[1]);
    size_t n = (1 << sn);

    // hard coded to 64 for now
    n = 64;
    std::cout << "Element num: " << n << std::endl;
    srand(time(nullptr));
    std::vector<__half> in_vec, out_vec, golden_vec;
    init_vec(in_vec, n);

    test_reduce_lds(in_vec, out_vec);
    golden(in_vec, golden_vec);
    std::cout << "out = " << (float)out_vec.at(0) << std::endl;
    std::cout << "golden = " << (float)golden_vec.at(0) << std::endl;

    return 0;
}
