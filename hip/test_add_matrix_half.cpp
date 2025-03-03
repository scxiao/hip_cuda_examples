// #include "hip_vec_add.hpp"
// #include "hip/hip_runtime.h"
// #include "timer.hpp"
#include <iostream>
#include <typeinfo>
#include <cassert>
#include <unistd.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <utilities.hpp>

using namespace std;

#define BLOCK_M 16
#define BLOCK_N 64


template<class T>
void init_vec(std::vector<T>& vec, std::size_t num)
{
    vec.resize(num);
    srand(time(nullptr));
    // srand(1);
    for (size_t i = 0; i < num; ++i)
    {
        float v = 10.0 * rand() / (1.0 * RAND_MAX);
        vec[i] = v;
    }
}


__global__ void kernel_add_matrix(__half *a, __half *b, __half *c, int m, int n)
{
    extern __shared__ __half lds[];

    __half *sa = lds;
    __half *sb = lds + BLOCK_M * BLOCK_N;

    // read input matrix to lds
    int blk_i = blockIdx.x * BLOCK_M;
    int blk_j = blockIdx.y * BLOCK_N;

    int tid = threadIdx.y;
    for (int i = 0; i < BLOCK_M; ++i) {
        for (int j = tid; j < BLOCK_N; j += blockDim.y) {
            sa[i * BLOCK_N + j] = a[(blk_i + i) * n + blk_j + j];
            sb[i * BLOCK_N + j] = b[(blk_i + i) * n + blk_j + j];
        }
    }
    __syncthreads();

    for (int i = 0; i < BLOCK_M; ++i) {
        for (int j = tid; j < BLOCK_N; j += blockDim.y) {
            __half sum = sa[i * BLOCK_N + j] + sb[i * BLOCK_N + j];
            c[(blk_i + i) * n + blk_j + j] = sum;
        }
    }

    return;
}

inline
hipError_t checkHip(hipError_t result)
{
  if (result != hipSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", hipGetErrorString(result));
    std::abort();
  }
  return result;
}

template<class T>
bool hip_add_matrix(const std::vector<T>& in1, const std::vector<T>& in2, std::vector<T>& res, int m, int n) {
    if (in1.size() != in2.size() or in1.size() != m * n)
    {
        std::cout << "Input matrix sizes are different!" << std::endl;
        return false;
    }

    std::size_t mem_size = m * n * sizeof(T);
    T *cu_in1, *cu_in2, *cu_res;
    hipMalloc((void **)&cu_in1, mem_size);
    hipMalloc((void **)&cu_in2, mem_size);
    hipMalloc((void **)&cu_res, mem_size);

    hipMemcpy(cu_in1, in1.data(), mem_size, hipMemcpyHostToDevice);
    hipMemcpy(cu_in2, in2.data(), mem_size, hipMemcpyHostToDevice);

    const std::size_t lds_size = sizeof(T) * BLOCK_M * BLOCK_N;
    // warm up to run the kernel for 10 times
    // for (int i = 0; i < 10; ++i) {
        hipLaunchKernelGGL(kernel_add_matrix, dim3(m / BLOCK_M, n / BLOCK_N),
                            dim3(1, BLOCK_N), 2 * lds_size, 0,
                            cu_in1, cu_in2, cu_res, m, n);
    // }
    checkHip(hipDeviceSynchronize());

    hipMemcpy((void *)res.data(), cu_res, mem_size, hipMemcpyDeviceToHost);
    hipFree(cu_in1); hipFree(cu_in2); hipFree(cu_res);

    return true;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " m, n" << std::endl;
        return 1;
    }

    int logm2 = std::atoi(argv[1]);
    int logn2 = std::atoi(argv[2]);

    int m = 1 << logm2;
    int n = 1 << logn2;
    std::cout << "m = " << m << ", n = " << n << std::endl;

    std::vector<__half> a, b;
    std::vector<__half> c;
    c.resize(m * n);
    init_vec(a, m * n);
    usleep(2000000);
    init_vec(b, m * n);

    hip_add_matrix(a, b, c, m, n);

    for (int i = 0; i < 10; ++i) {
        std::cout << "a = " << (float)a[i];
        std::cout << ", b = " << (float)b[i];
        std::cout << ", c = " << (float)c[i] << std::endl;
    }

    return 0;
}
