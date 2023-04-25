#include "hip_vec_add.hpp"
#include "hip/hip_runtime.h"
#include "timer.hpp"
#include <typeinfo>
#include <cassert>

using namespace std;



// half data type
__global__ 
void vec_add(__half *in1, __half *in2, __half *res, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        res[i] = __float2half(__half2float(in1[i]) + __half2float(in2[i]));
    }

    return;
}


__global__ void vec_add(float *in1, float *in2, float *res, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        res[i] = in1[i] + in2[i];
    }

    return;
}

__global__ void vec_add(double *in1, double *in2, double *res, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        res[i] = in1[i] + in2[i];
    }

    return;
}

// half2 data type
__global__ 
void vec_addh2(__half *in1, __half *in2, __half *res, int n)
{
    half2* h1 = (half2*)in1;
    half2* h2 = (half2*)in2;
    half2 *r = (half2*)res;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
       r[i] = __hadd2(h1[i], h2[i]);
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
bool hip_vec_addT(const std::vector<T>& in1, const std::vector<T>& in2, std::vector<T>& res) {
    if (in1.size() != in2.size())
    {
        std::cout << "Input vector sizes are different!" << std::endl;
        return false;
    }

    int n = in1.size();
    res.resize(n);
    std::size_t mem_size = n * sizeof(T);
    T *cu_in1, *cu_in2, *cu_res;
    hipMalloc((void **)&cu_in1, mem_size);
    hipMalloc((void **)&cu_in2, mem_size);
    hipMalloc((void **)&cu_res, mem_size);

    hipMemcpy(cu_in1, in1.data(), mem_size, hipMemcpyHostToDevice);
    hipMemcpy(cu_in2, in2.data(), mem_size, hipMemcpyHostToDevice);
    //hipMemcpy(res.data(), in2.data(), mem_size, hipMemcpyDeviceToHost);

    std::size_t cache_size = (1 << 28);
    float *cache_d;
    hipMalloc((void**)&cache_d, cache_size);

    std::size_t block_size = 1024;
    // warm up to run the kernel for 10 times
    for (int i = 0; i < 10; ++i) {
        hipMemset(cache_d, 0, cache_size);
        hipLaunchKernelGGL(vec_add, dim3(((n - 1) / block_size + 1)),
                            dim3(block_size), 0, 0,
                            cu_in1, cu_in2, cu_res, n);
    }
    checkHip(hipDeviceSynchronize());

    HRTimer timer;
    int repeat_num = 50;
    timer.start();
    for (int i = 0; i < repeat_num; ++i) {
        hipMemset(cache_d, 0, cache_size);
        // vec_add<<<((n - 1) / block_size + 1), block_size>>>(cu_in1, cu_in2, cu_res, n);
        hipLaunchKernelGGL(vec_add, dim3(((n - 1) / block_size + 1)),
                            dim3(block_size), 0, 0,
                            cu_in1, cu_in2, cu_res, n);
    }
    checkHip(hipDeviceSynchronize());
    timer.stop();

    hipMemcpy(res.data(), cu_res, mem_size, hipMemcpyDeviceToHost);
    hipFree(cu_in1); hipFree(cu_in2); hipFree(cu_res);

    std::size_t usec = timer.gettime_us();
    mem_size *= 3;
    mem_size += cache_size;
    float throughput = mem_size * repeat_num / usec / 1.0e3;
    std::cout << typeid(T).name() << " vec_add, time = " << usec << "us, throughput = " << throughput << "GB/s" << std::endl;

    return true;
}

bool hip_vec_addh2(const std::vector<__half>& in1, const std::vector<__half>& in2, std::vector<__half>& res) {
    if (in1.size() != in2.size())
    {
        std::cout << "Input vector sizes are different!" << std::endl;
        return false;
    }

    using T = __half;

    int n = in1.size();
    res.resize(n);
    std::size_t mem_size = n * sizeof(T);
    T *cu_in1, *cu_in2, *cu_res;
    hipMalloc((void **)&cu_in1, mem_size);
    hipMalloc((void **)&cu_in2, mem_size);
    hipMalloc((void **)&cu_res, mem_size);

    std::size_t cache_size = (1 << 28);
    float *cache_d;
    hipMalloc((void**)&cache_d, cache_size);

    hipMemcpy(cu_in1, in1.data(), mem_size, hipMemcpyHostToDevice);
    hipMemcpy(cu_in2, in2.data(), mem_size, hipMemcpyHostToDevice);
    //hipMemcpy(res.data(), in2.data(), mem_size, hipMemcpyDeviceToHost);

    // warm up
    std::size_t block_size = 1024;
    for (int i = 0; i < 10; ++i) {
        // vec_addh2<<<((n/2- 1) / block_size + 1), block_size>>>(cu_in1, cu_in2, cu_res, n);
        hipLaunchKernelGGL(vec_addh2, dim3(((n/2 - 1) / block_size + 1)),
                            dim3(block_size), 0, 0,
                            cu_in1, cu_in2, cu_res, n/2);

    }
    checkHip(hipDeviceSynchronize());

    HRTimer timer;
    timer.start();
    // repeat 100 times
    int repeat_num = 50;
    for (int i = 0; i < repeat_num; ++i) {
        // invalidate cache
        hipMemset(cache_d, 0, cache_size);
        // vec_addh2<<<((n/2- 1) / block_size + 1), block_size>>>(cu_in1, cu_in2, cu_res, n);
        hipLaunchKernelGGL(vec_addh2, dim3(((n/2 - 1) / block_size + 1)),
                            dim3(block_size), 0, 0,
                            cu_in1, cu_in2, cu_res, n/2);
    }
    checkHip(hipDeviceSynchronize());
    timer.stop();
    hipMemcpy(res.data(), cu_res, mem_size, hipMemcpyDeviceToHost);
    hipFree(cu_in1); hipFree(cu_in2); hipFree(cu_res);

    mem_size *= 3;
    mem_size += cache_size;
    std::size_t usec = timer.gettime_us();
    float throughput = mem_size * repeat_num / usec / 1000.0f ;
    std::cout << "half2, vec_add time = " << usec << "us, throughput = " << throughput << "GB/s" << std::endl;

    return true;
}


bool hip_vec_add(const std::vector<__half>& in1, const std::vector<__half>& in2, std::vector<__half>& res)
{
    return hip_vec_addT(in1, in2, res);
}

bool hip_vec_add(const std::vector<float>& in1, const std::vector<float>& in2, std::vector<float>& res)
{
    return hip_vec_addT(in1, in2, res);
}

bool hip_vec_add(const std::vector<double>& in1, const std::vector<double>& in2, std::vector<double>& res)
{
    return hip_vec_addT(in1, in2, res);
}


