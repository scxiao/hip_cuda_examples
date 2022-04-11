#include "hip_sqrt.hpp"
#include "hip/hip_runtime.h"
#include "timer.hpp"
#include <typeinfo>
#include <cassert>

using namespace std;



// half data type
__global__ 
void vec_sqrt(__half *in, __half *res, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        res[i] = hsqrt(in[i]);
        //res[i] = hrsqrt(in[i]);
    }

    return;
}


__global__ void vec_sqrt(float *in, float *res, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        res[i] = __fsqrt_rn(in[i]);
        //res[i] = __frsqrt_rn(in[i]);
    }

    return;
}

__global__ void vec_sqrt(double *in, double *res, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        res[i] = sqrt(in[i]);
        //res[i] = rsqrt(in[i]);
    }

    return;
}

// half2 data type
__global__ 
void vec_sqrth2(__half *in, __half *res, int n)
{
    half2* h = (half2*)in;
    half2 *r = (half2*)res;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
       r[i] = h2sqrt(h[i]);
       //r[i] = h2rsqrt(h[i]);
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
bool hip_sqrtT(const std::vector<T>& in, std::vector<T>& res) {
    int n = in.size();
    res.resize(n);
    std::size_t mem_size = n * sizeof(T);
    T *cu_in, *cu_res;
    hipMalloc((void **)&cu_in, mem_size);
    hipMalloc((void **)&cu_res, mem_size);

    hipMemcpy(cu_in, in.data(), mem_size, hipMemcpyHostToDevice);
    std::size_t block_size = 512;
    HRTimer timer;
    timer.start();
    vec_sqrt<<<((n - 1) / block_size + 1), block_size>>>(cu_in, cu_res, n);
    checkHip(hipDeviceSynchronize());
    timer.stop();

    hipMemcpy(res.data(), cu_res, mem_size, hipMemcpyDeviceToHost);
    hipFree(cu_in); hipFree(cu_res);

    std::size_t usec = timer.gettime_us();
    float throughput = 2.0f * mem_size / usec / 1.0e3;
    std::cout << typeid(T).name() << " sqrt, time = " << usec << "us, throughput = " << throughput << "GB/s" << std::endl;

    return true;
}

bool hip_sqrth2(const std::vector<__half>& in, std::vector<__half>& res) {
    using T = __half;
    int n = in.size();
    res.resize(n);
    std::size_t mem_size = n * sizeof(T);
    T *cu_in, *cu_res;
    hipMalloc((void **)&cu_in, mem_size);
    hipMalloc((void **)&cu_res, mem_size);

    hipMemcpy(cu_in, in.data(), mem_size, hipMemcpyHostToDevice);

    std::size_t block_size = 512;
    HRTimer timer;
    timer.start();
    vec_sqrth2<<<((n/2 - 1) / block_size + 1), block_size>>>(cu_in, cu_res, n);
    checkHip(hipDeviceSynchronize());
    timer.stop();

    hipMemcpy(res.data(), cu_res, mem_size, hipMemcpyDeviceToHost);
    hipFree(cu_in); hipFree(cu_res);

    std::size_t usec = timer.gettime_us();
    float throughput = 2.0f * mem_size / usec / 1.0e3;
    std::cout << typeid(T).name() << " sqrt, time = " << usec << "us, throughput = " << throughput << "GB/s" << std::endl;

    return true;
}


bool hip_sqrt(const std::vector<__half>& in, std::vector<__half>& res)
{
    return hip_sqrtT(in, res);
}

bool hip_sqrt(const std::vector<float>& in, std::vector<float>& res)
{
    return hip_sqrtT(in, res);
}

bool hip_sqrt(const std::vector<double>& in, std::vector<double>& res)
{
    return hip_sqrtT(in, res);
}


