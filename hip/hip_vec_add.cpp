#include "hip_vec_add.hpp"
#include "hip/hip_runtime.h"
#include "timer.hpp"
#include <typeinfo>
#include <cassert>

using namespace std;

//__global__
//void haxpy(int n, half a, const half *x, half *y)
//{
//	int start = threadIdx.x + blockDim.x * blockIdx.x;
//	int stride = blockDim.x * gridDim.x;
//
//#if __CUDA_ARCH__ >= 530
//  int n2 = n/2;
//  half2 *x2 = (half2*)x, *y2 = (half2*)y;
//
//  for (int i = start; i < n2; i+= stride) 
//    y2[i] = __hfma2(__halves2half2(a, a), x2[i], y2[i]);
//
//	// first thread handles singleton for odd arrays
//  if (start == 0 && (n%2))
//  	y[n-1] = __hfma(a, x[n-1], y[n-1]);   
//
//#else
//  for (int i = start; i < n; i+= stride) {
//    y[i] = __float2half(__half2float(a) * __half2float(x[i]) 
//      + __half2float(y[i]));
//  }
//#endif
//}

__global__ 
void vec_add(__half *in1, __half *in2, __half *res, int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int nglobal = blockDim.x * gridDim.x;

	//__half2* x1 = (__half2*)in1;
    //__half2* x2 = (__half2*)in2;
    //__half2* r = (__half2*)res;
    for (int i = tid; i < n; i += nglobal)
    {
        res[i] = __float2half(__half2float(in1[i]) + __half2float(in2[i]));
        //res[i] = __hadd(in1[i], in2[i]);
        float r = (float)res[i];
        //r[i] = __hadd(x1[i], x2[i]);
    }

    return;
}

__global__ void vec_add(float *in1, float *in2, float *res, int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int nglobal = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += nglobal)
    {
        res[i] = in1[i] + in2[i];
    }

    return;
}

__global__ void vec_add(double *in1, double *in2, double *res, int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int nglobal = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += nglobal)
    {
        res[i] = in1[i] + in2[i];
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
    hipMemcpy(res.data(), in2.data(), mem_size, hipMemcpyDeviceToHost);

    std::size_t block_size = 1024;
    HRTimer timer;
    timer.start();
    vec_add<<<((n - 1) / block_size + 1), block_size>>>(cu_in1, cu_in2, cu_res, n);
    checkHip(hipDeviceSynchronize());
    timer.stop();

    hipMemcpy(res.data(), cu_res, mem_size, hipMemcpyDeviceToHost);
    hipFree(cu_in1); hipFree(cu_in2); hipFree(cu_res);

    std::size_t usec = timer.gettime_us();
    float throughput = 3.0f * mem_size / usec / 1.0e3;
    std::cout << typeid(T).name() << " vec_add, time = " << usec << "us, throughput = " << throughput << "GB/s" << std::endl;

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


