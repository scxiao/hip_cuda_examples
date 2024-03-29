#include "cu_vec_add.hpp"
#include "timer.hpp"
#include <cuda.h>
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

    half2* x1 = (half2*)in1;
    half2* x2 = (half2*)in2;
    half2* r = (half2*)res;
    n = n / 2;
    int i = tid;
    if (tid < n)
    {
        //res[i] = __float2half(__half2float(in1[i]) + __half2float(in2[i]));
        //res[i] = __hadd(in1[i], in2[i]);
        //float r = (float)res[i];
        r[i] = __hadd2(x1[i], x2[i]);
    }

    return;
}

__global__ void vec_add(float *in1, float *in2, float *res, int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    //for (int i = tid; i < n; i += nglobal)
    int i = tid;
    if (i < n)
    {
        res[i] = in1[i] + in2[i];
    }

    return;
}

__global__ void vec_add(double *in1, double *in2, double *res, int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    //for (int i = tid; i < n; i += nglobal)
    int i = tid;
    if (i < n)
    {
        res[i] = in1[i] + in2[i];
    }

    return;
}

inline
cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    std::abort();
  }
  return result;
}

template<class T>
bool cu_vec_addT(const std::vector<T>& in1, const std::vector<T>& in2, std::vector<T>& res) {
    if (in1.size() != in2.size())
    {
        std::cout << "Input vector sizes are different!" << std::endl;
        return false;
    }

    int n = in1.size();
    res.resize(n);
    std::size_t mem_size = n * sizeof(T);
    T *cu_in1, *cu_in2, *cu_res;
    cudaMalloc((void **)&cu_in1, mem_size);
    cudaMalloc((void **)&cu_in2, mem_size);
    cudaMalloc((void **)&cu_res, mem_size);

    cudaMemcpy(cu_in1, in1.data(), mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_in2, in2.data(), mem_size, cudaMemcpyHostToDevice);

    float *cache;
    std::size_t cache_size = (1 << 28);
    cudaMalloc((void**)&cache, cache_size);

    // warm up for 10 times
    std::size_t block_size = 1024;
    for (int i = 0; i < 10; ++i)
    {
        cudaMemset(cache, 0, cache_size);
        vec_add<<<((n/2 - 1) / block_size + 1), block_size>>>(cu_in1, cu_in2, cu_res, n);
    }
    checkCuda(cudaDeviceSynchronize());

    int repeat_num = 50;
    HRTimer timer;
    timer.start();
    for (int i = 0; i < repeat_num; ++i) {
        cudaMemset(cache, 0, cache_size);
        vec_add<<<((n - 1) / block_size + 1), block_size>>>(cu_in1, cu_in2, cu_res, n);
    }
    checkCuda(cudaDeviceSynchronize());
    timer.stop();

    cudaMemcpy(res.data(), cu_res, mem_size, cudaMemcpyDeviceToHost);
    cudaFree(cu_in1); cudaFree(cu_in2); cudaFree(cu_res);

    std::size_t usec = timer.gettime_us();
    mem_size *= 3;
    mem_size += cache_size;
    float throughput = 1.0f * mem_size * repeat_num / usec / 1.0e3;
    std::cout << typeid(T).name() << " vec_add, time = " << usec << "us, throughput = " << throughput << "GB/s" << std::endl;

    return true;
}

bool cu_vec_add1(const std::vector<__half>& in1, const std::vector<__half>& in2, std::vector<__half>& res) {
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
    cudaMalloc((void **)&cu_in1, mem_size);
    cudaMalloc((void **)&cu_in2, mem_size);
    cudaMalloc((void **)&cu_res, mem_size);

    cudaMemcpy(cu_in1, in1.data(), mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_in2, in2.data(), mem_size, cudaMemcpyHostToDevice);

    float *cache;
    std::size_t cache_size = (1 << 28);
    cudaMalloc((void**)&cache, cache_size);

    // warm up for 10 times
    std::size_t block_size = 1024;
    for (int i = 0; i < 10; ++i)
    {
        cudaMemset(cache, 0, cache_size);
        vec_add<<<((n/2 - 1) / block_size + 1), block_size>>>(cu_in1, cu_in2, cu_res, n);
    }
    checkCuda(cudaDeviceSynchronize());

    int repeat_num = 50;
    HRTimer timer;
    timer.start();
    for (int i = 0; i < repeat_num; ++i) {
        cudaMemset(cache, 0, cache_size);
        vec_add<<<((n/2 - 1) / block_size + 1), block_size>>>(cu_in1, cu_in2, cu_res, n);
    }
    checkCuda(cudaDeviceSynchronize());
    timer.stop();

    cudaMemcpy(res.data(), cu_res, mem_size, cudaMemcpyDeviceToHost);
    cudaFree(cu_in1); cudaFree(cu_in2); cudaFree(cu_res);

    std::size_t usec = timer.gettime_us();
    mem_size *= 3;
    mem_size += cache_size;
    float throughput = 1.0f * mem_size * repeat_num / usec / 1.0e3;
    std::cout << typeid(T).name() << " vec_addH, time = " << usec << "us, throughput = " << throughput << "GB/s" << std::endl;

    return true;
}


bool cu_vec_add(const std::vector<__half>& in1, const std::vector<__half>& in2, std::vector<__half>& res)
{
    return cu_vec_addT(in1, in2, res);
}

bool cu_vec_addH(const std::vector<__half>& in1, const std::vector<__half>& in2, std::vector<__half>& res)
{
    return cu_vec_add1(in1, in2, res);
}


bool cu_vec_add(const std::vector<float>& in1, const std::vector<float>& in2, std::vector<float>& res)
{
    return cu_vec_addT(in1, in2, res);
}

bool cu_vec_add(const std::vector<double>& in1, const std::vector<double>& in2, std::vector<double>& res)
{
    return cu_vec_addT(in1, in2, res);
}


