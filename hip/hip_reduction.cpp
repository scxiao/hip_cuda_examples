#include "timer.hpp"
#include "hip_reduction.hpp"
#include "hip/hip_runtime.h"

void calc_mem_throughput(const std::string& prefix, int in_size, int out_size, std::size_t us_num)
{
    double throughput = 1.0 * (in_size + out_size) / us_num / 1000.0;
    std::cout << prefix << " time: " << us_num << "us, throughput: " << throughput << "GB/s" << std::endl;
}

// Reduction 0
template<class T>
__global__ void reduce0(T* g_idata, T* g_odata)
{
    extern __shared__ T sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template<>
__global__ void reduce0(__half* g_idata, __half* g_odata)
{
    extern __shared__ __half sdatah[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdatah[tid] = g_idata[i];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0)
        {
            sdatah[tid] += sdatah[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdatah[0];
}


template<class T>
bool reduction0(const std::vector<T>& in, std::vector<T>& out)
{
    auto in_size = sizeof(T) * in.size();
    int block_size = 1024;
    auto out_size = in_size / block_size;
    out.resize(out_size);

    HRTimer timer;

    T* in_d, *out_d;
    hipMalloc((void**)&in_d, in_size);
    hipMalloc((void**)&out_d, out_size);

    hipMemcpy(in_d, in.data(), in_size, hipMemcpyHostToDevice);
    int block_num = in.size() / block_size;
    std::cout << "block_num = " << block_num << ", block_size = " << block_size << std::endl;
    timer.start();
    reduce0<T><<<block_num, block_size, block_size * sizeof(T)>>>(in_d, out_d);
    hipDeviceSynchronize();
    timer.stop();
    auto num_us = timer.gettime_us();
    calc_mem_throughput("Reduction0", in_size, out_size, num_us);

    hipMemcpy(out.data(), out_d, out_size, hipMemcpyDeviceToHost);

    return true;
}

// Reduction 1
template<class T>
__global__ void reduce1(T* g_idata, T* g_odata)
{
    extern __shared__ T sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * tid;
        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template<>
__global__ void reduce1(__half* g_idata, __half* g_odata)
{
    extern __shared__ __half sdatah[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdatah[tid] = g_idata[i];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * tid;
        if (index < blockDim.x)
        {
            sdatah[index] += sdatah[index + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdatah[0];
}


template<class T>
bool reduction1(const std::vector<T>& in, std::vector<T>& out)
{
    auto in_size = sizeof(T) * in.size();
    int block_size = 1024;
    auto out_size = in_size / block_size;
    out.resize(out_size);

    HRTimer timer;

    T* in_d, *out_d;
    hipMalloc((void**)&in_d, in_size);
    hipMalloc((void**)&out_d, out_size);

    hipMemcpy(in_d, in.data(), in_size, hipMemcpyHostToDevice);
    int block_num = in.size() / block_size;
    std::cout << "block_num = " << block_num << ", block_size = " << block_size << std::endl;
    timer.start();
    reduce1<T><<<block_num, block_size, block_size * sizeof(T)>>>(in_d, out_d);
    hipDeviceSynchronize();
    timer.stop();
    auto num_us = timer.gettime_us();
    calc_mem_throughput("Reduction1", in_size, out_size, num_us);

    hipMemcpy(out.data(), out_d, out_size, hipMemcpyDeviceToHost);

    return true;
}

// Reduction 2
template<class T>
__global__ void reduce2(T* g_idata, T* g_odata)
{
    extern __shared__ T sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s /= 2)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template<>
__global__ void reduce2(__half* g_idata, __half* g_odata)
{
    extern __shared__ __half sdatah[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdatah[tid] = g_idata[i];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s /= 2)
    {
        if (tid < s)
        {
            sdatah[tid] += sdatah[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdatah[0];
}

template<class T>
bool reduction2(const std::vector<T>& in, std::vector<T>& out)
{
    auto in_size = sizeof(T) * in.size();
    int block_size = 1024;
    auto out_size = in_size / block_size;
    out.resize(out_size);

    HRTimer timer;

    T* in_d, *out_d;
    hipMalloc((void**)&in_d, in_size);
    hipMalloc((void**)&out_d, out_size);

    hipMemcpy(in_d, in.data(), in_size, hipMemcpyHostToDevice);
    int block_num = in.size() / block_size;
    std::cout << "block_num = " << block_num << ", block_size = " << block_size << std::endl;
    timer.start();
    reduce2<T><<<block_num, block_size, block_size * sizeof(T)>>>(in_d, out_d);
    hipDeviceSynchronize();
    timer.stop();
    auto num_us = timer.gettime_us();
    calc_mem_throughput("Reduction2", in_size, out_size, num_us);

    hipMemcpy(out.data(), out_d, out_size, hipMemcpyDeviceToHost);

    return true;
}

// Reduction 3
template<class T>
__global__ void reduce3(T* g_idata, T* g_odata)
{
    extern __shared__ T sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s /= 2)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template<>
__global__ void reduce3(__half* g_idata, __half* g_odata)
{
    extern __shared__ __half sdatah[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    sdatah[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s /= 2)
    {
        if (tid < s)
        {
            sdatah[tid] += sdatah[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdatah[0];
}


template<class T>
bool reduction3(const std::vector<T>& in, std::vector<T>& out)
{
    auto in_size = sizeof(T) * in.size();
    int block_size = 1024;
    auto out_size = in_size / block_size;
    out.resize(out_size);

    HRTimer timer;

    T* in_d, *out_d;
    hipMalloc((void**)&in_d, in_size);
    hipMalloc((void**)&out_d, out_size);

    hipMemcpy(in_d, in.data(), in_size, hipMemcpyHostToDevice);
    int block_num = in.size() / block_size;
    std::cout << "block_num = " << block_num << ", block_size = " << block_size << std::endl;
    timer.start();
    reduce3<T><<<block_num, block_size / 2, block_size * sizeof(T) / 2>>>(in_d, out_d);
    hipDeviceSynchronize();
    timer.stop();
    auto num_us = timer.gettime_us();
    calc_mem_throughput("Reduction3", in_size, out_size, num_us);

    hipMemcpy(out.data(), out_d, out_size, hipMemcpyDeviceToHost);

    return true;
}

// Reduction 4
template<class T>
__device__ void warpReduce(T *sdata, int tid)
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

template<>
__device__ void warpReduce(__half *sdatav, int tid)
{
    __half* sdata = (__half*)sdatav;
    sdata[tid] += sdata[tid + 32];
    __syncthreads();
    sdata[tid] += sdata[tid + 16];
    __syncthreads();
    sdata[tid] += sdata[tid + 8];
    __syncthreads();
    sdata[tid] += sdata[tid + 4];
    __syncthreads();
    sdata[tid] += sdata[tid + 2];
    __syncthreads();
    sdata[tid] += sdata[tid + 1];
    __syncthreads();

    //sdata[tid] = __float2half(__half2float(sdata[tid]) +  __half2float(sdata[tid + 32]));
    //sdata[tid] = __float2half(__half2float(sdata[tid]) +  __half2float(sdata[tid + 16]));
    //sdata[tid] = __float2half(__half2float(sdata[tid]) +  __half2float(sdata[tid + 8]));
    //sdata[tid] = __float2half(__half2float(sdata[tid]) +  __half2float(sdata[tid + 4]));
    //sdata[tid] = __float2half(__half2float(sdata[tid]) +  __half2float(sdata[tid + 2]));
    //sdata[tid] = __float2half(__half2float(sdata[tid]) +  __half2float(sdata[tid + 1]));
}

template<class T>
__global__ void reduce4(T* g_idata, T* g_odata)
{
    extern __shared__ T sdata4[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    sdata4[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s /= 2)
    {
        if (tid < s)
        {
            sdata4[tid] += sdata4[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) warpReduce(sdata4, tid);

    if (tid == 0) g_odata[blockIdx.x] = sdata4[0];
}

template<>
__global__ void reduce4(__half* g_idata, __half* g_odata)
{
    extern __shared__ __half sdatah[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    sdatah[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s /= 2)
    {
        if (tid < s)
        {
            sdatah[tid] += sdatah[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) warpReduce(sdatah, tid);

    if (tid == 0) g_odata[blockIdx.x] = sdatah[0];
}

template<class T>
bool reduction4(const std::vector<T>& in, std::vector<T>& out)
{
    auto in_size = sizeof(T) * in.size();
    int block_size = 1024;
    auto out_size = in_size / block_size;
    out.resize(out_size);

    HRTimer timer;

    T* in_d, *out_d;
    hipMalloc((void**)&in_d, in_size);
    hipMalloc((void**)&out_d, out_size);

    hipMemcpy(in_d, in.data(), in_size, hipMemcpyHostToDevice);
    int block_num = in.size() / block_size;
    std::cout << "block_num = " << block_num << ", block_size = " << block_size << std::endl;
    timer.start();
    reduce4<T><<<block_num, block_size / 2, block_size * sizeof(T) / 2>>>(in_d, out_d);
    hipDeviceSynchronize();
    timer.stop();
    auto num_us = timer.gettime_us();
    calc_mem_throughput("Reduction4", in_size, out_size, num_us);

    hipMemcpy(out.data(), out_d, out_size, hipMemcpyDeviceToHost);

    return true;
}

bool reduction0(const std::vector<float>& in, std::vector<float>& out)
{
    return reduction0<float>(in, out);
}

bool reduction1(const std::vector<float>& in, std::vector<float>& out)
{
    return reduction1<float>(in, out);
}

bool reduction2(const std::vector<float>& in, std::vector<float>& out)
{
    return reduction2<float>(in, out);
}

bool reduction3(const std::vector<float>& in, std::vector<float>& out)
{
    return reduction3<float>(in, out);
}

bool reduction4(const std::vector<float>& in, std::vector<float>& out)
{
    return reduction4<float>(in, out);
}



bool reduction0(const std::vector<__half>& in, std::vector<__half>& out)
{
    return reduction0<__half>(in, out);
}

bool reduction1(const std::vector<__half>& in, std::vector<__half>& out)
{
    return reduction1<__half>(in, out);
}

bool reduction2(const std::vector<__half>& in, std::vector<__half>& out)
{
    return reduction2<__half>(in, out);
}

bool reduction3(const std::vector<__half>& in, std::vector<__half>& out)
{
    return reduction3<__half>(in, out);
}

bool reduction4(const std::vector<__half>& in, std::vector<__half>& out)
{
    return reduction4<__half>(in, out);
}

