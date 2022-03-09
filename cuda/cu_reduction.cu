#include "timer.hpp"
#include "cu_reduction.hpp"

void calc_mem_throughput(const std::string& prefix, int in_size, int out_size, std::size_t us_num)
{
    double throughput = 1.0 * (in_size + out_size) / us_num / 1000.0;
    std::cout << prefix << " time: " << us_num << "us, throughput: " << throughput << "GB/s" << std::endl;
}

// Reduction 0
__global__ void reduce0(float* g_idata, float* g_odata)
{
    extern __shared__ float sdata[];
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

bool reduction0(const std::vector<float>& in, std::vector<float>& out)
{
    auto in_size = sizeof(float) * in.size();
    int block_size = 1024;
    auto out_size = in_size / block_size;
    out.resize(out_size);

    HRTimer timer;

    float* in_d, *out_d;
    cudaMalloc((void**)&in_d, in_size);
    cudaMalloc((void**)&out_d, out_size);

    cudaMemcpy(in_d, in.data(), in_size, cudaMemcpyHostToDevice);
    int block_num = in.size() / block_size;
    std::cout << "block_num = " << block_num << ", block_size = " << block_size << std::endl;
    timer.start();
    reduce0<<<block_num, block_size, block_size * sizeof(float)>>>(in_d, out_d);
    cudaDeviceSynchronize();
    timer.stop();
    auto num_us = timer.gettime_us();
    calc_mem_throughput("Reduction0", in_size, out_size, num_us);

    cudaMemcpy(out.data(), out_d, out_size, cudaMemcpyDeviceToHost);

    return true;
}

// Reduction 1
__global__ void reduce1(float* g_idata, float* g_odata)
{
    extern __shared__ float sdata[];
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

bool reduction1(const std::vector<float>& in, std::vector<float>& out)
{
    auto in_size = sizeof(float) * in.size();
    int block_size = 1024;
    auto out_size = in_size / block_size;
    out.resize(out_size);

    HRTimer timer;

    float* in_d, *out_d;
    cudaMalloc((void**)&in_d, in_size);
    cudaMalloc((void**)&out_d, out_size);

    cudaMemcpy(in_d, in.data(), in_size, cudaMemcpyHostToDevice);
    int block_num = in.size() / block_size;
    std::cout << "block_num = " << block_num << ", block_size = " << block_size << std::endl;
    timer.start();
    reduce1<<<block_num, block_size, block_size * sizeof(float)>>>(in_d, out_d);
    cudaDeviceSynchronize();
    timer.stop();
    auto num_us = timer.gettime_us();
    calc_mem_throughput("Reduction1", in_size, out_size, num_us);

    cudaMemcpy(out.data(), out_d, out_size, cudaMemcpyDeviceToHost);

    return true;
}

// Reduction 2
__global__ void reduce2(float* g_idata, float* g_odata)
{
    extern __shared__ float sdata[];
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

bool reduction2(const std::vector<float>& in, std::vector<float>& out)
{
    auto in_size = sizeof(float) * in.size();
    int block_size = 1024;
    auto out_size = in_size / block_size;
    out.resize(out_size);

    HRTimer timer;

    float* in_d, *out_d;
    cudaMalloc((void**)&in_d, in_size);
    cudaMalloc((void**)&out_d, out_size);

    cudaMemcpy(in_d, in.data(), in_size, cudaMemcpyHostToDevice);
    int block_num = in.size() / block_size;
    std::cout << "block_num = " << block_num << ", block_size = " << block_size << std::endl;
    timer.start();
    reduce2<<<block_num, block_size, block_size * sizeof(float)>>>(in_d, out_d);
    cudaDeviceSynchronize();
    timer.stop();
    auto num_us = timer.gettime_us();
    calc_mem_throughput("Reduction2", in_size, out_size, num_us);

    cudaMemcpy(out.data(), out_d, out_size, cudaMemcpyDeviceToHost);

    return true;
}

// Reduction 3
__global__ void reduce3(float* g_idata, float* g_odata)
{
    extern __shared__ float sdata[];
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

bool reduction3(const std::vector<float>& in, std::vector<float>& out)
{
    auto in_size = sizeof(float) * in.size();
    int block_size = 1024;
    auto out_size = in_size / block_size;
    out.resize(out_size);

    HRTimer timer;

    float* in_d, *out_d;
    cudaMalloc((void**)&in_d, in_size);
    cudaMalloc((void**)&out_d, out_size);

    cudaMemcpy(in_d, in.data(), in_size, cudaMemcpyHostToDevice);
    int block_num = in.size() / block_size;
    std::cout << "block_num = " << block_num << ", block_size = " << block_size << std::endl;
    timer.start();
    reduce3<<<block_num, block_size / 2, block_size * sizeof(float) / 2>>>(in_d, out_d);
    cudaDeviceSynchronize();
    timer.stop();
    auto num_us = timer.gettime_us();
    calc_mem_throughput("Reduction3", in_size, out_size, num_us);

    cudaMemcpy(out.data(), out_d, out_size, cudaMemcpyDeviceToHost);

    return true;
}

// Reduction 4
__device__ void warpReduce(volatile float *sdata, int tid)
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduce4(float* g_idata, float* g_odata)
{
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s /= 2)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) warpReduce(sdata, tid);

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

bool reduction4(const std::vector<float>& in, std::vector<float>& out)
{
    auto in_size = sizeof(float) * in.size();
    int block_size = 1024;
    auto out_size = in_size / block_size;
    out.resize(out_size);

    HRTimer timer;

    float* in_d, *out_d;
    cudaMalloc((void**)&in_d, in_size);
    cudaMalloc((void**)&out_d, out_size);

    cudaMemcpy(in_d, in.data(), in_size, cudaMemcpyHostToDevice);
    int block_num = in.size() / block_size;
    std::cout << "block_num = " << block_num << ", block_size = " << block_size << std::endl;
    timer.start();
    reduce4<<<block_num, block_size / 2, block_size * sizeof(float) / 2>>>(in_d, out_d);
    cudaDeviceSynchronize();
    timer.stop();
    auto num_us = timer.gettime_us();
    calc_mem_throughput("Reduction4", in_size, out_size, num_us);

    cudaMemcpy(out.data(), out_d, out_size, cudaMemcpyDeviceToHost);

    return true;
}

