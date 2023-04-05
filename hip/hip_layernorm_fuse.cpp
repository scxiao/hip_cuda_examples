#include <iostream>
#include <vector>
#include <numeric>
#include <functional>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "timer.hpp"
#include "hip_layernorm_fuse.hpp"

static size_t compute_block_size(int n, int max_block_size)
{
    size_t block_size = 64;
    while(block_size < max_block_size and block_size < n)
        block_size *= 2;
    return block_size;
}

static inline
hipError_t checkHip(hipError_t result)
{
  if (result != hipSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", hipGetErrorString(result));
    std::abort();
  }
  return result;
}

#define DEVICE_CONSTEXPR constexpr __device__ __host__ // NOLINT

/////////////////////////// Half data type ////////////////////////////////////

template<class T>
__device__ T block_reduce_half(T* buffer, int batch_size)
{
    __syncthreads();
    int block_size = blockDim.x;
    int tid = threadIdx.x;
    for(int s = block_size/2; s > 0; s /= 2)
    {
        if(tid < s and tid + s < batch_size and tid + s < block_size)
        {
            buffer[tid] += buffer[tid + s];
        }
        __syncthreads();
    }

    return buffer[0];
}

// m = x - mean(x)
// m / sqrt(mean(m ^ 2) + 1e-12)
__device__ void layernorm_kernel_half(__half* input,
                                      float* in_data,
                                      __half* w,
                                      __half* b,
                                      float* m_data,
                                      float* v_data,
                                      __half* out,
                                      int batch_size,
                                      float rnum)
{
    int block_size = blockDim.x;
    // initialize the data, this is for reduce_sum
    const int start = blockIdx.x * batch_size;
    in_data[threadIdx.x] = 0;
    for(int i = threadIdx.x; i < batch_size; i += block_size)
    {
        int idx = i + start;
        in_data[threadIdx.x] += __half2float(input[idx]);
    }

    auto m = block_reduce_half(in_data, batch_size);
    m *= rnum;
    float mv = m;
    if (threadIdx.x == 0) {
        m_data[blockIdx.x] = m;
    }

    __syncthreads();
    in_data[threadIdx.x] = 0;
    for(int i = threadIdx.x; i < batch_size; i += block_size)
    {
        int idx    = i + start;
        float d = __half2float(input[idx]) - mv;
        in_data[threadIdx.x] += (d * d);
    }

    auto vv = block_reduce_half(in_data, batch_size);
    vv *= rnum;
    vv += 1.0e-12f;
    auto rstd = rsqrt(vv);
    if (threadIdx.x == 0) {
        v_data[blockIdx.x] = rstd;
    }

    for(int i = threadIdx.x; i < batch_size; i += block_size)
    {
        int idx  = i + start;
        float o2 = (__half2float(input[idx]) - mv) * rstd;
        out[idx] = __float2half(o2 * __half2float(w[i]) + __half2float(b[i]));
    }
}

#define BLOCK_SIZE 1024
// m = x - mean(x)
// m / sqrt(mean(m ^ 2) + 1e-12)
__global__ void layernorm_half(void* in, void *w, void *b, float *m, float *v, void* data_out, int batch_size)
{
    __half* input = reinterpret_cast<__half*>(in);
    __half* ww = reinterpret_cast<__half*>(w);
    __half* bb = reinterpret_cast<__half*>(b);
    __half* output = reinterpret_cast<__half*>(data_out);
    float rnum     = 1.0f / batch_size;
    extern __shared__ float bufferh[];
    float* in_data        = bufferh;

    layernorm_kernel_half(input, in_data, ww, bb, m, v, output, batch_size, rnum);
}

void calc_layernorm_fuse_half(void* in_d,
                               void* w_d,
                               void* bias_d,
                               float* mean_d,
                               float* var_d,
                               void* out_d,
                               int block_num,
                               int batch_size,
                               int block_size,
                               int shared_size)
{
    layernorm_half<<<block_num, block_size, shared_size>>>(
        in_d, w_d, bias_d, mean_d, var_d, out_d, batch_size);
}

//////////////////////////////////  half2 data type /////////////////////////////

struct half2_sum
{
    DEVICE_CONSTEXPR __half2 operator()(__half2 x, __half2 y) const { return __hadd2(x, y); }
};

// in_data is in shared memory
template <class Op>
__device__ __half2 block_reduce_half2(
    __half2* buffer, int batch_item_num, int tid, int block_size, Op op)
{
    __syncthreads();
    for(int s = block_size; s > 0; s >>= 1)
    {
        if(tid < s and tid + s < batch_item_num and tid + s < block_size)
        {
            buffer[tid] = op(buffer[tid], buffer[tid + s]);
        }
        __syncthreads();
    }

    auto lows2  = __low2half2(buffer[0]);
    auto highs2 = __high2half2(buffer[0]);

    return op(lows2, highs2);
}

// m = x - mean(x)
// m / sqrt(mean(m ^ 2) + 1e-12)
__device__ void layernorm_kernel_half2(__half2* input,
                                       __half2* in_data,
                                       __half2* w,
                                       __half2* b,
                                       float*   m_data,
                                       float*   v_data,
                                       __half2* out,
                                       int batch_size,
                                       int block_size,
                                       float rbatch_num)
{
    auto rnum = __float2half2_rn(rbatch_num);
    const int bid = blockIdx.x;
    const int start = bid * batch_size;
    const int tid = threadIdx.x;
    in_data[tid] = 0;
    for(int i = tid; i < batch_size; i += block_size)
    {
        int idx           = i + start;
        in_data[tid] = __hadd2(in_data[tid], input[idx]);
    }

    auto m =
        block_reduce_half2(in_data, batch_size, tid, block_size, half2_sum{});
    m = __hmul2(m, rnum);
    if (tid == 0) {
        m_data[bid] = __low2float(m);
    }
    __half2 mv = m;

    __syncthreads();
    in_data[tid] = 0;
    for(int i = tid; i < batch_size; i += block_size)
    {
        int idx           = i + start;
        __half2 diff = __hsub2(input[idx], mv);
        in_data[tid] = __hadd2(in_data[tid], __hmul2(diff, diff));
    }

    m = block_reduce_half2(in_data, batch_size, tid, block_size, half2_sum{});
    m = __hmul2(m, rnum);

    auto eps = __float2half2_rn(1.0e-12f);
    auto r   = __hadd2(m, eps);
    r        = h2rsqrt(r);
    if (tid == 0) {
        v_data[blockIdx.x] = __low2float(r);
    }

    for(int i = tid; i < batch_size; i += block_size)
    {
        int idx  = i + start;
        auto o2 = __hmul2(__hsub2(input[idx], mv), r);
        out[idx] = __hadd2(__hmul2(o2, w[i]), b[i]);
    }
}

__global__ void layernorm_half2(void* in, void* w, void* b, float* m, float *v, void* data_out, int batch_size, int block_size)
{
    __half2* input = reinterpret_cast<__half2*>(in);
    __half2* ww = reinterpret_cast<__half2*>(w);
    __half2* bb = reinterpret_cast<__half2*>(b);

    __half2* output = reinterpret_cast<__half2*>(data_out);
    float rnum      = 1.0f / batch_size;
    batch_size /= 2;
    extern __shared__ __half2 buffer2[];
    __half2* in_data        = buffer2;

    layernorm_kernel_half2(input, in_data, ww, bb, m, v, output, batch_size, block_size, rnum);
}

void calc_layernorm_fuse_half2(void* in_d,
                               void* w_d,
                               void* bias_d,
                               float* mean_d,
                               float* var_d,
                               void* out_d,
                               int block_num,
                               int batch_size,
                               int block_size,
                               int shared_size)
{
    // block_size /= 2;
    layernorm_half2<<<block_num, block_size, shared_size>>>(
        in_d, w_d, bias_d, mean_d, var_d, out_d, batch_size, block_size);

}

// Wrapper functions
using func = std::function<void(void* in_d,
                               void* w_d,
                               void* bias_d,
                               float* mean_d,
                               float* var_d,
                               void* out_d,
                               int block_num,
                               int batch_size,
                               int block_size,
                               int shared_size)>;

float layernorm_fuse_wrapper(func fn,
                             const std::vector<__half>& in,
                             const std::vector<__half>& w,
                             const std::vector<__half>& bias,
                             std::vector<float>& mean,
                             std::vector<float>& var,
                             std::vector<__half>& out,
                             int batch_size,
                             int repeat_num = 50)
{
    int elem_num = in.size();
    out.resize(elem_num);
    int block_num         = elem_num / batch_size;
    auto block_size       = compute_block_size(batch_size, 1024);
    int shared_size       = block_size * 2 * sizeof(float);
    mean.resize(block_num);
    var.resize(block_num);

    __half *in_d, *out_d;
    size_t io_size = elem_num * sizeof(__half);
    hipMalloc((void**)&in_d, io_size);
    hipMalloc((void**)&out_d, io_size);

    __half *w_d, *b_d;
    size_t wb_size = batch_size * sizeof(__half);
    hipMalloc((void**)&w_d, wb_size);
    hipMalloc((void**)&b_d, wb_size);
    float *mean_d, *var_d;
    size_t mv_size = block_num * sizeof(float);
    hipMalloc((void**)&mean_d, mv_size);
    hipMalloc((void**)&var_d, mv_size);

    size_t cache_size = 256 * 1024 * 1024;
    void* cache;
    hipMalloc((void**)&cache, cache_size);
    hipMemcpy(in_d, in.data(), io_size, hipMemcpyHostToDevice);
    hipMemcpy(w_d, w.data(), wb_size, hipMemcpyHostToDevice);
    hipMemcpy(b_d, bias.data(), wb_size, hipMemcpyHostToDevice);

    // warm up for 10 times of calculation
    for (int i = 0; i < 10; ++i) {
        fn(in_d, w_d, b_d, mean_d, var_d, out_d,
           block_num, batch_size, block_size, shared_size);
    }
    checkHip(hipDeviceSynchronize());

    HRTimer timer;
    timer.start();
    for (int i = 0; i < repeat_num; ++i) {    
        fn(in_d, w_d, b_d, mean_d, var_d, out_d,
           block_num, batch_size, block_size, shared_size);
    }
    checkHip(hipDeviceSynchronize());
    timer.stop();

    size_t us = timer.gettime_us();
    size_t data_size = 2 * elem_num * sizeof(__half);
    float throughput = 1.0 * repeat_num * data_size / us * 1.0e-3; // GB/s

    hipMemcpy((void*)mean.data(), mean_d, mv_size, hipMemcpyDeviceToHost);
    hipMemcpy((void*)var.data(), var_d, mv_size, hipMemcpyDeviceToHost);
    hipMemcpy((void*)out.data(), out_d, io_size, hipMemcpyDeviceToHost);

    hipFree(in_d);
    hipFree(out_d);
    hipFree(w_d);
    hipFree(b_d);
    hipFree(mean_d);
    hipFree(var_d);
    hipFree(cache);

    return throughput;
}

float layernorm_fuse_half2_wrapper(const std::vector<__half>& in, 
                                    const std::vector<__half>& w,
                                    const std::vector<__half>& bias,
                                    std::vector<float>& mean,
                                    std::vector<float>& var,
                                    std::vector<__half>& out,
                                    int batch_size,
                                    int repeat_num = 50)
{
    return layernorm_fuse_wrapper(calc_layernorm_fuse_half2,
                in, w, bias, mean, var, out, batch_size, repeat_num);
}

float layernorm_fuse_half_wrapper(const std::vector<__half>& in, 
                                    const std::vector<__half>& w,
                                    const std::vector<__half>& bias,
                                    std::vector<float>& mean,
                                    std::vector<float>& var,
                                    std::vector<__half>& out,
                                    int batch_size,
                                    int repeat_num = 50)
{
    return layernorm_fuse_wrapper(calc_layernorm_fuse_half,
                in, w, bias, mean, var, out, batch_size, repeat_num);
}
