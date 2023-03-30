#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define DEVICE_CONSTEXPR constexpr __device__ __host__ // NOLINT

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
        if(tid < s and tid + s < batch_item_num)
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
__device__ void layernorm_kernel_half2(__half2* in_data,
                                       __half2* in_data_reduce,
                                       __half2* w_data,
                                       __half2* b_data,
                                       float*   m_data,
                                       float*   v_data,
                                       __half2* out,
                                       int batch_item_num,
                                       int block_size,
                                       float rbatch_num)
{
    auto rnum = __float2half2_rn(rbatch_num);
    extern __shared__ __half2 buffer2[];
    auto m =
        block_reduce_half2(in_data_reduce, batch_item_num, threadIdx.x, block_size, half2_sum{});
    m = __hmul2(m, rnum);
    *m_data = __low2float(m);

    for(int i = threadIdx.x; i < batch_item_num; i += block_size)
    {
        in_data[i]        = __hsub2(in_data[i], m);
        in_data_reduce[i] = __hmul2(in_data[i], in_data[i]);
    }

    m = block_reduce_half2(in_data_reduce, batch_item_num, threadIdx.x, block_size, half2_sum{});
    m = __hmul2(m, rnum);

    auto eps = __float2half2_rn(1.0e-12f);
    auto r   = __hadd2(m, eps);
    r        = h2rsqrt(r);
    *v_data = __low2float(r);

    int start = blockIdx.x * batch_item_num;
    for(int i = threadIdx.x; i < batch_item_num; i += block_size)
    {
        int idx  = i + start;
        auto o2 = __hmul2(in_data[i], r);
        out[idx] = __hadd2(__hmul2(o2, w_data[i]), b_data[i]);
    }
}

__global__ void layernorm_half2(void* in, void* w, void* b, float* m, float *v, void* data_out, int batch_item_num, int block_size)
{
    __half2* input = reinterpret_cast<__half2*>(in);
    __half2* ww = reinterpret_cast<__half2*>(w);
    __half2* bb = reinterpret_cast<__half2*>(b);

    __half2* output = reinterpret_cast<__half2*>(data_out);
    float rnum      = 1.0f / batch_item_num;
    batch_item_num /= 2;
    extern __shared__ __half2 buffer2[];
    __half2* in_data_reduce = buffer2;
    __half2* in_data        = buffer2 + batch_item_num;

    int bid = blockIdx.x;
    int start = bid * batch_item_num;
    for(int i = threadIdx.x; i < batch_item_num; i += block_size)
    {
        int idx           = i + start;
        in_data[i]        = input[idx];
        in_data_reduce[i] = in_data[i];
    }

    layernorm_kernel_half2(in_data, in_data_reduce, ww, bb, &m[bid], &v[bid], output, batch_item_num, block_size, rnum);
}

template <class T>
__device__ T
block_reduce_half(T* buffer, int batch_item_num, int tid, int block_size)
{
    __syncthreads();
    for(int s = block_size; s > 0; s >>= 1)
    {
        if(tid < s and tid + s < batch_item_num)
        {
            buffer[tid] = __float2half(__half2float(buffer[tid]) + __half2float(buffer[tid + s]));
        }
        __syncthreads();
    }

    return buffer[0];
}

// m = x - mean(x)
// m / sqrt(mean(m ^ 2) + 1e-12)
__device__ void layernorm_kernel_half(__half* in_data,
                                      __half* in_data_reduce,
                                      __half* w,
                                      __half* b,
                                      float* m_data,
                                      float* v_data,
                                      __half* out,
                                      int batch_item_num,
                                      int block_size,
                                      float rnum)
{
    auto m = block_reduce_half(in_data_reduce, batch_item_num, threadIdx.x, block_size);
    m *= rnum;
    *m_data = m;

    for(int i = threadIdx.x; i < batch_item_num; i += block_size)
    {
        in_data[i]        = __float2half(__half2float(in_data[i]) - __half2float(m));
        in_data_reduce[i] = __float2half(__half2float(in_data[i]) * __half2float(in_data[i]));
    }

    m = block_reduce_half(in_data_reduce, batch_item_num, threadIdx.x, block_size);
    m *= rnum;
    m += 1.0e-12f;
    auto rstd = rsqrt(__half2float(m));
    *v_data = rstd;

    int start = blockIdx.x * batch_item_num;
    for(int i = threadIdx.x; i < batch_item_num; i += block_size)
    {
        int idx  = i + start;
        auto o2 = __half2float(in_data[i]) * rstd;
        out[idx] = __float2half(o2 * __half2float(w[i]) + __half2float(b[i]));
    }
}

// m = x - mean(x)
// m / sqrt(mean(m ^ 2) + 1e-12)
__global__ void layernorm_half(void* in, void *w, void *b, float *m, float *v, void* data_out, int batch_item_num, int block_size)
{
    __half* input = reinterpret_cast<__half*>(in);
    __half* ww = reinterpret_cast<__half*>(w);
    __half* bb = reinterpret_cast<__half*>(b);
    __half* output = reinterpret_cast<__half*>(data_out);
    float rnum     = 1.0f / batch_item_num;
    extern __shared__ __half bufferh[];
    __half* in_data_reduce = bufferh;
    __half* in_data        = bufferh + batch_item_num;

    int start = blockIdx.x * batch_item_num;
    for(int i = threadIdx.x; i < batch_item_num; i += block_size)
    {
        int idx    = i + start;
        in_data[i] = input[idx];
        in_data_reduce[i] = in_data[i];
    }

    layernorm_kernel_half(in_data, in_data_reduce, ww, bb, &m[blockIdx.x], &v[blockIdx.x], output, batch_item_num, block_size, rnum);
}

static size_t compute_block_size(int n, int max_block_size)
{
    size_t block_size = 64;
    while(block_size < max_block_size and block_size < n)
        block_size *= 2;
    return block_size;
}

void layernorm_fuse_half2_wrapper(const std::vector<__half>& in, 
                                    const std::vector<__half>& w,
                                    const std::vector<__half>& bias,
                                    std::vector<float>& mean,
                                    std::vector<float>& var,
                                    std::vector<__half>& out,
                                    int batch_size) 
{
    int elem_num = in.size();
    out.resize(elem_num);
    auto block_size       = compute_block_size(batch_size, 1024);
    int block_num         = elem_num / batch_size;
    int shared_size       = batch_size * 2 * sizeof(__half);
    auto half2_block_size = block_size / 4;

    __half *in_d, *out_d;
    size_t size = elem_num * sizeof(__half);
    hipMalloc((void**)&in_d, size);
    hipMalloc((void**)&out_d, size);

    __half *w_d, *b_d;
    size_t wb_size = batch_size * sizeof(__half);
    hipMalloc((void**)&w_d, wb_size);
    hipMalloc((void**)&b_d, wb_size);

    float *mean_d, *var_d;
    size_t mv_size = block_num * sizeof(float);
    hipMalloc((void**)&mean_d, mv_size);
    hipMalloc((void**)&var_d, mv_size);

    hipMemcpy(in_d, in.data(), size, hipMemcpyHostToDevice);
    hipMemcpy(w_d, w.data(), wb_size, hipMemcpyHostToDevice);
    hipMemcpy(b_d, bias.data(), wb_size, hipMemcpyHostToDevice);

    layernorm_half2<<<block_num, half2_block_size, shared_size>>>(
        in_d, w_d, b_d, mean_d, var_d, out_d, batch_size, half2_block_size);

    mean.resize(block_num);
    hipMemcpy((void*)mean.data(), mean_d, mv_size, hipMemcpyDeviceToHost);
    var.resize(block_num);
    hipMemcpy((void*)var.data(), var_d, mv_size, hipMemcpyDeviceToHost);
    out.resize(elem_num);
    hipMemcpy((void*)out.data(), out_d, size, hipMemcpyDeviceToHost);
}

void layernorm_fuse_half_wrapper(const std::vector<__half>& in, 
                                   const std::vector<__half>& w,
                                   const std::vector<__half>& bias,
                                   std::vector<float>& mean,
                                   std::vector<float>& var,
                                   std::vector<__half>& out,
                                   int batch_size) 
{
    int elem_num = in.size();
    auto block_size       = compute_block_size(batch_size, 1024);
    int block_num         = elem_num / batch_size;
    int shared_size       = batch_size * 2 * sizeof(__half);
    auto half_block_size = block_size / 2;

    __half *in_d, *out_d;
    size_t size = elem_num * sizeof(__half);
    hipMalloc((void**)&in_d, size);
    hipMalloc((void**)&out_d, size);

    __half *w_d, *b_d;
    size_t wb_size = batch_size * sizeof(__half);
    hipMalloc((void**)&w_d, wb_size);
    hipMalloc((void**)&b_d, wb_size);

    float *mean_d, *var_d;
    size_t mv_size = block_num * sizeof(float);
    hipMalloc((void**)&mean_d, mv_size);
    hipMalloc((void**)&var_d, mv_size);

    hipMemcpy(in_d, in.data(), size, hipMemcpyHostToDevice);
    hipMemcpy(w_d, w.data(), wb_size, hipMemcpyHostToDevice);
    hipMemcpy(b_d, bias.data(), wb_size, hipMemcpyHostToDevice);

    layernorm_half<<<block_num, half_block_size, shared_size>>>(
        in_d, w_d, b_d, mean_d, var_d, out_d, batch_size, half_block_size);

    mean.resize(block_num);
    hipMemcpy((void*)mean.data(), mean_d, mv_size, hipMemcpyDeviceToHost);
    var.resize(block_num);
    hipMemcpy((void*)var.data(), var_d, mv_size, hipMemcpyDeviceToHost);
    out.resize(elem_num);
    hipMemcpy((void*)out.data(), out_d, size, hipMemcpyDeviceToHost);
}
