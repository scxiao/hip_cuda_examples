#include <functional>
#include "matrixmul.hpp"
#include "hip_matrix_mul.hpp"
#include "hip/hip_runtime.h"
#include "timer.hpp"

using namespace std;

__global__ void hipkernel_matrix_mul_naive(float *in1, float *in2, float *res,
        size_t row, size_t dim, size_t column) {
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx_x < row && idx_y < column) {
        float sum = 0.0;
        for (size_t k = 0; k < dim; ++k) {
            sum += in1[idx_x * dim + k] * in2[k * dim + idx_y];
        }
        res[idx_x * column + idx_y] = sum;
    }

    return;
}

#define BLOCK_SIZE_M 32
#define BLOCK_SIZE_N 32
#define BLOCK_SIZE_K 32

__global__ void hipkernel_matrix_mul_shared(float *in1, float *in2, float *res,
        size_t row, size_t dim, size_t column) {
    __shared__ float in_shared1[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ float in_shared2[BLOCK_SIZE_K][BLOCK_SIZE_N];
    //HIP_DYNAMIC_SHARED(float, in_shared1);
    //HIP_DYNAMIC_SHARED(float, in_shared2);

    size_t b_idx = blockIdx.x;
    size_t b_idy = blockIdx.y;
    size_t t_idx = threadIdx.x;
    size_t t_idy = threadIdx.y;

    float sum = 0.0;
    for (size_t tile_idx = 0; tile_idx < dim; tile_idx += BLOCK_SIZE_K) {
        in_shared1[t_idx][t_idy] = in1[(b_idx * BLOCK_SIZE_M + t_idx) * dim + tile_idx + t_idy];
        in_shared2[t_idx][t_idy] = in2[(tile_idx + t_idx) * column + b_idy * BLOCK_SIZE_N + t_idy];
        __syncthreads();

        for (size_t idx = 0; idx < BLOCK_SIZE_K; idx++) {
            sum += in_shared1[t_idx][idx] * in_shared2[idx][t_idy];
        }
        __syncthreads();
    }
    
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    res[idx_x * column + idx_y] = sum;

    return;
}

__global__ void hipkernel_matrix_mul_dynamic_shared(float *in1, float *in2, float *res,
        size_t row, size_t dim, size_t column) {
    extern __shared__ float shared_mem[];
    float *in_shared1 = shared_mem;
    float *in_shared2 = shared_mem + BLOCK_SIZE_M * (BLOCK_SIZE_K + 1);
    // HIP_DYNAMIC_SHARED(float, in_shared1);
    // HIP_DYNAMIC_SHARED(float, in_shared2);

    size_t b_idx = blockIdx.x;
    size_t b_idy = blockIdx.y;
    size_t t_idx = threadIdx.x;
    size_t t_idy = threadIdx.y;

    float sum = 0.0;
    size_t stride1 = BLOCK_SIZE_K + 1;
    size_t stride2 = BLOCK_SIZE_N + 1;
    for (size_t tile_idx = 0; tile_idx < dim; tile_idx += BLOCK_SIZE_K) {
        in_shared1[t_idx * stride1 + t_idy] = in1[(b_idx * BLOCK_SIZE_M + t_idx) * dim + tile_idx + t_idy];
        in_shared2[t_idx * stride2 + t_idy] = in2[(tile_idx + t_idx) * column + b_idy * BLOCK_SIZE_N + t_idy];
        __syncthreads();

        for (size_t idx = 0; idx < BLOCK_SIZE_K; idx++) {
            sum += in_shared1[t_idx * stride1 + idx] * in_shared2[idx * stride2 + t_idy];
        }
        __syncthreads();
    }
    
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    res[idx_x * column + idx_y] = sum;

    return;
}

// Version_1: 1 wave to handle tile size 16x16
__global__ void sgemm_fp32_16x16x4_fp32_v1(const float *A, const float *B, float *D, int M, int N, int K) {
    int LDA = K;
    int LDB = N;
    int LDD = N;

    int a_idx = threadIdx.x * LDA + threadIdx.y;
    int b_idx = threadIdx.x + threadIdx.y * LDB;

#if __gfx90a__ || __gfx908__
    using float4 = __attribute__((__vector_size__(4 * sizeof(float)) )) float;
    float4 d = {0};
    for (int i = 0; i < 4; ++i) {
        float a = A[a_idx];
        float b = B[b_idx];
        d = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, d, 0, 0, 0);
        a_idx += 4;
        b_idx += 4 * LDB;
    }

    for (int i = 0; i < 4; ++i) {
        int d_idx = threadIdx.x + i * LDD + threadIdx.y * 4 * LDD;
        D[d_idx] = d[i];
    }
#endif
}

// Version_2: 4 waves to handle tile size 32 x 32
// The input tile size 32 is divided into 2 parts, so A becomes 2  16 x 32
// and B becomes 2 32 x 16. Combine them together to get 4 submatrices
// Each wave handles one submatrix
__global__ void sgemm_fp32_16x16x4_fp32_v2(const float *A, const float *B, float *D, int M, int N, int K) {
    int LDA = K;
    int LDB = N;
    int LDD = N;
    const int mfma_m = 16
    const int mfma_n = 16
    const int mfma_k = 4;

    // first 64 threads are in wave 1, and second 64 threads are in wave 2
    int a_idx = (threadIdx.x + (threadIdx.y / mfma_k * mfma_m)) * LDA + (threadIdx.y % mfma_k);
    int b_idx = (threadIdx.x + threadIdx.y / mfma_k * mfma_n) + (threadIdx.y % mfma_k) * LDB;

#if __gfx90a__ || __gfx908__
    using float4 = __attribute__((__vector_size__(4 * sizeof(float)) )) float;
    float4 d = {0};
    for (int i = 0; i < K / mfma_k; ++i) {
        float a = A[a_idx];
        float b = B[b_idx];
        d = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, d, 0, 0, 0);
        a_idx += mfma_k;
        b_idx += mfma_k * LDB;
    }

    for (int i = 0; i < 4; ++i) {
        int d_idx = threadIdx.x + i * LDD + threadIdx.y * 4 * LDD;
        D[d_idx] = d[i];
    }
#endif
}

// __global__ void sgemm_16x16x4_tile(const float *A, const float *B, float *D, int M, int K, int N) {
// #if __gfx90a__ || __gfx908__
//     extern __shared__ float shared_mem[];
//     float *in_shared1 = shared_mem;
//     float *in_shared2 = shared_mem + BLOCK_SIZE_M * (BLOCK_SIZE_K + 1);

//     size_t b_idx = blockIdx.x;
//     size_t b_idy = blockIdx.y;
//     size_t t_idx = threadIdx.x;
//     size_t t_idy = threadIdx.y;

//     float4 sum = 0.0;
//     size_t stride1 = BLOCK_SIZE_K + 1;
//     size_t stride2 = BLOCK_SIZE_N + 1;
//     for (size_t tile_idx = 0; tile_idx < dim; tile_idx += BLOCK_SIZE_K) {
//         in_shared1[t_idx * stride1 + t_idy] = in1[(b_idx * BLOCK_SIZE_M + t_idx) * dim + tile_idx + t_idy];
//         in_shared2[t_idx * stride2 + t_idy] = in2[(tile_idx + t_idx) * column + b_idy * BLOCK_SIZE_N + t_idy];
//         __syncthreads();
//         sum = 
//     }

// #endif
// }

template<class T>
bool run_kernel(T kernel, const dim3& grid_size, const dim3& block_size, size_t lds_size,
                CMatrix<float> &in1, CMatrix<float> &in2, CMatrix<float> &res, double& perf_Tflops) {
    size_t row, dim1, dim2, column;
    in1.get_size(row, dim1);
    in2.get_size(dim2, column);

    if (dim1 != dim2) {
        cout << "Matrix dimensions mismatch!" << endl;
        return false;
    }
    res.resize(row, column);

    float *hip_in1, *hip_in2, *hip_res;
    hipMalloc((void **)&hip_in1, sizeof(float) * row * dim1);
    hipMalloc((void **)&hip_in2, sizeof(float) * dim2 * column);
    hipMalloc((void **)&hip_res, sizeof(float) * row * column);

    hipMemcpy(hip_in1, in1.get_buffer(), sizeof(float) * row * dim1, hipMemcpyHostToDevice);
    hipMemcpy(hip_in2, in2.get_buffer(), sizeof(float) * dim2 * column, hipMemcpyHostToDevice);

    // warm up execution
    for (int i = 0; i < 5; ++i) {
        hipLaunchKernelGGL(kernel, grid_size, block_size, lds_size, 0, hip_in1, hip_in2, hip_res, row, dim1, column);
    }
    hipDeviceSynchronize();
    hipError_t ret = hipGetLastError();
    if (ret != hipSuccess) {
        std::cout << "matrix_mul, kernel launch error, code = " << ret << std::endl;
        std::cout << "Error info: " << hipGetErrorString(ret) << std::endl;
    }

    HRTimer timer;
    timer.start();
    hipLaunchKernelGGL(kernel, grid_size, block_size, lds_size, 0, hip_in1, hip_in2, hip_res, row, dim1, column);
    hipDeviceSynchronize();
    timer.stop();
    size_t kernel_time = timer.gettime_us();
    perf_Tflops = 2.0 * row * column * dim1 / kernel_time / 1000000;

    hipMemcpy(res.get_buffer(), hip_res, sizeof(float) * row * column, hipMemcpyDeviceToHost);
    hipFree(hip_in1); hipFree(hip_in2); hipFree(hip_res);

    return true;
}


bool hip_matrix_mul_naive_fp32(CMatrix<float> &in1, CMatrix<float> &in2, CMatrix<float> &res, double& flops) {
    size_t row, dim1, dim2, column;
    in1.get_size(row, dim1);
    in2.get_size(dim2, column);
    size_t block_dimx = BLOCK_SIZE_M;
    size_t block_dimy = BLOCK_SIZE_N;

    if (dim1 != dim2) {
        cout << "Matrix dimensions mismatch!" << endl;
        return false;
    }

    dim3 grid_dim = dim3((row + block_dimx - 1)/block_dimx, (column + block_dimy - 1)/block_dimy);
    dim3 block_dim = dim3(block_dimx, block_dimy);

    return run_kernel(hipkernel_matrix_mul_naive, grid_dim, block_dim, 0, in1, in2, res, flops);
}

bool hip_matrix_mul_shared_fp32(CMatrix<float> &in1, CMatrix<float> &in2, CMatrix<float> &res, double& flops) {
    size_t row, dim1, dim2, column;
    in1.get_size(row, dim1);
    in2.get_size(dim2, column);
    if (dim1 != dim2) {
        cout << "Matrix dimensions mismatch!" << endl;
        return false;
    }

    dim3 grid_dim = dim3((row + BLOCK_SIZE_M - 1)/BLOCK_SIZE_M, (column + BLOCK_SIZE_N - 1)/BLOCK_SIZE_N);
    dim3 block_dim = dim3(BLOCK_SIZE_M, BLOCK_SIZE_N);
    size_t lds_size = BLOCK_SIZE_M * (BLOCK_SIZE_K + 1) + BLOCK_SIZE_K * (BLOCK_SIZE_N + 1);
    lds_size *= sizeof(float);

    return run_kernel(hipkernel_matrix_mul_dynamic_shared, grid_dim, block_dim, lds_size, in1, in2, res, flops);
}

bool hip_matrix_mul_sgemm_16x16x16_fp32(CMatrix<float> &in1, CMatrix<float> &in2, CMatrix<float> &res, double& flops) {
    size_t row, dim1, dim2, column;
    in1.get_size(row, dim1);
    in2.get_size(dim2, column);
    if (dim1 != dim2) {
        cout << "Matrix dimensions mismatch!" << endl;
        return false;
    }

    dim3 grid_dim_v1 = dim3(1, 1);
    dim3 block_dim_v1 = dim3(16, 4);

    run_kernel(sgemm_fp32_16x16x4_fp32_v1, grid_dim_v1, block_dim_v1, 0, in1, in2, res, flops);
}

bool hip_matrix_mul_sgemm_32x32x32_fp32(CMatrix<float> &in1, CMatrix<float> &in2, CMatrix<float> &res, double& flops) {
    size_t row, dim1, dim2, column;
    in1.get_size(row, dim1);
    in2.get_size(dim2, column);
    if (dim1 != dim2) {
        cout << "Matrix dimensions mismatch!" << endl;
        return false;
    }

    dim3 grid_dim_v2 = dim(1, 1);
    dim3 block_dim_v2 = dim(16, 16);
    return run_kernel(sgemm_fp32_16x16x4_fp32_v2, grid_dim_v2, block_dim_v2, 0, in1, in2, res, flops);
}
