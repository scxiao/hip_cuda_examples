#include <functional>
#include "matrixmul.hpp"
#include "hip_matrix_mul.hpp"
#include "hip/hip_runtime.h"
#include "timer.hpp"

using namespace std;

// wrap up function to call different kernels
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

// --------------------------- 1. ---------------------------
// Matrix multiplication naive implementation
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

#define BLOCK_SIZE 32
bool hip_matrix_mul_naive_fp32(CMatrix<float> &in1, CMatrix<float> &in2, CMatrix<float> &res, double& flops) {
    size_t row, dim1, dim2, column;
    in1.get_size(row, dim1);
    in2.get_size(dim2, column);
    size_t block_dimx = BLOCK_SIZE;
    size_t block_dimy = BLOCK_SIZE;

    if (dim1 != dim2) {
        cout << "Matrix dimensions mismatch!" << endl;
        return false;
    }

    dim3 grid_dim = dim3((row + block_dimx - 1)/block_dimx, (column + block_dimy - 1)/block_dimy);
    dim3 block_dim = dim3(block_dimx, block_dimy);

    return run_kernel(hipkernel_matrix_mul_naive, grid_dim, block_dim, 0, in1, in2, res, flops);
}

// --------------------------- 2. ---------------------------
// Matrix multiplication shared memory implementation, fixed shared memory size
__global__ void hipkernel_matrix_mul_shared(float *in1, float *in2, float *res,
        size_t row, size_t dim, size_t column) {
    __shared__ float in_shared1[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float in_shared2[BLOCK_SIZE][BLOCK_SIZE];
    //HIP_DYNAMIC_SHARED(float, in_shared1);
    //HIP_DYNAMIC_SHARED(float, in_shared2);

    size_t b_idx = blockIdx.x;
    size_t b_idy = blockIdx.y;
    size_t t_idx = threadIdx.x;
    size_t t_idy = threadIdx.y;

    float sum = 0.0;
    for (size_t tile_idx = 0; tile_idx < dim; tile_idx += BLOCK_SIZE) {
        in_shared1[t_idx][t_idy] = in1[(b_idx * BLOCK_SIZE + t_idx) * dim + tile_idx + t_idy];
        in_shared2[t_idx][t_idy] = in2[(tile_idx + t_idx) * column + b_idy * BLOCK_SIZE + t_idy];
        __syncthreads();

        for (size_t idx = 0; idx < BLOCK_SIZE; idx++) {
            sum += in_shared1[t_idx][idx] * in_shared2[idx][t_idy];
        }
        __syncthreads();
    }
    
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    res[idx_x * column + idx_y] = sum;

    return;
}

// Matrix multiplication shared memory implementation, dynamic shared memory size
__global__ void hipkernel_matrix_mul_dynamic_shared(float *in1, float *in2, float *res,
        size_t row, size_t dim, size_t column) {
    extern __shared__ float shared_mem[];
    float *in_shared1 = shared_mem;
    float *in_shared2 = shared_mem + BLOCK_SIZE * (BLOCK_SIZE + 1);
    // HIP_DYNAMIC_SHARED(float, in_shared1);
    // HIP_DYNAMIC_SHARED(float, in_shared2);

    size_t b_idx = blockIdx.x;
    size_t b_idy = blockIdx.y;
    size_t t_idx = threadIdx.x;
    size_t t_idy = threadIdx.y;

    float sum = 0.0;
    size_t stride1 = BLOCK_SIZE + 1;
    size_t stride2 = BLOCK_SIZE + 1;
    for (size_t tile_idx = 0; tile_idx < dim; tile_idx += BLOCK_SIZE) {
        in_shared1[t_idx * stride1 + t_idy] = in1[(b_idx * BLOCK_SIZE + t_idx) * dim + tile_idx + t_idy];
        in_shared2[t_idx * stride2 + t_idy] = in2[(tile_idx + t_idx) * column + b_idy * BLOCK_SIZE + t_idy];
        __syncthreads();

        for (size_t idx = 0; idx < BLOCK_SIZE; idx++) {
            sum += in_shared1[t_idx * stride1 + idx] * in_shared2[idx * stride2 + t_idy];
        }
        __syncthreads();
    }
    
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    res[idx_x * column + idx_y] = sum;

    return;
}

bool hip_matrix_mul_shared_fp32(CMatrix<float> &in1, CMatrix<float> &in2, CMatrix<float> &res, double& flops) {
    size_t row, dim1, dim2, column;
    in1.get_size(row, dim1);
    in2.get_size(dim2, column);
    if (dim1 != dim2) {
        cout << "Matrix dimensions mismatch!" << endl;
        return false;
    }

    dim3 grid_dim = dim3((row + BLOCK_SIZE - 1)/BLOCK_SIZE, (column + BLOCK_SIZE - 1)/BLOCK_SIZE);
    dim3 block_dim = dim3(BLOCK_SIZE, BLOCK_SIZE);
    size_t lds_size = BLOCK_SIZE * (BLOCK_SIZE + 1) + BLOCK_SIZE * (BLOCK_SIZE + 1);
    lds_size *= sizeof(float);

    return run_kernel(hipkernel_matrix_mul_dynamic_shared, grid_dim, block_dim, lds_size, in1, in2, res, flops);
}

// --------------------------- 3. ---------------------------
// Matrix multiplication, specific for M = N = K = 16
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

bool hip_matrix_mul_sgemm_16x16x16_fp32(CMatrix<float> &in1, CMatrix<float> &in2, CMatrix<float> &res, double& flops) {
    size_t row, dim1, dim2, column;
    in1.get_size(row, dim1);
    in2.get_size(dim2, column);
    if (dim1 != dim2) {
        cout << "Matrix dimensions mismatch!" << endl;
        return false;
    }

    // 1 block, with 1 wavefornt to handle a matrix multiplication with M = N = K = 16
    dim3 grid_dim_v1 = dim3(1, 1);
    dim3 block_dim_v1 = dim3(16, 4);

    return run_kernel(sgemm_fp32_16x16x4_fp32_v1, grid_dim_v1, block_dim_v1, 0, in1, in2, res, flops);
}

// --------------------------- 4. ---------------------------
// Matrix multiplication, call MFMA instructions
#define TILE_SIZE_M 32
#define TILE_SIZE_N 32
#define TILE_SIZE_K 16

// kernel to handle matrix multiplication with 
// size: TILE_SIZE_M, TILE_SIZE_N, and TILE_SIZE_K (32, 32, 32)
// The input tile size 32 is divided into 2 parts, so A becomes 2 16 x 32
// and B becomes 2 32 x 16. Combine them together to get 4 submatrices
// Each wave handles one submatrix
__global__ void sgemm_fp32_16x16x4_fp32_v2(const float *A, const float *B, float *D, int M, int N, int K) {
    int LDA = K;
    int LDB = N;
    int LDD = N;
    const int mfma_m = 16;
    const int mfma_n = 16;
    const int mfma_k = 4;

#if __gfx90a__ || __gfx908__
    // first 64 threads are in wave 1, and second 64 threads are in wave 2
    int wi = threadIdx.y / mfma_k;
    int wii = threadIdx.y % mfma_k;
    int wi_m = wi % 2;
    int wi_n = wi / 2;

    int a_idx = (threadIdx.x + wi_m * mfma_m) * LDA + wii;
    int b_idx = threadIdx.x + wi_n * mfma_n + wii * LDB;

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
        int d_idx = threadIdx.x + wi_n * mfma_n + (wi_m * mfma_m + wii * mfma_k + i) * LDD;
        D[d_idx] = d[i];
    }
#endif
}

// --------------------------- 5. ---------------------------
// device function to handle matrix multiplication with size: TILE_SIZE_M, 
// TILE_SIZE_N, and TILE_SIZE_K (32, 32, 32)
// This kernel uses the mfma instruction 16x16x4_1 block. The input 
// tile size 32 is divided into 2 parts, so A becomes 2 16 x 32 and B becomes 
// 2 32 x 16. Combine them together to get 4 submatrices
// We have 4 waves configed in the kernel and each wave handles one submatrix
__device__ void sgemm_fp32_16x16x4_fp32_device(const float *sa, const float *sb, float *bd, int lda, int ldb, int ldd, int K) {
    const int mfma_m = 16;
    const int mfma_n = 16;
    const int mfma_k = 4;

#if __gfx90a__ || __gfx908__
    // first 64 threads are in wave 1, and second 64 threads are in wave 2
    int wi = threadIdx.y / mfma_k;
    int wii = threadIdx.y % mfma_k;
    int wi_m = wi % 2;
    int wi_n = wi / 2;

    int a_idx = (threadIdx.x + wi_m * mfma_m) * lda + wii;
    int b_idx = threadIdx.x + wi_n * mfma_n + wii * ldb;

    using float4 = __attribute__((__vector_size__(4 * sizeof(float)) )) float;
    float4 d = {0};
    for (int i = 0; i < K / mfma_k; ++i) {
        float a = sa[a_idx];
        float b = sb[b_idx];
        d = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, d, 0, 0, 0);
        a_idx += mfma_k;
        b_idx += mfma_k * ldb;
    }    

    for (int i = 0; i < 4; ++i) {
        int d_idx = threadIdx.x + wi_n * mfma_n + (wi_m * mfma_m + wii * mfma_k + i) * ldd;
        bd[d_idx] += d[i];
    }
#endif
}

__global__ void sgemm_32x32xK_tile_v1(const float *A, const float *B, float *D, int M, int K, int N) {
    extern __shared__ float shared_mem[];
    int slda = TILE_SIZE_K + 1;
    int sldb = TILE_SIZE_N + 1;
    int sldc = sldb;
    float *in_shared_a = shared_mem;
    float *in_shared_b = shared_mem + TILE_SIZE_M * slda;
    float *in_shared_c = shared_mem + TILE_SIZE_M * slda + TILE_SIZE_K * sldc;

    size_t b_idx = blockIdx.x;
    size_t b_idy = blockIdx.y;
    size_t t_idx = threadIdx.x;
    size_t t_idy = threadIdx.y;
    size_t block_size_m = blockDim.x;
    size_t block_size_n = blockDim.y;

    for (int i = 0; i < TILE_SIZE_M; i += block_size_m) {
        for (int j = 0; j < TILE_SIZE_N; j += block_size_n) {
            int idx = (t_idx + i) * sldc + t_idy + j;
            in_shared_c[idx] = 0.0f;
        }
    }

    for (size_t tile_idx = 0; tile_idx < K; tile_idx += TILE_SIZE_K) {
        // copy A from global memory to LDS
        for (int i = 0; i < TILE_SIZE_M; i += block_size_m) {
            for (int j = 0; j < TILE_SIZE_K; j += block_size_n) {
                int s_idx = (t_idx + i) * slda + (t_idy + j);
                int g_idx = (b_idx * TILE_SIZE_M + t_idx + i) * K + (tile_idx + t_idy + j);
                in_shared_a[s_idx] = A[g_idx];
            }
        }

        // copy B from global memory to LDS
        for (int j = 0; j < TILE_SIZE_N; j += block_size_n) {
            for (int i = 0; i < TILE_SIZE_K; i += block_size_m) {
                int s_idx = (t_idx + i) * sldb + t_idy + j;
                int g_idx = (tile_idx + t_idx + i) * N + (b_idy * TILE_SIZE_N + t_idy + j);
                in_shared_b[s_idx] = B[g_idx];
            }
        }

        __syncthreads();
        sgemm_fp32_16x16x4_fp32_device(in_shared_a, in_shared_b, in_shared_c, slda, sldb, sldc, TILE_SIZE_K);
        __syncthreads();
    }

    for (int i = 0; i < TILE_SIZE_M; i += block_size_m) {
        for (int j = 0; j < TILE_SIZE_N; j += block_size_n) {
            int s_idx = (t_idx + i) * sldc + t_idy + j;
            int g_idx = (b_idx * TILE_SIZE_M + t_idx + i) * N + (b_idy * TILE_SIZE_N + t_idy + j);
            D[g_idx] = in_shared_c[s_idx];
        }
    }
}


bool hip_matrix_mul_sgemm_32x32xK_fp32_v1(CMatrix<float> &in1, CMatrix<float> &in2, CMatrix<float> &res, double& flops) {
    size_t row, dim1, dim2, column;
    in1.get_size(row, dim1);
    in2.get_size(dim2, column);
    if (dim1 != dim2) {
        cout << "Matrix dimensions mismatch!" << endl;
        return false;
    }

    size_t lds_size = TILE_SIZE_M * (TILE_SIZE_K + 1) + TILE_SIZE_K * (TILE_SIZE_N + 1) + TILE_SIZE_M * (TILE_SIZE_N + 1);
    lds_size *= sizeof(float);

    dim3 grid_dim_v2 = dim3((row + TILE_SIZE_M - 1) / TILE_SIZE_M, (column + TILE_SIZE_N - 1) / TILE_SIZE_N);
    dim3 block_dim_v2 = dim3(TILE_SIZE_M/2, TILE_SIZE_N/2);
    return run_kernel(sgemm_32x32xK_tile_v1, grid_dim_v2, block_dim_v2, lds_size, in1, in2, res, flops);
}

// --------------------------- 6. ---------------------------
// device function to handle matrix multiplication with size: TILE_SIZE_M, 
// TILE_SIZE_N, and TILE_SIZE_K (32, 32, 32)
// This kernel uses the mfma instruction 32x32x2_1. The input 
// tile size 32 x BLOCK_SIZE_K and BLOCK_SIZE_K * 32, so each instruction needs
// BLOCK_SIZE_K / 2 interations
// bock_size (32, 2), so only 1 wave
__device__ void sgemm_fp32_32x32x2_fp32_device(const float *sa, const float *sb, float *bd, int lda, int ldb, int ldd, int K) {
    const int mfma_m = 32;
    const int mfma_n = 32;
    const int mfma_k = 2;

#if __gfx90a__ || __gfx908__
    int a_idx = threadIdx.x * lda + threadIdx.y;
    int b_idx = threadIdx.x + threadIdx.y * ldb;

    using float16 = __attribute__((__vector_size__(16 * sizeof(float)) )) float;
    float16 d = {0};
    for (int i = 0; i < K / mfma_k; ++i) {
        float a = sa[a_idx];
        float b = sb[b_idx];
        d = __builtin_amdgcn_mfma_f32_32x32x2f32(a, b, d, 0, 0, 0);
        a_idx += mfma_k;
        b_idx += mfma_k * ldb;
    }    

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            bd[threadIdx.x + (4 * threadIdx.y + j + 8 * i)* ldd] += d[j + 4 * i];
        }
    }
#endif
}

__global__ void sgemm_32x32xK_tile_v2(const float *A, const float *B, float *D, int M, int K, int N) {
    extern __shared__ float shared_mem[];
    int slda = TILE_SIZE_K + 1;
    int sldb = TILE_SIZE_N + 1;
    int sldc = sldb;
    float *in_shared_a = shared_mem;
    float *in_shared_b = shared_mem + TILE_SIZE_M * slda;
    float *in_shared_c = shared_mem + TILE_SIZE_M * slda + TILE_SIZE_K * sldc;

    size_t b_idx = blockIdx.x;
    size_t b_idy = blockIdx.y;
    size_t t_idx = threadIdx.x;
    size_t t_idy = threadIdx.y;
    size_t block_size_m = blockDim.x;
    size_t block_size_n = blockDim.y;

    // initialize the output
    for (int j = 0; j < TILE_SIZE_N; j += block_size_n) {
        int idx = t_idx * sldc + t_idy + j;
        in_shared_c[idx] = 0.0f;
    }

    for (int tile_idx = 0; tile_idx < K; tile_idx += TILE_SIZE_K) {
        // copy A from global memory to LDS
        for (int i = 0; i < TILE_SIZE_K; i += block_size_n) {
            int s_idx = t_idx * slda + t_idy + i;
            int g_idx = (b_idx * TILE_SIZE_M + t_idx) * K + (tile_idx + t_idy + i);
            in_shared_a[s_idx] = A[g_idx];
        }

        // copy B from global memory to LDS
        for (int j = 0; j < TILE_SIZE_K; j += block_size_n) {
            int s_idx = (t_idy + j) * sldb + t_idx;
            int g_idx = (tile_idx + t_idy + j) * N + (b_idy * TILE_SIZE_N + t_idx);
            in_shared_b[s_idx] = B[g_idx];
        }

        __syncthreads();
        sgemm_fp32_32x32x2_fp32_device(in_shared_a, in_shared_b, in_shared_c, slda, sldb, sldc, TILE_SIZE_K);
        __syncthreads();
    }

    for (int i = 0; i < TILE_SIZE_N; i += block_size_n) {
        int s_idx = t_idx * sldc + t_idy + i;
        int g_idx = (b_idx * TILE_SIZE_M + t_idx) * N + (b_idy * TILE_SIZE_N + t_idy + i);
        D[g_idx] = in_shared_c[s_idx];
    }
}

bool hip_matrix_mul_sgemm_32x32xK_fp32_v2(CMatrix<float> &in1, CMatrix<float> &in2, CMatrix<float> &res, double& flops) {
    size_t row, dim1, dim2, column;
    in1.get_size(row, dim1);
    in2.get_size(dim2, column);
    if (dim1 != dim2) {
        cout << "Matrix dimensions mismatch!" << endl;
        return false;
    }

    const int block_size_m = TILE_SIZE_M;
    const int block_size_n = 2;

    size_t lds_size = TILE_SIZE_M * (TILE_SIZE_K + 1);
    lds_size += TILE_SIZE_K * (TILE_SIZE_N + 1);
    lds_size += TILE_SIZE_M * (TILE_SIZE_N + 1);
    lds_size *= sizeof(float);

    dim3 grid_dim_v2 = dim3((row + TILE_SIZE_M - 1) / TILE_SIZE_M, (column + TILE_SIZE_N - 1) / TILE_SIZE_N);
    dim3 block_dim_v2 = dim3(block_size_m, block_size_n);
    return run_kernel(sgemm_32x32xK_tile_v2, grid_dim_v2, block_dim_v2, lds_size, in1, in2, res, flops);
}

// --------------------------- 7. ---------------------------
// device function to handle matrix multiplication with size: TILE_SIZE_M, 
// TILE_SIZE_N, and TILE_SIZE_K (32, 32, 32)
// This kernel uses the mfma instruction 16X16X1_4fp32. The input tile size 
// 32 x BLOCK_SIZE_K and BLOCK_SIZE_K * 32, so each instruction needs
// BLOCK_SIZE_K / 1 interations. Bock_size is (16, 4), so only 1 wave
__device__ void sgemm_fp32_16x16x1_fp32_device(const float *sa, const float *sb, float *bd, 
                                               const int lda, const int ldb, const int ldd, 
                                               const int stride_a, const int stride_b, const int stride_c, 
                                               const int K) {
    const int mfma_m = 16;
    const int mfma_n = 16;
    const int mfma_k = 1;

#if __gfx90a__ || __gfx908__
    int a_idx = (threadIdx.x + mfma_m * (threadIdx.y / 2)) * lda;
    int b_idx = threadIdx.x + (threadIdx.y % 2) * mfma_n;

    using float16 = __attribute__((__vector_size__(16 * sizeof(float)) )) float;
    float16 d = {0};
    for (int i = 0; i < K / mfma_k; ++i) {
        float a = sa[a_idx];
        float b = sb[b_idx];
        d = __builtin_amdgcn_mfma_f32_16x16x1f32(a, b, d, 0, 0, 0);
        a_idx += mfma_k;
        b_idx += mfma_k * ldb;
    }    

    for (int i = 0; i < 4; ++i) {
        int mi = i / 2;
        int mj = i % 2;
        for (int j = 0; j < 4; ++j) {
            int s_idx = (4 * threadIdx.y + mi * mfma_m + j) * ldd + threadIdx.x + mj * mfma_n; 
            bd[s_idx] += d[4 * i + j];
        }
    }
#endif
}

__global__ void sgemm_32x32xK_tile_v3(const float *A, const float *B, float *D, int M, int K, int N) {
    extern __shared__ float shared_mem[];
    const int slda = TILE_SIZE_K + 1;
    const int sldb = TILE_SIZE_N + 1;
    const int sldc = sldb;
    const int stride_a = TILE_SIZE_M * slda;
    const int stride_b = TILE_SIZE_K * sldb;
    const int stride_c = TILE_SIZE_M * sldc;
    float *in_shared_a = shared_mem;
    float *in_shared_b = shared_mem + TILE_SIZE_M * slda;
    float *in_shared_c = shared_mem + TILE_SIZE_M * slda + TILE_SIZE_K * sldc;

    size_t b_idx = blockIdx.x;
    size_t b_idy = blockIdx.y;
    size_t t_idx = threadIdx.x;
    size_t t_idy = threadIdx.y;
    size_t block_size_m = blockDim.x;
    size_t block_size_n = blockDim.y;

    // initialize the output
    for (int i = 0; i < TILE_SIZE_M; i += block_size_m) {
        for (int j = 0; j < TILE_SIZE_N; j += block_size_n) {
            int idx = (t_idx + i) * sldc + t_idy + j;
            in_shared_c[idx] = 0.0f;
        }
    }

    for (int tile_idx = 0; tile_idx < K; tile_idx += TILE_SIZE_K) {
        // copy A from global memory to LDS
        for (int i = 0; i < TILE_SIZE_M; i += block_size_m) {
            for (int j = 0; j < TILE_SIZE_K; j += block_size_n) {
                int s_idx = (t_idx + i) * slda + t_idy + j;
                int g_idx = (b_idx * TILE_SIZE_M + t_idx + i) * K + (tile_idx + t_idy + j);
                in_shared_a[s_idx] = A[g_idx];
            }
        }

        // copy B from global memory to LDS
        for (int i = 0; i < TILE_SIZE_N; i += block_size_m) {
            for (int j = 0; j < TILE_SIZE_K; j += block_size_n) {
                int s_idx = (t_idy + j) * sldb + t_idx + i;
                int g_idx = (tile_idx + t_idy + j) * N + (b_idy * TILE_SIZE_N + t_idx);
                in_shared_b[s_idx] = B[g_idx];
            }
        }

        __syncthreads();
        sgemm_fp32_16x16x1_fp32_device(in_shared_a, in_shared_b, in_shared_c, 
                                       slda, sldb, sldc, 
                                       stride_a, stride_b, stride_c, TILE_SIZE_K);
        __syncthreads();
    }

    for (int i = 0; i < TILE_SIZE_M; i += block_size_m) {
        for (int j = 0; j < TILE_SIZE_N; j += block_size_n) {
            int sidx = (t_idx + i) * sldc + t_idy + j;
            int gidx = (b_idx * TILE_SIZE_M + t_idx + i) * N + b_idy * TILE_SIZE_N + t_idy + j;
            D[gidx] = in_shared_c[sidx];
        }
    }
}

bool hip_matrix_mul_sgemm_32x32xK_fp32_v3(CMatrix<float> &in1, CMatrix<float> &in2, CMatrix<float> &res, double& flops) {
    size_t row, dim1, dim2, column;
    in1.get_size(row, dim1);
    in2.get_size(dim2, column);
    if (dim1 != dim2) {
        cout << "Matrix dimensions mismatch!" << endl;
        return false;
    }

    const int block_size_m = 16;
    const int block_size_n = 4;

    size_t lds_size = TILE_SIZE_M * (TILE_SIZE_K + 1);
    lds_size += TILE_SIZE_K * (TILE_SIZE_N + 1);
    lds_size += TILE_SIZE_M * (TILE_SIZE_N + 1);
    lds_size *= sizeof(float);

    dim3 grid_dim_v2 = dim3((row + TILE_SIZE_M - 1) / TILE_SIZE_M, (column + TILE_SIZE_N - 1) / TILE_SIZE_N);
    dim3 block_dim_v2 = dim3(block_size_m, block_size_n);
    return run_kernel(sgemm_32x32xK_tile_v3, grid_dim_v2, block_dim_v2, lds_size, in1, in2, res, flops);
}
