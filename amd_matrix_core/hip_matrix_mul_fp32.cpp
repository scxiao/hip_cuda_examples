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

    run_kernel(sgemm_fp32_16x16x4_fp32_v1, grid_dim_v1, block_dim_v1, 0, in1, in2, res, flops);
}

// Matrix multiplication, call MFMA instructions
#define TILE_SIZE_M 32
#define TILE_SIZE_N 32
#define TILE_SIZE_K 32

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


// device function to handle matrix multiplication with 
// size: TILE_SIZE_M, TILE_SIZE_N, and TILE_SIZE_K (32, 32, 32)
// This kernel uses the mfma instruction 16x16x4 with 1 block.
// The input tile size 32 is divided into 2 parts, so A becomes 2 16 x 32
// and B becomes 2 32 x 16. Combine them together to get 4 submatrices
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

__global__ void sgemm_16x16x4_tile(const float *A, const float *B, float *D, int M, int K, int N) {
    extern __shared__ float shared_mem[];
    int slda = TILE_SIZE_K + 1;
    int sldb = TILE_SIZE_N + 1;
    int sldc = sldb;
    float *in_shared_a = shared_mem;
    float *in_shared_b = shared_mem + TILE_SIZE_M * slda;
    float *in_shared_c = shared_mem + TILE_SIZE_K * sldc;

    size_t b_idx = blockIdx.x;
    size_t b_idy = blockIdx.y;
    size_t t_idx = threadIdx.x;
    size_t t_idy = threadIdx.y;
    size_t block_size_m = blockDim.x;
    size_t block_size_n = blockDim.y;

    // int idx00 = t_idx * sldc + t_idy;
    // in_shared_c[idx00] = 0.0f;
    // int idx01 = idx00 + block_size_n;
    // in_shared_c[idx01] = 0.0f;
    // int idx10 = idx00 + block_size_m * sldc;
    // in_shared_c[idx10] = 0.0f;
    // int idx11 = idx10 + block_size_n;
    // in_shared_c[idx11] = 0.0f;


    int idx00 = (TILE_SIZE_M * b_idx + t_idx) * N + TILE_SIZE_N * b_idy + t_idy;
    D[idx00] = 0.0f;
    int idx01 = idx00 + block_size_n;
    D[idx01] = 0.0f;
    int idx10 = idx00 + block_size_m * N;
    D[idx10] = 0.0f;
    int idx11 = idx10 + block_size_n;
    D[idx11] = 0.0f;

    for (size_t tile_idx = 0; tile_idx < K; tile_idx += TILE_SIZE_K) {
        in_shared_a[t_idx * slda + t_idy] = A[(b_idx * TILE_SIZE_M + t_idx) * K + tile_idx + t_idy];
        in_shared_a[t_idx * slda + t_idy + block_size_n] = A[(b_idx * TILE_SIZE_M + t_idx) * K + tile_idx + t_idy + block_size_n];
        in_shared_a[(t_idx + block_size_m) * slda + t_idy] = A[(b_idx * TILE_SIZE_M + t_idx + block_size_m) * K + tile_idx + t_idy];
        in_shared_a[(t_idx + block_size_m) * slda + t_idy + block_size_n] = A[(b_idx * TILE_SIZE_M + t_idx + block_size_m) * K + tile_idx + t_idy + block_size_n];

        in_shared_b[t_idx * sldb + t_idy] = B[(tile_idx + t_idx) * N + b_idy * TILE_SIZE_N + t_idy];
        in_shared_b[t_idx * sldb + t_idy + block_size_n] = B[(tile_idx + t_idx) * N + b_idy * TILE_SIZE_N + t_idy + block_size_n];
        in_shared_b[(t_idx + block_size_m) * sldb + t_idy] = B[(tile_idx + t_idx + block_size_m) * N + b_idy * TILE_SIZE_N + t_idy];
        in_shared_b[(t_idx + block_size_m) * sldb + t_idy + block_size_n] = B[(tile_idx + t_idx + block_size_m) * N + b_idy * TILE_SIZE_N + t_idy + block_size_n];

        __syncthreads();
        int block_offset = (b_idx * TILE_SIZE_M) * N + b_idy * TILE_SIZE_N;
        float *bd = D + block_offset;
        sgemm_fp32_16x16x4_fp32_device(in_shared_a, in_shared_b, bd, slda, sldb, N, TILE_SIZE_K);
        __syncthreads();
    }
}


bool hip_matrix_mul_sgemm_32x32x32_fp32(CMatrix<float> &in1, CMatrix<float> &in2, CMatrix<float> &res, double& flops) {
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
    return run_kernel(sgemm_16x16x4_tile, grid_dim_v2, block_dim_v2, lds_size, in1, in2, res, flops);
}

