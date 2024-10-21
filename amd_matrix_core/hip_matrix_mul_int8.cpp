#include <iostream>
#include <cstdlib>
#include <cstdio>
#include "matrixmul.hpp"
#include "hip_matrix_mul.hpp"
#include "timer.hpp"

__global__ void hip_hgemm_naive_int8(int8_t *A, int8_t *B, int32_t *C, int M, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    int32_t sum = 0;
    for (int i = 0; i < K; ++i) {
        sum += static_cast<int32_t>(A[idx * K + i]) * static_cast<int32_t>(B[idy * K + i]);
    }
    C[idx * N + idy] = sum;
}

bool hip_matrix_mul_int8_naive(CMatrix<int8_t> &in1, CMatrix<int8_t> &in2, CMatrix<int32_t> &res, double& flops) {
    size_t M, N, K1, K2, K;
    in1.get_size(M, K1);
    in2.get_size(K2, N);
    assert(K1 == K2);
    K = K1;
    int8_t *da, *db;
    int32_t *dc;
    hipMalloc((void**)&da, sizeof(int8_t) * M * K);
    hipMalloc((void**)&db, sizeof(int8_t) * K * N);
    hipMalloc((void**)&dc, sizeof(int32_t) * M * N);

    hipMemcpy(da, in1.get_buffer(), sizeof(int8_t) * M * K, hipMemcpyHostToDevice);
    hipMemcpy(db, in2.get_buffer(), sizeof(int8_t) * K * N, hipMemcpyHostToDevice);
    const int block_size = 32;
    dim3 grid(M/block_size, N/block_size);
    dim3 block(block_size, block_size);
    hip_hgemm_naive_int8<<<grid, block>>>(da, db, dc, M, N, K);
    hipError_t ret = hipGetLastError();
    if (ret != hipSuccess) {
        std::cout << "hip_matrix_mul_int8_naive, kernel launch error, code = " << ret << std::endl;
        std::cout << "Error info: " << hipGetErrorString(ret) << std::endl;
    }

    hipMemcpy((void*)res.get_buffer(), dc, sizeof(int32_t) * M * N, hipMemcpyDeviceToHost);

    return true;
}

// /*
//  * use 1 wave and the mfma32x32x8xf16 instruction to do the computation, tile_size is 64 x 64
//  * block dim(32, 2)
// */
// __device__ void hgemm_32x32x8_fp16_device(__half *sa, __half *sb, float *sc, int sam, int sak, int sbn, int sbk, int scm, int scn) {
// #if __gfx90a__ || __gfx908__ || __gfx942__
//     // using float16x8 = __attribute__((__vector_size__(8 * sizeof(__half)))) __half;
//     using float16x4 = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
//     using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
//     // threadsPerWarp [32, 2]
//     int tidx = threadIdx.x % 32;
//     int tidy = threadIdx.x / 32;

//     float16x4 a[2];
//     float16x4 b[2];
//     floatx16 d[4] = {0};
//     // first quarter
//     for (int k = 0; k < 64; k += 8) {
//         for (int i = 0; i < 4; ++i) {
//             // a[0] is for upper half of sa, a[1] is for lower half of sa
//             a[0][i] = sa[tidx * sak + i + 4 * tidy + k];
//             a[1][i] = sa[(tidx + 32) * sak + i + 4 * tidy + k];
//             b[0][i] = sb[tidx * sbk + i + 4 * tidy + k];
//             b[1][i] = sb[(tidx + 32) * sbk + i + 4 * tidy + k];
//         }

//         d[0] = __builtin_amdgcn_mfma_f32_32x32x8f16(a[0], b[0], d[0], 0, 0, 0);
//         d[1] = __builtin_amdgcn_mfma_f32_32x32x8f16(a[0], b[1], d[1], 0, 0, 0);
//         d[2] = __builtin_amdgcn_mfma_f32_32x32x8f16(a[1], b[0], d[2], 0, 0, 0);
//         d[3] = __builtin_amdgcn_mfma_f32_32x32x8f16(a[1], b[1], d[3], 0, 0, 0);
//     }

//     for (int i = 0; i < 4; ++i) {
//         int i1 = i % 2;
//         int i2 = i / 2;
//         for (int j = 0; j < 16; ++j) {
//             int j1 = j % 4;
//             int j2 = j / 4;
//             int idx = tidx + i1 * 32 + (4 * tidy + j1 + 8 * j2 + i2 * 32) * scn;
//             sc[idx] += d[i][j];
//         }
//     }
// #endif
// }


#define SHARED_SIZE 64
#define SM SHARED_SIZE
#define SK (SHARED_SIZE + 1)
#define SN (SHARED_SIZE + 1)
#define SIZE (SHARED_SIZE * (SHARED_SIZE + 1))


// __global__ void hip_hgemm_kernel_32x32x8f16(__half *A, __half *B, __half *C, int M, int N, int K) {
//     const size_t size = SHARED_SIZE * (SHARED_SIZE + 1);
//     __shared__ __half sa[SIZE], sb[SIZE];
//     __shared__ float sc[SIZE];
//         int sam = SHARED_SIZE;
//         int sak = SHARED_SIZE + 1;
//         int sbn = SHARED_SIZE;
//         int sbk = SHARED_SIZE + 1;
//         int scm = SHARED_SIZE;
//         int scn = SHARED_SIZE + 1;
//     for (int j = 0; j < scn; ++j) {
//         sc[j * scn + threadIdx.x] = 0.0f;
//     }

//     for (int i = 0; i < K; i += 64) {
//         for (int j = 0; j < 64; ++j) {
//             sa[j * sak + threadIdx.x] = A[(blockIdx.x * 64 + j) * K + threadIdx.x + i];
//             sb[j * sbk + threadIdx.x] = B[(blockIdx.y * 64 + j) * K + threadIdx.x + i];
//         }

//         hgemm_32x32x8_fp16_device(sa, sb, sc, sam, sak, sbn, sbk, scm, scn);
//     }

//     for (int j = 0; j < 64; j++) {
//         C[(blockIdx.x * 64 + j) * N + blockIdx.y * 64 + threadIdx.x] = sc[j * scn + threadIdx.x];
//     }
// }


// bool hip_matrix_mul_32x32x8_fp16(CMatrix<__half> &in1, CMatrix<__half> &in2, CMatrix<__half> &res, double& flops) {
//     size_t M, N, K1, K2, K;
//     in1.get_size(M, K1);
//     in2.get_size(K2, N);
//     assert(K1 == K2);
//     K = K1;
//     __half *da, *db, *dc;
//     hipMalloc((void**)&da, sizeof(__half) * M * K);
//     hipMalloc((void**)&db, sizeof(__half) * K * N);
//     hipMalloc((void**)&dc, sizeof(__half) * M * N);

//     hipMemcpy(da, in1.get_buffer(), sizeof(__half) * M * K, hipMemcpyHostToDevice);
//     hipMemcpy(db, in2.get_buffer(), sizeof(__half) * K * N, hipMemcpyHostToDevice);

//     dim3 grid(M/64, N/64);
//     dim3 block(64, 1);
//     hip_hgemm_kernel_32x32x8f16<<<grid, block>>>(da, db, dc, M, N, K);
//     hipError_t ret = hipGetLastError();
//     if (ret != hipSuccess) {
//         std::cout << "hip_matrix_mul_fp16_464, kernel launch error, code = " << ret << std::endl;
//         std::cout << "Error info: " << hipGetErrorString(ret) << std::endl;
//     }

//     hipMemcpy((void*)res.get_buffer(), dc, sizeof(__half) * M * N, hipMemcpyDeviceToHost);

//     return true;
// }

/*
 * use 1 wave and the mfma4x4x16xf16 instruction to do the computation, tile_size is 64 x 64
 * block dim(64, 1)
 */
__device__ void hgemm_4x4x16_int8_device(int8_t *sa, int8_t *sb, int32_t *sc, int sam, int sak, int sbn, int sbk, int scm, int scn) {
    using int8x4 = __attribute__((__vector_size__(4 * sizeof(int8_t)))) int8_t;
    using int32x4 = __attribute__((__vector_size__(4 * sizeof(int32_t)))) int32_t;

    int8x4 a8x4, b8x4;
    int32x4 d[16] = {0};
    for (int j = 0; j < 64; j += 4) {
        a8x4[0] = sa[threadIdx.x * sak + j];
        a8x4[1] = sa[threadIdx.x * sak + j + 1];
        a8x4[2] = sa[threadIdx.x * sak + j + 2];
        a8x4[3] = sa[threadIdx.x * sak + j + 3];

        b8x4[0] = sb[threadIdx.x * sbk + j];
        b8x4[1] = sb[threadIdx.x * sbk + j + 1];
        b8x4[2] = sb[threadIdx.x * sbk + j + 2];
        b8x4[3] = sb[threadIdx.x * sbk + j + 3];

        int a = (int)a8x4;
        int b = (int)b8x4;
        d[0] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, d[0], 4, 0, 0);
        d[1] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, d[1], 4, 1, 0);
        d[2] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, d[2], 4, 2, 0);
        d[3] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, d[3], 4, 3, 0);
        d[4] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, d[4], 4, 4, 0);
        d[5] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, d[5], 4, 5, 0);
        d[6] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, d[6], 4, 6, 0);
        d[7] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, d[7], 4, 7, 0);
        d[8] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, d[8], 4, 8, 0);
        d[9] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, d[9], 4, 9, 0);
        d[10] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, d[10], 4, 10, 0);
        d[11] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, d[11], 4, 11, 0);
        d[12] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, d[12], 4, 12, 0);
        d[13] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, d[13], 4, 13, 0);
        d[14] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, d[14], 4, 14, 0);
        d[15] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, d[15], 4, 15, 0);
    }

    for (int i = 0; i < 16; ++i) {
        sc[(4 * i + 0) * scn + threadIdx.x] += d[i][0];
        sc[(4 * i + 1) * scn + threadIdx.x] += d[i][1];
        sc[(4 * i + 2) * scn + threadIdx.x] += d[i][2];
        sc[(4 * i + 3) * scn + threadIdx.x] += d[i][3];
    }
}

__global__ void hip_hgemm_kernel_4x4x4_int8_464(int8_t *A, int8_t *B, int32_t *C, int M, int N, int K) {
    const size_t size = SHARED_SIZE * (SHARED_SIZE + 1);
    __shared__ int8_t sa[SIZE], sb[SIZE];
    __shared__ int32_t sc[SIZE];
    int sam = SHARED_SIZE;
    int sak = SHARED_SIZE + 1;
    int sbn = SHARED_SIZE;
    int sbk = SHARED_SIZE + 1;
    int scm = SHARED_SIZE;
    int scn = SHARED_SIZE + 1;

    for (int j = 0; j < scn; ++j) {
        sc[j * scn + threadIdx.x] = 0.0f;
    }

    for (int i = 0; i < K; i += 64) {
        for (int j = 0; j < 64; ++j) {
            sa[j * sak + threadIdx.x] = A[(blockIdx.x * 64 + j) * K + threadIdx.x + i];
            sb[j * sbk + threadIdx.x] = B[(blockIdx.y * 64 + j) * K + threadIdx.x + i];
        }

        hgemm_4x4x16_int8_device(sa, sb, sc, sam, sak, sbn, sbk, scm, scn);
    }

    for (int j = 0; j < 64; j++) {
        C[(blockIdx.x * 64 + j) * N + blockIdx.y * 64 + threadIdx.x] = sc[j * scn + threadIdx.x];
    }
}

bool hip_matrix_mul_4x4x4_int8_464(CMatrix<int8_t> &in1, CMatrix<int8_t> &in2, CMatrix<int32_t> &res, double& flops) {
    size_t M, N, K1, K2, K;
    in1.get_size(M, K1);
    in2.get_size(K2, N);
    assert(K1 == K2);
    K = K1;
    int8_t *da, *db;
    int32_t *dc;
    hipMalloc((void**)&da, sizeof(int8_t) * M * K);
    hipMalloc((void**)&db, sizeof(int8_t) * K * N);
    hipMalloc((void**)&dc, sizeof(int32_t) * M * N);

    hipMemcpy(da, in1.get_buffer(), sizeof(int8_t) * M * K, hipMemcpyHostToDevice);
    hipMemcpy(db, in2.get_buffer(), sizeof(int8_t) * K * N, hipMemcpyHostToDevice);

    dim3 grid(M/64, N/64);
    dim3 block(64, 1);
    hip_hgemm_kernel_4x4x4_int8_464<<<grid, block>>>(da, db, dc, M, N, K);
    hipError_t ret = hipGetLastError();
    if (ret != hipSuccess) {
        std::cout << "hip_hgemm_kernel_4x4x4_int8_464, kernel launch error, code = " << ret << std::endl;
        std::cout << "Error info: " << hipGetErrorString(ret) << std::endl;
    }

    hipMemcpy((void*)res.get_buffer(), dc, sizeof(int32_t) * M * N, hipMemcpyDeviceToHost);

    return true;
}

// bypass LDS implementation, block_size (4, 64)
__global__ void hip_hgemm_kernel_4x4x4_int8_464_bpl(int8_t *A, int8_t *B, int32_t *C, int M, int N, int K) {
    using int8x8 = __attribute__((__vector_size__(8 * sizeof(int8_t)))) int8_t;
    using int32x4 = __attribute__((__vector_size__(4 * sizeof(int32_t)))) int32_t;

    int8x4 a8x4, b8x4;
    int32x4 d[16] = {0};

    for (int k1 = 0; k1 < K; k1 += 4) {
        for (int k2 = 0; k2 < 4; ++k2) {
            a8x4[k2] = A[(blockIdx.x * 64 + threadIdx.x) * K + k1 + k2];
            b8x4[k2] = B[(blockIdx.y * 64 + threadIdx.x) * K + k1 + k2];
        }

        int a = (int)a8x4;
        int b = (int)b8x4;
        d[0] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, d[0], 4, 0, 0);
        d[1] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, d[1], 4, 1, 0);
        d[2] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, d[2], 4, 2, 0);
        d[3] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, d[3], 4, 3, 0);
        d[4] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, d[4], 4, 4, 0);
        d[5] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, d[5], 4, 5, 0);
        d[6] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, d[6], 4, 6, 0);
        d[7] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, d[7], 4, 7, 0);
        d[8] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, d[8], 4, 8, 0);
        d[9] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, d[9], 4, 9, 0);
        d[10] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, d[10], 4, 10, 0);
        d[11] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, d[11], 4, 11, 0);
        d[12] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, d[12], 4, 12, 0);
        d[13] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, d[13], 4, 13, 0);
        d[14] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, d[14], 4, 14, 0);
        d[15] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, d[15], 4, 15, 0);
    }

    for (int i = 0; i < 16; ++i) {
        C[(blockIdx.x * 64 + i * 4 + 0) * N + blockIdx.y * 64 + threadIdx.x] = d[i][0];
        C[(blockIdx.x * 64 + i * 4 + 1) * N + blockIdx.y * 64 + threadIdx.x] = d[i][1];
        C[(blockIdx.x * 64 + i * 4 + 2) * N + blockIdx.y * 64 + threadIdx.x] = d[i][2];
        C[(blockIdx.x * 64 + i * 4 + 3) * N + blockIdx.y * 64 + threadIdx.x] = d[i][3];
    }
}
