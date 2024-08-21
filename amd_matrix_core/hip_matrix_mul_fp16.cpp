#include <iostream>
#include <cstdlib>
#include <cstdio>
#include "matrixmul.hpp"
#include "hip_matrix_mul.hpp"
#include "timer.hpp"

/*
 * use 1 wave and the mfma32x32x8xf16 instruction to do the computation, tile_size is 64 x 64
 * block dim(32, 2)
*/
__device__ void hgemm_32x32x8_fp16_device(__half *A, __half *B, __half *C, int M, int K, int N) {
#if __gfx90a__ || __gfx908__ || __gfx942__
    // using float16x8 = __attribute__((__vector_size__(8 * sizeof(__half)))) __half;
    using float16x8 = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
    using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;

    float16x8 a[2];
    float16x8 b[2];
    floatx16 d[4] = {0};
    // first quarter
    for (int k = 0; k < 64; k += 8) {
        for (int i = 0; i < 4; ++i) {
            a[0][2 * i + threadIdx.y] = A[threadIdx.x * K + threadIdx.y + 2 * i + k];
            a[1][2 * i + threadIdx.y] = A[(threadIdx.x + 32) * K + threadIdx.y + 2 * i + k];
            b[0][2 * i + threadIdx.y] = B[threadIdx.x * N + threadIdx.y + 2 * i + k];
            b[1][2 * i + threadIdx.y] = B[(threadIdx.x + 32) * N + threadIdx.y + 2 * i + k];
        }

        d[0] = __builtin_amdgcn_mfma_f32_32x32x8f16(a[0], b[0], d[0], 0, 0, 0);
        d[1] = __builtin_amdgcn_mfma_f32_32x32x8f16(a[0], b[1], d[1], 0, 0, 0);
        d[2] = __builtin_amdgcn_mfma_f32_32x32x8f16(a[1], b[0], d[2], 0, 0, 0);
        d[3] = __builtin_amdgcn_mfma_f32_32x32x8f16(a[1], b[1], d[3], 0, 0, 0);
    }

    for (int i = 0; i < 4; ++i) {
        int i1 = i % 2;
        int i2 = i / 2;
        int offset = i1 * 32 + i2 * 32 * N;
        for (int j = 0; j < 16; ++j) {
            int j1 = j % 4;
            int j2 = j / 4;
            int idx = threadIdx.x + (4 * threadIdx.y + j1 + 8 * j2) * N + offset;
            C[idx] = d[i][j];
        }
    }
#endif
}


/*
 * use 1 wave and the mfma4x4x16xf16 instruction to do the computation, tile_size is 64 x 64
 * block dim(32, 2)
 */
__device__ void hgemm_4x4x16_fp16_device(__half *A, __half *B, __half *C, int M, int K, int N) {
    using float16x4 = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
    using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

    float16x4 a, b;
    floatx4 d[16] = {0};
    for (int j = 0; j < 64; j += 4) {
        a[0] = A[threadIdx.x * K + j];
        a[1] = A[threadIdx.x * K + j + 1];
        a[2] = A[threadIdx.x * K + j + 2];
        a[3] = A[threadIdx.x * K + j + 3];

        b[0] = B[threadIdx.x + j * N];
        b[1] = B[threadIdx.x + (j + 1) * N];
        b[2] = B[threadIdx.x + (j + 2) * N];
        b[3] = B[threadIdx.x + (j + 3) * N];

        d[0] = __builtin_amdgcn_mfma_f32_4x4x4f16(a, b, d[0], 4, 0, 0);
        d[1] = __builtin_amdgcn_mfma_f32_4x4x4f16(a, b, d[1], 4, 1, 0);
        d[2] = __builtin_amdgcn_mfma_f32_4x4x4f16(a, b, d[2], 4, 2, 0);
        d[3] = __builtin_amdgcn_mfma_f32_4x4x4f16(a, b, d[3], 4, 3, 0);
        d[4] = __builtin_amdgcn_mfma_f32_4x4x4f16(a, b, d[4], 4, 4, 0);
        d[5] = __builtin_amdgcn_mfma_f32_4x4x4f16(a, b, d[5], 4, 5, 0);
        d[6] = __builtin_amdgcn_mfma_f32_4x4x4f16(a, b, d[6], 4, 6, 0);
        d[7] = __builtin_amdgcn_mfma_f32_4x4x4f16(a, b, d[7], 4, 7, 0);
        d[8] = __builtin_amdgcn_mfma_f32_4x4x4f16(a, b, d[8], 4, 8, 0);
        d[9] = __builtin_amdgcn_mfma_f32_4x4x4f16(a, b, d[9], 4, 9, 0);
        d[10] = __builtin_amdgcn_mfma_f32_4x4x4f16(a, b, d[10], 4, 10, 0);
        d[11] = __builtin_amdgcn_mfma_f32_4x4x4f16(a, b, d[11], 4, 11, 0);
        d[12] = __builtin_amdgcn_mfma_f32_4x4x4f16(a, b, d[12], 4, 12, 0);
        d[13] = __builtin_amdgcn_mfma_f32_4x4x4f16(a, b, d[13], 4, 13, 0);
        d[14] = __builtin_amdgcn_mfma_f32_4x4x4f16(a, b, d[14], 4, 14, 0);
        d[15] = __builtin_amdgcn_mfma_f32_4x4x4f16(a, b, d[15], 4, 15, 0);
    }

    for (int i = 0; i < 16; ++i) {
        C[(4 * i + 0) * N + threadIdx.x] = d[i][0];
        C[(4 * i + 1) * N + threadIdx.x] = d[i][1];
        C[(4 * i + 2) * N + threadIdx.x] = d[i][2];
        C[(4 * i + 3) * N + threadIdx.x] = d[i][3];
    }
}


#define SHARED_SIZE 64
#define SM SHARED_SIZE
#define SK (SHARED_SIZE + 1)
#define SN (SHARED_SIZE + 1)
#define SIZE (SHARED_SIZE * (SHARED_SIZE + 1))


__global__ void hip_hgemm_kernel_32x32x8f16(__half *A, __half *B, __half *C, int M, int N, int K) {
    const size_t size = SHARED_SIZE * (SHARED_SIZE + 1);
    __shared__ __half a[SIZE], b[SIZE], c[SIZE];
    for (int i = 0; i < K; i += 64) {
        for (int j = 0; j < 64; ++j) {
            a[i * SK + threadIdx.x] = A[(blockIdx.x * 64 + i) * K + threadIdx.x + j];
            b[threadIdx.x * SN + i] = B[blockIdx.y * 64 + threadIdx.x + (j + i) * N];
        }

        hgemm_32x32x8_fp16_device(a, b, c, SHARED_SIZE, SHARED_SIZE + 1, N);

        for (int i = 0; i < 64; i++) {
            C[(blockIdx.x * SHARED_SIZE + i) * N + blockIdx.y * SHARED_SIZE + threadIdx.x] = c[i * SN + threadIdx.x];
        }
    }
}


bool hip_matrix_mul_fp16_464(CMatrix<__half> &in1, CMatrix<__half> &in2, CMatrix<__half> &res, double& flops) {
    size_t M, N, K1, K2, K;
    in1.get_size(M, K1);
    in2.get_size(K2, N);
    assert(K1 == K2);
    K = K1;
    __half *da, *db, *dc;
    hipMalloc((void**)&da, sizeof(__half) * M * K);
    hipMalloc((void**)&db, sizeof(__half) * K * N);
    hipMalloc((void**)&dc, sizeof(__half) * M * N);

    hipMemcpy(da, in1.get_buffer(), sizeof(__half) * M * K, hipMemcpyHostToDevice);
    hipMemcpy(db, in2.get_buffer(), sizeof(__half) * K * N, hipMemcpyHostToDevice);

    dim3 grid(M/64, N/64);
    dim3 block(1, 1);
    hip_hgemm_kernel_32x32x8f16<<<grid, block>>>(da, db, dc, M, N, K);
    hipMemcpy((void*)in2.get_buffer(), dc, sizeof(__half) * M * N, hipMemcpyDeviceToHost);

    return true;
}

