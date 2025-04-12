#include <iostream>
#include <cstdlib>
#include <cstdio>
#include "matrixmul.hpp"
#include "hip_matrix_mul.hpp"
#include "timer.hpp"

__global__ void hip_sparse_hgemm_naive_f16(__half *A, int *sparse_idx, __half *B, float *C, int M, int N, int Ka, int Kb) {
    int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int id_y = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0;
    // process 4 element for each interation
    for (int k = 0; k < Kb; k += 4) {
        int a_id = k / 2;
        for (int j = 0; j < 2; j++) {
            sum += (float)(A[id_x * Ka + a_id + j] * B[id_y * Kb + k + sparse_idx[id_x * Ka + a_id + j]]);
        }
    }
    C[id_x * N + id_y] = sum;
}

bool hip_sparse_matrix_mul_f16_naive(CMatrix<__half> &in1, CMatrix<int>& idx, CMatrix<__half> &in2, CMatrix<float> &res, double& flops) {
    size_t M, N, K1, K2;
    in1.get_size(M, K1);
    in2.get_size(K2, N);
    __half *da, *db;
    int *da_i;
    float *dc;
    hipMalloc((void**)&da, sizeof(__half) * M * K1);
    hipMalloc((void**)&da_i, sizeof(int) * M * K1);
    hipMalloc((void**)&db, sizeof(__half) * K2 * N);
    hipMalloc((void**)&dc, sizeof(float) * M * N);

    hipMemcpy(da, in1.get_buffer(), sizeof(__half) * M * K1, hipMemcpyHostToDevice);
    hipMemcpy(da_i, idx.get_buffer(), sizeof(int) * M * K1, hipMemcpyHostToDevice);
    hipMemcpy(db, in2.get_buffer(), sizeof(__half) * K2 * N, hipMemcpyHostToDevice);
    dim3 grid(1, 1);
    dim3 block(32, 32);
    hip_sparse_hgemm_naive_f16<<<grid, block>>>(da, da_i, db, dc, M, N, K1, K2);
    hipError_t ret = hipGetLastError();
    if (ret != hipSuccess) {
        std::cout << "hip_sparse_hgemm_naive_f16, kernel launch error, code = " << ret << std::endl;
        std::cout << "Error info: " << hipGetErrorString(ret) << std::endl;
    }

    res.resize(M, N);
    hipMemcpy((void*)res.get_buffer(), dc, sizeof(float) * M * N, hipMemcpyDeviceToHost);

    return true;
}

/*
 * use 1 wave and the mfma32x32x8xf16 instruction to do the computation, tile_size is 64 x 64
 * block dim(32, 2)
*/
__device__ void sparse_gemm_32x32x16_fp16_device(__half *sa, int *sparse_idx, __half *sb, float *sc, int am, int ak, int bn, int bk, int cm, int cn) {
#if __gfx90a__ || __gfx908__ || __gfx942__
    // For A input
    using float16x4 = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
    // For B input
    using float16x8 = __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
    using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
    // threadsPerWarp [32, 2]
    int tid = threadIdx.x;
    int tidx = threadIdx.x % 32;
    int tidy = threadIdx.x / 32;

    float16x4 a;
    float16x8 b;
    int idx = sparse_idx[tid]; // sparse_idx
    floatx16 d = {0};

    for (int i = 0; i < 4; ++i) {
        // a[0] is for left half of sa, a[1] is for right half of sa
        a[i] = sa[tidx * ak + i + 4 * tidy];
    }
    for (int i = 0; i < 8; ++i) {
        b[i] = sb[tidx * bk + i + 8 * tidy];
    }
    d = __builtin_amdgcn_smfmac_f32_32x32x16_f16(a, b, d, idx, 0, 0);

    for (int j = 0; j < 16; ++j) {
        int j1 = j % 4;
        int j2 = j / 4;
        int idx = tidx + (4 * tidy + j1 + 8 * j2) * cn;
        sc[idx] = d[j];
    }
#endif
}


#define SHARED_SIZE 32
#define SM SHARED_SIZE
#define SK 16
#define SN SHARED_SIZE
#define SIZE (SHARED_SIZE * (SK + 1))


__global__ void hip_sparse_hgemm_kernel_32x32x16f16(__half *A, int *idx, __half *B, float *C, int M, int Ka, int N, int Kb) {
    sparse_gemm_32x32x16_fp16_device(A, idx, B, C, M, Ka, N, Kb, M, N);
}

std::vector<int> compressSparseIndex(CMatrix<int> &sparse_indices) {
    std::vector<int> result;
    std::size_t rowNum, colNum;
    sparse_indices.get_size(rowNum, colNum);
    assert(rowNum % 4 == 0);
    for (int c = 0; c < colNum; c += 4) {
        for (int r = 0; r < rowNum; ++r) {
            int v = 0;
            int idx = sparse_indices.get_elem(r, c);
            v |= (idx & 0x3);
            idx = sparse_indices.get_elem(r, c + 1);
            v |= ((idx & 0x3) << 2);
            idx = sparse_indices.get_elem(r, c + 2);
            v |= ((idx & 0x3) << 4);
            idx = sparse_indices.get_elem(r, c + 3);
            v |= ((idx & 0x3) << 6);
            result.push_back(v);
        }
    }

    return result;
}

bool hip_sparse_matrix_mul_32x32x16_fp16(CMatrix<__half> &in1, CMatrix<int>& idx, CMatrix<__half> &in2, CMatrix<float> &res, double& flops) {
    size_t M, N, K1, K2;
    in1.get_size(M, K1);
    in2.get_size(K2, N);
    __half *da, *db;
    int *da_i;
    float *dc;

    // comparess index input
    std::vector<int> compressed_idx = compressSparseIndex(idx);
    hipMalloc((void**)&da, sizeof(__half) * M * K1);
    hipMalloc((void**)&da_i, sizeof(int) * compressed_idx.size());
    hipMalloc((void**)&db, sizeof(__half) * K2 * N);
    hipMalloc((void**)&dc, sizeof(float) * M * N);

    hipMemcpy(da, in1.get_buffer(), sizeof(__half) * M * K1, hipMemcpyHostToDevice);
    hipMemcpy(da_i, compressed_idx.data(), sizeof(int) * compressed_idx.size(), hipMemcpyHostToDevice);
    hipMemcpy(db, in2.get_buffer(), sizeof(__half) * K2 * N, hipMemcpyHostToDevice);

    dim3 grid(1, 1);
    dim3 block(64, 1);
    hip_sparse_hgemm_kernel_32x32x16f16<<<grid, block>>>(da, da_i, db, dc, M, K1, N, K2);
    hipError_t ret = hipGetLastError();
    if (ret != hipSuccess) {
        std::cout << "hip_hgemm_kernel_32x32x16f16, kernel launch error, code = " << ret << std::endl;
        std::cout << "Error info: " << hipGetErrorString(ret) << std::endl;
    }

    res.resize(M, N);
    hipMemcpy((void*)res.get_buffer(), dc, sizeof(float) * M * N, hipMemcpyDeviceToHost);

    return true;
}

