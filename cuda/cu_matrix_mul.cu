#include "matrixmul.hpp"
#include "cu_matrix_mul.hpp"
#include <cuda.h>

using namespace std;

__global__ void kernel_matrix_mul_naive(double *in1, double *in2, double *res,
        size_t row, size_t dim, size_t column) {
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx_x < row && idx_y < column) {
        double sum = 0.0;
        for (size_t k = 0; k < dim; ++k) {
            sum += in1[idx_x * dim + k] * in2[k + idx_y * dim];
        }
        res[idx_x * column + idx_y] = sum;
    }

    return;
}

bool cu_matrix_mul_naive(CMatrix<double> &in1, CMatrix<double> &in2, CMatrix<double> &res) {
    size_t row, dim1, dim2, column;
    in1.get_size(row, dim1);
    in2.get_size(dim2, column);

    if (dim1 != dim2) {
        cout << "Matrix dimensions mismatch!" << endl;
        return false;
    }
    res.resize(row, column);
    res.reset();

    double* column_buffer2 = new double[dim2 * column];
    size_t i, j;
    for (i = 0; i < dim2; ++i) {
        for (j = 0; j < column; ++j) {
            column_buffer2[i + j * dim2] = in2.get_elem(i, j);
        }
    }

    double *cu_in1, *cu_in2, *cu_res;
    cudaMalloc((void **)&cu_in1, sizeof(double) * row * dim1);
    cudaMalloc((void **)&cu_in2, sizeof(double) * dim2 * column);
    cudaMalloc((void **)&cu_res, sizeof(double) * row * column);

    cudaMemcpy(cu_in1, in1.get_buffer(), sizeof(double) * row * dim1, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_in2, column_buffer2, sizeof(double) * dim2 * column, cudaMemcpyHostToDevice);
    size_t block_dimx = 32;
    size_t block_dimy = 32;

    kernel_matrix_mul_naive<<<dim3((row + block_dimx - 1)/block_dimx, (column + block_dimy - 1)/block_dimy),
            dim3(block_dimx, block_dimy)>>> (cu_in1, cu_in2, cu_res, row, dim1, column);
    cudaMemcpy(res.get_buffer(), cu_res, sizeof(double) * row * column, cudaMemcpyDeviceToHost);

    delete[] column_buffer2;
    cudaFree(cu_in1); cudaFree(cu_in2); cudaFree(cu_res);

    return true;
}

#define BLOCK_SIZE 32

__global__ void kernel_matrix_mul_shared(double *in1, double *in2, double *res,
        size_t row, size_t dim, size_t column) {
    __shared__ double in_shared1[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ double in_shared2[BLOCK_SIZE][BLOCK_SIZE + 1];

    size_t b_idx = blockIdx.x;
    size_t b_idy = blockIdx.y;
    size_t t_idx = threadIdx.x;
    size_t t_idy = threadIdx.y;

    // There might be a bug here to be fixed
    double sum = 0.0;
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


bool cu_matrix_mul_shared(CMatrix<double> &in1, CMatrix<double> &in2, CMatrix<double> &res) {
    size_t row, dim1, dim2, column;
    in1.get_size(row, dim1);
    in2.get_size(dim2, column);

    if (dim1 != dim2) {
        cout << "Matrix dimensions mismatch!" << endl;
        return false;
    }
    res.resize(row, column);
    res.reset();

    double *cu_in1, *cu_in2, *cu_res;
    cudaMalloc((void **)&cu_in1, sizeof(double) * row * dim1);
    cudaMalloc((void **)&cu_in2, sizeof(double) * dim2 * column);
    cudaMalloc((void **)&cu_res, sizeof(double) * row * column);

    cudaMemcpy(cu_in1, in1.get_buffer(), sizeof(double) * row * dim1, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_in2, in2.get_buffer(), sizeof(double) * dim2 * column, cudaMemcpyHostToDevice);
    size_t tile_size = BLOCK_SIZE;

    kernel_matrix_mul_shared<<<dim3((row + tile_size - 1)/tile_size, (column + tile_size - 1)/tile_size), 
            dim3(tile_size, tile_size)>>>(cu_in1, cu_in2, cu_res, row, dim1, column);
    cudaMemcpy(res.get_buffer(), cu_res, sizeof(double) * row * column, cudaMemcpyDeviceToHost);

    cudaFree(cu_in1); cudaFree(cu_in2); cudaFree(cu_res);

    return true;
}

// block size (16, 4), tile_size = 64 X 16
__global__ void kernel_matrix_mul_cublas(double *in1, double *in2, double *res,
        size_t row, size_t dim, size_t column) {
    const size_t tile_size = 16;
    __shared__ double in_shared[tile_size][tile_size];

    size_t b_idx = blockIdx.x;
    size_t b_idy = blockIdx.y;
    size_t t_idx = threadIdx.x;
    size_t t_idy = threadIdx.y;

    double sum[16] = {0.0};
    for (size_t tile_idx = 0; tile_idx < dim; tile_idx += 16) {
        in_shared[t_idx][t_idy] = in2[(tile_idx + t_idx) * column + (b_idy * 16 + t_idy)];
        in_shared[t_idx][t_idy + 4]  = in2[(tile_idx + t_idx) * column + (b_idy * 16 + t_idy + 4)];
        in_shared[t_idx][t_idy + 8]  = in2[(tile_idx + t_idx) * column + (b_idy * 16 + t_idy + 8)];
        in_shared[t_idx][t_idy + 12] = in2[(tile_idx + t_idx) * column + (b_idy * 16 + t_idy + 12)];
        __syncthreads();

        for (size_t l = 0; l < 16; l++) {
            double r = in1[(b_idx * 64 + 16 * t_idy + t_idx) * dim + tile_idx + l];
            sum[0] += r * in_shared[l][0];
            sum[1] += r * in_shared[l][1];
            sum[2] += r * in_shared[l][2];
            sum[3] += r * in_shared[l][3];
            sum[4] += r * in_shared[l][4];
            sum[5] += r * in_shared[l][5];
            sum[6] += r * in_shared[l][6];
            sum[7] += r * in_shared[l][7];
            sum[8] += r * in_shared[l][8];
            sum[9] += r * in_shared[l][9];
            sum[10] += r * in_shared[l][10];
            sum[11] += r * in_shared[l][11];
            sum[12] += r * in_shared[l][12];
            sum[13] += r * in_shared[l][13];
            sum[14] += r * in_shared[l][14];
            sum[15] += r * in_shared[l][15];
        }
        __syncthreads();
    }

    for (size_t l = 0; l < 16; l++) {
        res[(b_idx * 64 + 16 * t_idy + t_idx) * column + b_idy * 16 + l] = sum[l];
    }

    return;
}

// block size (32, 4), tile_size = 128 X 32
//__global__ void kernel_matrix_mul_cublas(double *in1, double *in2, double *res,
//        size_t row, size_t dim, size_t column) {
//    const size_t tile_size = 32;
//    __shared__ double in_shared[tile_size][tile_size];
//
//    size_t b_idx = blockIdx.x;
//    size_t b_idy = blockIdx.y;
//    size_t t_idx = threadIdx.x;
//    size_t t_idy = threadIdx.y;
//
//    double sum[32] = {0.0};
//    for (size_t tile_idx = 0; tile_idx < dim; tile_idx += 32) {
//        in_shared[t_idx][t_idy] = in2[(tile_idx + t_idx) * column + (b_idy * 32 + t_idy)];
//        in_shared[t_idx][t_idy + 4]  = in2[(tile_idx + t_idx) * column + (b_idy * 32 + t_idy + 4)];
//        in_shared[t_idx][t_idy + 8]  = in2[(tile_idx + t_idx) * column + (b_idy * 32 + t_idy + 8)];
//        in_shared[t_idx][t_idy + 12] = in2[(tile_idx + t_idx) * column + (b_idy * 32 + t_idy + 12)];
//        in_shared[t_idx][t_idy + 16] = in2[(tile_idx + t_idx) * column + (b_idy * 32 + t_idy + 16)];
//        in_shared[t_idx][t_idy + 20] = in2[(tile_idx + t_idx) * column + (b_idy * 32 + t_idy + 20)];
//        in_shared[t_idx][t_idy + 24] = in2[(tile_idx + t_idx) * column + (b_idy * 32 + t_idy + 24)];
//        in_shared[t_idx][t_idy + 28] = in2[(tile_idx + t_idx) * column + (b_idy * 32 + t_idy + 28)];
//        __syncthreads();
//
//        for (size_t l = 0; l < 32; l++) {
//            double r = in1[(b_idx * 128 + 32 * t_idy + t_idx) * dim + tile_idx + l];
//            sum[0] += r * in_shared[l][0];
//            sum[1] += r * in_shared[l][1];
//            sum[2] += r * in_shared[l][2];
//            sum[3] += r * in_shared[l][3];
//            sum[4] += r * in_shared[l][4];
//            sum[5] += r * in_shared[l][5];
//            sum[6] += r * in_shared[l][6];
//            sum[7] += r * in_shared[l][7];
//            sum[8] += r * in_shared[l][8];
//            sum[9] += r * in_shared[l][9];
//            sum[10] += r * in_shared[l][10];
//            sum[11] += r * in_shared[l][11];
//            sum[12] += r * in_shared[l][12];
//            sum[13] += r * in_shared[l][13];
//            sum[14] += r * in_shared[l][14];
//            sum[15] += r * in_shared[l][15];
//            sum[16] += r * in_shared[l][16];
//            sum[17] += r * in_shared[l][17];
//            sum[18] += r * in_shared[l][18];
//            sum[19] += r * in_shared[l][19];
//            sum[20] += r * in_shared[l][20];
//            sum[21] += r * in_shared[l][21];
//            sum[22] += r * in_shared[l][22];
//            sum[23] += r * in_shared[l][23];
//            sum[24] += r * in_shared[l][24];
//            sum[25] += r * in_shared[l][25];
//            sum[26] += r * in_shared[l][26];
//            sum[27] += r * in_shared[l][27];
//            sum[28] += r * in_shared[l][28];
//            sum[29] += r * in_shared[l][29];
//            sum[30] += r * in_shared[l][30];
//            sum[31] += r * in_shared[l][31];
//        }
//        __syncthreads();
//    }
//
//    for (size_t l = 0; l < 32; l++) {
//        res[(b_idx * 128 + 32 * t_idy + t_idx) * column + b_idy * 32 + l] = sum[l];
//    }
//
//    return;
//}

// column-major format of input and output matrix
//__global__ void kernel_matrix_mul_cublas(double *in1, double *in2, double *res,
//        size_t row, size_t dim, size_t column) {
//    const size_t tile_size = 16;
//    __shared__ double in_shared[tile_size][tile_size];
//
//    size_t b_idx = blockIdx.x;
//    size_t b_idy = blockIdx.y;
//    size_t t_idx = threadIdx.x;
//    size_t t_idy = threadIdx.y;
//
//    double sum[16] = {0.0};
//    for (size_t tile_idx = 0; tile_idx < dim; tile_idx += 16) {
//        in_shared[t_idx][t_idy]      = in2[(tile_idx + t_idx) + (b_idy * 16 + t_idy) * dim];
//        in_shared[t_idx][t_idy + 4]  = in2[(tile_idx + t_idx) + (b_idy * 16 + t_idy + 4) * dim];
//        in_shared[t_idx][t_idy + 8]  = in2[(tile_idx + t_idx) + (b_idy * 16 + t_idy + 8) * dim];
//        in_shared[t_idx][t_idy + 12] = in2[(tile_idx + t_idx) + (b_idy * 16 + t_idy + 12) * dim];
//        __syncthreads();
//
//        for (size_t l = 0; l < 16; l++) {
//            double r = in1[(b_idx * 64 + 16 * t_idy + t_idx) + (tile_idx + l) * dim];
//            sum[0] += r * in_shared[l][0];
//            sum[1] += r * in_shared[l][1];
//            sum[2] += r * in_shared[l][2];
//            sum[3] += r * in_shared[l][3];
//            sum[4] += r * in_shared[l][4];
//            sum[5] += r * in_shared[l][5];
//            sum[6] += r * in_shared[l][6];
//            sum[7] += r * in_shared[l][7];
//            sum[8] += r * in_shared[l][8];
//            sum[9] += r * in_shared[l][9];
//            sum[10] += r * in_shared[l][10];
//            sum[11] += r * in_shared[l][11];
//            sum[12] += r * in_shared[l][12];
//            sum[13] += r * in_shared[l][13];
//            sum[14] += r * in_shared[l][14];
//            sum[15] += r * in_shared[l][15];
//        }
//        __syncthreads();
//    }
//
//    for (size_t l = 0; l < 16; l++) {
//        res[(b_idx * 64 + 16 * t_idy + t_idx) + (b_idy * 16 + l) * column] = sum[l];
//    }
//
//    return;
//}


bool cu_matrix_mul_cublas(CMatrix<double> &in1, CMatrix<double> &in2, CMatrix<double> &res) {
    size_t row, dim1, dim2, column;
    in1.get_size(row, dim1);
    in2.get_size(dim2, column);

    if (dim1 != dim2) {
        cout << "Matrix dimensions mismatch!" << endl;
        return false;
    }
    res.resize(row, column);
    res.reset();

    double *cu_in1, *cu_in2, *cu_res;
    cudaMalloc((void **)&cu_in1, sizeof(double) * row * dim1);
    cudaMalloc((void **)&cu_in2, sizeof(double) * dim2 * column);
    cudaMalloc((void **)&cu_res, sizeof(double) * row * column);

    cudaMemcpy(cu_in1, in1.get_buffer(), sizeof(double) * row * dim1, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_in2, in2.get_buffer(), sizeof(double) * dim2 * column, cudaMemcpyHostToDevice);

    kernel_matrix_mul_cublas<<<dim3((row + 63)/64, (column + 15)/16), 
            dim3(16, 4)>>>(cu_in1, cu_in2, cu_res, row, dim1, column);
    cudaMemcpy(res.get_buffer(), cu_res, sizeof(double) * row * column, cudaMemcpyDeviceToHost);

    cudaFree(cu_in1); cudaFree(cu_in2); cudaFree(cu_res);

    return true;
}

__global__ void kernel_matrix_mul_magma(double *in1, double *in2, double *res,
        size_t row, size_t dim, size_t column) {
    __shared__ double in_shared1[64][16];
    __shared__ double in_shared2[16][64];

    size_t b_idx = blockIdx.x;
    size_t b_idy = blockIdx.y;
    size_t t_idx = threadIdx.x;
    size_t t_idy = threadIdx.y;

    // There might be a bug here to be fixed
    double sum[16] = {0.0};
    for (size_t tile_idx = 0; tile_idx < dim; tile_idx += 16) {
        for (size_t sidx = 0; sidx < 64; sidx += 16) {
            in_shared1[t_idx + sidx][t_idy] = in1[(b_idx * 64 + t_idx + sidx) * dim + tile_idx + t_idy];
            in_shared2[t_idx][sidx + t_idy] = in2[(tile_idx + t_idx) * column + b_idy * 64 + sidx + t_idy];
        }
        __syncthreads();

        for (size_t l = 0; l < 16; l++) {
            double r1[4], r2[4];
            r1[0] = in_shared1[t_idx][l];
            r1[1] = in_shared1[t_idx + 16][l];
            r1[2] = in_shared1[t_idx + 32][l];
            r1[3] = in_shared1[t_idx + 48][l];
            r2[0] = in_shared2[l][t_idy];
            r2[1] = in_shared2[l][t_idy + 16];
            r2[2] = in_shared2[l][t_idy + 32];
            r2[3] = in_shared2[l][t_idy + 48];

            sum[0] += r1[0] * r2[0];
            sum[1] += r1[0] * r2[1];
            sum[2] += r1[0] * r2[2];
            sum[3] += r1[0] * r2[3];
            sum[4] += r1[1] * r2[0];
            sum[5] += r1[1] * r2[1];
            sum[6] += r1[1] * r2[2];
            sum[7] += r1[1] * r2[3];
            sum[8] += r1[2] * r2[0];
            sum[9] += r1[2] * r2[1];
            sum[10] += r1[2] * r2[2];
            sum[11] += r1[2] * r2[3];
            sum[12] += r1[3] * r2[0];
            sum[13] += r1[3] * r2[1];
            sum[14] += r1[3] * r2[2];
            sum[15] += r1[3] * r2[3];
        }
       __syncthreads();
    }
    
    res[(b_idx * 64 + t_idx) * column + b_idy * 64 + t_idy] = sum[0];
    res[(b_idx * 64 + t_idx) * column + b_idy * 64 + t_idy + 16] = sum[1];
    res[(b_idx * 64 + t_idx) * column + b_idy * 64 + t_idy + 32] = sum[2];
    res[(b_idx * 64 + t_idx) * column + b_idy * 64 + t_idy + 48] = sum[3];
    res[(b_idx * 64 + t_idx + 16) * column + b_idy * 64 + t_idy] = sum[4];
    res[(b_idx * 64 + t_idx + 16) * column + b_idy * 64 + t_idy + 16] = sum[5];
    res[(b_idx * 64 + t_idx + 16) * column + b_idy * 64 + t_idy + 32] = sum[6];
    res[(b_idx * 64 + t_idx + 16) * column + b_idy * 64 + t_idy + 48] = sum[7];
    res[(b_idx * 64 + t_idx + 32) * column + b_idy * 64 + t_idy] = sum[8];
    res[(b_idx * 64 + t_idx + 32) * column + b_idy * 64 + t_idy + 16] = sum[9];
    res[(b_idx * 64 + t_idx + 32) * column + b_idy * 64 + t_idy + 32] = sum[10];
    res[(b_idx * 64 + t_idx + 32) * column + b_idy * 64 + t_idy + 48] = sum[11];
    res[(b_idx * 64 + t_idx + 48) * column + b_idy * 64 + t_idy] = sum[12];
    res[(b_idx * 64 + t_idx + 48) * column + b_idy * 64 + t_idy + 16] = sum[13];
    res[(b_idx * 64 + t_idx + 48) * column + b_idy * 64 + t_idy + 32] = sum[14];
    res[(b_idx * 64 + t_idx + 48) * column + b_idy * 64 + t_idy + 48] = sum[15];

    return;
}

bool cu_matrix_mul_magma(CMatrix<double> &in1, CMatrix<double> &in2, CMatrix<double> &res) {
    size_t row, dim1, dim2, column;
    in1.get_size(row, dim1);
    in2.get_size(dim2, column);

    if (dim1 != dim2) {
        cout << "Matrix dimensions mismatch!" << endl;
        return false;
    }
    res.resize(row, column);
    res.reset();

    double *cu_in1, *cu_in2, *cu_res;
    cudaMalloc((void **)&cu_in1, sizeof(double) * row * dim1);
    cudaMalloc((void **)&cu_in2, sizeof(double) * dim2 * column);
    cudaMalloc((void **)&cu_res, sizeof(double) * row * column);

    cudaMemcpy(cu_in1, in1.get_buffer(), sizeof(double) * row * dim1, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_in2, in2.get_buffer(), sizeof(double) * dim2 * column, cudaMemcpyHostToDevice);

    kernel_matrix_mul_magma<<<dim3((row + 63)/64, (column + 63)/64), 
            dim3(16, 16)>>>(cu_in1, cu_in2, cu_res, row, dim1, column);
    cudaMemcpy(res.get_buffer(), cu_res, sizeof(double) * row * column, cudaMemcpyDeviceToHost);

    cudaFree(cu_in1); cudaFree(cu_in2); cudaFree(cu_res);

    return true;
}

