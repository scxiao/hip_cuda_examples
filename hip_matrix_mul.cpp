#include "matrixmul.h"
#include "hip_matrix_mul.h"
#include "hip/hip_runtime.h"

using namespace std;

__global__ void hipkernel_matrix_mul_naive(double *in1, double *in2, double *res,
        size_t row, size_t dim, size_t column) {
    int idx_x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int idx_y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (idx_x < row && idx_y < column) {
        double sum = 0.0;
        for (size_t k = 0; k < dim; ++k) {
            sum += in1[idx_x * dim + k] * in2[k + idx_y * dim];
        }
        res[idx_x * column + idx_y] = sum;
    }

    return;
}

bool hip_matrix_mul_naive(CMatrix<double> &in1, CMatrix<double> &in2, CMatrix<double> &res) {
    size_t row, dim1, dim2, column;
    in1.get_size(row, dim1);
    in2.get_size(dim2, column);

    if (dim1 != dim2) {
        cout << "Matrix dimensions mismatch!" << endl;
        return false;
    }
    res.resize(row, column);

    double* column_buffer2 = new double[dim2 * column];
    size_t i, j;
    for (i = 0; i < dim2; ++i) {
        for (j = 0; j < column; ++j) {
            column_buffer2[i + j * dim2] = in2.get_elem(i, j);
        }
    }

    double *hip_in1, *hip_in2, *hip_res;
    hipMalloc((void **)&hip_in1, sizeof(double) * row * dim1);
    hipMalloc((void **)&hip_in2, sizeof(double) * dim2 * column);
    hipMalloc((void **)&hip_res, sizeof(double) * row * column);

    hipMemcpy(hip_in1, in1.get_buffer(), sizeof(double) * row * dim1, hipMemcpyHostToDevice);
    hipMemcpy(hip_in2, column_buffer2, sizeof(double) * dim2 * column, hipMemcpyHostToDevice);
    size_t block_dimx = 32;
    size_t block_dimy = 32;

    hipLaunchKernelGGL(hipkernel_matrix_mul_naive, 
            dim3((row + block_dimx - 1)/block_dimx, (column + block_dimy - 1)/block_dimy), 
            dim3(block_dimx, block_dimy),
                    0, 0, hip_in1, hip_in2, hip_res, row, dim1, column);
    hipMemcpy(res.get_buffer(), hip_res, sizeof(double) * row * column, hipMemcpyDeviceToHost);

    delete[] column_buffer2;
    hipFree(hip_in1); hipFree(hip_in2); hipFree(hip_res);

    return true;
}

#define BLOCK_SIZE 8

__global__ void hipkernel_matrix_mul_shared(double *in1, double *in2, double *res,
        size_t row, size_t dim, size_t column) {
    //__shared__ double in_shared1[BLOCK_SIZE][BLOCK_SIZE];
    //__shared__ double in_shared2[BLOCK_SIZE][BLOCK_SIZE];
    HIP_DYNAMIC_SHARED(double, in_shared1);
    HIP_DYNAMIC_SHARED(double, in_shared2);

    size_t b_idx = hipBlockIdx_x;
    size_t b_idy = hipBlockIdx_y;
    size_t t_idx = hipThreadIdx_x;
    size_t t_idy = hipThreadIdx_y;

    double sum = 0.0;
    for (size_t tile_idx = 0; tile_idx < dim; tile_idx += BLOCK_SIZE) {
        in_shared1[t_idx * BLOCK_SIZE + t_idy] = in1[(b_idx * BLOCK_SIZE + t_idx) * dim + tile_idx + t_idy];
        in_shared2[t_idx * BLOCK_SIZE + t_idy] = in2[(tile_idx + t_idx) * column + b_idy * BLOCK_SIZE + t_idy];
        __syncthreads();

        for (size_t idx = 0; idx < BLOCK_SIZE; idx++) {
            sum += in_shared1[t_idx * BLOCK_SIZE + idx] * in_shared2[idx * BLOCK_SIZE + t_idy];
        }
        __syncthreads();
    }
    
    int idx_x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int idx_y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    res[idx_x * column + idx_y] = sum;

    return;
}


bool hip_matrix_mul_shared(CMatrix<double> &in1, CMatrix<double> &in2, CMatrix<double> &res) {
    size_t row, dim1, dim2, column;
    in1.get_size(row, dim1);
    in2.get_size(dim2, column);

    if (dim1 != dim2) {
        cout << "Matrix dimensions mismatch!" << endl;
        return false;
    }
    res.resize(row, column);

    double *hip_in1, *hip_in2, *hip_res;
    hipMalloc((void **)&hip_in1, sizeof(double) * row * dim1);
    hipMalloc((void **)&hip_in2, sizeof(double) * dim2 * column);
    hipMalloc((void **)&hip_res, sizeof(double) * row * column);

    hipMemcpy(hip_in1, in1.get_buffer(), sizeof(double) * row * dim1, hipMemcpyHostToDevice);
    hipMemcpy(hip_in2, in2.get_buffer(), sizeof(double) * dim2 * column, hipMemcpyHostToDevice);
    size_t tile_size = BLOCK_SIZE;

    hipLaunchKernelGGL(hipkernel_matrix_mul_shared, 
            dim3((row + tile_size - 1)/tile_size, (column + tile_size - 1)/tile_size), 
            dim3(tile_size, tile_size),
                    sizeof(double) * tile_size * tile_size, 0, hip_in1, hip_in2, hip_res, row, dim1, column);
    hipMemcpy(res.get_buffer(), hip_res, sizeof(double) * row * column, hipMemcpyDeviceToHost);

    hipFree(hip_in1); hipFree(hip_in2); hipFree(hip_res);

    return true;
}

