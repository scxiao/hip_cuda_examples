// compile with: hipcc --std=c++11 matmulbasic.cpp -o matmul
// run with: ./matmul
#include <cassert>
#include <iostream>
#include <vector>
#include "matrixmul.hpp"
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

__global__ void hipkernel_matrix_mul_naive(double *in1, double *in2, double *res,
        size_t row, size_t dim, size_t column) {
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx_x < row && idx_y < column) {
        double sum = 0.0;
        for (size_t k = 0; k < dim; ++k) {
            sum += in1[idx_x * dim + k] * in2[k * column + idx_y];
        }
        res[idx_x * column + idx_y] = sum;
    }

    return;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        cout << "Usage: " << argv[0] << " row k col" << endl;
        return 0;
    }

    size_t row = atoi(argv[1]);
    size_t dim = atoi(argv[2]);
    size_t col = atoi(argv[3]);
    size_t thread_num = atoi(argv[4]);

    cout << "row = " << row << ", col = " << col << ", dim = " << dim << endl;
    CMatrix<double> matrixa(row, dim), matrixb(dim, col), res_matrix1, res_matrix2;
    bool ret = matrixa.multiply_optim(matrixb, res_matrix1);
    if (ret == false) {
        cout << "matrix dimension is incorrect, cannot multiplication." << endl;
        return 1;
    }

    hipDeviceProp_t props;
    int device = 0;
    hipGetDeviceProperties(&props, device);

    double *mat_ad, *mat_bd, *mat_cd;    
    assert( hipMalloc(&mat_ad, sizeof(double) * row * dim) == 0);
    assert( hipMalloc(&mat_bd, sizeof(double) * dim * col) == 0);
    assert( hipMalloc(&mat_cd, sizeof(double) * row * col) == 0);
    hipMemcpyAsync(mat_ad, matrixa.get_buffer(), sizeof(double) * row * dim, hipMemcpyHostToDevice,0);
    hipMemcpyAsync(mat_bd, matrixb.get_buffer(), sizeof(double) * dim * col, hipMemcpyHostToDevice,0);
    // Initializing variable Done

    // Launching Kernel Begin
    dim3 blockDim(32,32,1);
    int gridX = (row + blockDim.x - 1)/blockDim.x;
    int gridY = (col + blockDim.y - 1)/blockDim.y;
    dim3 gridDim(gridX,gridY,1);
    hipLaunchKernelGGL(hipkernel_matrix_mul_naive, gridDim, blockDim, 
                       0/*dynamicShared*/, 0/*stream*/, 
                       mat_ad, mat_bd, mat_cd, row, dim, col);
    hipStreamSynchronize(0); 
    assert(hipGetLastError() == 0);
    // Launching Kernel Done
    res_matrix2.resize(row, col);
    hipMemcpy(res_matrix2.get_buffer(), mat_cd, sizeof(double) * row * col, hipMemcpyDeviceToHost);

    // Clean up
    assert (hipFree(mat_ad) == 0);
    assert (hipFree(mat_bd) == 0);
    assert (hipFree(mat_cd) == 0);

    ret = (res_matrix1 == res_matrix2);
    if (ret == false) {
        cout << "Matrix multiplcation results mismatch!" << endl;
        std::cout << "FAILED!" << std::endl;
        return 1;
    }

    std::cout << "PASSED!" << std::endl;

    return 0;
}
