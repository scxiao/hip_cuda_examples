// compile with: hipcc --std=c++11 matmulbasic.cpp -o matmul
// run with: ./matmul
#include <cassert>
#include <iostream>
#include <vector>
#include "matrixmul.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void cuda_kernel_matrix_mul_naive(double *in1, double *in2, double *res,
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

    cout << "row = " << row << ", col = " << col << ", dim = " << dim << endl;
    CMatrix<double> matrixa(row, dim), matrixb(dim, col), res_matrix1, res_matrix2;
    bool ret = matrixa.multiply_optim(matrixb, res_matrix1);
    if (ret == false) {
        cout << "matrix dimension is incorrect, cannot multiplication." << endl;
        return 1;
    }

    cudaDeviceProp props;
    int device = 0;
    cudaGetDeviceProperties(&props, device);

    double *mat_ad, *mat_bd, *mat_cd;    
    assert( cudaMalloc(&mat_ad, sizeof(double) * row * dim) == 0);
    assert( cudaMalloc(&mat_bd, sizeof(double) * dim * col) == 0);
    assert( cudaMalloc(&mat_cd, sizeof(double) * row * col) == 0);
    cudaMemcpyAsync(mat_ad, matrixa.get_buffer(), sizeof(double) * row * dim, cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(mat_bd, matrixb.get_buffer(), sizeof(double) * dim * col, cudaMemcpyHostToDevice, 0);
    // Initializing variable Done

    int iter_num = 100;
    std::vector<cudaEvent_t> start(iter_num), stop(iter_num);
    for (int iter = 0; iter < iter_num; ++iter) {
        cudaEventCreate(&start[iter]);
        cudaEventCreate(&stop[iter]);
    }

	std::vector<void*> kernel_args;
	kernel_args.push_back(&mat_ad);
	kernel_args.push_back(&mat_bd);
	kernel_args.push_back(&mat_cd);
	kernel_args.push_back(&row);
	kernel_args.push_back(&dim);
	kernel_args.push_back(&col);


    // Launching Kernel Begin
    dim3 blockDim(32,32,1);
    int gridX = (row + blockDim.x - 1)/blockDim.x;
    int gridY = (col + blockDim.y - 1)/blockDim.y;
    dim3 gridDim(gridX,gridY,1);
    for (int iter = 0; iter < iter_num; ++iter) {
        cudaEventRecord(start[iter], 0);
        cudaLaunchKernel((void*)cuda_kernel_matrix_mul_naive, gridDim, blockDim, 
                        (void**)kernel_args.data(), (size_t)0, (cudaStream_t)0);
        cudaEventRecord(stop[iter], 0);
    }
    cudaStreamSynchronize(0); 
    assert(cudaGetLastError() == 0);
    // Launching Kernel Done
    res_matrix2.resize(row, col);
    cudaMemcpy(res_matrix2.get_buffer(), mat_cd, sizeof(double) * row * col, cudaMemcpyDeviceToHost);

    std::vector<float> gpu_time(iter_num, 0.0f);
    for (int iter = 0; iter < iter_num; ++iter) {
        cudaEventElapsedTime(&gpu_time[iter], start[iter], stop[iter]);
        std::cout << gpu_time[iter] << std::endl;
    }

    // Clean up
    assert (cudaFree(mat_ad) == 0);
    assert (cudaFree(mat_bd) == 0);
    assert (cudaFree(mat_cd) == 0);

    ret = (res_matrix1 == res_matrix2);
    if (ret == false) {
        cout << "Matrix multiplcation results mismatch!" << endl;
        std::cout << "FAILED!" << std::endl;
        return 1;
    }

    std::cout << "PASSED!" << std::endl;

    return 0;
}
