#include <functional>
#include "matrixmul.hpp"
#include "hip_matrix_mul.hpp"
#include "hip/hip_runtime.h"
#include "timer.hpp"

using namespace std;

__global__ void hipkernel_matrix_mul_naive(double *in1, double *in2, double *res,
        size_t row, size_t dim, size_t column) {
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx_x < row && idx_y < column) {
        double sum = 0.0;
        for (size_t k = 0; k < dim; ++k) {
            sum += in1[idx_x * dim + k] * in2[k * dim + idx_y];
        }
        res[idx_x * column + idx_y] = sum;
    }

    return;
}

#define BLOCK_SIZE 32

__global__ void hipkernel_matrix_mul_shared(double *in1, double *in2, double *res,
        size_t row, size_t dim, size_t column) {
    __shared__ double in_shared1[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ double in_shared2[BLOCK_SIZE][BLOCK_SIZE + 1];
    //HIP_DYNAMIC_SHARED(double, in_shared1);
    //HIP_DYNAMIC_SHARED(double, in_shared2);

    size_t b_idx = blockIdx.x;
    size_t b_idy = blockIdx.y;
    size_t t_idx = threadIdx.x;
    size_t t_idy = threadIdx.y;

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

__global__ void hipkernel_matrix_mul_dynamic_shared(double *in1, double *in2, double *res,
        size_t row, size_t dim, size_t column) {
    extern __shared__ double shared_mem[];
    double *in_shared1 = shared_mem;
    double *in_shared2 = shared_mem + BLOCK_SIZE * (BLOCK_SIZE + 1);
    // HIP_DYNAMIC_SHARED(double, in_shared1);
    // HIP_DYNAMIC_SHARED(double, in_shared2);

    size_t b_idx = blockIdx.x;
    size_t b_idy = blockIdx.y;
    size_t t_idx = threadIdx.x;
    size_t t_idy = threadIdx.y;

    double sum = 0.0;
    size_t stride = BLOCK_SIZE + 1;
    for (size_t tile_idx = 0; tile_idx < dim; tile_idx += BLOCK_SIZE) {
        in_shared1[t_idx * stride + t_idy] = in1[(b_idx * BLOCK_SIZE + t_idx) * dim + tile_idx + t_idy];
        in_shared2[t_idx * stride + t_idy] = in2[(tile_idx + t_idx) * column + b_idy * BLOCK_SIZE + t_idy];
        __syncthreads();

        for (size_t idx = 0; idx < BLOCK_SIZE; idx++) {
            sum += in_shared1[t_idx * stride + idx] * in_shared2[idx * stride + t_idy];
        }
        __syncthreads();
    }
    
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    res[idx_x * column + idx_y] = sum;

    return;
}

template<class T>
bool run_kernel(T kernel, const dim3& grid_size, const dim3& block_size, size_t lds_size,
                CMatrix<double> &in1, CMatrix<double> &in2, CMatrix<double> &res, double& perf_Tflops) {
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

    hipMemcpy(res.get_buffer(), hip_res, sizeof(double) * row * column, hipMemcpyDeviceToHost);
    hipFree(hip_in1); hipFree(hip_in2); hipFree(hip_res);

    return true;
}


bool hip_matrix_mul_naive_fp64(CMatrix<double> &in1, CMatrix<double> &in2, CMatrix<double> &res, double& flops) {
    size_t row, dim1, dim2, column;
    in1.get_size(row, dim1);
    in2.get_size(dim2, column);
    size_t block_dimx = 32;
    size_t block_dimy = 32;

    if (dim1 != dim2) {
        cout << "Matrix dimensions mismatch!" << endl;
        return false;
    }

    dim3 grid_dim = dim3((row + block_dimx - 1)/block_dimx, (column + block_dimy - 1)/block_dimy);
    dim3 block_dim = dim3(block_dimx, block_dimy);

    return run_kernel(hipkernel_matrix_mul_naive, grid_dim, block_dim, 0, in1, in2, res, flops);
}

bool hip_matrix_mul_shared_fp64(CMatrix<double> &in1, CMatrix<double> &in2, CMatrix<double> &res, double& flops) {
    size_t row, dim1, dim2, column;
    in1.get_size(row, dim1);
    in2.get_size(dim2, column);
    if (dim1 != dim2) {
        cout << "Matrix dimensions mismatch!" << endl;
        return false;
    }

    size_t tile_size = BLOCK_SIZE;
    dim3 grid_dim = dim3((row + tile_size - 1)/tile_size, (column + tile_size - 1)/tile_size);
    dim3 block_dim = dim3(tile_size, tile_size);
    size_t lds_size = 2 * tile_size * (tile_size + 1) * sizeof(double);

    return run_kernel(hipkernel_matrix_mul_dynamic_shared, grid_dim, block_dim, lds_size, in1, in2, res, flops);
}
