#include <iostream>
#include <cstdlib>
#include <cstdio>
#include "matrixmul.hpp"
#include "hip_matrix_mul.hpp"
#include "timer.hpp"

using namespace std;

int main(int argc, char **argv) {
    if (argc != 5) {
        cout << "Usage: " << argv[0] << " i k j thread_num" << endl;
        return 0;
    }

    size_t i = atoi(argv[1]);
    size_t k = atoi(argv[2]);
    size_t j = atoi(argv[3]);
    size_t thread_num = atoi(argv[4]);

    cout << "i = " << i << ", j = " << j << ", k = " << k << endl;

    CMatrix<float> matrix1(i, k), matrix2(k, j), res_matrix1, res_matrix2;
    HRTimer timer;
    double flops;

    timer.start();
    bool ret = matrix1.multiply_optim(matrix2, res_matrix1);
    if (ret == false) {
        cout << "matrix dimension is incorrect, cannot multiplication." << endl;
        return 1;
    }
    timer.stop();
    cout << "Sequential time = ";
    timer.printtime_ms();

    size_t num = 4;
    while (num <= thread_num) {
        timer.start();
        ret = matrix1.multiply_parallel(matrix2, num, res_matrix2);
        if (ret == false) {
            cout << "matrix dimension is incorrect, cannot multiplication." << endl;
            return 1;
        }
        timer.stop();
        size_t kernel_time_us = timer.gettime_us();
        double flops = 2.0 * i * j * k / kernel_time_us / 1000000;

        cout << "Parallel time with " << num 
             << " threads = " << kernel_time_us 
             << " us, flops = " << flops << " TFLOPS" << std::endl;

        ret = (res_matrix1 == res_matrix2);
        if (ret == false) {
            cout << "FAILED with " << num << " threads." << endl;
            return 1;
        }

        num *= 2;
    }

    // call hip naive implementation to run on GPU
    timer.start();
    ret = hip_matrix_mul_naive_fp32(matrix1, matrix2, res_matrix2, flops);
    if (ret == false) {
        cout << "cu_matrix_mul_naive failed, Matrix dimension mismatch!" << endl;
        return 1;
    }
    timer.stop();
    size_t kernel_time_us = timer.gettime_us(); 
    cout << "hip naive implementation = " << kernel_time_us << " us, flops = " << flops << " TFLOPS" << std::endl;
    ret = (res_matrix1 == res_matrix2);
    if (ret == false) {
        cout << "hip naive implementation failed." << endl;
        return 1;
    }

    // call hip shared implementation to run on GPU
    timer.start();
    ret = hip_matrix_mul_shared_fp32(matrix1, matrix2, res_matrix2, flops);
    if (ret == false) {
        cout << "hip_matrix_mul_shared failed, Matrix dimension mismatch!" << endl;
        return 1;
    }
    timer.stop();
    kernel_time_us = timer.gettime_us(); 
    cout << "hip shared implementation = " << kernel_time_us << " us, flops = " << flops << " TFLOPS" << std::endl;
    ret = (res_matrix1 == res_matrix2);
    if (ret == false) {
        cout << "hip shared implementation failed." << endl;
        return 1;
    }

    // call mfma fp32_16x16x4fp32 implementation to run on GPU
    res_matrix2.reset();
    timer.start();
    ret = hip_matrix_mul_sgemm_32x32x32_fp32(matrix1, matrix2, res_matrix2, flops);
    if (ret == false) {
        cout << "hip_matrix_mul_sgemm_32x32x32_fp32 failed, Matrix dimension mismatch!" << endl;
        return 1;
    }
    timer.stop();
    kernel_time_us = timer.gettime_us(); 
    cout << "hip hip_matrix_mul_sgemm_32x32x32_fp32 implementation = " << kernel_time_us << " us, flops = " << flops << " TFLOPS" << std::endl;
    ret = (res_matrix1 == res_matrix2);
    if (ret == false) {
        cout << "hip_matrix_mul_sgemm_32x32x32_fp32 failed." << endl;
        return 1;
    }

 //   // call hip rocblas implementation to run on GPU
 //   timer.start();
 //   ret = cu_matrix_mul_cublas(matrix1, matrix2, res_matrix2);
 //   if (ret == false) {
 //       cout << "hip_matrix_mul_cublas failed, Matrix dimension mismatch!" << endl;
 //       return 1;
 //   }
 //   timer.stop();
 //   cout << "hip cublas implementation = ";
 //   timer.printtime_ms();
 //   ret = (res_matrix1 == res_matrix2);
 //   if (ret == false) {
 //       cout << "hip rocblas implementation failed." << endl;
 //       return 1;
 //   }

 //   // call hip magma implementation to run on GPU
 //   timer.start();
 //   ret = cu_matrix_mul_magma(matrix1, matrix2, res_matrix2);
 //   if (ret == false) {
 //       cout << "hip_matrix_mul_magma failed, Matrix dimension mismatch!" << endl;
 //       return 1;
 //   }
 //   timer.stop();
 //   cout << "hip magma implementation = ";
 //   timer.printtime_ms();
 //   ret = (res_matrix1 == res_matrix2);
 //   if (ret == false) {
 //       cout << "hip magma implementation failed." << endl;
 //       return 1;
 //   }

    cout << "PASSED" << endl;


    return 0;

}

