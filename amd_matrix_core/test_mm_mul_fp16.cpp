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

    size_t m = atoi(argv[1]);
    size_t k = atoi(argv[2]);
    size_t n = atoi(argv[3]);
    size_t thread_num = atoi(argv[4]);

    cout << "m = " << m << ", n = " << n << ", k = " << k << endl;

    CMatrix<float> matrix1(m, k), matrix2(k, n), res_matrix1, res_matrix2;
    CMatrix<__half> matrix1_half(m, k), matrix2_half(k, n), matrix_res_half(m, n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            matrix1_half.get_elem(i, j) = matrix1.get_elem(i, j);
            matrix1.get_elem(i, j) = matrix1_half.get_elem(i, j);
        }
    } 

    matrix2_half.set_row_major(false);
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix2_half.get_elem(i, j) = matrix2.get_elem(i, j);
            matrix2.get_elem(i, j) = matrix2_half.get_elem(i, j);
        }
    } 

    HRTimer timer;
    double flops;

    timer.start();
    bool ret = matrix1.multiply_optim(matrix2, res_matrix1);
    // bool ret = matrix1.multiply_parallel(matrix2, 16, res_matrix1);
    if (ret == false) {
        cout << "matrix dimension is incorrect, cannot multiplication." << endl;
        return 1;
    }
    timer.stop();
    cout << "Sequential time = ";
    timer.printtime_ms();

    size_t num = 8;
    while (num <= thread_num) {
        timer.start();
        ret = matrix1.multiply_parallel(matrix2, num, res_matrix2);
        if (ret == false) {
            cout << "matrix dimension is incorrect, cannot multiplication." << endl;
            return 1;
        }
        timer.stop();
        size_t kernel_time_us = timer.gettime_us();
        double flops = 2.0 * m * n * k / kernel_time_us / 1000000;

        cout << "Parallel time with " << num 
             << " threads = " << kernel_time_us 
             << " us, flops = " << flops << " TFLOPS, PASSED!" << std::endl;

        ret = (res_matrix1 == res_matrix2);
        if (ret == false) {
            cout << "FAILED with " << num << " threads." << endl;
            return 1;
        }

        num *= 2;
    }

    // naive implementation
    timer.start();
    ret = hip_matrix_mul_f16_naive(matrix1_half, matrix2_half, matrix_res_half, flops);
    if (ret == false) {
        cout << "half naive impl failed!" << endl;
        return 1;
    }
    timer.stop();
    size_t kernel_time_us = timer.gettime_us(); 
    cout << "hip_matrix_mul_f16_naive implementation = " << kernel_time_us << " us, flops = " << flops << " TFLOPS";
    CMatrix<float> matrix_res_float(m, n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix_res_float.get_elem(i, j) = matrix_res_half.get_elem(i, j);
        }
    }
    ret = (res_matrix1 == matrix_res_float);
    if (ret == false) {
        cout << "hip_matrix_mul_f16_naive implementation failed." << endl;
        return 1;
    }
    else {
        std::cout << ", PASSED!" << std::endl;
    }

    // mfma32X32X8
    timer.start();
    ret = hip_matrix_mul_32x32x8_fp16(matrix1_half, matrix2_half, matrix_res_half, flops);
    if (ret == false) {
        cout << "hip_matrix_mul_32x32x8_fp16 failed, Matrix dimension mismatch!" << endl;
        return 1;
    }
    timer.stop();
    kernel_time_us = timer.gettime_us(); 
    cout << "hip_matrix_mul_32x32x8_fp16 implementation = " << kernel_time_us << " us, flops = " << flops << " TFLOPS";

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix_res_float.get_elem(i, j) = matrix_res_half.get_elem(i, j);
        }
    }

    ret = (res_matrix1 == matrix_res_float);
    if (ret == false) {
        cout << "hip hip_matrix_mul_32x32x8_fp16 implementation failed." << endl;
        return 1;
    }
    else {
        std::cout << ", PASSED!" << std::endl;
    }

    // mfma4X4X4_16
    timer.start();
    ret = hip_matrix_mul_4x4x4_fp16_464(matrix1_half, matrix2_half, matrix_res_half, flops);
    if (ret == false) {
        cout << "hip_matrix_mul_4x4x4_fp16_464 failed, Matrix dimension mismatch!" << endl;
        return 1;
    }
    timer.stop();
    kernel_time_us = timer.gettime_us(); 
    cout << "hip_matrix_mul_4x4x4_fp16_464 implementation = " << kernel_time_us << " us, flops = " << flops << " TFLOPS";

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix_res_float.get_elem(i, j) = matrix_res_half.get_elem(i, j);
        }
    }

    ret = (res_matrix1 == matrix_res_float);
    if (ret == false) {
        cout << "hip hip_matrix_mul_4x4x4_fp16_464 implementation failed." << endl;
        return 1;
    }
    else {
        std::cout << ", PASSED!" << std::endl;
    }

    // // mfma32X32X8
    // timer.start();
    // ret = hip_matrix_mul_double_rate_32x32x16_fp16(matrix1_half, matrix2_half, matrix_res_half, flops);
    // if (ret == false) {
    //     cout << "hip_matrix_mul_double_rate_32x32x16_fp16 failed, Matrix dimension mismatch!" << endl;
    //     return 1;
    // }
    // timer.stop();
    // kernel_time_us = timer.gettime_us(); 
    // cout << "hip_matrix_mul_double_rate_32x32x16_fp16 implementation = " << kernel_time_us << " us, flops = " << flops << " TFLOPS";

    // for (int i = 0; i < m; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         matrix_res_float.get_elem(i, j) = matrix_res_half.get_elem(i, j);
    //     }
    // }

    // ret = (res_matrix1 == matrix_res_float);
    // if (ret == false) {
    //     cout << "hip hip_matrix_mul_double_rate_32x32x16_fp16 implementation failed." << endl;
    //     return 1;
    // }
    // else {
    //     std::cout << ", PASSED!" << std::endl;
    // }


    return 0;
}


