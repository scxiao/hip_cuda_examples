#include <iostream>
#include <cstdlib>
#include <cstdio>
#include "matrixmul.h"
#include "cu_matrix_mul.h"
#include "timer.h"

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

    CMatrix<double> matrix1(i, k), matrix2(k, j), res_matrix1, res_matrix2;
    HRTimer timer;

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

        cout << "Parallel time with " << num << " threads = ";
        timer.printtime_ms();

        ret = (res_matrix1 == res_matrix2);
        if (ret == false) {
            cout << "FAILED with " << num << " threads." << endl;
            return 1;
        }

        num *= 2;
    }

    // call cuda naive implementation to run on GPU
    timer.start();
    ret = cu_matrix_mul_naive(matrix1, matrix2, res_matrix2);
    if (ret == false) {
        cout << "cu_matrix_mul_naive failed, Matrix dimension mismatch!" << endl;
        return 1;
    }
    timer.stop();
    cout << "cuda naive implementation = ";
    timer.printtime_ms();
    ret = (res_matrix1 == res_matrix2);
    if (ret == false) {
        cout << "cuda naive implementation failed." << endl;
        return 1;
    }

    // call cuda shared implementation to run on GPU
    timer.start();
    ret = cu_matrix_mul_shared(matrix1, matrix2, res_matrix2);
    if (ret == false) {
        cout << "cuda_matrix_mul_shared failed, Matrix dimension mismatch!" << endl;
        return 1;
    }
    timer.stop();
    cout << "cuda shared implementation = ";
    timer.printtime_ms();
    ret = (res_matrix1 == res_matrix2);
    if (ret == false) {
        cout << "cuda shared implementation failed." << endl;
        return 1;
    }

    // call cuda cublas implementation to run on GPU
    timer.start();
    ret = cu_matrix_mul_cublas(matrix1, matrix2, res_matrix2);
    if (ret == false) {
        cout << "cuda_matrix_mul_cublas failed, Matrix dimension mismatch!" << endl;
        return 1;
    }
    timer.stop();
    cout << "cuda cublas implementation = ";
    timer.printtime_ms();
    ret = (res_matrix1 == res_matrix2);
    if (ret == false) {
        cout << "cuda cublas implementation failed." << endl;
        return 1;
    }

    // call cuda magma implementation to run on GPU
    timer.start();
    ret = cu_matrix_mul_magma(matrix1, matrix2, res_matrix2);
    if (ret == false) {
        cout << "cuda_matrix_mul_magma failed, Matrix dimension mismatch!" << endl;
        return 1;
    }
    timer.stop();
    cout << "cuda magma implementation = ";
    timer.printtime_ms();
    ret = (res_matrix1 == res_matrix2);
    if (ret == false) {
        cout << "cuda magma implementation failed." << endl;
        return 1;
    }



    cout << "PASSED" << endl;


    return 0;

}

