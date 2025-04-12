#include <iostream>
#include <cstdlib>
#include <cstdio>
#include "matrixmul.hpp"
#include "hip_matrix_mul.hpp"
#include "timer.hpp"

using namespace std;


CMatrix<float> sparseMatMul(CMatrix<__half> &a, CMatrix<int> &idx, CMatrix<__half> &b) {
    size_t am, ak, bn, bk;
    a.get_size(am, ak);
    b.get_size(bk, bn);

    CMatrix<float> result(am, bn);
    for (int i = 0; i < am; ++i) {
        for (int j = 0; j < bn; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < bk; k += 4) {
                int ka = k / 2;
                int idx0 = k + idx.get_elem(i, ka);
                sum += (float)a.get_elem(i, ka) * (float)b.get_elem(idx0, j);
                int idx1 = k + idx.get_elem(i, ka + 1);
                sum += (float)a.get_elem(i, ka + 1) * (float)b.get_elem(idx1, j);
            }

            result.get_elem(i, j) = sum;
        }
    }

    return result;
}


int main(int argc, char **argv) {
    // if (argc != 4) {
    //     cout << "Usage: " << argv[0] << " i k j thread_num" << endl;
    //     return 0;
    // }

    // size_t m = atoi(argv[1]);
    // size_t k = atoi(argv[2]);
    // size_t n = atoi(argv[3]);

    size_t m = 32;
    size_t n = 32;
    size_t k = 16;

    cout << "m = " << m << ", n = " << n << ", k = " << k << endl;

    CMatrix<float> matrix1(m, k/2), matrix2(k, n), res_matrix1, res_matrix2;
    CMatrix<__half> matrix1_half(m, k/2), matrix2_half(k, n), matrix_res_half(m, n);
    CMatrix<int> matrix_idx(m, k/2);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k/2; ++j) {
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

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k/2; j += 2) {
            matrix_idx.get_elem(i, j) = (i + j) % 3;
            matrix_idx.get_elem(i, j + 1) = (i + j) % 3 + 1;
        }
    }

    CMatrix<float> res_ref = sparseMatMul(matrix1_half, matrix_idx, matrix2_half);

    HRTimer timer;
    double flops;

    // naive implementation
    bool ret = hip_sparse_matrix_mul_f16_naive(matrix1_half, matrix_idx, matrix2_half, res_matrix1, flops);
    if (ret == false) {
        cout << "half sparse matrix, naive impl failed!" << endl;
        return 1;
    }

    ret = hip_sparse_matrix_mul_32x32x16_fp16(matrix1_half, matrix_idx, matrix2_half, res_matrix2, flops);
    if (ret == false) {
        cout << "hip_sparse_matrix_mul_32x32x16_fp16 failed, Matrix dimension mismatch!" << endl;
        return 1;
    }

    std::cout << "res_ref = " << res_ref.get_elem(0, 0) << ", res1 = " << res_matrix1.get_elem(0, 0) << ", res2 = " << res_matrix2.get_elem(0, 0) << std::endl;
    if (res_ref != res_matrix1) {
        std::cout << "FAILED: naive implementation mismatch!" << std::endl;
    }
    else {
        std::cout << "PASSED: naive implementation, results match!" << std::endl;
    }

    if (res_ref != res_matrix2) {
        std::cout << "FAILED: hip_sparse_matrix_mul_32x32x16_fp16 mismatch!" << std::endl;
    }
    else {
        std::cout << "PASSED: hip_sparse_matrix_mul_32x32x16_fp16, results match!" << std::endl;
    }

    return 0;
}


