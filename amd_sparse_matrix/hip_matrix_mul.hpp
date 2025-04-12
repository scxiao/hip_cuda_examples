#ifndef __HIP_SPARSE_MATRIX_MUL_HPP__
#define __HIP_SPARSE_MATRIX_MUL_HPP__
#include "matrixmul.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

bool hip_sparse_matrix_mul_f16_naive(CMatrix<__half> &in1, CMatrix<int>& idx, CMatrix<__half> &in2, CMatrix<float> &res, double& flops);

bool hip_sparse_matrix_mul_32x32x16_fp16(CMatrix<__half> &in1, CMatrix<int>& idx, CMatrix<__half> &in2, CMatrix<float> &res, double& flops);

#endif

