#ifndef __HIP_MATRIX_MUL_HPP__
#define __HIP_MATRIX_MUL_HPP__
#include "matrixmul.hpp"

bool hip_matrix_mul_naive_fp64(CMatrix<double> &in1, CMatrix<double> &in2, CMatrix<double> &res, double& flops);
bool hip_matrix_mul_shared_fp64(CMatrix<double> &in1, CMatrix<double> &in2, CMatrix<double> &res, double& flops);

bool hip_matrix_mul_naive_fp32(CMatrix<float> &in1, CMatrix<float> &in2, CMatrix<float> &res, double& flops);
bool hip_matrix_mul_shared_fp32(CMatrix<float> &in1, CMatrix<float> &in2, CMatrix<float> &res, double& flops);
bool hip_matrix_mul_sgemm_16x16x16_fp32(CMatrix<float> &in1, CMatrix<float> &in2, CMatrix<float> &res, double& flops);
bool hip_matrix_mul_sgemm_32x32x32_fp32(CMatrix<float> &in1, CMatrix<float> &in2, CMatrix<float> &res, double& flops);

#endif

