#ifndef __HIP_MATRIX_MUL_HPP__
#define __HIP_MATRIX_MUL_HPP__
#include "matrixmul.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

bool hip_matrix_mul_naive_fp64(CMatrix<double> &in1, CMatrix<double> &in2, CMatrix<double> &res, double& flops);
bool hip_matrix_mul_shared_fp64(CMatrix<double> &in1, CMatrix<double> &in2, CMatrix<double> &res, double& flops);

bool hip_matrix_mul_naive_fp32(CMatrix<float> &in1, CMatrix<float> &in2, CMatrix<float> &res, double& flops);
bool hip_matrix_mul_shared_fp32(CMatrix<float> &in1, CMatrix<float> &in2, CMatrix<float> &res, double& flops);
bool hip_matrix_mul_sgemm_16x16x16_fp32(CMatrix<float> &in1, CMatrix<float> &in2, CMatrix<float> &res, double& flops);
bool hip_matrix_mul_sgemm_32x32xK_fp32_v1(CMatrix<float> &in1, CMatrix<float> &in2, CMatrix<float> &res, double& flops);
bool hip_matrix_mul_sgemm_32x32xK_fp32_v2(CMatrix<float> &in1, CMatrix<float> &in2, CMatrix<float> &res, double& flops);
bool hip_matrix_mul_sgemm_32x32xK_fp32_v3(CMatrix<float> &in1, CMatrix<float> &in2, CMatrix<float> &res, double& flops);
bool hip_matrix_mul_sgemm_32x32xK_fp32_v4(CMatrix<float> &in1, CMatrix<float> &in2, CMatrix<float> &res, double& flops);

bool hip_matrix_mul_4x4x4_fp16_464(CMatrix<__half> &in1, CMatrix<__half> &in2, CMatrix<__half> &res, double& flops);
bool hip_matrix_mul_32x32x8_fp16(CMatrix<__half> &in1, CMatrix<__half> &in2, CMatrix<__half> &res, double& flops);
bool hip_matrix_mul_f16_naive(CMatrix<__half> &in1, CMatrix<__half> &in2, CMatrix<__half> &res, double& flops);

bool hip_matrix_mul_int8_naive(CMatrix<int8_t> &in1, CMatrix<int8_t> &in2, CMatrix<int32_t> &res, double& flops);
bool hip_matrix_mul_4x4x4_int8_464(CMatrix<int8_t> &in1, CMatrix<int8_t> &in2, CMatrix<int32_t> &res, double& flops);


bool hip_matrix_mul_double_rate_32x32x16_fp16(CMatrix<__half> &in1, CMatrix<__half> &in2, CMatrix<__half> &res, double& flops);

#endif

