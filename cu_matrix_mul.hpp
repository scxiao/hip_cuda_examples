#ifndef __HIP_MATRIX_MUL_HPP__
#define __HIP_MATRIX_MUL_HPP__
#include "matrixmul.hpp"

bool cu_matrix_mul_naive(CMatrix<double> &in1, CMatrix<double> &in2, CMatrix<double> &res);
bool cu_matrix_mul_shared(CMatrix<double> &in1, CMatrix<double> &in2, CMatrix<double> &res);
bool cu_matrix_mul_cublas(CMatrix<double> &in1, CMatrix<double> &in2, CMatrix<double> &res);
bool cu_matrix_mul_magma(CMatrix<double> &in1, CMatrix<double> &in2, CMatrix<double> &res);

#endif

