#ifndef __HIP_MATRIX_MUL_H__
#define __HIP_MATRIX_MUL_H__
#include "matrixmul.h"

bool hip_matrix_mul_naive(CMatrix<double> &in1, CMatrix<double> &in2, CMatrix<double> &res);
bool hip_matrix_mul_shared(CMatrix<double> &in1, CMatrix<double> &in2, CMatrix<double> &res);

#endif

