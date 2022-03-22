#ifndef __ROCBLA_MATMUL_HPP__
#define __ROCBLA_MATMUL_HPP__
#include "matrixmul.hpp"

bool rocblas_matmul(CMatrix<double> &in1, 
                    CMatrix<double> &in2, 
                    CMatrix<double> &in3, 
                    CMatrix<double> &res, 
                    double alpha = 1.0, 
                    double beta = 0.0);

bool rocblas_batch_matmul(int batch, 
                          CMatrix<double> &in1, 
                          CMatrix<double> &in2, 
                          CMatrix<double> &in3, 
                          CMatrix<double> &res,
                          double alpha = 1.0,
                          double beta = 0.0);

#endif

