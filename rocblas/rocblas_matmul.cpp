#include "matrixmul.hpp"
#include "rocblas_matmul.hpp"
#include <hip/hip_runtime.h>
#include <rocblas.h>
#include "timer.hpp"

using namespace std;

rocblas_handle create_rocblas_handle()
{
    rocblas_handle handle;
    rocblas_create_handle(&handle);
    return handle;
}

bool rocblas_matmul(CMatrix<double> &in1, CMatrix<double> &in2, CMatrix<double> &in3, CMatrix<double> &res, double alpha, double beta) {
    size_t row, dim1, dim2, column;
    in1.get_size(row, dim1);
    in2.get_size(dim2, column);

    if (dim1 != dim2) {
        cout << "Matrix dimensions mismatch!" << endl;
        return false;
    }
    res.resize(row, column);

    auto handle = create_rocblas_handle();

    double *hip_in1, *hip_in2, *hip_in3, *hip_res;
    hipMalloc((void **)&hip_in1, sizeof(double) * row * dim1);
    hipMalloc((void **)&hip_in2, sizeof(double) * dim2 * column);
    hipMalloc((void **)&hip_in3, sizeof(double) * row * column);
    hipMalloc((void **)&hip_res, sizeof(double) * row * column);
    hipMemset(hip_res, 0, sizeof(double) * row * column);

    hipMemcpy(hip_in1, in1.get_buffer(), sizeof(double) * row * dim1, hipMemcpyHostToDevice);
    hipMemcpy(hip_in2, in2.get_buffer(), sizeof(double) * dim2 * column, hipMemcpyHostToDevice);
    hipMemcpy(hip_in3, in3.get_buffer(), sizeof(double) * row * column, hipMemcpyHostToDevice);

    // warm up of the api call
    auto ret = rocblas_gemm_ex(handle,
                               rocblas_operation_none,
                               rocblas_operation_none,
                               column,
                               row,
                               dim1,
                               &alpha,
                               hip_in2,
                               rocblas_datatype_f64_r,
                               column,
                               hip_in1,
                               rocblas_datatype_f64_r,
                               dim1,
                               &beta,
                               hip_res,
                               rocblas_datatype_f64_r,
                               column,
                               hip_res,
                               rocblas_datatype_f64_r,
                               column,
                               rocblas_datatype_f64_r,
                               rocblas_gemm_algo_standard,
                               0,
                               0);
    hipDeviceSynchronize();
    hipMemset(hip_res, 0, sizeof(double) * row * column);

    HRTimer timer;
    timer.start();
    ret = rocblas_gemm_ex(handle,
                               rocblas_operation_none,
                               rocblas_operation_none,
                               column,
                               row,
                               dim1,
                               &alpha,
                               hip_in2,
                               rocblas_datatype_f64_r,
                               column,
                               hip_in1,
                               rocblas_datatype_f64_r,
                               dim1,
                               &beta,
                               hip_res,
                               rocblas_datatype_f64_r,
                               column,
                               hip_res,
                               rocblas_datatype_f64_r,
                               column,
                               rocblas_datatype_f64_r,
                               rocblas_gemm_algo_standard,
                               0,
                               0);
    hipDeviceSynchronize();
    timer.stop();
    cout << "rocblas matmul = ";
    timer.printtime_us();
    std::cout << " us" << std::endl;

    timer.start();
    ret = rocblas_gemm_ex(handle,
                               rocblas_operation_none,
                               rocblas_operation_none,
                               column,
                               row,
                               dim1,
                               &alpha,
                               hip_in2,
                               rocblas_datatype_f64_r,
                               column,
                               hip_in1,
                               rocblas_datatype_f64_r,
                               dim1,
                               &beta,
                               hip_in3,
                               rocblas_datatype_f64_r,
                               column,
                               hip_res,
                               rocblas_datatype_f64_r,
                               column,
                               rocblas_datatype_f64_r,
                               rocblas_gemm_algo_standard,
                               0,
                               0);
    hipDeviceSynchronize();
    timer.stop();
    cout << "rocblas matmul1 = ";
    timer.printtime_us();
    std::cout << " us" << std::endl;


    if (ret != rocblas_status_success)
    {
        std::cout << "rocblas api call error: " << ret << std::endl;
        return false;
    }
                              
    hipMemcpy(res.get_buffer(), hip_res, sizeof(double) * row * column, hipMemcpyDeviceToHost);
    hipFree(hip_in1); hipFree(hip_in2); hipFree(hip_res);

    return true;
}

bool rocblas_batch_matmul(int batch, CMatrix<double> &in1, CMatrix<double> &in2, CMatrix<double> &in3, CMatrix<double> &res, double alpha, double beta)
{
    size_t row, dim1, dim2, column;
    in1.get_size(row, dim1);
    in2.get_size(dim2, column);

    if (dim1 != dim2) {
        cout << "Matrix dimensions mismatch!" << endl;
        return false;
    }
    res.resize(row, column);

    auto handle = create_rocblas_handle();
    double *hip_in1, *hip_in2, *hip_in3, *hip_res, *hip_res_batch;
    size_t size1 = sizeof(double) * row * dim1;
    size_t size2 = sizeof(double) * dim2 * column;
    size_t size3 = sizeof(double) * row * column;
    hipMalloc((void **)&hip_in1, size1 * batch);
    hipMalloc((void **)&hip_in2, size2 * batch);
    hipMalloc((void **)&hip_in3, size3 * batch);
    hipMalloc((void **)&hip_res, size3 * batch);
    hipMalloc((void **)&hip_res_batch, size3 * batch);
    hipMemset(hip_res, 0, size3 * batch);
    hipMemset(hip_res_batch, 0, size3 * batch);

    size_t elem1 = row * dim1;
    size_t elem2 = dim2 * column;
    size_t elem3 = row * column;
    for (int i = 0; i < batch; ++i)
    {
        hipMemcpy(hip_in1 + i * elem1, in1.get_buffer(), size1, hipMemcpyHostToDevice);
        hipMemcpy(hip_in2 + i * elem2, in2.get_buffer(), size2, hipMemcpyHostToDevice);
        hipMemcpy(hip_in3 + i * elem3, in3.get_buffer(), size3, hipMemcpyHostToDevice);
    }

    // warm up of the api call
    auto ret = rocblas_gemm_ex(handle,
                               rocblas_operation_none,
                               rocblas_operation_none,
                               column,
                               row,
                               dim1,
                               &alpha,
                               hip_in2,
                               rocblas_datatype_f64_r,
                               column,
                               hip_in1,
                               rocblas_datatype_f64_r,
                               dim1,
                               &beta,
                               hip_in3,
                               rocblas_datatype_f64_r,
                               column,
                               hip_res,
                               rocblas_datatype_f64_r,
                               column,
                               rocblas_datatype_f64_r,
                               rocblas_gemm_algo_standard,
                               0,
                               0);
    hipDeviceSynchronize();
    hipMemset(hip_res, 0, size3 * batch);
    hipMemset(hip_res_batch, 0, size3 * batch);

    HRTimer timer;
    timer.start();
    for (int i = 0; i < batch; ++i)
    {
        ret = rocblas_gemm_ex(handle,
                               rocblas_operation_none,
                               rocblas_operation_none,
                               column,
                               row,
                               dim1,
                               &alpha,
                               hip_in2 + i * elem2,
                               rocblas_datatype_f64_r,
                               column,
                               hip_in1 + i * elem1,
                               rocblas_datatype_f64_r,
                               dim1,
                               &beta,
                               hip_in3 + i * elem3,
                               rocblas_datatype_f64_r,
                               column,
                               hip_res + i * elem3,
                               rocblas_datatype_f64_r,
                               column,
                               rocblas_datatype_f64_r,
                               rocblas_gemm_algo_standard,
                               0,
                               0);
    }
    hipDeviceSynchronize();
    timer.stop();
    auto total_time = timer.gettime_us();
    cout << "multiple calls of rocblas api, total exec time = " << total_time << " us" << std::endl;
    cout << "multiple calls of rocblas api, per call time = " << total_time / batch << " us" << std::endl;

    CMatrix<double> gold0(row, column);
    hipMemcpy(gold0.get_buffer(), hip_res, sizeof(double) * elem3, hipMemcpyDeviceToHost);
    for (int i = 0; i < batch; ++i)
    {
        CMatrix<double> result(row, column);
        hipMemcpy(result.get_buffer(), hip_res + i * elem3, sizeof(double) * elem3, hipMemcpyDeviceToHost);
        if (gold0 != result)
        {
            std::cout << "multiple rocblas call, index " << i << " outputs are wrong!" << std::endl;
            return false;
        }
    }

    // warm up call for the batch gemm
    ret = rocblas_gemm_strided_batched_ex(handle,
                           rocblas_operation_none,
                           rocblas_operation_none,
                           column,
                           row,
                           dim1,
                           &alpha,
                           hip_in2,
                           rocblas_datatype_f64_r,
                           column,
                           dim1 * column,
						   hip_in1,
                           rocblas_datatype_f64_r,
                           dim1,
                           row * dim1,
                           &beta,
						   hip_in3,
                           rocblas_datatype_f64_r,
                           column,
                           row * column,
                           hip_res_batch,
                           rocblas_datatype_f64_r,
                           column,
                           row * column,
                           batch,
                           rocblas_datatype_f64_r,
                           rocblas_gemm_algo_standard,
                           0,
                           0);

    // actual call of rocblas apis
    timer.start();
    ret = rocblas_gemm_strided_batched_ex(handle,
                           rocblas_operation_none,
                           rocblas_operation_none,
                           column,
                           row,
                           dim1,
                           &alpha,
                           hip_in2,
                           rocblas_datatype_f64_r,
                           column,
                           dim1 * column,
						   hip_in1,
                           rocblas_datatype_f64_r,
                           dim1,
                           row * dim1,
                           &beta,
						   hip_in3,
                           rocblas_datatype_f64_r,
                           column,
                           row * column,
                           hip_res_batch,
                           rocblas_datatype_f64_r,
                           column,
                           row * column,
                           batch,
                           rocblas_datatype_f64_r,
                           rocblas_gemm_algo_standard,
                           0,
                           0);
    hipDeviceSynchronize();
    timer.stop();
    cout << "batch rocblas call, total exec time = ";
    timer.printtime_us();
    std::cout << " us" << std::endl;
    auto total_time_us = timer.gettime_us();
    cout << "batch rocblas call, per call time = " << total_time_us / batch << " us" << std::endl;

    if (ret != rocblas_status_success)
    {
        std::cout << "batch_rocblas api call error: " << ret << std::endl;
        return false;
    }

    for (int i = 0; i < batch; ++i)
    {
        CMatrix<double> result(row, column);
        hipMemcpy(result.get_buffer(), hip_res_batch + i * elem3, sizeof(double) * elem3, hipMemcpyDeviceToHost);
        if (gold0 != result)
        {
            std::cout << "rocblas batch call, index " << i << " outputs are wrong!" << std::endl;
            return false;
        }
    }

    hipFree(hip_in1); hipFree(hip_in2); hipFree(hip_res); hipFree(hip_res_batch);

    return true;
}

