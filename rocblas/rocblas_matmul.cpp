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

//rocblas_handle_ptr create_rocblas_handle_ptr(hipStream_t s)
//{
//    rocblas_handle_ptr rb = create_rocblas_handle_ptr();
//    rocblas_set_stream(rb.get(), s);
//    return rb;
//}

bool rocblas_matmul(CMatrix<double> &in1, CMatrix<double> &in2, CMatrix<double> &in3, CMatrix<double> &res) {
    size_t row, dim1, dim2, column;
    in1.get_size(row, dim1);
    in2.get_size(dim2, column);

    if (dim1 != dim2) {
        cout << "Matrix dimensions mismatch!" << endl;
        return false;
    }
    res.resize(row, column);

    auto handle = create_rocblas_handle();
    double alpha = 3.0;
    double beta = 2.0;

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
                               0,
                               nullptr,
                               nullptr);
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
                               0,
                               nullptr,
                               nullptr);
    hipDeviceSynchronize();
    timer.stop();
    cout << "rocblas matmul = ";
    timer.printtime_us();
    std::cout << "us" << std::endl;

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
                               0,
                               nullptr,
                               nullptr);
    hipDeviceSynchronize();
    timer.stop();
    cout << "rocblas matmul1 = ";
    timer.printtime_us();
    std::cout << "us" << std::endl;


    if (ret != rocblas_status_success)
    {
        std::cout << "rocblas api call error: " << ret << std::endl;
        return false;
    }
                              
    hipMemcpy(res.get_buffer(), hip_res, sizeof(double) * row * column, hipMemcpyDeviceToHost);
    hipFree(hip_in1); hipFree(hip_in2); hipFree(hip_res);

    return true;
}

