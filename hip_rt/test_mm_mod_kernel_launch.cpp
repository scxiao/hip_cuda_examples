// compile with: hipcc --std=c++11 matmulbasic.cpp -o matmul
// run with: ./matmul
#include <cassert>
#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include "matrixmul.hpp"

static constexpr auto mm_naive{
R"(
#include <hip/hip_runtime.h>
extern "C" __global__ void hipkernel_matrix_mul_naive(double *in1, double *in2, double *res,
        size_t row, size_t dim, size_t column) {
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx_x == 0 && idx_y ==0 ) {
        printf("r:%zu\n",row);
        printf("c:%zu\n",column);
        printf("dim:%zu\n",dim);
    }

    if (idx_x < row && idx_y < column) {
        double sum = 0.0;
        for (size_t k = 0; k < dim; ++k) {
            sum += in1[idx_x * dim + k] * in2[k * dim + idx_y];
        }
        res[idx_x * column + idx_y] = sum;
    }


    return;
}
)"};

int main(int argc, char **argv) {
    if (argc != 4) {
        cout << "Usage: " << argv[0] << " row k col" << endl;
        return 0;
    }

    size_t row = atoi(argv[1]);
    size_t dim = atoi(argv[2]);
    size_t col = atoi(argv[3]);

    cout << "row = " << row << ", col = " << col << ", dim = " << dim << endl;
    CMatrix<double> matrixa(row, dim), matrixb(dim, col), res_matrix1, res_matrix2;
    bool ret = matrixa.multiply_optim(matrixb, res_matrix1);
    if (ret == false) {
        cout << "matrix dimension is incorrect, cannot multiplication." << endl;
        return 1;
    }

    hipDeviceProp_t props;
    int device = 0;
    hipGetDeviceProperties(&props, device);

    hiprtcProgram mm_prog;
    hiprtcCreateProgram(&mm_prog,   // prog
                        mm_naive,   // buffer
                        "mm.abc", // name
                        0,          // numHeaders
                        nullptr,    // headers
                        nullptr);   // includeNames

    // Compiling Matmul Code Begin
    std::string sarg = "-DCUPY_JIT_MODE";
    const int num_options = 1;
    const char* options[1];
    options[0] = sarg.c_str();
    for (int i = 0; i < num_options; ++i) {
        printf("option %d is %s\n", i, options[i]);
    }

    hiprtcResult compileResult = hiprtcCompileProgram(mm_prog, num_options, options);
    assert (compileResult == 0);

    size_t logSize;
    hiprtcGetProgramLogSize(mm_prog, &logSize);
    if (logSize) {
        std::string log(logSize, '\0');
        hiprtcGetProgramLog(mm_prog, &log[0]);
        std::cout << log << '\n';
    }

    size_t codeSize;
    hiprtcGetCodeSize(mm_prog, &codeSize);

    std::vector<char> code(codeSize);
    hiprtcGetCode(mm_prog, code.data());

    hiprtcDestroyProgram(&mm_prog);
    // Compiling Done

    FILE* file = fopen( "myfile.bin", "wb" );
    fwrite(code.data(), 1, codeSize, file );
    fclose(file);

    // Loading HSACO Begin
    hipModule_t module;
    hipFunction_t kernel;
    hipModuleLoadData(&module, code.data());
    hipModuleGetFunction(&kernel, module, "hipkernel_matrix_mul_naive");
    // Loading HSACO Done

    double *mat_ad, *mat_bd, *mat_cd;
    assert( hipMalloc(&mat_ad, sizeof(double) * row * dim) == 0);
    assert( hipMalloc(&mat_bd, sizeof(double) * dim * col) == 0);
    assert( hipMalloc(&mat_cd, sizeof(double) * row * col) == 0);
    hipMemcpy(mat_ad, matrixa.get_buffer(), sizeof(double) * row * dim, hipMemcpyHostToDevice);
    hipMemcpy(mat_bd, matrixb.get_buffer(), sizeof(double) * dim * col, hipMemcpyHostToDevice);

    // Launching Kernel Begin
    size_t block_dimx = 32;
    size_t block_dimy = 32;
    int gridX = (row + block_dimx - 1)/block_dimx;
    int gridY = (col + block_dimy - 1)/block_dimy;
    //void* kernelParam[] = {mat_a,mat_b,mat_c,row_d,dim_d,col_d};
	std::vector<void*> kernelargs;
	kernelargs.push_back(mat_ad);
	kernelargs.push_back(mat_bd);
	kernelargs.push_back(mat_cd);
	kernelargs.push_back((void*)row);
	kernelargs.push_back((void*)dim);
	kernelargs.push_back((void*)col);

	size_t size = sizeof(void*) * kernelargs.size();
	void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, kernelargs.data(),
					  HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, 
					  HIP_LAUNCH_PARAM_END};

    assert (hipModuleLaunchKernel(
        kernel, gridX, gridY, 1, block_dimx, block_dimy, 1,
        0, nullptr, nullptr, (void**)&config) == 0);
    // Launching Kernel Done
    res_matrix2.resize(row, col);
    hipMemcpy(res_matrix2.get_buffer(), mat_cd, 
        sizeof(double) * row * col, hipMemcpyDeviceToHost);

    // Clean up
    assert (hipFree(mat_ad) == 0);
    assert (hipFree(mat_bd) == 0);
    assert (hipFree(mat_cd) == 0);

    ret = (res_matrix1 == res_matrix2);
    if (ret == false) {
        cout << "Matrix multiplcation results mismatch!" << endl;
        std::cout << "FAILED!" << std::endl;
        return 1;
    }

    std::cout << "PASSED!" << std::endl;

    return 0;
}
