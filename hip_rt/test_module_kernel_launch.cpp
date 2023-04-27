// compile with: hipcc --std=c++11 matmulbasic.cpp -o matmul
// run with: ./matmul
#include <cassert>
#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

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
            sum += in1[idx_x * dim + k] * in2[k + idx_y * dim];
        }
        res[idx_x * column + idx_y] = sum;
    }


    return;
}
)"};

int main() {
    hiprtcProgram mm_prog;
    hiprtcCreateProgram(&mm_prog,      // prog
                        mm_naive,      // buffer
                        "mm.hsaco", // name
                        0,          // numHeaders
                        nullptr,    // headers
                        nullptr);   // includeNames
    hipDeviceProp_t props;
    int device = 0;
    hipGetDeviceProperties(&props, device);
    // Compiling Matmul Code Begin
    std::string sarg = "-DCUPY_JIT_MODE";
    const int num_options = 1;
    const char* options[1];
    options[0] = sarg.c_str();
    for (int i = 0; i < num_options; ++i)
        printf("option %d is %s\n", i, options[i]);
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

    // Loading HSACO Begin
    hipModule_t module;
    hipFunction_t kernel;
    FILE* file = fopen( "myfile.bin", "wb" );
    fwrite(code.data(), 1, codeSize, file );
    hipModuleLoadData(&module, code.data());
    hipModuleGetFunction(&kernel, module, "hipkernel_matrix_mul_naive");
    // Loading HSACO Done

    // Setting up Variables begin
    // Matrix A = ones(row x d1)
    // Matrix B = ones(d2 x col)
    // Matrix C = A*B  with size (row x col)

    double *mat_a, *mat_b, *mat_c;
    //size_t *row_d, *dim_d, *col_d;
    size_t row = 32;
    size_t d1 = 32;
    size_t d2 = d1;
    size_t col = 32;
    
    size_t *row_ptr = &row;
    size_t *dim_ptr = &d1;
    size_t *col_ptr = &col;

    assert( hipMalloc(&mat_a, sizeof(double) * row * d1) == 0);
    assert( hipMalloc(&mat_b, sizeof(double) * d2 * col) == 0);
    assert( hipMalloc(&mat_c, sizeof(double) * row * col) == 0);
    //assert( hipMalloc(&row_d, sizeof(size_t)) == 0);
    //assert( hipMalloc(&dim_d, sizeof(size_t)) == 0);
    //assert( hipMalloc(&col_d, sizeof(size_t)) == 0);
    double* h_mata = new double[row*d1];
    double* h_matb = new double[d2*col];
    double* h_matc = new double[row*col];
    for (int i = 0; i<row*d1; i++) h_mata[i] = 1;
    for (int i = 0; i<d2*col; i++) h_matb[i] = 1;
    for (int i = 0; i<row*col; i++) h_matc[i] = 0;
    hipMemcpy(mat_a, h_mata, sizeof(double) * row * d1, hipMemcpyHostToDevice);
    hipMemcpy(mat_b, h_matb, sizeof(double) * d2 * col, hipMemcpyHostToDevice);
    //hipMemcpy(row_d, row_ptr, sizeof(size_t), hipMemcpyHostToDevice);
    //hipMemcpy(dim_d, dim_ptr, sizeof(size_t), hipMemcpyHostToDevice);
    //hipMemcpy(co_d, col_ptr, sizeof(size_t), hipMemcpyHostToDevice);
    // Initializing variable Done

    // Launching Kernel Begin
    size_t block_dimx = 32;
    size_t block_dimy = 32;
    int gridX = (row + block_dimx - 1)/block_dimx;
    int gridY = (col + block_dimy - 1)/block_dimy;
    //void* kernelParam[] = {mat_a,mat_b,mat_c,row_d,dim_d,col_d};
	std::vector<void*> kernelargs;
	kernelargs.push_back(mat_a);
	kernelargs.push_back(mat_b);
	kernelargs.push_back(mat_c);
	kernelargs.push_back((void*)row);
	kernelargs.push_back((void*)d1);
	kernelargs.push_back((void*)col);

	size_t size = sizeof(void*) * kernelargs.size();
	void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, kernelargs.data(),
					  HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, 
					  HIP_LAUNCH_PARAM_END};

    assert (hipModuleLaunchKernel(
        kernel, gridX, gridY, 1, block_dimx, block_dimy, 1,
        0, nullptr, nullptr, (void**)&config) == 0);
    hipStreamSynchronize(0);
    // Launching Kernel Done
    hipMemcpy(h_matc, mat_c, sizeof(double) * row * col, hipMemcpyDeviceToHost);

	for (int i = 0; i < row * col; ++i) {
		std::cout << "c = " << h_matc[i] << std::endl;
	}

    // Clean up
    assert (hipFree(mat_a) == 0);
    assert (hipFree(mat_b) == 0);
    assert (hipFree(mat_c) == 0);
    //assert (hipFree(row_d) == 0);
    //assert (hipFree(col_d) == 0);
    //assert (hipFree(dim_d) == 0);
    std::cout << "done" << std::endl;
    return 0;
}
