// Wave Matrix Multiply Accumulate (WMMA) using HIP compiler intrinsic
// Does a matrix multiplication of two 16x16, fp16 matrices, and stores them into a 16x16 fp16 result matrix

#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

using namespace std;



// Use half16 as an alias of the internal clang vector type of 16 fp16 values
typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));

__global__ void wmma_matmul(__half* a, __half* b, float* c)
{
    const int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const int lIdx = threadIdx.x;

    // a and b fragments are stored in 8 VGPRs each, in packed format, so 16 elements each for a and b
    // a_frag will store one column of the 16x16 matrix A tile
    // b_frag will store one row of the 16x16 matrix B tile
    half16 a_frag;
    half16 b_frag;
    // initialize c fragment to 0
    float8 c_frag = {};

    // lane is (0-31) mod 16 instead of 0-31 due to matrix replication in RDNA 3
    const int lane = lIdx % 16;

    for (int ele = 0; ele < 16; ++ele)
    {
        b_frag[ele] = b[16*ele + lane];
    }

    for (int ele = 0; ele < 16; ++ele)
    {
        a_frag[ele] = a[16 * lane + ele];
    }

    // call the WMMA intrinsic with OPSEL set to "false"
    // c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag, b_frag, c_frag, false);
    c_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, c_frag);

    for (int ele = 0; ele < 8; ++ele)
    {
        const int r = ele * 2 + (lIdx / 16);
        // store results from unpacked c_frag output
        c[16 * r + lane] = c_frag[ele];
        // if OPSEL was set to "true", the line above would instead be
        // c[16 * r + lane] = c_frag[ele*2 + 1];
    }
}

void matmul_ref(__half *a, __half *b, float *c, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int s = 0; s < k; ++s) {
                sum += (float)a[i * k + s] * (float)b[s * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

int main(int argc, char* argv[])
{
    __half a[16 * 16] = {};
    __half b[16 * 16] = {};
    float c[16 * 16] = {};
    float c_ref[16 * 16] = {};
    __half *a_gpu, *b_gpu;
    float *c_gpu;
    hipMalloc(&a_gpu, 16*16 * sizeof(__half));
    hipMalloc(&b_gpu, 16*16 * sizeof(__half));
    hipMalloc(&c_gpu, 16*16 * sizeof(float));

    // fill in some data into matrices A and B
    int k = 0;
    for (int i = 0; i < 16; ++i)
    {
        for (int j = 0; j < 16; ++j)
        {
            a[i * 16 + j] = (__half)((k + 1) * 0.1f);
            b[i * 16 + j] = (__half)((k + 2) * 0.2f);
            k++;
        }
    }

    hipMemcpy(a_gpu, a, (16*16) * sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(b_gpu, b, (16*16) * sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(c_gpu, c, (16*16) * sizeof(float), hipMemcpyHostToDevice);

    wmma_matmul<<<dim3(1), dim3(32, 1, 1), 0, 0>>>(a_gpu, b_gpu, c_gpu);
    auto err = hipGetLastError();
    std::cout << "err = " << err << std::endl;
    std::cout << "err: " << hipGetErrorString(err) << std::endl;
    hipMemcpy(c, c_gpu, (16 * 16) * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(a_gpu);
    hipFree(b_gpu);
    hipFree(c_gpu);

    matmul_ref(a, b, c_ref, 16, 16, 16);

    std::cout << "wmma_output:" << std::endl;
    for (int i = 0; i < 16; ++i)
    {
        for (int j = 0; j < 16; ++j)
        {
            printf("%f ", (float)c[i * 16 + j]);
        }
        printf("\n");
    }

    std::cout << "ref_output:" << std::endl;
    for (int i = 0; i < 16; ++i)
    {
        for (int j = 0; j < 16; ++j)
        {
            printf("%f ", (float)c_ref[i * 16 + j]);
        }
        printf("\n");
    }

    return 0;
}

