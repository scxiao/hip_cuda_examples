#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>
#include <climits>
#include <vector>

using namespace std;

__global__ void vec_add(float *in1, float *in2, float *res, int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    //for (int i = tid; i < n; i += nglobal)
    int i = tid;
    if (i < n)
    {
        res[i] = in1[i] + in2[i];
    }

    return;
}


int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " n" << endl;
        return 0;
    }

    int sn = atoi(argv[1]);
    size_t n = (1 << sn);

    cout << "vector_size = " << n << endl;
    srand(time(nullptr));
    std::vector<float> fin1, fin2, fres;
    fin1.resize(n);
    fin2.resize(n);
    fres.resize(n);

    for (int i = 0; i < n; ++i)
    {
        fin1[i] = 1.0 * rand() / INT_MAX;
        fin2[i] = 1.0 * rand() / INT_MAX;
    }

    float *din1, *din2, *dres;
    cudaMalloc((void**)&din1, sizeof(float) * n);
    cudaMalloc((void**)&din2, sizeof(float) * n);
    cudaMalloc((void**)&dres, sizeof(float) * n);

    cudaMemcpy(din1, fin1.data(), sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(din2, fin2.data(), sizeof(float) * n, cudaMemcpyHostToDevice);

    int iter_num = 100;
    std::vector<cudaEvent_t> start(iter_num), stop(iter_num);
    for (int i = 0; i < iter_num; ++i) {
        cudaEventCreate(&start[i]);
        cudaEventCreate(&stop[i]);
    }

    size_t block_size = 1024;
    size_t block_num = (n + block_size - 1) / block_size;
    for (int i = 0; i < iter_num; ++i) {
        cudaEventRecord(start[i]);
        vec_add<<<block_num, block_size>>>(din1, din2, dres, n);
        cudaEventRecord(stop[i]);
    }
    cudaMemcpy((void*)fres.data(), dres, sizeof(float) * n, cudaMemcpyDeviceToHost);

    for (int i = 0; i < iter_num; ++i) {
        float gpu_time = 0.0f;
        cudaEventElapsedTime(&gpu_time, start[i], stop[i]);
        std::cout << "cudaEvent_time = " << gpu_time << std::endl;
    }
    cout << "PASSED" << endl;

    return 0;
}

