#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>

using namespace std;

__global__ void change_value(volatile int *lock) {
    atomicCAS((int*)lock, 0, 1);
}

__global__ void serialized_add(float *data, volatile int* lock, int elem_num) {
    int idx = threadIdx.x;
    if (idx == 0) {
        while (atomicCAS((int*)lock, 0, 1) == 1) {
        }
    }
    __syncthreads();
    if (idx < elem_num) {
        data[idx] += 1.0;
    }

    __syncthreads();
    __threadfence();
    if (idx == 0)
        //atomicExch((int*)lock, 0);
        atomicCAS((int*)lock, 1, 0);
}

__global__ void atomic_add(int *data) {
    atomicAdd(data, 1);
}

int main() {
    int* lock, lock_cpu = 0.0f;
    hipMalloc((void **)&lock, sizeof(int));
    hipMemset(lock, 0, sizeof(int));
    hipStream_t stream;
    hipStreamCreate(&stream);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    dim3 gridDim(200);
    dim3 blockDim(1);

    // warm up
    for (int i = 0; i < 200; ++i) {
        atomic_add<<<gridDim, blockDim, 0, stream>>>(lock);
    }
    hipStreamSynchronize(stream);
    hipMemcpy(&lock_cpu, lock, sizeof(int), hipMemcpyDeviceToHost);
    std::cout << "lock_val = " << lock_cpu << std::endl;

    std::vector<float> vecKernelTime;
    float ms;
    hipMemset(lock, 0, sizeof(int));
    for (int i = 0; i < 200; ++i) {
        dim3 gridDim1(i + 1);
        hipEventRecord(start, stream);
        atomic_add<<<gridDim1, blockDim, 0, stream>>>(lock);
        hipEventRecord(stop, stream);
        hipStreamSynchronize(stream);
        hipEventElapsedTime(&ms, start, stop);
        vecKernelTime.push_back(ms);
    }

    char c = '{';
    for (auto v : vecKernelTime) {
        std::cout << c;
        std::cout << v;
        if (c == '{') c = ',';
    }
    std::cout << '}';

    hipFree(lock);

    return 0;
}

