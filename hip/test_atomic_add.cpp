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


__device__ int atomic_add_block(int *data, int val) {
    __shared__ int count;
    int tid = threadIdx.x;
    if (tid == 0) {
        count = atomicAdd(data, val);
    }
    __syncthreads();
    return count;
}


__global__ void atomic_add(int *data, int *output) {
    int bid = blockIdx.x;
    int block_size = blockDim.x;
    int tid = threadIdx.x;

    int count = atomic_add_block(data, 1);
    int id = block_size * bid + threadIdx.x;
    output[id] = count;
}

void print_output(const std::vector<int>& vec_out, int block_num, int block_size) {
    for (int i = 0; i < block_num; ++i) {
        std::cout << "i = " << i << "\n";
        char c = '{';
        for (int j = 0; j < block_size; ++j) {
            std::cout << c << vec_out[i * block_size + j];
            if (c == '{') c = ',';
        }
        std:cout << '}' << std::endl;
    }
    std::cout << std::endl;
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

    int max_block_num = 200;
    int block_size = 128;
    dim3 gridDim(max_block_num);
    dim3 blockDim(block_size);

    int *outputd;
    std::vector<int> vec_out(max_block_num * block_size);
    hipMalloc((void**)&outputd, max_block_num * block_size * sizeof(int));

    // warm up
    for (int i = 0; i < 200; ++i) {
        atomic_add<<<gridDim, blockDim, 4, stream>>>(lock, outputd);
    }
    hipStreamSynchronize(stream);
    hipMemcpy(&lock_cpu, lock, sizeof(int), hipMemcpyDeviceToHost);
    std::cout << "lock_val = " << lock_cpu << std::endl;

    hipMemcpy((void*)vec_out.data(), outputd, max_block_num * block_size * sizeof(int), hipMemcpyDeviceToHost);
    print_output(vec_out, max_block_num, block_size);

    std::vector<float> vecKernelTime;
    float ms;
    for (int i = 0; i < max_block_num; ++i) {
        hipMemset(lock, 0, sizeof(int));
        dim3 gridDim1(i + 1);
        hipEventRecord(start, stream);
        atomic_add<<<gridDim1, blockDim, 0, stream>>>(lock, outputd);
        hipEventRecord(stop, stream);
        hipStreamSynchronize(stream);
        hipMemcpy(&lock_cpu, lock, sizeof(int), hipMemcpyDeviceToHost);
        hipEventElapsedTime(&ms, start, stop);
        vecKernelTime.push_back(ms);
        std::cout << "lock = " << lock_cpu << std::endl;
    }

    char c = '{';
    for (auto v : vecKernelTime) {
        std::cout << c;
        std::cout << v;
        if (c == '{') c = ',';
    }
    std::cout << '}';

    hipFree(lock);
    hipFree(outputd);

    return 0;
}

