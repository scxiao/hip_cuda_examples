#include <iostream>
#include <vector>
#include <cassert>
#include <numeric>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <init_vec.hpp>
#include <utilities.hpp>


__global__ void kernelPrefixSum(float *arr_in, float *arr_out, int elemNum) {
    int tid = threadIdx.x;
    int size = blockDim.x;
    extern __shared__ float tmp[];
    int bout = 0, bin = 1;

    tmp[bout * size + tid] = (tid < elemNum) ? arr_in[tid]: 0;
    __syncthreads();
    for (int offset = 1; offset < elemNum; offset *= 2) {
        bout = 1 - bout;
        bin = 1 - bin;
        if (tid >= offset) {
            tmp[bout * size + tid] = tmp[bin * size + tid] + tmp[bin * size + tid - offset];
        }
        else {
            tmp[bout * size + tid] = tmp[bin * size + tid];
        }
        __syncthreads();
    }
    arr_out[tid] = tmp[bout * size + tid];
}

void testPrefixSum(const std::vector<float> &vec, std::vector<float> &vec_out) {
    float *vecd, *vecd_out;
    int elemNum = vec.size();
    unsigned int size = elemNum * sizeof(float);
    vec_out.resize(elemNum, 0);

    int threadsPerBlock = 256;
    int blocksPerGrid = (elemNum + threadsPerBlock - 1) / threadsPerBlock;

    hipMalloc((void**)&vecd, size);
    hipMalloc((void**)&vecd_out, size);
    hipMemset(vecd_out, 0, size);
    hipMemcpy(vecd, vec.data(), size, hipMemcpyHostToDevice);
    std::size_t sharedSize = 2 * threadsPerBlock * sizeof(float);
    kernelPrefixSum<<<blocksPerGrid, threadsPerBlock, sharedSize>>>(vecd, vecd_out, elemNum);
    hipMemcpy((void*)vec_out.data(), vecd_out, size, hipMemcpyDeviceToHost);
    hipFree(vecd);
    hipFree(vecd_out);
}


void reduce_write(const std::vector<float>& vals, 
                  const std::vector<int>& indices,
                  std::vector<float> &outs) {
    assert(vals.size() == indices.size());
    for (int i = 0; i < vals.size(); ++i) {
        outs[indices[i]] += vals[i];
    }
}


__global__ void kernelDynamicPrefixSum(float *arr_in, int *idx, float *arr_out, int elemNum) {
    int tid = threadIdx.x;
    int size = blockDim.x;
    extern __shared__ float tmp[];
    int *idxb = ((int*)tmp) + 2 * size;
    int bout = 0, bin = 1;

    tmp[bout * size + tid] = (tid < elemNum) ? arr_in[tid]: 0;
    idxb[tid] = (tid < elemNum) ? idx[tid] : -1;
    __syncthreads();
    for (int offset = 1; offset < elemNum; offset *= 2) {
        bout = 1 - bout;
        bin = 1 - bin;
        if (tid >= offset and idx[tid - offset] == idx[tid]) {
            tmp[bout * size + tid] = tmp[bin * size + tid] + tmp[bin * size + tid - offset];
        }
        else {
            tmp[bout * size + tid] = tmp[bin * size + tid];
        }
        __syncthreads();
    }

    __shared__ bool master[1024];
    if ((tid < elemNum - 1 and idx[tid] != idx[tid + 1]) or (tid == elemNum - 1)) {
        master[tid] = true;
    }
    else {
        master[tid] = false;
    }

    if (master[tid]) {
        arr_out[idx[tid]] = tmp[bout * size + tid];
    }
}

void testDynamicPrefixSum(const std::vector<float> &vec, const std::vector<int> &idx, std::vector<float> &vec_out) {
    float *vecd, *vecd_out;
    int *idxd;
    int elemNum = vec.size();
    std::size_t size = elemNum * sizeof(float);
    std::size_t idxSize = elemNum * sizeof(int);
    vec_out.resize(elemNum, 0);

    int threadsPerBlock = 256;
    int blocksPerGrid = (elemNum + threadsPerBlock - 1) / threadsPerBlock;

    hipMalloc((void**)&vecd, size);
    hipMalloc((void**)&vecd_out, size);
    hipMalloc((void**)&idxd, idxSize);
    hipMemset(vecd_out, 0, size);
    hipMemcpy(vecd, vec.data(), size, hipMemcpyHostToDevice);
    hipMemcpy(idxd, idx.data(), idxSize, hipMemcpyHostToDevice);
    std::size_t sharedSize = 3 * threadsPerBlock * sizeof(float);
    kernelDynamicPrefixSum<<<blocksPerGrid, threadsPerBlock, sharedSize>>>(vecd, idxd, vecd_out, elemNum);
    hipMemcpy((void*)vec_out.data(), vecd_out, size, hipMemcpyDeviceToHost);
    hipFree(vecd);
    hipFree(vecd_out);
}


int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " n" << std::endl;
        return 0;
    }
    int size = std::atoi(argv[1]);
    std::vector<float> vecVal(size);
    std::vector<int> vecIdx(size);
    init_vec(vecVal, vecVal.size());
    init_vec(vecIdx, vecIdx.size());

    std::cout << "Before sorting:" << std::endl;
    std::cout << "val = \n" << vecVal << std::endl;

    std::sort(vecIdx.begin(), vecIdx.end());
    std::cout << "sorted_idx =" << std::endl;
    std::cout << vecIdx << std::endl;

    // golden output
    std::vector<float> golden_out(vecVal.size(), 0);
    reduce_write(vecVal, vecIdx, golden_out);
    std::cout << "\ngolden = \n" << golden_out << std::endl;

    // GPU bitonic sort
    std::vector<float> vecSum;
    testDynamicPrefixSum(vecVal, vecIdx, vecSum);

    std::cout << "\ngpuPrefixSum:" << std::endl;
    std::cout << "val = \n" << vecSum << std::endl;

    return 0;
}
