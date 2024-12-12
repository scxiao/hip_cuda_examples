#include <iostream>
#include <vector>
#include <cassert>
#include <numeric>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <init_vec.hpp>
#include <utilities.hpp>

using namespace std;

template<class T>
void reduce_write(const vector<T>& vals, 
                  const vector<int>& indices,
                  vector<T> &outs) {
    assert(vals.size() == indices.size());
    for (int i = 0; i < vals.size(); ++i) {
        outs[indices[i]] += vals[i];
    }
}

template<class T>
void mergeSortRecur(vector<T>& vec, int l, int r) {
    if (l + 1 >= r) return;
    else if (l + 2 == r) {
        if (vec[l] > vec[l + 1]) {
            swap(vec[l], vec[l + 1]);
        }
        return;
    }
    else {
        int m = (l + r) / 2;
        mergeSortRecur(vec, l, m);
        mergeSortRecur(vec, m, r);
        // merge the two sub vector
        std::vector<float> v1(vec.begin() + l, vec.begin() + m);
        std::vector<float> v2(vec.begin() + m, vec.begin() + r);
        int i1 = 0, i2 = 0;
        while (i1 < v1.size() and i2 < v2.size()) {
            if (v1[i1] <= v2[i2]) {
                vec[l++] = v1[i1++];
            }
            else {
                vec[l++] = v2[i2++];
            }
        }
        if (i1 < v1.size()) {
            std::copy(v1.begin() + i1, v1.end(), vec.begin() + l);
        }
        if (i2 < v2.size()) {
            std::copy(v2.begin() + i2, v2.end(), vec.begin() + l);
        }
    }
}

template<class T>
void mergeSort(vector<T>& vec) {
    mergeSortRecur(vec, 0, vec.size());
}


// // GPU kernel for bitonic sort
// __global__ void kernelBitonicSort(int *arr, int j, int k) {
//     unsigned i, ij;
//     i = threadIdx.x;
//     ij = i ^ j;
//     if (ij > i) {
//         if ((i & k) == 0) {
//             if (arr[i] > arr[ij]) {
//                 int temp = arr[i];
//                 arr[i] = arr[ij];
//                 arr[ij] = temp;
//             }
//         }
//         else {
//             if (arr[i] < arr[ij]) {
//                 int temp = arr[i];
//                 arr[i] = arr[ij];
//                 arr[ij] = temp;
//             }
//         }
//     }
// }

// void bitonicSort(const std::vector<int> &vec, std::vector<int> &sorted_vec) {
//     int *vecd;
//     int elemNum = vec.size();
//     unsigned int size = elemNum * sizeof(int);

//     int threadsPerBlock = 1024;
//     int blocksPerGrid = (elemNum + threadsPerBlock - 1) / threadsPerBlock;

//     hipMalloc((void**)&vecd, size);
//     hipMemcpy(vecd, vec.data(), size, hipMemcpyHostToDevice);
//     int j, k;
//     for (k = 2; k <= elemNum; k <<= 1) {
//         for (j = k >> 1; j > 0; j = j >> 1) {
//             kernelBitonicSort<<<blocksPerGrid, threadsPerBlock>>>(vecd, j, k);
//         }
//     }
//     sorted_vec.resize(vec.size());
//     hipMemcpy((void*)sorted_vec.data(), vecd, size, hipMemcpyDeviceToHost);
// }


// GPU kernel for bitonic sort
__global__ void kernelBitonicSort(int *arr, int elem_num) {
    unsigned i, j, k, ij;
    i = threadIdx.x;

    for (k = 2; k <= elem_num; k <<= 1) {
        for (j = k >> 1; j > 0; j = j >> 1) {
            ij = i ^ j;
            if (ij > i) {
                if ((i & k) == 0) {
                    if (arr[i] > arr[ij]) {
                        int temp = arr[i];
                        arr[i] = arr[ij];
                        arr[ij] = temp;
                    }
                }
                else {
                    if (arr[i] < arr[ij]) {
                        int temp = arr[i];
                        arr[i] = arr[ij];
                        arr[ij] = temp;
                    }
                }
            }
            __syncthreads();
        }
    }
}

void bitonicSort(const std::vector<int> &vec, std::vector<int> &sorted_vec) {
    int *vecd;
    int elemNum = vec.size();
    unsigned int size = elemNum * sizeof(int);

    int threadsPerBlock = 1024;
    int blocksPerGrid = (elemNum + threadsPerBlock - 1) / threadsPerBlock;

    hipMalloc((void**)&vecd, size);
    hipMemcpy(vecd, vec.data(), size, hipMemcpyHostToDevice);
    kernelBitonicSort<<<blocksPerGrid, threadsPerBlock>>>(vecd, elemNum);
    sorted_vec.resize(vec.size());
    hipMemcpy((void*)sorted_vec.data(), vecd, size, hipMemcpyDeviceToHost);
}


// GPU kernel for bitonic sort
__global__ void kernelIdxBitonicSort(int *arr, int *idx, int elemNum) {
    unsigned i, j, k, ij;
    i = threadIdx.x;

    for (k = 2; k <= elemNum; k <<= 1) {
        for (j = k >> 1; j > 0; j = j >> 1) {
            ij = i ^ j;
            if (ij > i) {
                if ((i & k) == 0) {
                    if (idx[i] > idx[ij]) {
                        int temp = arr[i];
                        arr[i] = arr[ij];
                        arr[ij] = temp;
                        temp = idx[i];
                        idx[i] = idx[ij];
                        idx[ij] = temp;
                    }
                }
                else {
                    if (idx[i] < idx[ij]) {
                        int temp = arr[i];
                        arr[i] = arr[ij];
                        arr[ij] = temp;
                        temp = idx[i];
                        idx[i] = idx[ij];
                        idx[ij] = temp;
                    }
                }
            }
            __syncthreads();
        }
    }
}


__global__ void kernelDynamicPrefixSum(int *arr_in, int *idx, int *arr_out, int elemNum) {
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


void testDynamicPrefixSum(const std::vector<int> &vec, const std::vector<int>& idx, 
                    std::vector<int> &vec_out, std::vector<int> &sorted_idx) {
    int *vecd, *idxd, *vecd_out;
    int elemNum = vec.size();
    unsigned int size = elemNum * sizeof(int);

    int threadsPerBlock = 1024;
    int blocksPerGrid = (elemNum + threadsPerBlock - 1) / threadsPerBlock;

    hipMalloc((void**)&vecd, size);
    hipMalloc((void**)&vecd_out, size);
    hipMalloc((void**)&idxd, size);
    hipMemset(vecd_out, 0, size);
    hipMemcpy(vecd, vec.data(), size, hipMemcpyHostToDevice);
    hipMemcpy(idxd, idx.data(), size, hipMemcpyHostToDevice);

    // sort value according to index
    kernelIdxBitonicSort<<<blocksPerGrid, threadsPerBlock>>>(vecd, idxd, elemNum);

    // dynamic reduce_sum for each individual index value
    std::size_t sharedSize = 3 * threadsPerBlock * sizeof(float);
    kernelDynamicPrefixSum<<<blocksPerGrid, threadsPerBlock, sharedSize>>>(vecd, idxd, vecd_out, elemNum);

    vec_out.resize(vec.size());
    sorted_idx.resize(idx.size());
    
    hipMemcpy((void*)vec_out.data(), vecd_out, size, hipMemcpyDeviceToHost);
    hipMemcpy((void*)sorted_idx.data(), idxd, size, hipMemcpyDeviceToHost);
    hipFree(vecd);
    hipFree(idxd);
}


int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " n" << std::endl;
        return 0;
    }
    int size = std::atoi(argv[1]);
    std::vector<int> vecVal(size), vecIdx(size);
    std::iota(vecVal.begin(), vecVal.end(), 0);
    std::iota(vecIdx.begin(), vecIdx.begin() + 4, 0);
    shuffle_vec(vecVal);
    init_vec(vecIdx, vecIdx.size());

    std::cout << "Before sorting:" << std::endl;
    std::cout << "val = \n" << vecVal << std::endl;
    std::cout << "idx = \n" << vecIdx << std::endl;

    // refer result
    std::vector<int> golden_out(vecVal.size(), 0);
    reduce_write(vecVal, vecIdx, golden_out);
    std::cout << "\ngolden = \n" << golden_out << std::endl;

    // GPU bitonic sort
    std::vector<int> vecVal_gpu, vecIdx_gpu;
    testDynamicPrefixSum(vecVal, vecIdx, vecVal_gpu, vecIdx_gpu);

    std::cout << "GPU sorted results:" << std::endl;
    std::cout << "val = \n" << vecVal_gpu << std::endl;
    std::cout << "idx = \n" << vecIdx_gpu << std::endl;

    return 0;
}
