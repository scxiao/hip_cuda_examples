#include <iostream>
#include <vector>
#include <cassert>
#include <numeric>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <init_vec.hpp>
#include <utilities.hpp>

using namespace std;

void reduce_write(const vector<float>& vals, 
                  const vector<int>& indices,
                  vector<float> &outs) {
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


__global__ void kernelDynamicReduceSum(int *arr, int *idx, int elemNum) {
    int i = threadIdx.x;
    __shared__ int master[1024];

    if (i >= elemNum) return;

    // find master thread
    if (i == 0 or idx[i] != idx[i - 1]) {
        master[i] = 1;
    }
    else {
        master[i] = 0;
    }

    // calculate the number of elements to be added by each master thread
    int sum = 0;
    if (master[i]) {
        sum = arr[i];
        int ii = i + 1;
        while (master[ii] == 0 and ii < elemNum) {
            sum += arr[ii];
        }
    }

    __syncthreads();
    arr[i] = 0;
    
    if (master[i]) {
        arr[idx[i]] = sum;
    }
}


void idxBitonicSort(const std::vector<int> &vec, const std::vector<int>& idx, 
                    std::vector<int> &sorted_vec, std::vector<int> &sorted_idx) {
    int *vecd, *idxd;
    int elemNum = vec.size();
    unsigned int size = elemNum * sizeof(int);

    int threadsPerBlock = 1024;
    int blocksPerGrid = (elemNum + threadsPerBlock - 1) / threadsPerBlock;

    hipMalloc((void**)&vecd, size);
    hipMalloc((void**)&idxd, size);
    hipMemcpy(vecd, vec.data(), size, hipMemcpyHostToDevice);
    hipMemcpy(idxd, idx.data(), size, hipMemcpyHostToDevice);

    // sort value according to index
    kernelIdxBitonicSort<<<blocksPerGrid, threadsPerBlock>>>(vecd, idxd, elemNum);

    // dynamic reduce_sum for each individual index value
    kernelDynamicReduceSum<<<blocksPerGrid, threadsPerBlock>>>(vecd, idxd, elemNum);

    sorted_vec.resize(vec.size());
    sorted_idx.resize(idx.size());
    hipMemcpy((void*)sorted_vec.data(), vecd, size, hipMemcpyDeviceToHost);
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
    // std::vector<int> vec_gold(vec);
    // mergeSort(vec_gold);

    // std::cout << "ref_result:" << std::endl;
    // std::cout << vec_gold << std::endl;

    // GPU bitonic sort
    std::vector<int> vecVal_gpu, vecIdx_gpu;
    idxBitonicSort(vecVal, vecIdx, vecVal_gpu, vecIdx_gpu);

    std::cout << "GPU sorted results:" << std::endl;
    std::cout << "val = \n" << vecVal_gpu << std::endl;
    std::cout << "idx = \n" << vecIdx_gpu << std::endl;

    return 0;
}
