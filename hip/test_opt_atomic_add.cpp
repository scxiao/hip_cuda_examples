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


int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " n" << std::endl;
        return 0;
    }
    int size = std::atoi(argv[1]);
    std::vector<int> vec(size);
    std::iota(vec.begin(), vec.end(), 0);
    shuffle_vec(vec);

    std::cout << "init_vec:" << std::endl;
    std::cout << vec << std::endl;

    // refer result
    std::vector<int> vec_gold(vec);
    mergeSort(vec_gold);

    std::cout << "ref_result:" << std::endl;
    std::cout << vec_gold << std::endl;

    // GPU bitonic sort
    std::vector<int> vec_gpu;
    bitonicSort(vec, vec_gpu);

    std::cout << "gpu_result:" << std::endl;
    std::cout << vec_gpu << std::endl;

    return 0;
}
