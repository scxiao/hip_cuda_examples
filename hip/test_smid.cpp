#include <hip/hip_runtime.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <cassert>


// __device__ void print_cu(const char * name) {
//     // Check which CU this kernel is running on
//     if (threadIdx.x == 0 && threadIdx.y == 0) {
//         int xcc_id;
//         asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcc_id));
//         int cu_id;
//         asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 8, 4)" : "=s"(cu_id));
//         int se_id;
//         asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 13, 3)" : "=s"(se_id));
//         printf("%s running on XCD %d SE %d CU %d\n", name, xcc_id, se_id, cu_id);
//     }
// }
// __global__ void whoami() {
//     print_cu("I am");
// }



__global__ void identify_cu_kernel(int *cu_ids, int *cu_ids1) {
    __shared__ int temp[32 * 1024];
    // Get the global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Use the built-in __smid() to get the Compute Unit ID
    // Note: Only threads within the same wavefront are guaranteed to see 
    // the same SMID at a given moment.

    int xcc = __smid();

    auto cu = (xcc & ((1 << HW_ID_CU_ID_SIZE) -1));
    // // cu = (cu >> 2)& ((1 << HW_ID_CU_ID_SIZE) - 1);
    int se = ((xcc >> HW_ID_CU_ID_SIZE) & ((1 <<(HW_ID_SE_ID_SIZE))-1));
    auto chip = (xcc >> (HW_ID_SE_ID_SIZE + HW_ID_CU_ID_SIZE));
    cu_ids1[tid] = cu;

    int cu_id;
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 8, 4)" : "=s"(cu_id));

    int se_id;
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 13, 2)" : "=s"(se_id));

    int xcc_id;
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 4)" : "=s"(xcc_id));

    // Store the CU ID for later analysis on the host
    cu_ids[tid] = cu_id;
}

int main() {
    int num_blocks = 1024;
    int num_threads = 1024;
    const int total_num_threads = num_threads * num_blocks;
    int *h_cu_ids = (int*)malloc(total_num_threads * sizeof(int));
    int *h_cu_ids1 = (int*)malloc(total_num_threads * sizeof(int));
    int *d_cu_ids, *d_cu_ids1;
    
    hipMalloc(&d_cu_ids, total_num_threads * sizeof(int));
    hipMalloc(&d_cu_ids1, total_num_threads * sizeof(int));
    
    // Launch kernel with 2 blocks of 128 threads each
    identify_cu_kernel<<<num_blocks, num_threads>>>(d_cu_ids, d_cu_ids1);
    
    hipMemcpy(h_cu_ids, d_cu_ids, total_num_threads * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_cu_ids1, d_cu_ids1, total_num_threads * sizeof(int), hipMemcpyDeviceToHost);
    
    // Print the results for the first few threads of each block
    printf("Thread 0 (Block 0) is on CU: %d\n", h_cu_ids[0]);
    printf("Thread 128 (Block 1) is on CU: %d\n", h_cu_ids[128]);
    std::vector<int> vec_cu_ids;
    std::vector<int> vec_cu_ids1;
    for (int block_id = 0; block_id < num_blocks; ++block_id) {
        std::vector<int> vec(h_cu_ids + block_id * num_threads, h_cu_ids + block_id * num_threads + num_threads);
        std::vector<int> vec1(h_cu_ids1 + block_id * num_threads, h_cu_ids1 + block_id * num_threads + num_threads);
        assert(std::all_of(vec.begin(), vec.end(), [&](auto v) {
            return v == vec.front();
        }));
        assert(std::all_of(vec1.begin(), vec1.end(), [&](auto v) {
            return v == vec1.front();
        }));


        vec_cu_ids.push_back(vec.front());
        vec_cu_ids1.push_back(vec1.front());
        // char c = '{';
        // for (int thread_id = 0; thread_id < 128; ++thread_id) {
        //     std::cout << c;
        //     std::cout << h_cu_ids[block_id * 128 + thread_id];
        //     if (c == '{') c = ',';
        // }
        // std::cout << '}' << std::endl;
        
    }
    // std::sort(vec_cu_ids.begin(), vec_cu_ids.end());

    // assert(vec_cu_ids1 == vec_cu_ids);

    char c = '{';
    for (auto cu_id : vec_cu_ids) {
        std::cout << c << cu_id;
        if (c == '{') c = ',';
    }
    std::cout << c << std::endl;
    std::cout << std::endl;

    c = '{';
    for (auto cu_id : vec_cu_ids1) {
        std::cout << c << cu_id;
        if (c == '{') c = ',';
    }
    std::cout << c << std::endl;
    std::cout << std::endl;



    int cu_num = 32;
    for (int cu = 0; cu < cu_num; ++cu) {
        std::cout << "CU: " << cu << ", count = " << std::count(vec_cu_ids.begin(), vec_cu_ids.end(), cu) << std::endl;
    }

    hipFree(d_cu_ids);
    free(h_cu_ids);
    return 0;
}
