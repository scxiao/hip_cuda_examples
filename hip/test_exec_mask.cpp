#include <iostream>
#include <vector>
#include <cassert>
#include <numeric>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <init_vec.hpp>
#include <utilities.hpp>
#include <iomanip>

__global__ void maskOper_1(uint64_t mask, int32_t *out) {
  int tid = threadIdx.x;
  if (tid % 2 == 1) {
    int index = __builtin_amdgcn_mbcnt_hi(static_cast<uint32_t>(mask >> 32),
                                      __builtin_amdgcn_mbcnt_lo(static_cast<uint32_t>(mask), 0));
    out[tid] = index;
  }
}


__global__ void maskOper_2(uint64_t* mask) {
  int tid = threadIdx.x;
  uint64_t m = 0;
  if (tid % 2 == 1) {
    m = __ballot(true);
  }
  mask[tid] = m;
}

void testMaskOper_2(std::vector<uint64_t> &output) {
  uint64_t *outd;
  int waveSize = 64;
  hipMalloc((void **)&outd, sizeof(uint64_t) * waveSize);
  maskOper_2<<<1, waveSize>>>(outd);
  output.resize(waveSize);
  hipMemcpy((void *)(output.data()), outd, sizeof(uint64_t) * waveSize, hipMemcpyDeviceToHost);
}

int main(int argc, char **argv) {
    if (argc != 1) {
        std::cout << "Usage: " << argv[0] << std::endl;
        return 0;
    }

    std::vector<uint64_t> output;
    // uint64_t mask = 0xAAAAAAAAAAAAAAAAll;
    testMaskOper_2(output);
    // std::cout << "output = \n" << output << std::endl;

    for (auto v: output) {
      std::cout << "0x" << std::hex << v << std::endl;
    }

    return 0;
}
