#include <iostream>
#include <vector>
#include <cassert>
#include <numeric>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <init_vec.hpp>
#include <utilities.hpp>


template <typename T>
__device__ void atomicAddNoRetGFX9(T *addr_in, T *addr_out, T value)
{
  auto readfirstlane = [](auto arg) -> auto
  {
    constexpr size_t count = (sizeof(arg) + 3) / 4;
    union
    {
      decltype(arg) v;
      int32_t i[count];
    } u{arg};
    for (size_t i = 0; i < count; i++)
    {
      u.i[i] = __builtin_amdgcn_readfirstlane(u.i[i]);
    }
    return u.v;
  };
  auto permute = [](uint32_t index, auto arg) -> auto
  {
    constexpr size_t count = (sizeof(arg) + 3) / 4;
    union
    {
      decltype(arg) v;
      int32_t i[count];
    } u{arg};
    for (size_t i = 0; i < count; i++)
    {
      u.i[i] = __builtin_amdgcn_ds_permute(index, u.i[i]);
    }
    return u.v;
  };
  auto bpermute = [](uint32_t index, auto arg) -> auto
  {
    constexpr size_t count = (sizeof(arg) + 3) / 4;
    union
    {
      decltype(arg) v;
      int32_t i[count];
    } u{arg};
    for (size_t i = 0; i < count; i++)
    {
      u.i[i] = __builtin_amdgcn_ds_bpermute(index, u.i[i]);
    }
    return u.v;
  };

  bool done = false;
  uint32_t base = 0;
  uint32_t index;
  uint32_t start;
  uint32_t count;
  bool leader;

  // group lanes by address
  while (!done)
  {
    auto chosen = readfirstlane(addr_in);
    bool done_ = chosen == addr_in;
    uint64_t mask = __ballot(done_);
    start = base;
    count = __popcll(mask);
    index = __builtin_amdgcn_mbcnt_hi(static_cast<uint32_t>(mask >> 32),
                                      __builtin_amdgcn_mbcnt_lo(static_cast<uint32_t>(mask), 0));
    base = start + count;
    leader = index == 0;
    count -= index;
    index += start;

    done = done_;
  }

  // coalesce
  uint32_t index_times_four = index * 4;
  // addr_in = (T*) permute(index_times_four, (T __attribute__((address_space(1)))*) addr_in);
  addr_out = (T*) permute(index_times_four, (T __attribute__((address_space(1)))*) addr_out);
  value = permute(index_times_four, value);

  uint32_t packed = permute(index_times_four, leader | index_times_four | (count << 8));

  // NOTE: bpermute will ignore other bits so we don't mask here
  index_times_four = packed;
  count = (packed >> 8) & 0xff;
  leader = (packed & 1) != 0;

  // reduce
  T acc = value;
  // TODO: Is this is helpfull?
  if (__ballot(count != 1) != 0)
  {
#pragma clang loop unroll(full)
    for (int i = 32; i != 0; i /= 2)
    {
      auto tmp = bpermute(index_times_four + i * 4, acc);
      acc += (i < count) ? tmp : T(0);
    }
  }

  // apply
  if (leader)
  {
     atomicAdd(addr_out, acc);
  }
}

__global__ void kernelPrefixSumImpl(float *arr_in, int *idx, float *arr_out, int elemNum) {
    int tid = threadIdx.x;
    atomicAddNoRetGFX9(arr_in + idx[tid], arr_out + idx[tid], arr_in[tid]);
}

void testPrefixSumImpl(const std::vector<float> &vec, const std::vector<int> &idx, std::vector<float> &vec_out) {
    float *vecd, *vecd_out;
    int *idxd;
    int elemNum = vec.size();
    std::size_t size = elemNum * sizeof(float);
    std::size_t idxSize = elemNum * sizeof(int);
    vec_out.resize(elemNum, 0);

    int threadsPerBlock = 64;
    int blocksPerGrid = (elemNum + threadsPerBlock - 1) / threadsPerBlock;

    hipMalloc((void**)&vecd, size);
    hipMalloc((void**)&vecd_out, size);
    hipMalloc((void**)&idxd, idxSize);
    hipMemset(vecd_out, 0, size);
    hipMemcpy(vecd, vec.data(), size, hipMemcpyHostToDevice);
    hipMemcpy(idxd, idx.data(), idxSize, hipMemcpyHostToDevice);
    // std::size_t sharedSize = 3 * threadsPerBlock * sizeof(float);
    kernelPrefixSumImpl<<<blocksPerGrid, threadsPerBlock>>>(vecd, idxd, vecd_out, elemNum);
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
    std::cout << "\ngpuPrefixSum:" << std::endl;
    std::vector<float> vecSum;
    testDynamicPrefixSum(vecVal, vecIdx, vecSum);
    std::cout << vecSum << std::endl;

    // GPU bitonic sort
    std::cout << "\ngpuPrefixSumImpl:" << std::endl;
    std::vector<float> vecSumImpl;
    testPrefixSumImpl(vecVal, vecIdx, vecSumImpl);
    std::cout << vecSumImpl << std::endl;

    return 0;
}
