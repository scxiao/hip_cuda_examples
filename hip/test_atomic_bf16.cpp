#include <chrono>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <iostream>

using namespace std::chrono;
using namespace std;

__global__ void atomic_dpp(__half *a, __half *b) {
  const int lIdx = threadIdx.x;
  const int dppCtrl = 0x101;
  int old = 1;
  half first = half(float(lIdx) / 10);
  int isecond = __builtin_amdgcn_update_dpp(old, *(int *)(&first), dppCtrl, 15,
                                            15, false);
  b[lIdx] = *(half *)(&isecond);
  half2 pairH = half2(first, *(half *)(&isecond));
  if (lIdx % 2 == 0) {
    half2 res =
        __builtin_amdgcn_global_atomic_fadd_v2f16((half2 *)(&a[lIdx]), pairH);
  }
}

int main(int argc, char *argv[]) {
  __half a[64] = {};
  __half b[64] = {};
  __half *a_gpu, *b_gpu;
  hipMalloc(&a_gpu, 64 * sizeof(__half));
  hipMalloc(&b_gpu, 64 * sizeof(__half));
  for (int i = 0; i < 64; ++i) {
    a[i] = (__half)1.f;
    b[i] = (__half)0.f;
  }
  hipMemcpy(a_gpu, a, (64) * sizeof(__half), hipMemcpyHostToDevice);
  hipMemcpy(b_gpu, b, (64) * sizeof(__half), hipMemcpyHostToDevice);
  auto start = high_resolution_clock::now();
  // for (int i = 0; i < 100; ++i){
  atomic_dpp<<<dim3(1), dim3(64), 0, 0>>>(a_gpu, b_gpu);
  hipMemcpy(a, a_gpu, (64) * sizeof(__half), hipMemcpyDeviceToHost);
  hipMemcpy(b, b_gpu, (64) * sizeof(__half), hipMemcpyDeviceToHost);
  //}
  auto stop = high_resolution_clock::now();
  hipFree(a_gpu);
  hipFree(b_gpu);

  printf("result threadId + 1:\n");
  for (int i = 0; i < 64; ++i)
  {
    printf("%f ", (float)a[i]);
  }
  printf("\n");
  printf("moved data:\n");
  for (int i = 0; i < 64; ++i) {
    printf("%f ", (float)b[i]);
  }
  printf("\n");

  auto duration = duration_cast<microseconds>((stop - start) / 100);
  cout << "time: " << duration.count() << endl;

  return 0;
}

