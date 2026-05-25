/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "Tensor/Tensor.h"
#include "Utils/CUDATypes.h"
#include "Utils/CUDAUtils.h"
#include "kernel/Dispatch.h"
#include "kernel/GemvOps.h"

namespace tinygpt::kernel {

static constexpr int kGemvWarpsPerBlock = 8;
static constexpr int kGemvBlockSize = kGemvWarpsPerBlock * 32;  // 256 threads

__device__ __forceinline__ float warpReduceSum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

template <typename T>
__global__ void __launch_bounds__(256) kGemvLmHead(const T* __restrict__ input,   // [K]
                                                   const T* __restrict__ weight,  // [N, K]
                                                   T* __restrict__ output,        // [N]
                                                   int N, int K) {
  const unsigned int tid = threadIdx.x;
  const unsigned int warpId = tid / 32;
  const unsigned int laneId = tid % 32;

  // which output row this warp is responsible for
  const unsigned int outIdx = blockIdx.x * kGemvWarpsPerBlock + warpId;
  if (outIdx >= static_cast<unsigned int>(N)) {
    return;
  }

  const T* wRow = weight + static_cast<int64_t>(outIdx) * K;
  float acc = 0.f;

  if constexpr (sizeof(T) == 2) {
    // BF16 / FP16: 8 elements per uint4 (128-bit) load
    constexpr int kElemsPerLoad = 8;
    const int totalLaneLoads = K / (32 * kElemsPerLoad);

    const auto* inVecBase = reinterpret_cast<const uint4*>(input);
    const auto* wVecBase = reinterpret_cast<const uint4*>(wRow);

#pragma unroll
    for (int iter = 0; iter < totalLaneLoads; iter++) {
      const unsigned int vecIdx = iter * 32 + laneId;
      uint4 in128 = __ldg(&inVecBase[vecIdx]);
      uint4 w128 = wVecBase[vecIdx];
      const T* inElem = reinterpret_cast<const T*>(&in128);
      const T* wElem = reinterpret_cast<const T*>(&w128);

      acc += static_cast<float>(inElem[0]) * static_cast<float>(wElem[0]);
      acc += static_cast<float>(inElem[1]) * static_cast<float>(wElem[1]);
      acc += static_cast<float>(inElem[2]) * static_cast<float>(wElem[2]);
      acc += static_cast<float>(inElem[3]) * static_cast<float>(wElem[3]);
      acc += static_cast<float>(inElem[4]) * static_cast<float>(wElem[4]);
      acc += static_cast<float>(inElem[5]) * static_cast<float>(wElem[5]);
      acc += static_cast<float>(inElem[6]) * static_cast<float>(wElem[6]);
      acc += static_cast<float>(inElem[7]) * static_cast<float>(wElem[7]);
    }

    // handle remainder (if K is not divisible by 256)
    const int processed = totalLaneLoads * 32 * kElemsPerLoad;
    for (unsigned int j = processed + laneId; j < K; j += 32) {
      acc += static_cast<float>(__ldg(&input[j])) * static_cast<float>(wRow[j]);
    }
  } else {
    // FP32: 4 elements per float4 (128-bit) load
    constexpr int kElemsPerLoad = 4;
    const int totalLaneLoads = K / (32 * kElemsPerLoad);

    const auto* inVecBase = reinterpret_cast<const float4*>(input);
    const auto* wVecBase = reinterpret_cast<const float4*>(wRow);

#pragma unroll
    for (int iter = 0; iter < totalLaneLoads; iter++) {
      const unsigned int vecIdx = iter * 32 + laneId;
      float4 in128 = __ldg(&inVecBase[vecIdx]);
      float4 w128 = wVecBase[vecIdx];

      acc += in128.x * w128.x;
      acc += in128.y * w128.y;
      acc += in128.z * w128.z;
      acc += in128.w * w128.w;
    }

    // handle remainder
    const int processed = totalLaneLoads * 32 * kElemsPerLoad;
    for (unsigned int j = processed + laneId; j < K; j += 32) {
      acc += __ldg(&input[j]) * wRow[j];
    }
  }

  // warp-level reduction
  acc = warpReduceSum(acc);
  if (laneId == 0) {
    output[outIdx] = static_cast<T>(acc);
  }
}

template <typename CudaT>
static tinytorch::Tensor gemvLmHeadImpl(const tinytorch::Tensor& input, const tinytorch::Tensor& weight) {
  const auto M = static_cast<int>(input.size(0));
  const auto K = static_cast<int>(input.size(1));
  const auto N = static_cast<int>(weight.size(0));
  ASSERT(M == 1);
  ASSERT(weight.size(1) == K);

  auto outOpts = input.options().noGrad();
  tinytorch::Tensor output({static_cast<int64_t>(M), static_cast<int64_t>(N)}, outOpts);

  const int numBlocks = (N + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
  auto stream = tinytorch::cuda::getCurrentCUDAStream(input.device().index).stream();
  kGemvLmHead<CudaT><<<numBlocks, kGemvBlockSize, 0, stream>>>(input.dataPtr<CudaT>(), weight.dataPtr<CudaT>(),
                                                               output.dataPtr<CudaT>(), N, K);
  CUDA_KERNEL_CHECK();

  return output;
}

tinytorch::Tensor gemvLmHead(const tinytorch::Tensor& input, const tinytorch::Tensor& weight) {
  ASSERT(input.dim() == 2);
  ASSERT(weight.dim() == 2);
  ASSERT(input.size(0) == 1);               // M=1 only
  ASSERT(input.size(1) == weight.size(1));  // K dimensions match

  tinytorch::Tensor output;
  TINYGPT_DISPATCH_FLOAT_DTYPE(input, { output = gemvLmHeadImpl<CudaT>(input, weight); });
  return output;
}

static int getGemvLinearMinN(int deviceIndex) {
  static int cachedMinN[8] = {};
  if (cachedMinN[deviceIndex] != 0) {
    return cachedMinN[deviceIndex];
  }

  int numSMs = 0;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceIndex);
  if (numSMs <= 0) {
    numSMs = 72;
  }

  const int minN = numSMs * 2 * kGemvWarpsPerBlock;
  cachedMinN[deviceIndex] = minN;
  return minN;
}

tinytorch::Tensor gemvLinear(const tinytorch::Tensor& input, const tinytorch::Tensor& weight) {
  ASSERT(input.device().isCuda());
  ASSERT(input.dim() == 2 && weight.dim() == 2);
  ASSERT(input.size(0) == 1);
  ASSERT(input.size(1) == weight.size(1));

  const auto K = input.size(1);
  ASSERT(K % 8 == 0);

  const auto N = weight.size(0);
  ASSERT(N >= getGemvLinearMinN(input.device().index));

  tinytorch::Tensor output;
  TINYGPT_DISPATCH_FLOAT_DTYPE(input, { output = gemvLmHeadImpl<CudaT>(input, weight); });
  return output;
}

}  // namespace tinygpt::kernel
