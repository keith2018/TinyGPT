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

// =============================================================================
// High-bandwidth GEMV kernel: output[i] = dot(input, weight[i, :])
//
// Design: Warp-cooperative, 1 row per warp, no shared memory.
//
// Each warp (32 threads) computes one output element via coalesced 128-bit
// vectorized loads from both input and weight.  The input vector (2KB for
// K=1024 BF16) fits in L1 cache and is loaded via __ldg(); weight rows
// stream from HBM one row at a time.
//
// This design maximizes TLP (thread-level parallelism) by keeping the grid
// large: numBlocks = ceil(N/8).  On A10 (72 SMs), N=3072 gives 384 blocks
// = 5.3 blocks/SM, which is sufficient for memory-bound workloads.
//
// Performance: 86% peak bandwidth for large N (lm_head), 63-74% for medium N
// (qkv, gate_up).  The 63% floor for K=1024 is due to only 4 vectorized
// iterations per warp — insufficient ILP to fully hide HBM latency.
// Attempts at multi-row-per-warp (more ILP) fail because the resulting
// register pressure reduces occupancy and TLP, which matters more.
//
// Config: blockDim = 256, warpsPerBlock = 8, grid = ceil(N / 8)
// Launch bounds: minBlocksPerSM = 8 (sm_80+) / 4 (sm_75) / 2 (older)
// =============================================================================

static constexpr int kGemvWarpsPerBlock = 8;
static constexpr int kGemvBlockSize = kGemvWarpsPerBlock * 32;  // 256 threads

// Warp-level reduction via shuffle
__device__ __forceinline__ float warpReduceSum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// T4 (sm_75): max 1024 threads/SM → at most 4 blocks of 256
// A10/A100 (sm_80+): max 2048 threads/SM → can fit 8 blocks of 256
#if __CUDA_ARCH__ >= 800
#define GEMV_MIN_BLOCKS_PER_SM 8
#elif __CUDA_ARCH__ >= 750
#define GEMV_MIN_BLOCKS_PER_SM 4
#else
#define GEMV_MIN_BLOCKS_PER_SM 2
#endif

template <typename T>
__global__ void __launch_bounds__(256, GEMV_MIN_BLOCKS_PER_SM) kGemvLmHead(const T* __restrict__ input,   // [K]
                                                                           const T* __restrict__ weight,  // [N, K]
                                                                           T* __restrict__ output,        // [N]
                                                                           int N, int K) {
  const unsigned int tid = threadIdx.x;
  const unsigned int warpId = tid / 32;
  const unsigned int laneId = tid % 32;

  // Which output row this warp is responsible for
  const unsigned int outIdx = blockIdx.x * kGemvWarpsPerBlock + warpId;
  if (outIdx >= static_cast<unsigned int>(N)) return;

  const T* wRow = weight + static_cast<int64_t>(outIdx) * K;
  float acc = 0.f;

  if constexpr (sizeof(T) == 2) {
    // BF16 / FP16: 8 elements per uint4 (128-bit) load
    // Both input and weight loaded via vectorized 128-bit reads.
    // Input reads hit L1 cache via __ldg; weight reads stream from HBM.
    constexpr int kElemsPerLoad = 8;
    const int totalLaneLoads = K / (32 * kElemsPerLoad);

    const uint4* inVecBase = reinterpret_cast<const uint4*>(input);
    const uint4* wVecBase = reinterpret_cast<const uint4*>(wRow);

#pragma unroll
    for (int iter = 0; iter < totalLaneLoads; iter++) {
      const int vecIdx = iter * 32 + laneId;
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

    // Handle remainder (if K is not divisible by 256)
    const int processed = totalLaneLoads * 32 * kElemsPerLoad;
    for (int j = processed + laneId; j < K; j += 32) {
      acc += static_cast<float>(__ldg(&input[j])) * static_cast<float>(wRow[j]);
    }
  } else {
    // FP32: 4 elements per float4 (128-bit) load
    constexpr int kElemsPerLoad = 4;
    const int totalLaneLoads = K / (32 * kElemsPerLoad);

    const float4* inVecBase = reinterpret_cast<const float4*>(input);
    const float4* wVecBase = reinterpret_cast<const float4*>(wRow);

#pragma unroll
    for (int iter = 0; iter < totalLaneLoads; iter++) {
      const int vecIdx = iter * 32 + laneId;
      float4 in128 = __ldg(&inVecBase[vecIdx]);
      float4 w128 = wVecBase[vecIdx];

      acc += in128.x * w128.x;
      acc += in128.y * w128.y;
      acc += in128.z * w128.z;
      acc += in128.w * w128.w;
    }

    // Handle remainder
    const int processed = totalLaneLoads * 32 * kElemsPerLoad;
    for (int j = processed + laneId; j < K; j += 32) {
      acc += __ldg(&input[j]) * wRow[j];
    }
  }

  // Warp-level reduction — sum partial products across 32 lanes
  acc = warpReduceSum(acc);

  // Lane 0 writes the final result
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

  auto stream = tinytorch::cuda::getCurrentCUDAStream(input.device().index).stream();

  const int numBlocks = (N + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;

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

// =============================================================================
// Threshold logic for gemvLinear.
//
// Our kernel: 1 row/warp, 8 warps/block → numBlocks = ceil(N/8).
// Need enough blocks per SM to hide memory latency via TLP.
// With __launch_bounds__(256, minBlocks), up to minBlocks blocks can be resident per SM.
// We need at least ~2 blocks/SM for basic saturation.
// minN = numSMs × 2 × kGemvWarpsPerBlock
//
// A10 (72 SMs): minN = 72 × 2 × 8 = 1152
//   → qkv(3072) ✓, gate_up(5632) ✓, o/down(1024) fallback ✓
// =============================================================================

static int getGemvLinearMinN(int deviceIndex) {
  static int cachedMinN[8] = {};
  if (cachedMinN[deviceIndex] != 0) {
    return cachedMinN[deviceIndex];
  }

  int numSMs = 0;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceIndex);
  if (numSMs <= 0) numSMs = 72;

  const int minN = numSMs * 2 * kGemvWarpsPerBlock;
  cachedMinN[deviceIndex] = minN;
  return minN;
}

tinytorch::Tensor gemvLinear(const tinytorch::Tensor& input, const tinytorch::Tensor& weight) {
  // Guard: only handle M=1, 2D, CUDA, matching K
  if (input.dim() != 2 || weight.dim() != 2) return {};
  if (input.size(0) != 1) return {};
  if (!input.device().isCuda()) return {};
  if (input.size(1) != weight.size(1)) return {};

  // Ensure K is divisible by 8 (uint4 vectorized loads require 16-byte alignment on BF16/FP16)
  // and divisible by 4 for FP32.  K for LLMs is always a multiple of 128, so this is safe.
  const auto K = input.size(1);
  if (K % 8 != 0) return {};

  // Only engage when we have enough output rows to saturate the GPU.
  // Below the threshold, cuBLAS's multi-warp-per-row GEMV (gemvx) is faster.
  const auto N = weight.size(0);
  if (N < getGemvLinearMinN(input.device().index)) return {};

  tinytorch::Tensor output;
  TINYGPT_DISPATCH_FLOAT_DTYPE(input, { output = gemvLmHeadImpl<CudaT>(input, weight); });
  return output;
}

}  // namespace tinygpt::kernel
