/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "Tensor/Tensor.h"
#include "Utils/CUDATypes.h"
#include "Utils/CUDAUtils.h"
#include "flash_attn/flash_api.cuh"
#include "kernel/AttentionOps.h"
#include "kernel/Dispatch.h"

namespace tinygpt::kernel {

template <typename CudaT>
static tinytorch::Tensor flashAttentionPagedVarLenImpl(
    const tinytorch::Tensor& query, const tinytorch::Tensor& kCachePool, const tinytorch::Tensor& vCachePool,
    const tinytorch::Tensor& cuSeqLensQ, const tinytorch::Tensor& cuSeqLensKV, const tinytorch::Tensor& blockTable,
    int maxSeqLenQ, int maxSeqLenKV, int pageSize, int maxBlocksPerSeq, bool isCausal, float* externalTmpO,
    float* externalTmpLse) {
  // query: [totalTokens, numHeadsQ, headDim]
  ASSERT(query.dim() == 3);
  const auto totalQ = static_cast<int>(query.size(0));
  const auto numHeadsQ = static_cast<int>(query.size(1));
  const auto headDim = static_cast<int>(query.size(2));
  const auto numHeadsKV = static_cast<int>(kCachePool.size(1));
  const auto batchSize = static_cast<int>(cuSeqLensQ.size(0) - 1);

  ASSERT(numHeadsQ % numHeadsKV == 0);  // GQA

  tinytorch::Tensor out({query.size(0), query.size(1), query.size(2)}, query.options().noGrad());

  // split-KV
  const int numPartitions = tfa::splitKvNumPartitions(maxSeqLenKV, kSplitKvPartitionSize);

  float* tmpOPtr = nullptr;
  float* tmpLsePtr = nullptr;
  tinytorch::Tensor tmpO;
  tinytorch::Tensor tmpLse;

  if (numPartitions > 1) {
    if (externalTmpO != nullptr && externalTmpLse != nullptr) {
      tmpOPtr = externalTmpO;
      tmpLsePtr = externalTmpLse;
    } else {
      const auto tmpOSize = static_cast<int64_t>(tfa::splitKvTmpOSize(totalQ, numHeadsQ, numPartitions, headDim));
      const auto tmpLseSize = static_cast<int64_t>(tfa::splitKvTmpLseSize(totalQ, numHeadsQ, numPartitions));
      const auto floatOpts = query.options().noGrad().dtype(tinytorch::DType::Float32);
      tmpO = tinytorch::Tensor({tmpOSize}, floatOpts);
      tmpLse = tinytorch::Tensor({tmpLseSize}, floatOpts);
      tmpOPtr = tmpO.dataPtr<float>();
      tmpLsePtr = tmpLse.dataPtr<float>();
    }
  }

  auto stream = tinytorch::cuda::getCurrentCUDAStream(query.device().index).stream();
  tfa::flashAttnPagedVarLen<CudaT>(query.dataPtr<CudaT>(), out.dataPtr<CudaT>(), kCachePool.dataPtr<CudaT>(),
                                   vCachePool.dataPtr<CudaT>(), cuSeqLensQ.dataPtr<int>(), cuSeqLensKV.dataPtr<int>(),
                                   blockTable.dataPtr<int>(), batchSize, maxSeqLenQ, maxSeqLenKV, numHeadsQ, numHeadsKV,
                                   headDim, pageSize, maxBlocksPerSeq, totalQ, isCausal, tmpOPtr, tmpLsePtr,
                                   kSplitKvPartitionSize, stream);
  CUDA_KERNEL_CHECK();
  return out;
}

tinytorch::Tensor flashAttentionPagedVarLen(const tinytorch::Tensor& query, const tinytorch::Tensor& kCachePool,
                                            const tinytorch::Tensor& vCachePool, const tinytorch::Tensor& cuSeqLensQ,
                                            const tinytorch::Tensor& cuSeqLensKV, const tinytorch::Tensor& blockTable,
                                            int maxSeqLenQ, int maxSeqLenKV, int pageSize, int maxBlocksPerSeq,
                                            bool isCausal, float* externalTmpO, float* externalTmpLse) {
  tinytorch::Tensor out;
  TINYGPT_DISPATCH_FLOAT_DTYPE(query, {
    out = flashAttentionPagedVarLenImpl<CudaT>(query, kCachePool, vCachePool, cuSeqLensQ, cuSeqLensKV, blockTable,
                                               maxSeqLenQ, maxSeqLenKV, pageSize, maxBlocksPerSeq, isCausal,
                                               externalTmpO, externalTmpLse);
  });
  return out;
}

template <typename scalar_t>
__global__ void kScatterKVToCache(const scalar_t* __restrict__ key, const scalar_t* __restrict__ value,
                                  scalar_t* __restrict__ keyCache, scalar_t* __restrict__ valueCache,
                                  const int* __restrict__ slotMapping, int numKvHeads, int headDim, int blockSize) {
  // grid: (numTokens), block: (threads per token row)
  const unsigned int tokenIdx = blockIdx.x;
  const int slot = slotMapping[tokenIdx];
  if (slot < 0) {
    return;  // padding token
  }
  const int blockId = slot / blockSize;
  const int blockOffset = slot % blockSize;

  for (int h = 0; h < numKvHeads; ++h) {
    const unsigned int srcOffset = tokenIdx * numKvHeads * headDim + h * headDim;
    const int dstOffset = blockId * numKvHeads * blockSize * headDim + h * blockSize * headDim + blockOffset * headDim;
    for (unsigned int d = threadIdx.x; d < headDim; d += blockDim.x) {
      keyCache[dstOffset + d] = key[srcOffset + d];
      valueCache[dstOffset + d] = value[srcOffset + d];
    }
  }
}

template <typename CudaT>
static void scatterKVToCacheImpl(const tinytorch::Tensor& key, const tinytorch::Tensor& value,
                                 tinytorch::Tensor& keyCache, tinytorch::Tensor& valueCache,
                                 const tinytorch::Tensor& slotMapping, int blockSize) {
  // key / value: [numTokens, numKvHeads, headDim]
  ASSERT(key.dim() == 3);
  const auto numTokens = static_cast<int>(key.size(0));
  const auto numKvHeads = static_cast<int>(key.size(1));
  const auto headDim = static_cast<int>(key.size(2));

  if (numTokens == 0) return;

  constexpr int kThreadsPerBlock = 128;
  const dim3 grid(numTokens);
  const dim3 block(kThreadsPerBlock);

  auto stream = tinytorch::cuda::getCurrentCUDAStream(key.device().index).stream();
  kScatterKVToCache<CudaT><<<grid, block, 0, stream>>>(key.dataPtr<CudaT>(), value.dataPtr<CudaT>(),
                                                       keyCache.dataPtr<CudaT>(), valueCache.dataPtr<CudaT>(),
                                                       slotMapping.dataPtr<int>(), numKvHeads, headDim, blockSize);
  CUDA_KERNEL_CHECK();
}

void scatterKVToCache(const tinytorch::Tensor& key, const tinytorch::Tensor& value, tinytorch::Tensor& keyCache,
                      tinytorch::Tensor& valueCache, const tinytorch::Tensor& slotMapping, int blockSize) {
  TINYGPT_DISPATCH_FLOAT_DTYPE(
      key, { scatterKVToCacheImpl<CudaT>(key, value, keyCache, valueCache, slotMapping, blockSize); });
}

// =============================================================================
// Fused RoPE(K) + ScatterKVToCache kernel
//
// Combines rotary position embedding on K with scattered write of K,V to the
// paged KV cache.  Eliminates the intermediate rotated-K tensor and one kernel
// launch per layer per decode step.
//
// For each token:
//   1. Read K from dense input, apply RoPE rotation, write to keyCache
//   2. Read V from dense input, write to valueCache (no rotation)
// =============================================================================

template <typename scalar_t>
__global__ void kRopeScatterKV(const scalar_t* __restrict__ key,       // [numTokens, numKvHeads, headDim]
                               const scalar_t* __restrict__ value,     // [numTokens, numKvHeads, headDim]
                               scalar_t* __restrict__ keyCache,        // paged pool
                               scalar_t* __restrict__ valueCache,      // paged pool
                               const int* __restrict__ slotMapping,    // [numTokens]
                               const float* __restrict__ ropeCache,    // [maxPos, headDim, 2]
                               const int64_t* __restrict__ positions,  // [numTokens]
                               int numKvHeads, int headDim, int blockSize) {
  const unsigned int tokenIdx = blockIdx.x;
  const int slot = slotMapping[tokenIdx];
  if (slot < 0) return;  // padding token

  const int blockId = slot / blockSize;
  const int blockOffset = slot % blockSize;
  const int halfDim = headDim >> 1;

  // Load position for RoPE
  const int64_t pos = positions[tokenIdx];
  const float* ropeRow = ropeCache + pos * headDim * 2;

  for (int h = 0; h < numKvHeads; ++h) {
    const unsigned int srcOffset = tokenIdx * numKvHeads * headDim + h * headDim;
    const int dstOffset = blockId * numKvHeads * blockSize * headDim + h * blockSize * headDim + blockOffset * headDim;

    for (unsigned int d = threadIdx.x; d < static_cast<unsigned>(halfDim); d += blockDim.x) {
      // --- K: apply RoPE rotation and scatter ---
      const float k1 = static_cast<float>(key[srcOffset + d]);
      const float k2 = static_cast<float>(key[srcOffset + halfDim + d]);
      const float cos_val = ropeRow[d * 2];
      const float sin_val = ropeRow[d * 2 + 1];
      keyCache[dstOffset + d] = static_cast<scalar_t>(k1 * cos_val - k2 * sin_val);
      keyCache[dstOffset + halfDim + d] = static_cast<scalar_t>(k2 * cos_val + k1 * sin_val);

      // --- V: pass-through scatter (no rotation) ---
      valueCache[dstOffset + d] = value[srcOffset + d];
      valueCache[dstOffset + halfDim + d] = value[srcOffset + halfDim + d];
    }
  }
}

template <typename CudaT>
static void ropeScatterKVToCacheImpl(const tinytorch::Tensor& key, const tinytorch::Tensor& value,
                                     tinytorch::Tensor& keyCache, tinytorch::Tensor& valueCache,
                                     const tinytorch::Tensor& slotMapping, int blockSize,
                                     const tinytorch::Tensor& ropeCache, const tinytorch::Tensor& positions) {
  ASSERT(key.dim() == 3);
  const auto numTokens = static_cast<int>(key.size(0));
  const auto numKvHeads = static_cast<int>(key.size(1));
  const auto headDim = static_cast<int>(key.size(2));

  if (numTokens == 0) return;

  constexpr int kThreadsPerBlock = 128;
  const dim3 grid(numTokens);
  const dim3 block(kThreadsPerBlock);

  auto stream = tinytorch::cuda::getCurrentCUDAStream(key.device().index).stream();
  kRopeScatterKV<CudaT><<<grid, block, 0, stream>>>(key.dataPtr<CudaT>(), value.dataPtr<CudaT>(),
                                                    keyCache.dataPtr<CudaT>(), valueCache.dataPtr<CudaT>(),
                                                    slotMapping.dataPtr<int>(), ropeCache.dataPtr<float>(),
                                                    positions.dataPtr<int64_t>(), numKvHeads, headDim, blockSize);
  CUDA_KERNEL_CHECK();
}

void ropeScatterKVToCache(const tinytorch::Tensor& key, const tinytorch::Tensor& value, tinytorch::Tensor& keyCache,
                          tinytorch::Tensor& valueCache, const tinytorch::Tensor& slotMapping, int blockSize,
                          const tinytorch::Tensor& ropeCache, const tinytorch::Tensor& positions) {
  TINYGPT_DISPATCH_FLOAT_DTYPE(key, {
    ropeScatterKVToCacheImpl<CudaT>(key, value, keyCache, valueCache, slotMapping, blockSize, ropeCache, positions);
  });
}

// =============================================================================
// Fused K-Norm + RoPE(K) + ScatterKVToCache kernel
//
// Combines RMSNorm on K + rotary embedding + scattered write to paged cache
// in a single kernel.  This eliminates the separate kNormSmall kernel for K.
//
// RMSNorm: out[i] = (input[i] / sqrt(mean(input^2) + eps)) * weight[i]
//
// The kernel processes one head per thread block warp-group to allow the
// block-level reduction needed for RMSNorm (computing sum of squares).
// =============================================================================

template <typename scalar_t>
__global__ void kNormRopeScatterKV(const scalar_t* __restrict__ key,         // [numTokens, numKvHeads, headDim]
                                   const scalar_t* __restrict__ value,       // [numTokens, numKvHeads, headDim]
                                   scalar_t* __restrict__ keyCache,          // paged pool
                                   scalar_t* __restrict__ valueCache,        // paged pool
                                   const int* __restrict__ slotMapping,      // [numTokens]
                                   const float* __restrict__ ropeCache,      // [maxPos, headDim, 2]
                                   const int64_t* __restrict__ positions,    // [numTokens]
                                   const scalar_t* __restrict__ normWeight,  // [headDim]
                                   float eps, int numKvHeads, int headDim, int blockSize) {
  // Grid: (numTokens, numKvHeads), Block: (128 threads)
  // Each block handles one (token, head) pair for K, and the same for V pass-through.
  const int tokenIdx = blockIdx.x;
  const int headIdx = blockIdx.y;
  const int tid = threadIdx.x;
  const int halfDim = headDim >> 1;

  const int slot = slotMapping[tokenIdx];
  if (slot < 0) return;

  const int blockId = slot / blockSize;
  const int blockOffset = slot % blockSize;

  const unsigned int srcOffset = tokenIdx * numKvHeads * headDim + headIdx * headDim;
  const int dstOffset =
      blockId * numKvHeads * blockSize * headDim + headIdx * blockSize * headDim + blockOffset * headDim;

  // --- Phase 1: RMSNorm on K head ---
  // Compute sum of squares for this head
  float sumSq = 0.f;
  for (int d = tid; d < headDim; d += blockDim.x) {
    float val = static_cast<float>(key[srcOffset + d]);
    sumSq += val * val;
  }

  // Block-level reduction for sum of squares
  // Use warp shuffle + shared memory reduction
  __shared__ float sPartial[32];  // one per warp (up to 4 warps for 128 threads)
  const int warpId = tid / 32;
  const int laneId = tid % 32;

  // Warp-level reduction
  for (int offset = 16; offset > 0; offset >>= 1) {
    sumSq += __shfl_down_sync(0xffffffff, sumSq, offset);
  }
  if (laneId == 0) sPartial[warpId] = sumSq;
  __syncthreads();

  // First warp reduces across warps
  if (warpId == 0) {
    float val = (laneId < (blockDim.x / 32)) ? sPartial[laneId] : 0.f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (laneId == 0) sPartial[0] = val;
  }
  __syncthreads();

  const float invRms = rsqrtf(sPartial[0] / static_cast<float>(headDim) + eps);

  // --- Phase 2: Normalize K + Apply RoPE + Scatter to cache ---
  const int64_t pos = positions[tokenIdx];
  const float* ropeRow = ropeCache + pos * headDim * 2;

  for (int d = tid; d < halfDim; d += blockDim.x) {
    // RMSNorm: normalize and apply weight
    float k1_raw = static_cast<float>(key[srcOffset + d]);
    float k2_raw = static_cast<float>(key[srcOffset + halfDim + d]);
    float w1 = static_cast<float>(normWeight[d]);
    float w2 = static_cast<float>(normWeight[halfDim + d]);
    float k1 = k1_raw * invRms * w1;
    float k2 = k2_raw * invRms * w2;

    // RoPE rotation
    float cos_val = ropeRow[d * 2];
    float sin_val = ropeRow[d * 2 + 1];
    keyCache[dstOffset + d] = static_cast<scalar_t>(k1 * cos_val - k2 * sin_val);
    keyCache[dstOffset + halfDim + d] = static_cast<scalar_t>(k2 * cos_val + k1 * sin_val);

    // V pass-through scatter
    valueCache[dstOffset + d] = value[srcOffset + d];
    valueCache[dstOffset + halfDim + d] = value[srcOffset + halfDim + d];
  }
}

template <typename CudaT>
static void normRopeScatterKVToCacheImpl(const tinytorch::Tensor& key, const tinytorch::Tensor& value,
                                         tinytorch::Tensor& keyCache, tinytorch::Tensor& valueCache,
                                         const tinytorch::Tensor& slotMapping, int blockSize,
                                         const tinytorch::Tensor& ropeCache, const tinytorch::Tensor& positions,
                                         const tinytorch::Tensor& normWeight, float eps) {
  ASSERT(key.dim() == 3);
  const auto numTokens = static_cast<int>(key.size(0));
  const auto numKvHeads = static_cast<int>(key.size(1));
  const auto headDim = static_cast<int>(key.size(2));

  if (numTokens == 0) return;

  constexpr int kThreadsPerBlock = 128;
  const dim3 grid(numTokens, numKvHeads);
  const dim3 block(kThreadsPerBlock);

  auto stream = tinytorch::cuda::getCurrentCUDAStream(key.device().index).stream();
  kNormRopeScatterKV<CudaT><<<grid, block, 0, stream>>>(
      key.dataPtr<CudaT>(), value.dataPtr<CudaT>(), keyCache.dataPtr<CudaT>(), valueCache.dataPtr<CudaT>(),
      slotMapping.dataPtr<int>(), ropeCache.dataPtr<float>(), positions.dataPtr<int64_t>(), normWeight.dataPtr<CudaT>(),
      eps, numKvHeads, headDim, blockSize);
  CUDA_KERNEL_CHECK();
}

void normRopeScatterKVToCache(const tinytorch::Tensor& key, const tinytorch::Tensor& value, tinytorch::Tensor& keyCache,
                              tinytorch::Tensor& valueCache, const tinytorch::Tensor& slotMapping, int blockSize,
                              const tinytorch::Tensor& ropeCache, const tinytorch::Tensor& positions,
                              const tinytorch::Tensor& normWeight, float eps) {
  TINYGPT_DISPATCH_FLOAT_DTYPE(key, {
    normRopeScatterKVToCacheImpl<CudaT>(key, value, keyCache, valueCache, slotMapping, blockSize, ropeCache, positions,
                                        normWeight, eps);
  });
}

}  // namespace tinygpt::kernel
