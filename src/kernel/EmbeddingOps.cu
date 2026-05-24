/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "Tensor/Tensor.h"
#include "Utils/CUDATypes.h"
#include "Utils/CUDAUtils.h"
#include "kernel/Dispatch.h"
#include "kernel/EmbeddingOps.h"

namespace tinygpt::kernel {

// =============================================================================
// Fast embedding lookup: copies rows from the embedding table to output.
//
// For decode (numTokens=1, hiddenSize=1024, BF16):
//   - One thread block with 256 threads
//   - Each thread copies 4 elements (uint4 = 8 BF16 = 16 bytes)
//   - Total: 256 threads × 4 elements = 1024 elements = 1 row
//   - Single vectorized pass, no index computation overhead
//
// For small numTokens (≤32), we use one block per token.
// =============================================================================

template <typename T>
__global__ void kEmbeddingLookup(const T* __restrict__ weight,          // [vocabSize, hiddenSize]
                                 const int64_t* __restrict__ inputIds,  // [numTokens]
                                 T* __restrict__ output,                // [numTokens, hiddenSize]
                                 int hiddenSize) {
  const int tokenIdx = blockIdx.x;
  const int64_t tokenId = inputIds[tokenIdx];
  const T* srcRow = weight + tokenId * hiddenSize;
  T* dstRow = output + tokenIdx * hiddenSize;

  // Vectorized copy: use uint4 (16 bytes = 8 BF16 or 4 FP32) for coalesced access
  if constexpr (sizeof(T) == 2) {
    // BF16/FP16: 8 elements per uint4
    const int elemsPerVec = 8;
    const int numVecs = hiddenSize / elemsPerVec;
    const uint4* src4 = reinterpret_cast<const uint4*>(srcRow);
    uint4* dst4 = reinterpret_cast<uint4*>(dstRow);
    for (int i = threadIdx.x; i < numVecs; i += blockDim.x) {
      dst4[i] = src4[i];
    }
    // Handle remainder (hiddenSize not divisible by 8)
    const int processed = numVecs * elemsPerVec;
    for (int i = processed + threadIdx.x; i < hiddenSize; i += blockDim.x) {
      dstRow[i] = srcRow[i];
    }
  } else {
    // FP32: 4 elements per float4
    const int elemsPerVec = 4;
    const int numVecs = hiddenSize / elemsPerVec;
    const float4* src4 = reinterpret_cast<const float4*>(srcRow);
    float4* dst4 = reinterpret_cast<float4*>(dstRow);
    for (int i = threadIdx.x; i < numVecs; i += blockDim.x) {
      dst4[i] = src4[i];
    }
    const int processed = numVecs * elemsPerVec;
    for (int i = processed + threadIdx.x; i < hiddenSize; i += blockDim.x) {
      dstRow[i] = srcRow[i];
    }
  }
}

template <typename CudaT>
static tinytorch::Tensor embeddingLookupImpl(const tinytorch::Tensor& inputIds, const tinytorch::Tensor& weight) {
  const auto numTokens = static_cast<int>(inputIds.size(0));
  const auto hiddenSize = static_cast<int>(weight.size(1));

  auto outOpts = weight.options().noGrad();
  tinytorch::Tensor output({static_cast<int64_t>(numTokens), static_cast<int64_t>(hiddenSize)}, outOpts);

  // Choose block size: enough threads to cover hiddenSize in one pass
  constexpr int kThreadsPerBlock = 256;
  const dim3 grid(numTokens);
  const dim3 block(kThreadsPerBlock);

  auto stream = tinytorch::cuda::getCurrentCUDAStream(weight.device().index).stream();
  kEmbeddingLookup<CudaT><<<grid, block, 0, stream>>>(weight.dataPtr<CudaT>(), inputIds.dataPtr<int64_t>(),
                                                      output.dataPtr<CudaT>(), hiddenSize);
  CUDA_KERNEL_CHECK();

  return output;
}

tinytorch::Tensor embeddingLookup(const tinytorch::Tensor& inputIds, const tinytorch::Tensor& weight) {
  // Guard conditions
  if (!weight.device().isCuda()) return {};
  if (inputIds.dtype() != tinytorch::DType::Int64) return {};
  if (weight.dim() != 2) return {};
  if (inputIds.dim() != 1) return {};
  // Only optimize for small token counts (decode / small prefill chunks)
  if (inputIds.size(0) > 32) return {};

  tinytorch::Tensor output;
  TINYGPT_DISPATCH_FLOAT_DTYPE(weight, { output = embeddingLookupImpl<CudaT>(inputIds, weight); });
  return output;
}

}  // namespace tinygpt::kernel
