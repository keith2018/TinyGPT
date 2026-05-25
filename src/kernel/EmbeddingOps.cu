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

template <typename T>
__global__ void kEmbeddingLookup(const T* __restrict__ weight, const int64_t* __restrict__ inputIds,
                                 T* __restrict__ output, int hiddenSize) {
  const unsigned int tokenIdx = blockIdx.x;
  const int64_t tokenId = inputIds[tokenIdx];
  const T* srcRow = weight + tokenId * hiddenSize;
  T* dstRow = output + tokenIdx * hiddenSize;

  // vectorized copy
  if constexpr (sizeof(T) == 2) {
    // BF16/FP16: 8 elements per uint4
    constexpr int elemsPerVec = 8;
    const int numVecs = hiddenSize / elemsPerVec;
    const auto* src4 = reinterpret_cast<const uint4*>(srcRow);
    auto* dst4 = reinterpret_cast<uint4*>(dstRow);
    for (unsigned int i = threadIdx.x; i < numVecs; i += blockDim.x) {
      dst4[i] = src4[i];
    }
    const int processed = numVecs * elemsPerVec;
    for (unsigned int i = processed + threadIdx.x; i < hiddenSize; i += blockDim.x) {
      dstRow[i] = srcRow[i];
    }
  } else {
    // FP32: 4 elements per float4
    constexpr int elemsPerVec = 4;
    const int numVecs = hiddenSize / elemsPerVec;
    const auto* src4 = reinterpret_cast<const float4*>(srcRow);
    auto* dst4 = reinterpret_cast<float4*>(dstRow);
    for (unsigned int i = threadIdx.x; i < numVecs; i += blockDim.x) {
      dst4[i] = src4[i];
    }
    const int processed = numVecs * elemsPerVec;
    for (unsigned int i = processed + threadIdx.x; i < hiddenSize; i += blockDim.x) {
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
  ASSERT(weight.device().isCuda());
  ASSERT(inputIds.dtype() == tinytorch::DType::Int64);
  ASSERT(weight.dim() == 2);
  ASSERT(inputIds.dim() == 1);

  // only optimize for small token counts (decode / small prefill chunks)
  if (inputIds.size(0) > 32) {
    return {};
  }

  tinytorch::Tensor output;
  TINYGPT_DISPATCH_FLOAT_DTYPE(weight, { output = embeddingLookupImpl<CudaT>(inputIds, weight); });
  return output;
}

}  // namespace tinygpt::kernel
