/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include <algorithm>
#include <cmath>

#include "Tensor/Tensor.h"
#include "Utils/CUDATypes.h"
#include "Utils/CUDAUtils.h"
#include "kernel/Dispatch.h"
#include "kernel/RopeOps.h"

namespace tinygpt::kernel {

__global__ void kRopeComputeInvFreq(float* invFreqPtr, int64_t halfDim, float thetaBase) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= halfDim) {
    return;
  }
  invFreqPtr[idx] = 1.f / powf(thetaBase, static_cast<float>(idx << 1) / static_cast<float>(halfDim << 1));
}

__global__ void kRopeScaling(float* invFreqPtr, int64_t halfDim, float originalContextLength, float lowFreqFactor,
                             float highFreqFactor, float scalingFactor) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= halfDim) {
    return;
  }

  const float invFreq = invFreqPtr[idx];
  const float waveLen = 2.f * static_cast<float>(M_PI) / invFreq;
  const float lowWaveLen = originalContextLength / lowFreqFactor;
  const float highWaveLen = originalContextLength / highFreqFactor;

  if (waveLen > lowWaveLen) {
    // long wavelength: fully scaled down
    invFreqPtr[idx] = invFreq / scalingFactor;
  } else if (waveLen < highWaveLen) {
    // short wavelength: unchanged
  } else {
    // in-between: smooth interpolation
    const float smoothFactor = (originalContextLength / waveLen - lowFreqFactor) / (highFreqFactor - lowFreqFactor);
    const float scaled = invFreq / scalingFactor;
    invFreqPtr[idx] = (1.f - smoothFactor) * scaled + smoothFactor * invFreq;
  }
}

__global__ void kRopePrecomputeCosSin(const float* invFreqPtr, float* ropePtr, int64_t contextLength, int64_t headDim) {
  // grid: (contextLength), block: (threads over halfDim)
  const int64_t pos = blockIdx.x;
  if (pos >= contextLength) {
    return;
  }

  const auto halfDim = headDim >> 1;
  for (auto i = threadIdx.x; i < halfDim; i += blockDim.x) {
    const float angle = static_cast<float>(pos) * invFreqPtr[i];
    const float cosVal = ::cosf(angle);
    const float sinVal = ::sinf(angle);

    const int64_t offset1 = (pos * headDim + i) * 2;
    const int64_t offset2 = (pos * headDim + halfDim + i) * 2;

    ropePtr[offset1] = cosVal;
    ropePtr[offset1 + 1] = sinVal;
    ropePtr[offset2] = cosVal;
    ropePtr[offset2 + 1] = sinVal;
  }
}

tinytorch::Tensor ropeInit(int64_t headDim, int64_t contextLength, float thetaBase, const RopeScalingConfig* scaling,
                           tinytorch::Options options) {
  ASSERT(options.device_.type == tinytorch::DeviceType::CUDA);
  ASSERT(headDim % 2 == 0);

  options.dtype_ = tinytorch::DType::Float32;
  options.requiresGrad_ = false;

  const int64_t halfDim = headDim >> 1;

  // inverse frequency vector: [halfDim]
  tinytorch::Tensor invFreq({halfDim}, options);
  auto* invFreqPtr = invFreq.dataPtr<float>();

  const auto params = tinytorch::cuda::getKernelLaunchParams(options.device_.index, halfDim);
  CUDA_LAUNCH_KERNEL(kRopeComputeInvFreq, params, invFreqPtr, halfDim, thetaBase);

  // frequency rescaling
  if (scaling != nullptr) {
    CUDA_LAUNCH_KERNEL(kRopeScaling, params, invFreqPtr, halfDim, static_cast<float>(scaling->originalContextLength),
                       scaling->lowFreqFactor, scaling->highFreqFactor, scaling->factor);
  }

  // precompute cos/sin table: [contextLength, headDim, 2]
  tinytorch::Tensor rope({contextLength, headDim, 2}, options);
  auto* ropePtr = rope.dataPtr<float>();

  const auto blockSize = tinytorch::cuda::getKernelBlockSize(options.device_.index);
  auto stream = tinytorch::cuda::getCurrentCUDAStream(options.device_.index).stream();
  kRopePrecomputeCosSin<<<contextLength, blockSize, 0, stream>>>(invFreqPtr, ropePtr, contextLength, headDim);
  CUDA_KERNEL_CHECK();

  return rope;
}

template <typename T>
__global__ void kRopeApplyImpl(const T* __restrict__ input, const float* __restrict__ ropeCache,
                               const int64_t* __restrict__ positions, T* __restrict__ output, int numHeads,
                               int headDim) {
  const unsigned int tokenIdx = blockIdx.x;
  const unsigned int headIdx = blockIdx.y;
  const unsigned int halfDim = headDim >> 1;

  const int64_t pos = positions[tokenIdx];
  const float* ropeRow = ropeCache + pos * headDim * 2;

  const unsigned int base = tokenIdx * numHeads * headDim + headIdx * headDim;
  const T* xPtr = input + base;
  T* yPtr = output + base;

  for (unsigned int i = threadIdx.x; i < halfDim; i += blockDim.x) {
    const auto x1 = static_cast<float>(xPtr[i]);
    const auto x2 = static_cast<float>(xPtr[halfDim + i]);

    const unsigned int idx = i * 2;
    const float c = ropeRow[idx];      // cos
    const float s = ropeRow[idx + 1];  // sin

    yPtr[i] = static_cast<T>(x1 * c - x2 * s);
    yPtr[halfDim + i] = static_cast<T>(x2 * c + x1 * s);
  }
}

template <typename CudaT>
static tinytorch::Tensor ropeApplyImpl(const tinytorch::Tensor& input, const tinytorch::Tensor& ropeCache,
                                       const tinytorch::Tensor& positions) {
  // input: [totalTokens, numHeads, headDim]
  ASSERT(input.dim() == 3);
  ASSERT(positions.dim() == 1);

  const auto totalTokens = static_cast<int>(input.size(0));
  const auto numHeads = static_cast<int>(input.size(1));
  const auto headDim = static_cast<int>(input.size(2));

  tinytorch::Tensor out(input.shape(), input.options().noGrad());

  constexpr int kThreadsPerBlock = 128;
  const dim3 grid(totalTokens, numHeads);
  const dim3 block(std::min(kThreadsPerBlock, headDim / 2));

  auto stream = tinytorch::cuda::getCurrentCUDAStream(input.device().index).stream();
  kRopeApplyImpl<CudaT><<<grid, block, 0, stream>>>(input.dataPtr<CudaT>(), ropeCache.dataPtr<float>(),
                                                    positions.dataPtr<int64_t>(), out.dataPtr<CudaT>(), numHeads,
                                                    headDim);
  CUDA_KERNEL_CHECK();
  return out;
}

tinytorch::Tensor ropeApply(const tinytorch::Tensor& input, const tinytorch::Tensor& ropeCache,
                            const tinytorch::Tensor& positions) {
  tinytorch::Tensor out;
  TINYGPT_DISPATCH_FLOAT_DTYPE(input, { out = ropeApplyImpl<CudaT>(input, ropeCache, positions); });
  return out;
}

template <typename CudaT>
static void ropeApplyInplaceImpl(tinytorch::Tensor& input, const tinytorch::Tensor& ropeCache,
                                 const tinytorch::Tensor& positions) {
  ASSERT(input.dim() == 3);
  ASSERT(positions.dim() == 1);

  const auto totalTokens = static_cast<int>(input.size(0));
  const auto numHeads = static_cast<int>(input.size(1));
  const auto headDim = static_cast<int>(input.size(2));

  constexpr int kThreadsPerBlock = 128;
  const dim3 grid(totalTokens, numHeads);
  const dim3 block(std::min(kThreadsPerBlock, headDim / 2));

  auto stream = tinytorch::cuda::getCurrentCUDAStream(input.device().index).stream();
  kRopeApplyImpl<CudaT><<<grid, block, 0, stream>>>(input.dataPtr<CudaT>(), ropeCache.dataPtr<float>(),
                                                    positions.dataPtr<int64_t>(), input.dataPtr<CudaT>(), numHeads,
                                                    headDim);
  CUDA_KERNEL_CHECK();
}

void ropeApplyInplace(tinytorch::Tensor& input, const tinytorch::Tensor& ropeCache,
                      const tinytorch::Tensor& positions) {
  TINYGPT_DISPATCH_FLOAT_DTYPE(input, { ropeApplyInplaceImpl<CudaT>(input, ropeCache, positions); });
}

}  // namespace tinygpt::kernel
