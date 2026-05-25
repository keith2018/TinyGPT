/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "Sampler.h"

#include "Functions.h"
#include "Tensor/Storage.h"
#include "Utils/CUDAUtils.h"
#include "Utils/RandomGenerator.h"

namespace tt = tinytorch;

namespace tinygpt {

struct BatchSampler::Impl {
  tt::Device device;
  int32_t maxBatchTokens;

  // pinned host buffer
  tt::Tensor tokensHost;

  tt::cuda::CUDAEvent ready;

  Impl(tt::Device dev, int32_t maxTokens)
      : device(dev),
        maxBatchTokens(maxTokens),
        tokensHost({maxTokens}, tt::Options(tt::Device::cpu(), tt::DType::Int64).noGrad().pinnedMemory(true)),
        ready(tt::cuda::createCUDAEvent(dev.index)) {}
};

BatchSampler::BatchSampler(tt::Device device, int32_t maxBatchTokens)
    : impl_(std::make_unique<Impl>(device, maxBatchTokens)) {
  ASSERT(device.isCuda());
}

BatchSampler::~BatchSampler() = default;

bool BatchSampler::allGreedy(const std::vector<Sampler*>& samplers) {
  return std::all_of(samplers.begin(), samplers.end(), [](const Sampler* s) { return s && s->isGreedy(); });
}

void BatchSampler::sampleEager(const tt::Tensor& logits, const std::vector<Sampler*>& samplers, bool allGreedy,
                               tt::cuda::CUDAStream& stream) {
  const auto batch = static_cast<int32_t>(samplers.size());
  ASSERT(batch > 0 && batch <= impl_->maxBatchTokens);

  tt::Tensor sampledDev;
  if (allGreedy) {
    sampledDev = tt::function::argmax(logits, -1, true);  // [batch, 1] Int64
  } else {
    std::vector<kernel::SamplingParams> perRow(static_cast<size_t>(batch));
    for (int32_t i = 0; i < batch; i++) {
      perRow[static_cast<size_t>(i)] = samplers[static_cast<size_t>(i)]->params();
    }
    const auto globalSeed = tt::RandomGeneratorCUDA::getSeed();
    const auto globalSeq = tt::RandomGeneratorCUDA::nextSequence();
    sampledDev = kernel::fusedSample(logits, perRow.data(), batch, globalSeed, globalSeq);
  }

  tt::Storage::copyOnDevice(impl_->tokensHost.dataPtr<>(), tt::Device::cpu(), sampledDev.dataPtr<>(), impl_->device,
                            static_cast<int64_t>(batch) * static_cast<int64_t>(sizeof(int64_t)), &stream);
  impl_->ready.record(stream);
}

void BatchSampler::recordTokensReady(tt::cuda::CUDAStream& stream) { impl_->ready.record(stream); }

const int64_t* BatchSampler::consumeTokens() {
  CUDA_CHECK(cudaEventSynchronize(impl_->ready.event()));
  return impl_->tokensHost.dataPtr<int64_t>();
}

GraphSampleStage BatchSampler::makeGreedyStage(int32_t batchSize) {
  // capture by raw pointer — Impl outlives the graph runner
  Impl* impl = impl_.get();
  return [impl, batchSize](const tt::Tensor& logits, tt::cuda::CUDAStream& stream) {
    tt::Tensor sampledDev = tt::function::argmax(logits, -1, true);  // [batchSize, 1] Int64
    tt::Storage::copyOnDevice(impl->tokensHost.dataPtr<>(), tt::Device::cpu(), sampledDev.dataPtr<>(), impl->device,
                              static_cast<int64_t>(batchSize) * static_cast<int64_t>(sizeof(int64_t)), &stream);
  };
}

}  // namespace tinygpt
