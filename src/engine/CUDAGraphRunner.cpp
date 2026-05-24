/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "CUDAGraphRunner.h"

#include "Functions.h"
#include "Tensor/CachedAllocator.h"
#include "Utils/CUDAGraph.h"
#include "Utils/CUDAUtils.h"
#include "Utils/Logger.h"
#include "Utils/RandomGenerator.h"
#include "kernel/SamplerOps.h"

namespace tt = tinytorch;

namespace tinygpt {

struct CUDAGraphRunner::Impl {
  tt::cuda::CUDAGraph graph;
  int graphPoolId = -1;

  // non-greedy sampling: device-side RNG sequence counter updated via H2D before each replay
  tt::Tensor devGlobalSeqBuf;
  tt::Tensor hostGlobalSeqBuf;
  bool useSampling = false;

  // event recorded AFTER graph replay (not inside graph) for D2H sync
  tt::cuda::CUDAEvent* sampledEvent = nullptr;
};

CUDAGraphRunner::CUDAGraphRunner(GPTModel& model, tt::Device device, int32_t batchSize)
    : impl_(std::make_unique<Impl>()), model_(model), device_(device), batchSize_(batchSize) {
  impl_->graphPoolId = tt::CachedAllocator::newPoolId();
}

CUDAGraphRunner::~CUDAGraphRunner() {
  reset();
  auto* allocator = tt::getCUDACachedAllocator(device_.index);
  if (allocator && impl_->graphPoolId >= 0) {
    allocator->freePool(impl_->graphPoolId);
  }
}

bool CUDAGraphRunner::captured() const { return impl_->graph.valid(); }

void CUDAGraphRunner::capture(ForwardContext& ctx, const tt::Tensor& inputIds, tt::Tensor& sampledHostBuf,
                              void* sampledEventPtr, void* streamPtr, const kernel::SamplingParams* samplingParams) {
  ASSERT(!impl_->graph.valid() && "Graph already captured; call reset() first");
  ASSERT(device_.isCuda());

  auto& stream = *static_cast<tt::cuda::CUDAStream*>(streamPtr);
  auto& sampledEvent = *static_cast<tt::cuda::CUDAEvent*>(sampledEventPtr);

  auto* allocator = tt::getCUDACachedAllocator(device_.index);
  ASSERT(allocator != nullptr);

  impl_->useSampling = (samplingParams != nullptr);
  impl_->sampledEvent = &sampledEvent;

  // allocate RNG sequence buffers for non-greedy mode
  if (impl_->useSampling) {
    if (!impl_->devGlobalSeqBuf.defined()) {
      auto devOpts = tt::Options(device_, tt::DType::Int64).noGrad();
      auto pinnedOpts = tt::Options(tt::Device::cpu(), tt::DType::Int64).noGrad().pinnedMemory(true);
      impl_->devGlobalSeqBuf = tt::Tensor({1}, devOpts);
      impl_->hostGlobalSeqBuf = tt::Tensor({1}, pinnedOpts);
    }
    auto seq = tt::RandomGeneratorCUDA::nextSequence();
    impl_->hostGlobalSeqBuf.dataPtr<int64_t>()[0] = static_cast<int64_t>(seq);
    tt::Storage::copyOnDevice(impl_->devGlobalSeqBuf.dataPtr<>(), device_, impl_->hostGlobalSeqBuf.dataPtr<>(),
                              tt::Device::cpu(), static_cast<int64_t>(sizeof(uint64_t)), &stream);
    stream.synchronize();
  }

  // pool warm-up: run once without capture to populate graph pool
  allocator->beginAllocateToPool(impl_->graphPoolId);
  {
    tt::Tensor warmLogits;
    {
      ForwardContextGuard guard(&ctx);
      warmLogits = model_.forward(inputIds);
    }
    if (impl_->useSampling) {
      auto outOpts = tt::Options(device_, tt::DType::Int64).noGrad();
      tt::Tensor warmOutput({batchSize_, 1}, outOpts);
      kernel::fusedSampleGraphable(warmLogits, warmOutput, *samplingParams, tt::RandomGeneratorCUDA::getSeed(),
                                   reinterpret_cast<const uint64_t*>(impl_->devGlobalSeqBuf.dataPtr<int64_t>()));
      tt::Storage::copyOnDevice(sampledHostBuf.dataPtr<>(), tt::Device::cpu(), warmOutput.dataPtr<>(), device_,
                                static_cast<int64_t>(batchSize_) * static_cast<int64_t>(sizeof(int64_t)), &stream);
    } else {
      tt::Tensor warmSampled = tt::function::argmax(warmLogits, -1, true);
      tt::Storage::copyOnDevice(sampledHostBuf.dataPtr<>(), tt::Device::cpu(), warmSampled.dataPtr<>(), device_,
                                static_cast<int64_t>(batchSize_) * static_cast<int64_t>(sizeof(int64_t)), &stream);
    }
    stream.synchronize();
  }
  allocator->endAllocateToPool();

  // graph capture
  allocator->beginAllocateToPool(impl_->graphPoolId);
  impl_->graph.beginCapture(stream);

  tt::Tensor logits;
  {
    ForwardContextGuard guard(&ctx);
    logits = model_.forward(inputIds);  // [batchSize_, vocabSize]
  }

  tt::Tensor sampledDev;
  if (impl_->useSampling) {
    auto outOpts = tt::Options(device_, tt::DType::Int64).noGrad();
    sampledDev = tt::Tensor({batchSize_, 1}, outOpts);
    kernel::fusedSampleGraphable(logits, sampledDev, *samplingParams, tt::RandomGeneratorCUDA::getSeed(),
                                 reinterpret_cast<const uint64_t*>(impl_->devGlobalSeqBuf.dataPtr<int64_t>()));
  } else {
    sampledDev = tt::function::argmax(logits, -1, true);  // [batchSize_, 1]
  }

  tt::Storage::copyOnDevice(sampledHostBuf.dataPtr<>(), tt::Device::cpu(), sampledDev.dataPtr<>(), device_,
                            static_cast<int64_t>(batchSize_) * static_cast<int64_t>(sizeof(int64_t)), &stream);

  // event record is NOT inside the graph (cannot be externally synchronized)

  impl_->graph.endCapture(stream);
  allocator->endAllocateToPool();
}

void CUDAGraphRunner::replay(void* streamPtr) {
  ASSERT(impl_->graph.valid() && "No graph captured");
  auto& stream = *static_cast<tt::cuda::CUDAStream*>(streamPtr);

  // update RNG sequence counter before replay for non-greedy sampling
  if (impl_->useSampling) {
    auto seq = tt::RandomGeneratorCUDA::nextSequence();
    impl_->hostGlobalSeqBuf.dataPtr<int64_t>()[0] = static_cast<int64_t>(seq);
    tt::Storage::copyOnDevice(impl_->devGlobalSeqBuf.dataPtr<>(), device_, impl_->hostGlobalSeqBuf.dataPtr<>(),
                              tt::Device::cpu(), static_cast<int64_t>(sizeof(uint64_t)), &stream);
  }

  impl_->graph.replay(stream);

  // record event after graph launch for D2H sync (must be outside graph)
  if (impl_->sampledEvent) {
    impl_->sampledEvent->record(stream);
  }
}

void CUDAGraphRunner::reset() { impl_->graph.reset(); }

}  // namespace tinygpt
