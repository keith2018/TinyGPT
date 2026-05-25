/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "CUDAGraphRunner.h"

#include "Tensor/CachedAllocator.h"
#include "Utils/CUDAGraph.h"
#include "Utils/CUDAUtils.h"

namespace tt = tinytorch;

namespace tinygpt {

struct CUDAGraphRunner::Impl {
  tt::cuda::CUDAGraph graph;
  int graphPoolId = -1;
  PostReplayFn postReplay;
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

void CUDAGraphRunner::capture(ForwardContext& ctx, const tt::Tensor& inputIds, const SamplingStageFn& samplingStage,
                              tt::cuda::CUDAStream& stream, PostReplayFn postReplay) {
  ASSERT(!impl_->graph.valid() && "Graph already captured; call reset() first");
  ASSERT(device_.isCuda());
  ASSERT(samplingStage && "samplingStage must be provided");

  auto* allocator = tt::getCUDACachedAllocator(device_.index);
  ASSERT(allocator != nullptr);

  impl_->postReplay = std::move(postReplay);

  // warm-up
  allocator->beginAllocateToPool(impl_->graphPoolId);
  {
    tt::Tensor warmLogits;
    {
      ForwardContextGuard guard(&ctx);
      warmLogits = model_.forward(inputIds);
    }
    samplingStage(warmLogits, stream);
    stream.synchronize();
  }
  allocator->endAllocateToPool();

  // capture
  allocator->beginAllocateToPool(impl_->graphPoolId);
  impl_->graph.beginCapture(stream);
  {
    tt::Tensor logits;
    {
      ForwardContextGuard guard(&ctx);
      logits = model_.forward(inputIds);  // [batchSize_, vocabSize]
    }
    samplingStage(logits, stream);
  }
  impl_->graph.endCapture(stream);
  allocator->endAllocateToPool();
}

void CUDAGraphRunner::replay(tt::cuda::CUDAStream& stream) {
  ASSERT(impl_->graph.valid() && "No graph captured");
  impl_->graph.replay(stream);
  if (impl_->postReplay) {
    impl_->postReplay(stream);
  }
}

void CUDAGraphRunner::reset() { impl_->graph.reset(); }

}  // namespace tinygpt
