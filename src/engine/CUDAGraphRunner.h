/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <functional>
#include <memory>

#include "engine/ForwardContext.h"
#include "model/GPTModel.h"

namespace tinytorch::cuda {
struct CUDAStream;
}

namespace tinygpt {

class CUDAGraphRunner {
 public:
  using SamplingStageFn = std::function<void(const tinytorch::Tensor& logits, tinytorch::cuda::CUDAStream& stream)>;
  using PostReplayFn = std::function<void(tinytorch::cuda::CUDAStream& stream)>;

  CUDAGraphRunner(GPTModel& model, tinytorch::Device device, int32_t batchSize = 1);
  ~CUDAGraphRunner();

  CUDAGraphRunner(const CUDAGraphRunner&) = delete;
  CUDAGraphRunner& operator=(const CUDAGraphRunner&) = delete;

  bool captured() const;
  void capture(ForwardContext& ctx, const tinytorch::Tensor& inputIds, const SamplingStageFn& samplingStage,
               tinytorch::cuda::CUDAStream& stream, PostReplayFn postReplay = {});

  void replay(tinytorch::cuda::CUDAStream& stream);
  void reset();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  GPTModel& model_;
  tinytorch::Device device_;
  int32_t batchSize_;
};

}  // namespace tinygpt
