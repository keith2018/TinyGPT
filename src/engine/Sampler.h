/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "kernel/SamplerOps.h"

namespace tinytorch::cuda {
struct CUDAStream;
}

namespace tinygpt {

using SamplerConfig = kernel::SamplingParams;
using GraphSampleStage = std::function<void(const tinytorch::Tensor& logits, tinytorch::cuda::CUDAStream& stream)>;

class Sampler {
 public:
  explicit Sampler(SamplerConfig params) : params_(params) { params_.normalize(); }

  const SamplerConfig& params() const { return params_; }
  bool isGreedy() const { return kernel::isGreedy(params_); }

 private:
  SamplerConfig params_;
};

class BatchSampler {
 public:
  BatchSampler(tinytorch::Device device, int32_t maxBatchTokens);
  ~BatchSampler();

  BatchSampler(const BatchSampler&) = delete;
  BatchSampler& operator=(const BatchSampler&) = delete;

  static bool allGreedy(const std::vector<Sampler*>& samplers);

  void sampleEager(const tinytorch::Tensor& logits, const std::vector<Sampler*>& samplers, bool allGreedy,
                   tinytorch::cuda::CUDAStream& stream);

  void recordTokensReady(tinytorch::cuda::CUDAStream& stream);

  const int64_t* consumeTokens();

  GraphSampleStage makeGreedyStage(int32_t batchSize);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tinygpt
