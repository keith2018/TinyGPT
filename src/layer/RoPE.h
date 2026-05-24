/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <optional>

#include "kernel/RopeOps.h"

namespace tinytorch::nn {

class RoPE {
 public:
  RoPE(int64_t headDim, int64_t contextLength, float thetaBase,
       std::optional<tinygpt::RopeScalingConfig> scaling = std::nullopt, Options options = {})
      : headDim_(headDim), contextLength_(contextLength), thetaBase_(thetaBase), scaling_(scaling), options_(options) {
    initCache();
  }

  explicit RoPE(Tensor sharedCache)
      : headDim_(sharedCache.size(1)),
        contextLength_(sharedCache.size(0)),
        thetaBase_(0.f),
        options_(sharedCache.options()),
        cache_(std::move(sharedCache)) {}

  RoPE(RoPE&&) = default;
  RoPE& operator=(RoPE&&) = default;

  RoPE(const RoPE&) = delete;
  RoPE& operator=(const RoPE&) = delete;

  Tensor apply(const Tensor& input, const Tensor& positions) const {
    return tinygpt::kernel::ropeApply(input, cache_, positions);
  }

  void applyInplace(Tensor& input, const Tensor& positions) const {
    tinygpt::kernel::ropeApplyInplace(input, cache_, positions);
  }

  const Tensor& cache() const { return cache_; }

  void to(Device device) {
    if (cache_.device() != device) {
      cache_ = cache_.to(device);
    }
  }

 private:
  void initCache() {
    const tinygpt::RopeScalingConfig* scalingPtr = scaling_.has_value() ? &scaling_.value() : nullptr;
    cache_ = tinygpt::kernel::ropeInit(headDim_, contextLength_, thetaBase_, scalingPtr, options_);
  }

  int64_t headDim_;
  int64_t contextLength_;
  float thetaBase_;
  std::optional<tinygpt::RopeScalingConfig> scaling_;
  Options options_;
  Tensor cache_;
};

}  // namespace tinytorch::nn
