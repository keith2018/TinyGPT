/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Modules.h"
#include "kernel/GemvOps.h"

namespace tinytorch::nn {

class LinearRef : public Module {
 public:
  explicit LinearRef(TensorPtr weight, TensorPtr bias = nullptr) : weight_(weight), bias_(bias) {}

  void updateRefs(TensorPtr weight, TensorPtr bias = nullptr) {
    weight_ = weight;
    bias_ = bias;
  }

 protected:
  std::vector<std::pair<std::string, TensorPtr>> namedParameters_() override {
    if (bias_ && bias_->defined()) {
      return {{"weight", weight_}, {"bias", bias_}};
    }
    return {{"weight", weight_}};
  }

 private:
  TensorPtr weight_;
  TensorPtr bias_;
};

class GemvLinear : public Linear {
 public:
  GemvLinear(int64_t inFeatures, int64_t outFeatures, bool bias = false, Options options = {})
      : Linear(inFeatures, outFeatures, bias, options) {}

  GemvLinear(GemvLinear &&other) noexcept : Linear(std::move(other)) {}

  GemvLinear &operator=(GemvLinear &&other) noexcept {
    if (this != &other) {
      Linear::operator=(std::move(other));
    }
    return *this;
  }

  GemvLinear(const GemvLinear &) = delete;
  GemvLinear &operator=(const GemvLinear &) = delete;

  Tensor forward(const Tensor &input) override {
    if (!useBias_ && input.dim() == 2 && input.size(0) == 1 && input.device().isCuda()) {
      auto result = tinygpt::kernel::gemvLinear(input, weight_);
      if (result.defined()) return result;
    }
    return Linear::forward(input);
  }
};

class LmHeadLinear : public Linear {
 public:
  LmHeadLinear(int64_t inFeatures, int64_t outFeatures, bool bias = false, Options options = {})
      : Linear(inFeatures, outFeatures, bias, options) {}

  LmHeadLinear(LmHeadLinear &&other) noexcept : Linear(std::move(other)) {}

  LmHeadLinear &operator=(LmHeadLinear &&other) noexcept {
    if (this != &other) {
      Linear::operator=(std::move(other));
    }
    return *this;
  }

  LmHeadLinear(const LmHeadLinear &) = delete;
  LmHeadLinear &operator=(const LmHeadLinear &) = delete;

  Tensor forward(const Tensor &input) override {
    // fast path
    if (input.dim() == 2 && input.size(0) == 1 && input.device().isCuda()) {
      return tinygpt::kernel::gemvLmHead(input, weight_);
    }
    return Linear::forward(input);
  }
};

class MergedLinear : public Linear {
 public:
  MergedLinear(int64_t inputSize, IntArrayView outputSizes, bool bias = false, Options options = {})
      : Linear(inputSize, arraySum(outputSizes), bias, options), outputSizes_(outputSizes.begin(), outputSizes.end()) {
    initRefs();
  }

  MergedLinear(MergedLinear &&other) noexcept : Linear(std::move(other)), outputSizes_(std::move(other.outputSizes_)) {
    initRefs();
  }

  MergedLinear &operator=(MergedLinear &&other) noexcept {
    if (this != &other) {
      Linear::operator=(std::move(other));
      outputSizes_ = std::move(other.outputSizes_);
      initRefs();
    }
    return *this;
  }

  MergedLinear(const MergedLinear &) = delete;
  MergedLinear &operator=(const MergedLinear &) = delete;

  LinearRef &moduleRefs(int64_t idx) { return moduleRefs_[idx]; }

  // raw access for parallel subclasses that need to tag per-segment weights.
  std::vector<Tensor> &weightSegments() { return weightRefs_; }
  std::vector<Tensor> &biasSegments() { return biasRefs_; }

  Tensor forward(const Tensor &input) override {
    if (!useBias_ && input.dim() == 2 && input.size(0) == 1 && input.device().isCuda()) {
      auto result = tinygpt::kernel::gemvLinear(input, weight_);
      if (result.defined()) {
        return result;
      }
    }
    return Linear::forward(input);
  }

 protected:
  std::vector<std::pair<std::string, TensorPtr>> namedParameters_() override { return {}; }

 private:
  void initRefs() {
    weightRefs_ = weight_.split(outputSizes_, 0);
    if (useBias_) {
      biasRefs_ = bias_.split(outputSizes_, 0);
    }

    moduleRefs_.clear();
    moduleRefs_.reserve(outputSizes_.size());
    for (size_t idx = 0; idx < outputSizes_.size(); idx++) {
      if (useBias_) {
        moduleRefs_.emplace_back(&weightRefs_[idx], &biasRefs_[idx]);
      } else {
        moduleRefs_.emplace_back(&weightRefs_[idx]);
      }
    }
  }

  static int64_t arraySum(IntArrayView arr) {
    int64_t ret = 0;
    for (int64_t i : arr) {
      ret += i;
    }
    return ret;
  }

  std::vector<int64_t> outputSizes_;

  std::vector<Tensor> weightRefs_;
  std::vector<Tensor> biasRefs_;
  std::vector<LinearRef> moduleRefs_;
};

}  // namespace tinytorch::nn
