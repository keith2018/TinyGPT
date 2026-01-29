/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Modules.h"

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
    for (long i : arr) {
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
