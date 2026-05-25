/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Tensor/Tensor.h"

namespace tinygpt {

struct RopeScalingConfig {
  float factor = 1.0f;
  float highFreqFactor = 1.0f;
  float lowFreqFactor = 1.0f;
  int64_t originalContextLength = 0;

  RopeScalingConfig() = default;
  RopeScalingConfig(float factor, float highFreqFactor, float lowFreqFactor, int64_t originalContextLength)
      : factor(factor),
        highFreqFactor(highFreqFactor),
        lowFreqFactor(lowFreqFactor),
        originalContextLength(originalContextLength) {}
};

}  // namespace tinygpt

namespace tinygpt::kernel {

// output    : [contextLength, headDim, 2] (FP32)
tinytorch::Tensor ropeInit(int64_t headDim, int64_t contextLength, float thetaBase, const RopeScalingConfig* scaling,
                           tinytorch::Options options);

// input     : [totalTokens, numHeads, headDim]
// ropeCache : [contextLength, headDim, 2]  (FP32)
// positions : [totalTokens] int64
// output    : [totalTokens, numHeads, headDim]
tinytorch::Tensor ropeApply(const tinytorch::Tensor& input, const tinytorch::Tensor& ropeCache,
                            const tinytorch::Tensor& positions);

void ropeApplyInplace(tinytorch::Tensor& input, const tinytorch::Tensor& ropeCache, const tinytorch::Tensor& positions);

}  // namespace tinygpt::kernel
