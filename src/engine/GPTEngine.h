/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "huggingface/ModelLoader.h"

namespace tinygpt {

struct GPTConfig {
  std::string modelDir;  // huggingface repo
  tinytorch::Device device = tinytorch::DeviceType::CUDA;
  tinytorch::DType dtype = tinytorch::DType::BFloat16;

  SamplerConfig samplerConfig;
  int64_t maxNewTokens = 16;
};

struct GPTOutput {
  int64_t batch;
  std::vector<int32_t> tokenIds;
  std::vector<std::string> texts;
};

class GPTEngine {
 public:
  explicit GPTEngine(GPTConfig config);

  bool prepare();
  GPTOutput generateSync(tinytorch::ArrayView<std::string> texts);

 private:
  tinytorch::Tensor genNextToken(const tinytorch::Tensor& tokens, const tinytorch::Tensor& mask);

  tinytorch::TensorPair encodeTexts(tinytorch::ArrayView<std::string> texts) const;
  GPTOutput decodeTokens(const tinytorch::Tensor& tokens, int64_t offset) const;

  GPTConfig config_;
  Sampler sampler_;
  huggingface::GPTContext context_;
};

}  // namespace tinygpt