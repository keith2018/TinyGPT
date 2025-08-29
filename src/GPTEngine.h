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
  std::vector<int32_t> tokenIds;
  std::string text;
};

class GPTEngine {
 public:
  explicit GPTEngine(GPTConfig config);

  bool prepare();
  GPTOutput generateSync(const std::string& text);

 private:
  tinytorch::Tensor genNextToken(const tinytorch::Tensor& tokens);

  tinytorch::Tensor encodeTexts(const std::string& text) const;
  GPTOutput decodeTokens(const tinytorch::Tensor& tokens, int64_t offset = 0) const;

  GPTConfig config_;
  Sampler sampler_;
  huggingface::GPTContext context_;
};

}  // namespace tinygpt