/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "GPTModel.h"
#include "Sampler.h"
#include "tokenizer/Tokenizer.h"

namespace tinygpt {

struct GPTConfig {
  // model
  GPTModelType modelType;
  GPTModelSize modelSize;
  std::string modelFilePath;  // safetensors format

  // tokenizers
  std::string tokenizerPath;  // huggingface format
  std::string tokenizerConfigPath;

  // inference
  tinytorch::Device device = tinytorch::DeviceType::CUDA;
  SamplerConfig samplerConfig;
  int64_t maxNewTokens = 128;
};

struct GPTOutput {
  std::vector<int32_t> tokenIds;
  std::string text;
};

class GPTEngine {
 public:
  explicit GPTEngine(const GPTConfig& config);

  bool prepare();
  GPTOutput generateSync(const std::string& text);

 private:
  GPTConfig config_;
  Sampler sampler_;
  tokenizer::Tokenizer tokenizer_;
  std::unique_ptr<GPTModel> model_;
};

}  // namespace tinygpt