/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <functional>
#include <memory>

#include "huggingface/ModelLoader.h"

namespace tinygpt {

class AsyncTokenPipeline;

using GenerateCallback = std::function<bool(const std::string& tokenText)>;

enum class FinishReason {
  Stop,
  Length,
};

struct GPTConfig {
  std::string modelDir;  // huggingface repo
  tinytorch::Device device = tinytorch::DeviceType::CUDA;
  tinytorch::DType dtype = tinytorch::DType::BFloat16;

  SamplerConfig samplerConfig;
  int64_t maxNewTokens = 16;
};

struct GPTOutput {
  int64_t batch;
  int64_t newTokens;
  std::vector<int32_t> tokenIds;
  std::vector<std::string> texts;
  FinishReason finishReason = FinishReason::Stop;
};

class GPTEngine {
 public:
  explicit GPTEngine(GPTConfig config);
  ~GPTEngine();

  bool prepare();

  void reconfigure(const SamplerConfig& samplerConfig, int64_t maxNewTokens,
                   const std::vector<int32_t>& extraStopTokenIds = {});

  GPTOutput generateSync(tinytorch::ArrayView<std::string> texts);
  GPTOutput generateAsync(const std::string& text, const GenerateCallback& callback);

  bool hasChatTemplate() const;
  std::string applyChatTemplate(const std::vector<tokenizer::ChatMessage>& messages,
                                bool addGenerationPrompt = true) const;

 private:
  tinytorch::Tensor genNextToken(const tinytorch::Tensor& tokens, const tinytorch::Tensor& mask);
  bool isEosToken(int32_t tokenId) const;

  tinytorch::TensorPair encodeTexts(tinytorch::ArrayView<std::string> texts) const;
  GPTOutput decodeTokens(const tinytorch::Tensor& tokens, int64_t offset) const;

  GPTConfig config_;
  Sampler sampler_;
  huggingface::GPTContext context_;
  std::vector<int32_t> baseEosTokenIds_;
  std::vector<int32_t> eosTokenIds_;

  std::unique_ptr<AsyncTokenPipeline> tokenPipeline_;
};

}  // namespace tinygpt
