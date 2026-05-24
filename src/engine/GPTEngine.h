/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "PagedKVCache.h"
#include "Sampler.h"
#include "Scheduler.h"
#include "huggingface/ModelLoader.h"

namespace tinygpt {

using GenerateCallback = std::function<bool(const std::string& tokenText)>;

struct GPTConfig {
  std::string modelDir;  // huggingface repo
  tinytorch::Device device = tinytorch::DeviceType::CUDA;
  tinytorch::DType dtype = tinytorch::DType::BFloat16;

  SamplerConfig samplerConfig;
  PagedKVCacheConfig pagedConfig;
  int32_t maxNewTokens = 16;
  int32_t maxBatchTokens = 8192;

  // chunked prefill: max prefill tokens per step per sequence
  int32_t prefillChunkSize = 512;

  // max batch size for CUDA graph capture
  int32_t maxGraphBatch = 128;
};

struct GPTOutput {
  std::vector<int32_t> tokenIds;
  std::string text;
  int32_t promptTokens = 0;
  int32_t newTokens = 0;
  FinishReason finishReason = FinishReason::Stop;
};

class GPTEngine {
 public:
  explicit GPTEngine(GPTConfig config);
  ~GPTEngine();

  bool prepare();

  void reconfigure(const SamplerConfig& samplerConfig, int32_t maxNewTokens,
                   const std::vector<int32_t>& extraStopTokenIds = {});

  GPTOutput generate(const std::string& prompt, const GenerateCallback& callback = {});
  GPTOutput generate(const std::string& prompt, const SamplerConfig& samplerConfig, int32_t maxNewTokens,
                     const std::vector<int32_t>& extraStopTokenIds, const GenerateCallback& callback = {});

  bool hasChatTemplate() const;
  std::string applyChatTemplate(const std::vector<tokenizer::ChatMessage>& messages,
                                bool addGenerationPrompt = true) const;

  struct EngineStats {
    size_t numRunning = 0;
    size_t numWaiting = 0;
    int64_t kvTotalBlocks = 0;
    int64_t kvFreeBlocks = 0;
    int32_t kvBlockSize = 0;
  };
  EngineStats stats() const;

 private:
  GPTConfig config_;
  huggingface::GPTContext context_;
  std::vector<int32_t> baseEosTokenIds_;
  std::vector<int32_t> extraEosTokenIds_;  // from reconfigure()

  std::unique_ptr<PagedKVCache> pagedCache_;
  std::unique_ptr<Scheduler> scheduler_;
};

}  // namespace tinygpt
