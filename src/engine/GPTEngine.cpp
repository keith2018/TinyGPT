/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "GPTEngine.h"

#include <utility>

#include "Functions.h"

namespace tt = tinytorch;

namespace tinygpt {

class AsyncTokenPipeline {
 public:
  virtual ~AsyncTokenPipeline() = default;
  virtual void submitToken(const tt::Tensor& curToken) = 0;
  virtual int32_t fetchTokenId() = 0;
};

class DefaultTokenPipeline final : public AsyncTokenPipeline {
 public:
  void submitToken(const tt::Tensor& curToken) override { pending_ = curToken; }
  int32_t fetchTokenId() override { return pending_.to(tt::DType::Int32).item<int32_t>(); }

 private:
  tt::Tensor pending_;
};

inline std::unique_ptr<AsyncTokenPipeline> createTokenPipeline(const tt::Device& device) {
  return std::make_unique<DefaultTokenPipeline>();
}

GPTEngine::GPTEngine(GPTConfig config) : config_(std::move(config)), sampler_(config.samplerConfig) {}

GPTEngine::~GPTEngine() = default;

bool GPTEngine::prepare() {
  huggingface::ModelLoader loader;
  bool success = loader.load(config_.modelDir, config_.device, config_.dtype);
  if (!success) {
    LOGE("Prepare failed");
    return false;
  }
  context_ = loader.getContext();

  if (context_.generationConfig) {
    for (auto id : context_.generationConfig->eosTokenIds) {
      baseEosTokenIds_.push_back(static_cast<int32_t>(id));
    }
  }
  if (baseEosTokenIds_.empty()) {
    int32_t eosId = context_.tokenizer->eosTokenId();
    if (eosId >= 0) {
      baseEosTokenIds_.push_back(eosId);
    }
  }
  eosTokenIds_ = baseEosTokenIds_;

  tokenPipeline_ = createTokenPipeline(config_.device);
  return true;
}

void GPTEngine::reconfigure(const SamplerConfig& samplerConfig, int64_t maxNewTokens,
                            const std::vector<int32_t>& extraStopTokenIds) {
  // sampler
  sampler_ = Sampler(samplerConfig);
  config_.samplerConfig = samplerConfig;
  config_.maxNewTokens = maxNewTokens;

  // eos
  eosTokenIds_ = baseEosTokenIds_;
  for (auto id : extraStopTokenIds) {
    if (!isEosToken(id)) {
      eosTokenIds_.push_back(id);
    }
  }

  // kv cache
  context_.model->resetCache();
}

bool GPTEngine::hasChatTemplate() const { return context_.tokenizer && context_.tokenizer->hasChatTemplate(); }

std::string GPTEngine::applyChatTemplate(const std::vector<tokenizer::ChatMessage>& messages,
                                         bool addGenerationPrompt) const {
  if (!context_.tokenizer) return {};
  return context_.tokenizer->applyChatTemplate(messages, addGenerationPrompt);
}

tt::Tensor GPTEngine::genNextToken(const tt::Tensor& tokens, const tt::Tensor& mask) {
  // TODO padding mask
  auto logits = context_.model->forward(tokens);
  logits = tt::function::narrow(logits, 1, logits.size(1) - 1, 1).squeeze(1);
  return sampler_.sample(logits);
}

tt::TensorPair GPTEngine::encodeTexts(tt::ArrayView<std::string> texts) const {
  auto tokenLists = context_.tokenizer->encodeBatch(texts);
  int64_t maxLength = 0;
  for (auto& tokenIds : tokenLists) {
    maxLength = std::max(maxLength, static_cast<int64_t>(tokenIds.size()));
  }
  maxLength = std::min(maxLength, context_.model->contextSize());
  int32_t padToken = context_.tokenizer->padTokenId();
  if (padToken < 0) {
    padToken = context_.tokenizer->eosTokenId();
    if (padToken < 0) {
      padToken = 0;  // default pad token id
    }
  }

  std::vector<std::vector<int32_t>> alignedTokens;
  std::vector<std::vector<uint8_t>> attnMask;
  alignedTokens.resize(tokenLists.size());
  attnMask.resize(tokenLists.size());

  // truncation & padding (left)
  for (size_t i = 0; i < tokenLists.size(); i++) {
    auto& tokens = tokenLists[i];
    alignedTokens[i].resize(maxLength);
    attnMask[i].resize(maxLength);

    if (static_cast<int64_t>(tokens.size()) > maxLength) {
      std::copy(tokens.end() - maxLength, tokens.end(), alignedTokens[i].begin());
      std::fill(attnMask[i].begin(), attnMask[i].end(), 1);
    } else {
      int64_t padAmount = maxLength - static_cast<int64_t>(tokens.size());

      std::fill(alignedTokens[i].begin(), alignedTokens[i].begin() + padAmount, padToken);
      std::copy(tokens.begin(), tokens.end(), alignedTokens[i].begin() + padAmount);

      std::fill(attnMask[i].begin(), attnMask[i].begin() + padAmount, 0);
      std::fill(attnMask[i].begin() + padAmount, attnMask[i].end(), 1);
    }
  }

  auto tokensTensor = tt::Tensor(alignedTokens, tt::Options(config_.device, tt::DType::Int32)).to(tt::DType::Int64);
  auto maskTensor = tt::Tensor(attnMask, tt::Options(config_.device, tt::DType::Bool));
  return {tokensTensor, maskTensor};
}

GPTOutput GPTEngine::decodeTokens(const tt::Tensor& tokens, int64_t offset) const {
  auto batch = tokens.shape(0);
  auto outputIds = tokens.to(tt::DType::Int32).toList<int32_t>();
  auto texts = context_.tokenizer->decodeBatch(outputIds, batch, offset);
  auto seqLen = static_cast<int64_t>(outputIds.size()) / batch;
  return {batch, seqLen - offset, outputIds, texts};
}

GPTOutput GPTEngine::generateSync(tt::ArrayView<std::string> texts) {
  tt::NoGradGuard guard;

  auto [tokens, attnMask] = encodeTexts(texts);
  auto inputTokenCnt = tokens.size(1);

  // prefill
  auto nextToken = genNextToken(tokens, attnMask);
  tokens = tt::function::concat({tokens, nextToken}, 1);

  // decode
  for (int i = 1; i < config_.maxNewTokens; i++) {
    nextToken = genNextToken(nextToken, {});
    tokens = tt::function::concat({tokens, nextToken}, 1);
  }

  // skip eos check
  auto output = decodeTokens(tokens, inputTokenCnt);
  output.finishReason = FinishReason::Length;
  return output;
}

bool GPTEngine::isEosToken(int32_t tokenId) const {
  return std::any_of(eosTokenIds_.begin(), eosTokenIds_.end(), [tokenId](int32_t eosId) { return tokenId == eosId; });
}

GPTOutput GPTEngine::generateAsync(const std::string& text, const GenerateCallback& callback) {
  tt::NoGradGuard guard;

  // batch = 1
  std::vector<std::string> texts = {text};
  auto [tokens, attnMask] = encodeTexts(texts);
  auto inputTokenCnt = tokens.size(1);

  // prefill
  auto curToken = genNextToken(tokens, attnMask);
  tokens = tt::function::concat({tokens, curToken}, 1);

  bool hitEos = false;
  bool aborted = false;

  // decode
  for (int i = 1; i < config_.maxNewTokens; i++) {
    tokenPipeline_->submitToken(curToken);
    auto futureToken = genNextToken(curToken, {});

    int32_t tokenId = tokenPipeline_->fetchTokenId();
    if (isEosToken(tokenId)) {
      hitEos = true;
      break;
    }

    std::vector<int32_t> newIds = {tokenId};
    std::string chunk = context_.tokenizer->decodeStream(newIds);
    if (!chunk.empty() && callback) {
      if (!callback(chunk)) {
        aborted = true;
        break;
      }
    }

    curToken = futureToken;
    tokens = tt::function::concat({tokens, curToken}, 1);
  }

  // flush remaining bytes in stream cache (incomplete UTF-8 sequences)
  if (!aborted) {
    std::string remaining = context_.tokenizer->decodeStreamFlush();
    if (!remaining.empty() && callback) {
      if (!callback(remaining)) {
        // aborted = true;
      }
    }
  }

  auto output = decodeTokens(tokens, inputTokenCnt);
  output.finishReason = (hitEos || aborted) ? FinishReason::Stop : FinishReason::Length;
  return output;
}

}  // namespace tinygpt