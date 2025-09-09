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

GPTEngine::GPTEngine(GPTConfig config) : config_(std::move(config)), sampler_(config.samplerConfig) {}

bool GPTEngine::prepare() {
  huggingface::ModelLoader loader;
  bool success = loader.load(config_.modelDir, config_.device, config_.dtype);
  if (!success) {
    LOGE("Prepare failed");
    return false;
  }
  context_ = loader.getContext();
  return true;
}

tt::Tensor GPTEngine::genNextToken(const tt::Tensor& tokens, const tt::Tensor& mask) {
  // TODO padding mask
  auto logits = context_.model->forward(tokens);
  logits = tt::function::narrow(logits, 1, logits.size(1) - 1, 1).squeeze(1);
  return sampler_.sample(logits);
}

tt::TensorPair GPTEngine::encodeTexts(tinytorch::ArrayView<std::string> texts) const {
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
  return {batch, outputIds, texts};
}

GPTOutput GPTEngine::generateSync(tinytorch::ArrayView<std::string> texts) {
  tt::NoGradGuard guard;

  auto [tokens, attnMask] = encodeTexts(texts);
  auto inputTokenCnt = tokens.size(1);
  tt::Tensor nextToken;

  // prefill
  {
    nextToken = genNextToken(tokens, attnMask);
    tokens = tt::function::concat({tokens, nextToken}, 1);
    // TODO check eos
  }

  // decode
  for (int i = 1; i <= config_.maxNewTokens; i++) {
    nextToken = genNextToken(nextToken, {});
    tokens = tt::function::concat({tokens, nextToken}, 1);
    // TODO check eos
  }

  return decodeTokens(tokens, inputTokenCnt);
}

}  // namespace tinygpt