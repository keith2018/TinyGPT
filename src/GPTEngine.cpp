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

tt::Tensor GPTEngine::genNextToken(const tt::Tensor& tokens) {
  auto logits = context_.model->forward(tokens);
  logits = tt::function::narrow(logits, 1, logits.size(1) - 1, 1).squeeze(1);
  return sampler_.sample(logits);
}

tt::Tensor GPTEngine::encodeTexts(const std::string& text) const {
  auto inputIds = context_.tokenizer->encode(text);
  auto tokens = tt::Tensor(inputIds, tt::Options(config_.device, tt::DType::Int32)).to(tt::DType::Int64);
  tokens.unsqueeze_(0);  // batch=1
  return tokens;
}

GPTOutput GPTEngine::decodeTokens(const tt::Tensor& tokens, int64_t offset) const {
  auto outputIds = tokens.to(tt::DType::Int32).toList<int32_t>();
  return {outputIds, context_.tokenizer->decode(outputIds, offset)};
}

GPTOutput GPTEngine::generateSync(const std::string& text) {
  tt::NoGradGuard guard;

  auto tokens = encodeTexts(text);
  auto inputTokenCnt = tokens.numel();

  auto eosId = context_.tokenizer->eosTokenId();
  tt::Tensor nextToken;

  // prefill
  {
    nextToken = genNextToken(tokens);
    tokens = tt::function::concat({tokens, nextToken}, 1);
    if (nextToken.item<int64_t>() == eosId) {
      return decodeTokens(tokens);
    }
  }

  // decode
  for (int i = 1; i <= config_.maxNewTokens; i++) {
    nextToken = genNextToken(nextToken);
    tokens = tt::function::concat({tokens, nextToken}, 1);
    if (nextToken.item<int64_t>() == eosId) {
      break;
    }
  }

  return decodeTokens(tokens, inputTokenCnt);
}

}  // namespace tinygpt