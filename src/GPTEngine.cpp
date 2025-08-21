/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "GPTEngine.h"

#include "Functions.h"
#include "models/ModelGPT2.h"
#include "models/ModelLlama32.h"

namespace tt = tinytorch;

namespace tinygpt {

GPTEngine::GPTEngine(const GPTConfig& config) : config_(config), sampler_(config.samplerConfig) {}

bool GPTEngine::prepare() {
  LOGI("Init tokenizer ...");
  // load tokenizer
  bool success = tokenizer_.initWithConfigHF(config_.tokenizerPath, config_.tokenizerConfigPath);
  if (!success) {
    LOGE("Init tokenizer failed");
    return false;
  }
  LOGI("Init tokenizer done.");

  // create model
  switch (config_.modelType) {
    case GPTModelType::GPT2:
      model_ = std::make_unique<ModelGPT2>(config_.device);
      break;
    case GPTModelType::LLAMA32:
      model_ = std::make_unique<ModelLlama32>(config_.device, config_.modelSize);
      break;
    default:
      LOGE("Unsupported GPTModelType: %d", config_.modelType);
      return false;
  }

  // load from file
  LOGI("Load model ...");
  success = model_->load(config_.modelFilePath);
  if (!success) {
    LOGE("Load model failed: %s", config_.modelFilePath.c_str());
    return false;
  }
  LOGI("Load model done.");

  // set model eval
  model_->model().eval();
  return true;
}

GPTOutput GPTEngine::generateSync(const std::string& text) {
  tt::NoGradGuard guard;

  // encode
  auto inputIds = tokenizer_.encode(text);
  auto tokens = tt::Tensor(inputIds, tt::Options(config_.device, tt::DType::Int32));
  tokens = tokens.to(tt::DType::Int64);
  tokens.unsqueeze_(0);  // batch == 1

  // inference
  auto contextSize = model_->contextSize();
  auto eosId = tokenizer_.eosTokenId();

  for (auto i = 0; i < config_.maxNewTokens; i++) {
    auto seqLen = tokens.size(1);
    tt::Tensor tokensCond;
    if (seqLen > contextSize) {
      tokensCond = tt::function::narrow(tokens, 1, seqLen - contextSize, contextSize);
    } else {
      tokensCond = tokens;
    }
    auto logits = model_->forward(tokensCond);
    logits = tt::function::narrow(logits, 1, logits.size(1) - 1, 1).squeeze(1);

    const auto nextToken = sampler_.sample(logits);

    bool complete = false;
    ASSERT(nextToken.numel() == 1);  // batch == 1
    if (nextToken.item<int64_t>() == eosId) {
      complete = true;
    }

    tokens = tt::function::concat(tt::ArrayView<tt::Tensor>{tokens, nextToken}, 1);

    if (complete) {
      break;
    }
  }

  // decode
  tokens = tokens.to(tt::DType::Int32);
  auto outputIds = tokens.toList<int32_t>();
  auto outputText = tokenizer_.decode(outputIds);
  return {outputIds, outputText};
}

}  // namespace tinygpt