/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "GPTModel.h"
#include "huggingface/ModelConfig.h"
#include "layer/Attention.h"

namespace tinygpt {

namespace mistral {

namespace tt = tinytorch;

using Config = huggingface::model::MistralConfig;

using MistralForCausalLM = tt::nn::CausalLM<tt::nn::Attention, tt::nn::GatedMLP>;

inline std::unique_ptr<MistralForCausalLM> createModel(const Config &config, KVCacheManager &kvCache,
                                                       tt::Options options) {
  int64_t headDim = config.hiddenSize / config.numAttentionHeads;
  tt::nn::AttentionConfig attnConfig{config.hiddenSize, config.numAttentionHeads, headDim, config.numKeyValueHeads};

  auto attnFactory = [&](int layerIdx) {
    auto rope = tt::nn::RoPE(headDim, config.maxPositionEmbeddings, config.ropeTheta, std::nullopt, options);
    return tt::nn::Attention(&kvCache, layerIdx, attnConfig, std::move(rope), options);
  };

  auto mlpFactory = [&](int /*layerIdx*/) {
    return tt::nn::GatedMLP(config.hiddenSize, config.intermediateSize, options);
  };

  return std::make_unique<MistralForCausalLM>(config.vocabSize, config.hiddenSize, config.numHiddenLayers,
                                              config.rmsNormEps, config.tieWordEmbeddings, options, attnFactory,
                                              mlpFactory);
}

}  // namespace mistral

class ModelMistral : public GPTModel {
 public:
  ModelMistral(const huggingface::model::MistralConfig &config, tinytorch::Device device)
      : config_(config),
        device_(device),
        model_(mistral::createModel(config_, kvCache_, tinytorch::Options(device, config.torchDtype))) {
    init();
  }

  ~ModelMistral() override = default;

  GPTModelType type() override { return GPTModelType::MISTRAL; }

  int64_t numLayers() override { return config_.numHiddenLayers; }

  int64_t contextSize() override { return config_.maxPositionEmbeddings; }

  tinytorch::nn::Module &model() override { return *model_; }

  tinytorch::Device device() const override { return device_; }

 private:
  const huggingface::model::MistralConfig &config_;
  tinytorch::Device device_;
  std::unique_ptr<tinytorch::nn::CausalLM<tinytorch::nn::Attention, tinytorch::nn::GatedMLP>> model_;
};

}  // namespace tinygpt
