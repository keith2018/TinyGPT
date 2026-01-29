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

namespace llama {

namespace tt = tinytorch;

using Config = huggingface::model::LlamaConfig;

inline tt::RopeScalingConfig convertToRopeScalingConfig(const Config &config) {
  return {config.ropeScaling.factor, config.ropeScaling.highFreqFactor, config.ropeScaling.lowFreqFactor,
          config.ropeScaling.originalMaxPositionEmbeddings};
}

inline int64_t getContextSize(const Config &config) {
  if (config.ropeScaling.originalMaxPositionEmbeddings > 0) {
    return config.ropeScaling.originalMaxPositionEmbeddings;
  }
  return config.maxPositionEmbeddings;
}

using LlamaForCausalLM = tt::nn::CausalLM<tt::nn::Attention, tt::nn::GatedMLP>;

inline std::unique_ptr<LlamaForCausalLM> createModel(const Config &config, KVCacheManager &kvCache,
                                                     tt::Options options) {
  int64_t headDim = config.hiddenSize / config.numAttentionHeads;
  tt::nn::AttentionConfig attnConfig{config.hiddenSize, config.numAttentionHeads, headDim, config.numKeyValueHeads};

  auto attnFactory = [&](int layerIdx) {
    auto rope =
        tt::nn::RoPE(headDim, getContextSize(config), config.ropeTheta, convertToRopeScalingConfig(config), options);
    return tt::nn::Attention(&kvCache, layerIdx, attnConfig, std::move(rope), options);
  };

  auto mlpFactory = [&](int /*layerIdx*/) {
    return tt::nn::GatedMLP(config.hiddenSize, config.intermediateSize, options);
  };

  return std::make_unique<LlamaForCausalLM>(config.vocabSize, config.hiddenSize, config.numHiddenLayers,
                                            config.rmsNormEps, config.tieWordEmbeddings, options, attnFactory,
                                            mlpFactory);
}

}  // namespace llama

class ModelLlama : public GPTModel {
 public:
  ModelLlama(const huggingface::model::LlamaConfig &config, tinytorch::Device device)
      : config_(config),
        device_(device),
        model_(llama::createModel(config_, kvCache_, tinytorch::Options(device, config.torchDtype))) {
    init();
  }

  ~ModelLlama() override = default;

  GPTModelType type() override { return GPTModelType::LLAMA; }

  int64_t numLayers() override { return config_.numHiddenLayers; }

  int64_t contextSize() override { return llama::getContextSize(config_); }

  tinytorch::nn::Module &model() override { return *model_; }

  tinytorch::Device device() const override { return device_; }

 private:
  const huggingface::model::LlamaConfig &config_;
  tinytorch::Device device_;
  std::unique_ptr<tinytorch::nn::CausalLM<tinytorch::nn::Attention, tinytorch::nn::GatedMLP>> model_;
};

}  // namespace tinygpt
