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

inline RopeScalingConfig convertToRopeScalingConfig(const Config &config) {
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

inline std::unique_ptr<LlamaForCausalLM> createModel(const Config &config, tt::Options options) {
  int64_t headDim = config.hiddenSize / config.numAttentionHeads;
  tt::nn::AttentionConfig attnConfig{config.hiddenSize, config.numAttentionHeads, headDim, config.numKeyValueHeads};

  auto scalingCfg = convertToRopeScalingConfig(config);
  auto ropeCache = tinygpt::kernel::ropeInit(headDim, getContextSize(config), config.ropeTheta, &scalingCfg, options);

  auto attnFactory = [&](int64_t layerIdx) {
    auto rope = tt::nn::RoPE(ropeCache);
    return tt::nn::Attention(static_cast<size_t>(layerIdx), attnConfig, std::move(rope), options);
  };

  auto mlpFactory = [&](int64_t /*layerIdx*/) {
    return tt::nn::GatedMLP(config.hiddenSize, config.intermediateSize, options);
  };

  return std::make_unique<LlamaForCausalLM>(config.vocabSize, config.hiddenSize, config.numHiddenLayers,
                                            config.rmsNormEps, config.tieWordEmbeddings, options, attnFactory,
                                            mlpFactory);
}

inline GPTModel::ModelDims makeDims(const Config &config) {
  const int64_t headDim = config.hiddenSize / config.numAttentionHeads;
  return {config.numHiddenLayers, getContextSize(config), config.numAttentionHeads, config.numKeyValueHeads, headDim,
          config.hiddenSize};
}

}  // namespace llama

class ModelLlama : public GPTModel {
 public:
  ModelLlama(const huggingface::model::LlamaConfig &config, tinytorch::Device device)
      : GPTModel(llama::makeDims(config), device),
        model_(llama::createModel(config, tinytorch::Options(device, config.torchDtype))) {}

  ~ModelLlama() override = default;

  GPTModelType type() const override { return GPTModelType::LLAMA; }

  tinytorch::nn::Module &model() override { return *model_; }

 private:
  std::unique_ptr<tinytorch::nn::CausalLM<tinytorch::nn::Attention, tinytorch::nn::GatedMLP>> model_;
};

}  // namespace tinygpt
