/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "GPTModel.h"
#include "distributed/ParallelLayers.h"
#include "huggingface/ModelConfig.h"
#include "layer/Attention.h"

namespace tinygpt {

namespace mistral {

namespace tt = tinytorch;

using Config = huggingface::model::MistralConfig;

using AttnType = distributed::TPAttention;
using MLPType = distributed::TPGatedMLP;
using LMHeadType = distributed::TPLmHead;
using MistralForCausalLM = tt::nn::CausalLM<AttnType, MLPType, LMHeadType>;

inline std::unique_ptr<MistralForCausalLM> createModel(const Config &config, tt::Options options) {
  int64_t headDim = config.hiddenSize / config.numAttentionHeads;
  tt::nn::AttentionConfig attnConfig{config.hiddenSize, config.numAttentionHeads, headDim, config.numKeyValueHeads};

  auto ropeCache = tinygpt::kernel::ropeInit(headDim, config.maxPositionEmbeddings, config.ropeTheta, nullptr, options);

  auto attnFactory = [&](int64_t layerIdx) {
    auto rope = tt::nn::RoPE(ropeCache);
    return AttnType(static_cast<size_t>(layerIdx), attnConfig, std::move(rope), options);
  };

  auto mlpFactory = [&](int64_t /*layerIdx*/) { return MLPType(config.hiddenSize, config.intermediateSize, options); };

  return std::make_unique<MistralForCausalLM>(config.vocabSize, config.hiddenSize, config.numHiddenLayers,
                                              config.rmsNormEps, config.tieWordEmbeddings, options, attnFactory,
                                              mlpFactory);
}

inline GPTModel::ModelDims makeDims(const Config &config) {
  const int64_t headDim = config.hiddenSize / config.numAttentionHeads;
  return {
      config.numHiddenLayers, config.maxPositionEmbeddings, config.numAttentionHeads, config.numKeyValueHeads, headDim,
      config.hiddenSize};
}

}  // namespace mistral

class ModelMistral : public GPTModel {
 public:
  ModelMistral(const huggingface::model::MistralConfig &config, tinytorch::Device device)
      : GPTModel(mistral::makeDims(config), device),
        model_(mistral::createModel(config, tinytorch::Options(device, config.torchDtype))) {}

  ~ModelMistral() override = default;

  GPTModelType type() const override { return GPTModelType::MISTRAL; }

  tinytorch::nn::Module &model() override { return *model_; }

 private:
  std::unique_ptr<mistral::MistralForCausalLM> model_;
};

}  // namespace tinygpt
