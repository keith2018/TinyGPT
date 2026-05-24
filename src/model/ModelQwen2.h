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

namespace qwen2 {

namespace tt = tinytorch;

using Config = huggingface::model::QwenConfig;

using Qwen2ForCausalLM = tt::nn::CausalLM<tt::nn::Attention, tt::nn::GatedMLP>;

inline std::unique_ptr<Qwen2ForCausalLM> createModel(const Config &config, tt::Options options) {
  int64_t headDim = config.hiddenSize / config.numAttentionHeads;
  tt::nn::AttentionConfig attnConfig{
      config.hiddenSize, config.numAttentionHeads, headDim, config.numKeyValueHeads, true, false};

  auto ropeCache = tinygpt::kernel::ropeInit(headDim, config.maxPositionEmbeddings, config.ropeTheta, nullptr, options);

  auto attnFactory = [&](int64_t layerIdx) {
    auto rope = tt::nn::RoPE(ropeCache);
    return tt::nn::Attention(static_cast<size_t>(layerIdx), attnConfig, std::move(rope), options);
  };

  auto mlpFactory = [&](int64_t /*layerIdx*/) {
    return tt::nn::GatedMLP(config.hiddenSize, config.intermediateSize, options);
  };

  return std::make_unique<Qwen2ForCausalLM>(config.vocabSize, config.hiddenSize, config.numHiddenLayers,
                                            config.rmsNormEps, config.tieWordEmbeddings, options, attnFactory,
                                            mlpFactory);
}

inline GPTModel::ModelDims makeDims(const Config &config) {
  const int64_t headDim = config.hiddenSize / config.numAttentionHeads;
  return {
      config.numHiddenLayers, config.maxPositionEmbeddings, config.numAttentionHeads, config.numKeyValueHeads, headDim,
      config.hiddenSize};
}

}  // namespace qwen2

class ModelQwen2 : public GPTModel {
 public:
  ModelQwen2(const huggingface::model::QwenConfig &config, tinytorch::Device device)
      : GPTModel(qwen2::makeDims(config), device),
        model_(qwen2::createModel(config, tinytorch::Options(device, config.torchDtype))) {}

  ~ModelQwen2() override = default;

  GPTModelType type() const override { return GPTModelType::QWEN2; }

  tinytorch::nn::Module &model() override { return *model_; }

 private:
  std::unique_ptr<tinytorch::nn::CausalLM<tinytorch::nn::Attention, tinytorch::nn::GatedMLP>> model_;
};

}  // namespace tinygpt
