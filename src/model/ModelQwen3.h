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

namespace qwen3 {

namespace tt = tinytorch;

using Config = huggingface::model::QwenConfig;

using Qwen3ForCausalLM = tt::nn::CausalLM<tt::nn::AttentionWithQKNorm, tt::nn::GatedMLP>;

inline std::unique_ptr<Qwen3ForCausalLM> createModel(const Config &config, KVCacheManager &kvCache,
                                                     tt::Options options) {
  tt::nn::AttentionConfig attnConfig{config.hiddenSize, config.numAttentionHeads, config.headDim,
                                     config.numKeyValueHeads};

  auto attnFactory = [&](int layerIdx) {
    auto rope = tt::nn::RoPE(config.headDim, config.maxPositionEmbeddings, config.ropeTheta, std::nullopt, options);
    return tt::nn::AttentionWithQKNorm(&kvCache, layerIdx, attnConfig, std::move(rope), config.rmsNormEps, options);
  };

  auto mlpFactory = [&](int /*layerIdx*/) {
    return tt::nn::GatedMLP(config.hiddenSize, config.intermediateSize, options);
  };

  return std::make_unique<Qwen3ForCausalLM>(config.vocabSize, config.hiddenSize, config.numHiddenLayers,
                                            config.rmsNormEps, config.tieWordEmbeddings, options, attnFactory,
                                            mlpFactory);
}

}  // namespace qwen3

class ModelQwen3 : public GPTModel {
 public:
  ModelQwen3(const huggingface::model::QwenConfig &config, tinytorch::Device device)
      : config_(config),
        device_(device),
        model_(qwen3::createModel(config_, kvCache_, tinytorch::Options(device, config.torchDtype))) {
    init();
  }

  ~ModelQwen3() override = default;

  GPTModelType type() override { return GPTModelType::QWEN3; }

  int64_t numLayers() override { return config_.numHiddenLayers; }

  int64_t contextSize() override { return config_.maxPositionEmbeddings; }

  tinytorch::nn::Module &model() override { return *model_; }

  tinytorch::Device device() const override { return device_; }

 private:
  const huggingface::model::QwenConfig &config_;
  tinytorch::Device device_;
  std::unique_ptr<tinytorch::nn::CausalLM<tinytorch::nn::AttentionWithQKNorm, tinytorch::nn::GatedMLP>> model_;
};

}  // namespace tinygpt
