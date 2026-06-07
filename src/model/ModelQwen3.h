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

namespace qwen3 {

namespace tt = tinytorch;

using Config = huggingface::model::QwenConfig;

using AttnType = distributed::TPAttentionWithQKNorm;
using MLPType = distributed::TPGatedMLP;
using LMHeadType = distributed::TPLmHead;
using Qwen3ForCausalLM = tt::nn::CausalLM<AttnType, MLPType, LMHeadType>;

inline std::unique_ptr<Qwen3ForCausalLM> createModel(const Config &config, tt::Options options) {
  tt::nn::AttentionConfig attnConfig{config.hiddenSize, config.numAttentionHeads, config.headDim,
                                     config.numKeyValueHeads};

  auto ropeCache =
      tinygpt::kernel::ropeInit(config.headDim, config.maxPositionEmbeddings, config.ropeTheta, nullptr, options);

  auto attnFactory = [&](int64_t layerIdx) {
    auto rope = tt::nn::RoPE(ropeCache);
    return AttnType(static_cast<size_t>(layerIdx), attnConfig, std::move(rope), config.rmsNormEps, options);
  };

  auto mlpFactory = [&](int64_t /*layerIdx*/) { return MLPType(config.hiddenSize, config.intermediateSize, options); };

  return std::make_unique<Qwen3ForCausalLM>(config.vocabSize, config.hiddenSize, config.numHiddenLayers,
                                            config.rmsNormEps, config.tieWordEmbeddings, options, attnFactory,
                                            mlpFactory);
}

inline GPTModel::ModelDims makeDims(const Config &config) {
  return {config.numHiddenLayers,   config.maxPositionEmbeddings,
          config.numAttentionHeads, config.numKeyValueHeads,
          config.headDim,           config.hiddenSize};
}

}  // namespace qwen3

class ModelQwen3 : public GPTModel {
 public:
  ModelQwen3(const huggingface::model::QwenConfig &config, tinytorch::Device device)
      : GPTModel(qwen3::makeDims(config), device),
        model_(qwen3::createModel(config, tinytorch::Options(device, config.torchDtype))) {}

  ~ModelQwen3() override = default;

  GPTModelType type() const override { return GPTModelType::QWEN3; }

  tinytorch::nn::Module &model() override { return *model_; }

 private:
  std::unique_ptr<qwen3::Qwen3ForCausalLM> model_;
};

}  // namespace tinygpt
