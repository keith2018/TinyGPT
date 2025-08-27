/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <memory>
#include <string>

#include "Tensor.h"

namespace tinygpt::huggingface::model {

constexpr const char* MODEL_TYPE_GPT2 = "gpt2";
constexpr const char* MODEL_TYPE_LLAMA = "llama";
constexpr const char* MODEL_TYPE_QWEN2 = "qwen2";

struct ModelConfig {
  virtual ~ModelConfig() = default;

  std::string modelType;
  std::string hiddenAct;
  tinytorch::DType torchDtype;

  int64_t vocabSize;
  int64_t bosTokenId;
  int64_t eosTokenId;
  int64_t hiddenSize;
  int64_t intermediateSize;
  int64_t maxPositionEmbeddings;
  int64_t numAttentionHeads;
  int64_t numHiddenLayers;
  int64_t numKeyValueHeads;

  float rmsNormEps;
};

struct GPT2Config : ModelConfig {
  std::string activationFunction;
  float layerNormEpsilon;
  int64_t nCtx;
  int64_t nEmbd;
  int64_t nHead;
  int64_t nLayer;
  int64_t nPositions;
};

struct LlamaConfig : ModelConfig {
  bool attentionBias;
  int64_t headDim;

  struct RopeScalingConfig {
    float factor;
    float highFreqFactor;
    float lowFreqFactor;
    int64_t originalMaxPositionEmbeddings;
    std::string ropeType;
  } ropeScaling;

  float ropeTheta;
};

struct Qwen2Config : ModelConfig {
  float ropeTheta;
  int64_t slidingWindow;
  bool useMRope;
  bool useSlidingWindow;
};

struct GenerationConfig {
  int64_t bosTokenId;
  int64_t eosTokenId;

  bool doSample;
  float temperature;
  int64_t topK;
  float topP;
};

std::unique_ptr<ModelConfig> loadModelConfig(const std::string& cfgPath);

std::unique_ptr<GenerationConfig> loadGenerationConfig(const std::string& cfgPath);

}  // namespace tinygpt::huggingface::model
