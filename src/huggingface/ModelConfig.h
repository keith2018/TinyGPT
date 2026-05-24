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

constexpr const char* MODEL_TYPE_LLAMA = "llama";
constexpr const char* MODEL_TYPE_QWEN2 = "qwen2";
constexpr const char* MODEL_TYPE_QWEN3 = "qwen3";
constexpr const char* MODEL_TYPE_MISTRAL = "mistral";

struct ModelConfig {
  virtual ~ModelConfig() = default;

  std::string modelType;
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
  bool tieWordEmbeddings;
};

struct LlamaConfig : ModelConfig {
  int64_t headDim;

  struct RopeScalingConfig {
    float factor;
    float highFreqFactor;
    float lowFreqFactor;
    int64_t originalMaxPositionEmbeddings;
  } ropeScaling;

  float ropeTheta;
};

struct QwenConfig : ModelConfig {
  float ropeTheta;
  int64_t headDim;
};

struct MistralConfig : ModelConfig {
  float ropeTheta;
};

struct GenerationConfig {
  int32_t bosTokenId;
  std::vector<int32_t> eosTokenIds;

  bool doSample;
  float temperature;
  int32_t topK;
  float topP;
};

std::unique_ptr<ModelConfig> loadModelConfig(const std::string& cfgPath);

std::unique_ptr<GenerationConfig> loadGenerationConfig(const std::string& cfgPath);

}  // namespace tinygpt::huggingface::model
