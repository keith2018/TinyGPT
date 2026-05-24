/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "ModelConfig.h"

#include <fstream>
#include <memory>
#include <sstream>

#include "JsonHelper.h"
#include "Utils/Logger.h"

namespace tinygpt::huggingface::model {

using namespace tinygpt::json;

static tinytorch::DType cvtDtype(const std::string& typeStr) {
  if (typeStr == "bfloat16") {
    return tinytorch::DType::BFloat16;
  }

  if (typeStr == "float16") {
    return tinytorch::DType::Float16;
  }

  return tinytorch::DType::Float32;
}

static std::string readFileToString(const std::string& filePath) {
  std::ifstream ifs(filePath, std::ios::binary);
  if (!ifs.is_open()) {
    LOGE("Failed to open file: %s", filePath.c_str());
    return "";
  }
  std::stringstream buffer;
  buffer << ifs.rdbuf();
  return buffer.str();
}

std::unique_ptr<ModelConfig> loadModelConfig(const std::string& cfgPath) {
  std::string jsonStr = readFileToString(cfgPath);
  if (jsonStr.empty()) {
    return nullptr;
  }
  rapidjson::Document doc;
  doc.Parse(jsonStr.c_str());
  if (doc.HasParseError()) {
    LOGE("JSON parse error in: %s", cfgPath.c_str());
    return nullptr;
  }

  auto modelType = getJsonValue<std::string>(doc, "model_type", "");
  if (modelType.empty()) {
    LOGE("Missing or invalid model_type in config.json");
    return nullptr;
  }

  std::unique_ptr<ModelConfig> config;

  if (modelType == MODEL_TYPE_LLAMA) {
    auto cfg = std::make_unique<LlamaConfig>();
    cfg->headDim = getJsonValue<int64_t>(doc, "head_dim", -1);

    // rope
    if (doc.HasMember("rope_scaling") && doc["rope_scaling"].IsObject()) {
      const auto& rope = doc["rope_scaling"];
      cfg->ropeScaling.factor = getJsonValue<float>(rope, "factor", 1.f);
      cfg->ropeScaling.highFreqFactor = getJsonValue<float>(rope, "high_freq_factor", 1.f);
      cfg->ropeScaling.lowFreqFactor = getJsonValue<float>(rope, "low_freq_factor", 1.f);
      cfg->ropeScaling.originalMaxPositionEmbeddings =
          getJsonValue<int64_t>(rope, "original_max_position_embeddings", -1);
    }
    cfg->ropeTheta = getJsonValue<float>(doc, "rope_theta", 1.f);
    config = std::move(cfg);
  } else if (modelType == MODEL_TYPE_QWEN2 || modelType == MODEL_TYPE_QWEN3) {
    auto cfg = std::make_unique<QwenConfig>();
    cfg->ropeTheta = getJsonValue<float>(doc, "rope_theta", 10000.f);
    cfg->headDim = getJsonValue<int64_t>(doc, "head_dim", -1);
    config = std::move(cfg);
  } else if (modelType == MODEL_TYPE_MISTRAL) {
    auto cfg = std::make_unique<MistralConfig>();
    cfg->ropeTheta = getJsonValue<float>(doc, "rope_theta", 10000.0f);
    config = std::move(cfg);
  } else {
    LOGE("Unsupported model_type: %s", modelType.c_str());
    return nullptr;
  }

  config->bosTokenId = getJsonValue<int64_t>(doc, "bos_token_id", -1);
  config->eosTokenId = getJsonValue<int64_t>(doc, "eos_token_id", -1);
  config->hiddenSize = getJsonValue<int64_t>(doc, "hidden_size", -1);
  config->intermediateSize = getJsonValue<int64_t>(doc, "intermediate_size", -1);
  config->maxPositionEmbeddings = getJsonValue<int64_t>(doc, "max_position_embeddings", -1);
  config->modelType = modelType;
  config->numAttentionHeads = getJsonValue<int64_t>(doc, "num_attention_heads", -1);
  config->numHiddenLayers = getJsonValue<int64_t>(doc, "num_hidden_layers", -1);
  config->numKeyValueHeads = getJsonValue<int64_t>(doc, "num_key_value_heads", -1);
  config->rmsNormEps = getJsonValue<float>(doc, "rms_norm_eps", 1e-5f);
  config->tieWordEmbeddings = getJsonValue<bool>(doc, "tie_word_embeddings", false);
  config->torchDtype = cvtDtype(getJsonValue<std::string>(doc, "torch_dtype", ""));
  config->vocabSize = getJsonValue<int64_t>(doc, "vocab_size", -1);

  return config;
}

std::unique_ptr<GenerationConfig> loadGenerationConfig(const std::string& cfgPath) {
  std::string jsonStr = readFileToString(cfgPath);
  if (jsonStr.empty()) {
    return nullptr;
  }
  rapidjson::Document doc;
  doc.Parse(jsonStr.c_str());
  if (doc.HasParseError()) {
    LOGE("JSON parse error in: %s", cfgPath.c_str());
    return nullptr;
  }

  auto cfg = std::make_unique<GenerationConfig>();
  cfg->bosTokenId = getJsonValue<int32_t>(doc, "bos_token_id", -1);

  if (doc.HasMember("eos_token_id")) {
    auto& val = doc["eos_token_id"];
    if (val.IsArray()) {
      for (auto& v : val.GetArray()) {
        if (v.IsInt())
          cfg->eosTokenIds.push_back(v.GetInt());
        else if (v.IsInt64())
          cfg->eosTokenIds.push_back(static_cast<int32_t>(v.GetInt64()));
      }
    } else if (val.IsInt()) {
      cfg->eosTokenIds.push_back(val.GetInt());
    } else if (val.IsInt64()) {
      cfg->eosTokenIds.push_back(static_cast<int32_t>(val.GetInt64()));
    }
  }

  cfg->doSample = getJsonValue<bool>(doc, "do_sample", false);
  cfg->temperature = getJsonValue<float>(doc, "temperature", 0.f);
  cfg->topK = getJsonValue<int32_t>(doc, "top_k", 0);
  cfg->topP = getJsonValue<float>(doc, "top_p", 1.f);

  return cfg;
}

}  // namespace tinygpt::huggingface::model