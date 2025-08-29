/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "ModelLoader.h"

#include "model/ModelGPT2.h"
#include "model/ModelLlama.h"
#include "model/ModelQwen2.h"
#include "util/PathUtils.h"

namespace tinygpt::huggingface {

constexpr const char* kModelConfigPath = "config.json";
constexpr const char* kGenerationConfigPath = "generation_config.json";
constexpr const char* kTokenizerPath = "tokenizer.json";
constexpr const char* kTokenizerConfigPath = "tokenizer_config.json";
constexpr const char* kModelPath = "model.safetensors";
constexpr const char* kModelIndexPath = "model.safetensors.index.json";

bool ModelLoader::load(const std::string& dir, tinytorch::Device device, tinytorch::DType dtype) {
  // model config
  context_.modelConfig = model::loadModelConfig(PathUtils::joinPath(dir, kModelConfigPath));
  if (!context_.modelConfig) {
    LOGE("Failed to load model config: %s", kModelConfigPath);
    return false;
  }

  // generation config
  context_.generationConfig = model::loadGenerationConfig(PathUtils::joinPath(dir, kGenerationConfigPath));
  if (!context_.generationConfig) {
    LOGE("Failed to load generation config: %s", kGenerationConfigPath);
    return false;
  }

  // tokenizer
  context_.tokenizer = std::make_unique<tokenizer::Tokenizer>();
  bool success = context_.tokenizer->initWithConfig(PathUtils::joinPath(dir, kTokenizerPath),
                                                    PathUtils::joinPath(dir, kTokenizerConfigPath));
  if (!success) {
    LOGE("Failed to load tokenizer");
    return false;
  }

  // model
  if (context_.modelConfig->modelType == model::MODEL_TYPE_GPT2) {
    auto* config = dynamic_cast<model::GPT2Config*>(context_.modelConfig.get());
    context_.model = std::make_unique<ModelGPT2>(*config, device);
  } else if (context_.modelConfig->modelType == model::MODEL_TYPE_LLAMA) {
    auto* config = dynamic_cast<model::LlamaConfig*>(context_.modelConfig.get());
    context_.model = std::make_unique<ModelLlama>(*config, device);
  } else if (context_.modelConfig->modelType == model::MODEL_TYPE_QWEN2) {
    auto* config = dynamic_cast<model::Qwen2Config*>(context_.modelConfig.get());
    context_.model = std::make_unique<ModelQwen2>(*config, device);
  } else {
    LOGE("model type not support: %s", context_.modelConfig->modelType.c_str());
    return false;
  }

  // load model from file
  LOGI("Load model ...");
  auto modelPath = PathUtils::joinPath(dir, kModelPath);
  if (!PathUtils::fileExists(modelPath)) {
    modelPath = PathUtils::joinPath(dir, kModelIndexPath);
  }
  success = context_.model->load(modelPath);
  if (!success) {
    LOGE("Load model failed: %s", modelPath.c_str());
    return false;
  }
  LOGI("Load model done.");

  // convert dtype
  context_.model->model().to(dtype);

  // set model eval
  context_.model->model().eval();
  return true;
}

}  // namespace tinygpt::huggingface