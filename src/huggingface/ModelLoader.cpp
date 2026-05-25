/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "ModelLoader.h"

#include "model/ModelLlama.h"
#include "model/ModelMistral.h"
#include "model/ModelQwen2.h"
#include "model/ModelQwen3.h"
#include "util/FileUtils.h"

namespace tinygpt::huggingface {

constexpr const char* kModelConfigPath = "config.json";
constexpr const char* kGenerationConfigPath = "generation_config.json";
constexpr const char* kTokenizerPath = "tokenizer.json";
constexpr const char* kTokenizerConfigPath = "tokenizer_config.json";
constexpr const char* kModelPath = "model.safetensors";
constexpr const char* kModelIndexPath = "model.safetensors.index.json";

bool ModelLoader::load(const std::string& dir, tinytorch::Device device, tinytorch::DType dtype) {
  // model config
  context_.modelConfig = model::loadModelConfig(fileutil::join(dir, kModelConfigPath));
  if (!context_.modelConfig) {
    LOGE("Failed to load model config: %s", kModelConfigPath);
    return false;
  }

  // generation config
  context_.generationConfig = model::loadGenerationConfig(fileutil::join(dir, kGenerationConfigPath));
  if (!context_.generationConfig) {
    LOGE("Failed to load generation config: %s", kGenerationConfigPath);
    return false;
  }

  // tokenizer
  context_.tokenizer = std::make_unique<tokenizer::Tokenizer>();
  bool success = context_.tokenizer->initWithConfig(fileutil::join(dir, kTokenizerPath),
                                                    fileutil::join(dir, kTokenizerConfigPath));
  if (!success) {
    LOGE("Failed to load tokenizer");
    return false;
  }

  // model
  if (context_.modelConfig->modelType == model::MODEL_TYPE_LLAMA) {
    auto* config = dynamic_cast<model::LlamaConfig*>(context_.modelConfig.get());
    context_.model = std::make_unique<ModelLlama>(*config, device);
  } else if (context_.modelConfig->modelType == model::MODEL_TYPE_QWEN2) {
    auto* config = dynamic_cast<model::QwenConfig*>(context_.modelConfig.get());
    context_.model = std::make_unique<ModelQwen2>(*config, device);
  } else if (context_.modelConfig->modelType == model::MODEL_TYPE_QWEN3) {
    auto* config = dynamic_cast<model::QwenConfig*>(context_.modelConfig.get());
    context_.model = std::make_unique<ModelQwen3>(*config, device);
  } else if (context_.modelConfig->modelType == model::MODEL_TYPE_MISTRAL) {
    auto* config = dynamic_cast<model::MistralConfig*>(context_.modelConfig.get());
    context_.model = std::make_unique<ModelMistral>(*config, device);
  } else {
    LOGE("model type not support: %s", context_.modelConfig->modelType.c_str());
    return false;
  }

  // load model from file
  LOGI("Load model ...");
  auto modelPath = fileutil::join(dir, kModelPath);
  if (!fileutil::exists(modelPath)) {
    modelPath = fileutil::join(dir, kModelIndexPath);
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
