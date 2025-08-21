/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "GPTEngine.h"
#include "Utils/Timer.h"

// clone from https://huggingface.co/meta-llama/Llama-3.2-1B
const std::string LLAMA32_MODEL_DIR_1B = "path to llama3.2-1B model files";

// clone from https://huggingface.co/meta-llama/Llama-3.2-3B
const std::string LLAMA32_MODEL_DIR_3B = "path to llama3.2-3B model files";

const std::string INPUT_STR = "llamas eat";

void demo_llama32_impl(tinygpt::GPTModelSize size) {
  LOGI("demo_llama3(), size: %s", toString(size).c_str());

  std::string modelDir;
  std::string modelFile;
  switch (size) {
    case tinygpt::GPTModelSize::SIZE_1B:
      modelDir = LLAMA32_MODEL_DIR_1B;
      modelFile = "/model.safetensors";
      break;
    case tinygpt::GPTModelSize::SIZE_3B:
      modelDir = LLAMA32_MODEL_DIR_3B;
      modelFile = "/model.safetensors.index.json";
      break;
    default:
      break;
  }

  tinygpt::GPTConfig config;
  config.modelType = tinygpt::GPTModelType::LLAMA32;
  config.modelSize = size;
  config.modelFilePath = modelDir + modelFile;
  config.tokenizerPath = modelDir + "/tokenizer.json";
  config.tokenizerConfigPath = modelDir + "/tokenizer_config.json";
  config.device = tinytorch::DeviceType::CUDA;
  config.samplerConfig.temperature = 0.6f;
  config.samplerConfig.topP = 0.9f;
  config.maxNewTokens = 128;

  tinygpt::GPTEngine engine(config);
  bool success = engine.prepare();
  if (!success) {
    LOGE("Prepare engine failed");
    return;
  }

  tinytorch::Timer timer;
  timer.start();

  auto output = engine.generateSync(INPUT_STR);
  LOGI("output: \n%s", output.text.c_str());

  timer.mark();
  LOGI("Time cost: %lld ms, speed: %.2f token/s", timer.elapseMillis(),
       output.tokenIds.size() * 1000.0f / timer.elapseMillis());
}

void demo_llama32() {
  demo_llama32_impl(tinygpt::GPTModelSize::SIZE_1B);
  demo_llama32_impl(tinygpt::GPTModelSize::SIZE_3B);
}
