/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "GPTEngine.h"
#include "Utils/Timer.h"
#include "util/StringUtils.h"

// clone from huggingface
const std::string MODEL_DIR = "path to model files (huggingface repo)";
const std::string INPUT_STR = "The future of AI is";

void demo_gpt_impl(const std::string &modelDir) {
  LOGI("demo_gpt(), dir: %s", modelDir.c_str());

  tinygpt::GPTConfig config;
  config.modelDir = modelDir;
  config.device = tinytorch::DeviceType::CUDA;
  config.dtype = tinytorch::DType::BFloat16;
  config.samplerConfig.temperature = 0.8;
  config.samplerConfig.topP = 0.9;
  config.maxNewTokens = 32;

  tinygpt::GPTEngine engine(config);
  bool success = engine.prepare();
  if (!success) {
    LOGE("Prepare engine failed");
    return;
  }

  tinytorch::Timer timer;
  timer.start();

  auto output = engine.generateSync(INPUT_STR);

  LOGI("Prompt:\t'%s'", INPUT_STR.c_str());
  LOGI("Output:\t'%s'", tinygpt::StringUtils::repr(output.text).c_str());

  timer.mark();
  LOGI("Time cost: %lld ms, speed: %.2f token/s", timer.elapseMillis(),
       output.tokenIds.size() * 1000.0f / timer.elapseMillis());
}

void demo_gpt() { demo_gpt_impl(MODEL_DIR); }
