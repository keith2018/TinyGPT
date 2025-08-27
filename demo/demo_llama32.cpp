/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "GPTEngine.h"
#include "Utils/Timer.h"
#include "util/StringUtils.h"

// clone from https://huggingface.co/meta-llama/Llama-3.2-1B
const std::string LLAMA32_MODEL_DIR_1B = "path to llama3.2-1B model files";

// clone from https://huggingface.co/meta-llama/Llama-3.2-3B
const std::string LLAMA32_MODEL_DIR_3B = "path to llama3.2-3B model files";

const std::string INPUT_STR = "llamas eat";

void demo_llama32_impl(const std::string &modelDir) {
  LOGI("demo_llama3(), dir: %s", modelDir.c_str());

  tinygpt::GPTConfig config;
  config.modelDir = modelDir;
  config.device = tinytorch::DeviceType::CUDA;

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

void demo_llama32() {
  demo_llama32_impl(LLAMA32_MODEL_DIR_1B);
  demo_llama32_impl(LLAMA32_MODEL_DIR_3B);
}
