/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "GPTEngine.h"
#include "Utils/Timer.h"
#include "util/StringUtils.h"

// clone from https://huggingface.co/openai-community/gpt2
const std::string GPT2_MODEL_DIR = "path to gpt2 model files";
const std::string INPUT_STR = "Alan Turing theorized that computers would one day become";

void demo_gpt2() {
  LOGI("demo_gpt2()");

  tinygpt::GPTConfig config;
  config.modelDir = GPT2_MODEL_DIR;
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
