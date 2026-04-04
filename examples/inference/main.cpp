/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "Utils/Profiler.h"
#include "Utils/Timer.h"
#include "engine/GPTEngine.h"
#include "util/StringUtils.h"

const std::vector<std::string> INPUT_STRS = {
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
};

static void printUsage(const char* progName) {
  LOGI("Usage: %s [options]", progName);
  LOGI("Options:");
  LOGI("  --model <path>        Path to HuggingFace model directory (required)");
  LOGI("  --device <cpu|cuda>   Device type (default: cuda)");
  LOGI("  --dtype <fp32|fp16|bf16>  Data type (default: bf16)");
  LOGI("  --max-tokens <n>      Max new tokens (default: 32)");
  LOGI("  --temperature <f>     Sampling temperature (default: 0.8)");
  LOGI("  --top-p <f>           Top-p sampling (default: 0.9)");
  LOGI("  --help                Show this help message");
}

int main(int argc, char** argv) {
  std::string modelDir;
  std::string device = "cuda";
  std::string dtype = "bf16";
  int maxTokens = 32;
  float temperature = 0.8f;
  float topP = 0.9f;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      return 0;
    }
    if (arg == "--model" && i + 1 < argc) {
      modelDir = argv[++i];
    } else if (arg == "--device" && i + 1 < argc) {
      device = argv[++i];
    } else if (arg == "--dtype" && i + 1 < argc) {
      dtype = argv[++i];
    } else if (arg == "--max-tokens" && i + 1 < argc) {
      maxTokens = std::atoi(argv[++i]);
    } else if (arg == "--temperature" && i + 1 < argc) {
      temperature = std::strtof(argv[++i], nullptr);
    } else if (arg == "--top-p" && i + 1 < argc) {
      topP = std::strtof(argv[++i], nullptr);
    } else {
      LOGE("Unknown argument: %s", arg.c_str());
      printUsage(argv[0]);
      return 1;
    }
  }

  if (modelDir.empty()) {
    LOGE("Error: --model is required");
    printUsage(argv[0]);
    return 1;
  }

  tinygpt::GPTConfig config;
  config.modelDir = modelDir;
  config.samplerConfig.temperature = temperature;
  config.samplerConfig.topP = topP;
  config.maxNewTokens = maxTokens;

  if (device == "cpu") {
    config.device = tinytorch::DeviceType::CPU;
  } else {
    config.device = tinytorch::DeviceType::CUDA;
  }

  if (dtype == "fp32") {
    config.dtype = tinytorch::DType::Float32;
  } else if (dtype == "fp16") {
    config.dtype = tinytorch::DType::Float16;
  } else {
    config.dtype = tinytorch::DType::BFloat16;
  }

  tinygpt::GPTEngine engine(config);
  bool success = engine.prepare();
  if (!success) {
    LOGE("Prepare engine failed");
    return 1;
  }

  tinytorch::Timer timer;
  timer.start();

  PROFILE_START();
  auto output = engine.generateSync(INPUT_STRS);
  PROFILE_STOP();

  LOGI("Generated Outputs:");
  LOGI("------------------------------------------------------------");
  for (auto i = 0; i < output.batch; i++) {
    LOGI("Prompt:    '%s'", INPUT_STRS[i].c_str());
    LOGI("Output:    '%s'", tinygpt::StringUtils::repr(output.texts[i]).c_str());
    LOGI("------------------------------------------------------------");
  }

  timer.mark();
  LOGI("Time cost: %lld ms, speed: %.2f token/s", timer.elapseMillis(),
       output.tokenIds.size() * 1000.0f / timer.elapseMillis());

  return 0;
}
