/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <queue>
#include <thread>

#include "Utils/Profiler.h"
#include "Utils/Timer.h"
#include "engine/GPTEngine.h"

const std::string DEFAULT_INPUT = "The future of AI is";

static void printUsage(const char* progName) {
  LOGI("Usage: %s [options]", progName);
  LOGI("Options:");
  LOGI("  --model <path>        Path to HuggingFace model directory (required)");
  LOGI("  --device <cpu|cuda>   Device type (default: cuda)");
  LOGI("  --dtype <fp32|fp16|bf16>  Data type (default: bf16)");
  LOGI("  --max-tokens <n>      Max new tokens (default: 32)");
  LOGI("  --max-graph-batch <n> Max batch size for CUDA Graph capture (default: 64)");
  LOGI("  --temperature <f>     Sampling temperature (default: 0.8)");
  LOGI("  --top-p <f>           Top-p sampling (default: 0.9)");
  LOGI("  --input <text>        Input prompt (default: '%s')", DEFAULT_INPUT.c_str());
  LOGI("  --help                Show this help message");
}

int main(int argc, char** argv) {
  std::string modelDir;
  std::string device = "cuda";
  std::string dtype = "bf16";
  int maxTokens = 32;
  int maxGraphBatch = 64;
  float temperature = 0.8f;
  float topP = 0.9f;
  std::string input = DEFAULT_INPUT;

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
      maxTokens = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
    } else if (arg == "--max-graph-batch" && i + 1 < argc) {
      maxGraphBatch = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
    } else if (arg == "--temperature" && i + 1 < argc) {
      temperature = std::strtof(argv[++i], nullptr);
    } else if (arg == "--top-p" && i + 1 < argc) {
      topP = std::strtof(argv[++i], nullptr);
    } else if (arg == "--input" && i + 1 < argc) {
      input = argv[++i];
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
  config.maxGraphBatch = maxGraphBatch;

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

  LOGI("Prompt: '%s'", input.c_str());
  LOGI("Streaming output:");

  std::mutex printMutex;
  std::condition_variable printCV;
  std::queue<std::string> printQueue;
  std::atomic<bool> generationDone{false};

  std::thread printer([&]() {
    while (true) {
      std::string chunk;
      {
        std::unique_lock<std::mutex> lock(printMutex);
        printCV.wait(lock, [&] { return !printQueue.empty() || generationDone.load(); });
        if (printQueue.empty() && generationDone.load()) {
          break;
        }
        chunk = std::move(printQueue.front());
        printQueue.pop();
      }
      std::fwrite(chunk.data(), 1, chunk.size(), stdout);
      std::fflush(stdout);
    }
  });

  int tokenCount = 0;
  auto output = engine.generate(input, [&](const std::string& tokenText) -> bool {
    tokenCount++;
    {
      std::lock_guard<std::mutex> lock(printMutex);
      printQueue.push(tokenText);
    }
    printCV.notify_one();
    return true;
  });

  generationDone.store(true);
  printCV.notify_all();
  printer.join();

  // flush stream output
  std::fputc('\n', stdout);
  std::fflush(stdout);

  const char* reasonStr = "Unknown";
  switch (output.finishReason) {
    case tinygpt::FinishReason::Stop:
      reasonStr = "Stop";
      break;
    case tinygpt::FinishReason::Length:
      reasonStr = "Length";
      break;
    case tinygpt::FinishReason::Aborted:
      reasonStr = "Aborted";
      break;
  }
  LOGI("Finish reason: %s", reasonStr);

  PROFILE_STOP();
  timer.mark();
  LOGI("Time cost: %lld ms, tokens: %d, speed: %.2f token/s", timer.elapseMillis(), tokenCount,
       tokenCount * 1000.0f / timer.elapseMillis());

  return 0;
}
