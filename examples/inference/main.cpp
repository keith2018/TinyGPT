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

#include "Distributed/BackendNCCL.h"
#include "Distributed/DistributedProcessGroup.h"
#include "Utils/CUDAUtils.h"
#include "Utils/Profiler.h"
#include "Utils/Timer.h"
#include "distributed/Communicator.h"
#include "distributed/WorkerRuntime.h"
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
  LOGI("  --tensor-parallel <n> Tensor parallel size (default: 1)");
  LOGI("                        Each rank is launched as a separate process via env://");
  LOGI("                        (set RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT)");
  LOGI("  --tp-init <s>         Init method (default: env://; e.g. tcp://host:port)");
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
  int tensorParallelSize = 1;
  std::string tpInitMethod = "env://";

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
    } else if (arg == "--tensor-parallel" && i + 1 < argc) {
      tensorParallelSize = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
    } else if (arg == "--tp-init" && i + 1 < argc) {
      tpInitMethod = argv[++i];
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
  config.tensorParallelSize = tensorParallelSize;

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

  int tpRank = 0;
  if (tensorParallelSize > 1) {
    if (!config.device.isCuda()) {
      LOGE("Error: --tensor-parallel > 1 requires --device cuda");
      return 1;
    }
    auto& dpg = tinytorch::distributed::DistributedProcessGroup::getInstance();
    if (!dpg->initProcessGroup(tinytorch::distributed::NCCL, tpInitMethod, -1, tensorParallelSize)) {
      LOGE("Failed to init process group");
      return 1;
    }
    auto pg = dpg->getProcessGroup();
    // enable single-stream nccl so collectives run in-order on the compute stream
    auto backend =
        std::dynamic_pointer_cast<tinytorch::distributed::BackendNCCL>(pg->getBackend(tinytorch::distributed::NCCL));
    if (backend) backend->setUseComputeStream(true);

    tinygpt::distributed::Communicator::tp().init(pg);
    tpRank = pg->getRank();

    config.device = tinytorch::Device(tinytorch::DeviceType::CUDA, static_cast<tinytorch::DeviceIndex>(tpRank));
    tinytorch::cuda::setDevice(tpRank);
    LOGI("TP rank=%d worldSize=%d device=cuda:%d", tpRank, tensorParallelSize, tpRank);
  }

  auto teardownTP = [&]() {
    if (tensorParallelSize > 1) {
      tinygpt::distributed::Communicator::tp().reset();
      tinytorch::distributed::DistributedProcessGroup::getInstance()->destroyProcessGroup();
    }
  };

  if (tpRank != 0) {
    {
      tinygpt::GPTEngine engine(config);
      if (!engine.prepare()) {
        LOGE("Worker rank=%d: prepare failed", tpRank);
        teardownTP();
        return 1;
      }
      tinygpt::distributed::WorkerRuntime worker(*engine.model(), *engine.pagedCache(), engine.maxBatchTokens());
      worker.run();
    }
    teardownTP();
    return 0;
  }

  int rc = 0;
  {
    tinygpt::GPTEngine engine(config);
    bool success = engine.prepare();
    if (!success) {
      LOGE("Prepare engine failed");
      teardownTP();
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
  }

  teardownTP();
  return rc;
}
