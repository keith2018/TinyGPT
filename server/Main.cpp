/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include <csignal>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "Distributed/BackendNCCL.h"
#include "Distributed/DistributedProcessGroup.h"
#include "HttpServer.h"
#include "Utils/CUDAUtils.h"
#include "distributed/Communicator.h"
#include "distributed/WorkerRuntime.h"

using namespace tinygpt;
using namespace tinygpt::server;

static HttpServer* g_server = nullptr;

static void signalHandler(int sig) {
  LOGI("Received signal %d, shutting down...", sig);
  if (g_server) {
    g_server->stop();
  }
}

static void printUsage(const char* progName) {
  LOGI("Usage: %s [options]", progName);
  LOGI("Options:");
  LOGI("  --model <path>     Path to HuggingFace model directory (required)");
  LOGI("  --host <addr>      Server host address (default: 0.0.0.0)");
  LOGI("  --port <port>      Server port (default: 8080)");
  LOGI("  --max-tokens <n>   Max new tokens per request (default: 4096)");
  LOGI("  --max-batch-tokens <n>  Max tokens per batch step (default: 8192)");
  LOGI("  --prefill-chunk-size <n>  Max prefill tokens per step per seq (default: 512)");
  LOGI("  --max-graph-batch <n>  Max batch size for CUDA Graph capture (default: 64)");
  LOGI("  --temperature <f>  Sampling temperature (default: 0.7)");
  LOGI("  --top-p <f>        Top-p sampling (default: 0.9)");
  LOGI("  --min-p <f>        Min-p sampling (default: 0.0)");
  LOGI("  --chat-template <s> Custom chat template (Jinja2 string or file path)");
  LOGI("  --web-dir <path>   Path to web UI directory (auto-detected if omitted)");
  LOGI("  --tensor-parallel <n>  Tensor parallel size (default: 1)");
  LOGI("                     Each rank is launched as a separate process via env://");
  LOGI("                     (set RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT)");
  LOGI("  --tp-init <s>      Init method (default: env://; e.g. tcp://host:port)");
  LOGI("  --help             Show this help message");
}

int main(int argc, char** argv) {
  ServerConfig config;
  bool hasModel = false;

  auto parseInt = [](const char* s, auto& dst) {
    dst = static_cast<std::remove_reference_t<decltype(dst)>>(std::strtoll(s, nullptr, 10));
  };

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      return 0;
    }
    if (arg == "--model" && i + 1 < argc) {
      config.modelDir = argv[++i];
      hasModel = true;
    } else if (arg == "--host" && i + 1 < argc) {
      config.host = argv[++i];
    } else if (arg == "--port" && i + 1 < argc) {
      parseInt(argv[++i], config.port);
    } else if (arg == "--max-tokens" && i + 1 < argc) {
      parseInt(argv[++i], config.maxNewTokens);
    } else if (arg == "--max-batch-tokens" && i + 1 < argc) {
      parseInt(argv[++i], config.maxBatchTokens);
    } else if (arg == "--prefill-chunk-size" && i + 1 < argc) {
      parseInt(argv[++i], config.prefillChunkSize);
    } else if (arg == "--max-graph-batch" && i + 1 < argc) {
      parseInt(argv[++i], config.maxGraphBatch);
    } else if (arg == "--temperature" && i + 1 < argc) {
      config.samplerConfig.temperature = std::strtof(argv[++i], nullptr);
    } else if (arg == "--top-p" && i + 1 < argc) {
      config.samplerConfig.topP = std::strtof(argv[++i], nullptr);
    } else if (arg == "--min-p" && i + 1 < argc) {
      config.samplerConfig.minP = std::strtof(argv[++i], nullptr);
    } else if (arg == "--web-dir" && i + 1 < argc) {
      config.webDir = argv[++i];
    } else if (arg == "--tensor-parallel" && i + 1 < argc) {
      parseInt(argv[++i], config.tensorParallelSize);
    } else if (arg == "--tp-init" && i + 1 < argc) {
      config.tpInitMethod = argv[++i];
    } else if (arg == "--chat-template" && i + 1 < argc) {
      std::string val = argv[++i];
      bool isFile = false;
      auto dotPos = val.rfind('.');
      if (dotPos != std::string::npos) {
        std::string ext = val.substr(dotPos);
        if (ext == ".jinja" || ext == ".jinja2" || ext == ".txt" || ext == ".json") {
          isFile = true;
        }
      }
      if (!isFile && (val.find('/') != std::string::npos || val.find('\\') != std::string::npos)) {
        isFile = true;
      }
      if (isFile) {
        std::ifstream ifs(val);
        if (!ifs.is_open()) {
          LOGE("Error: cannot open chat template file: %s", val.c_str());
          return 1;
        }
        config.chatTemplate = std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
      } else {
        config.chatTemplate = val;
      }
    } else {
      LOGE("Unknown argument: %s", arg.c_str());
      printUsage(argv[0]);
      return 1;
    }
  }

  if (!hasModel) {
    LOGE("Error: --model is required");
    printUsage(argv[0]);
    return 1;
  }

  int tpRank = 0;
  if (config.tensorParallelSize > 1) {
    auto& dpg = tinytorch::distributed::DistributedProcessGroup::getInstance();
    if (!dpg->initProcessGroup(tinytorch::distributed::NCCL, config.tpInitMethod, -1, config.tensorParallelSize)) {
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
    LOGI("TP rank=%d worldSize=%d device=cuda:%d", tpRank, config.tensorParallelSize, tpRank);
  }

  // register signal handlers
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  auto teardownTP = [&]() {
    if (config.tensorParallelSize > 1) {
      tinygpt::distributed::Communicator::tp().reset();
      tinytorch::distributed::DistributedProcessGroup::getInstance()->destroyProcessGroup();
    }
  };

  if (tpRank != 0) {
    {
      GPTConfig gptCfg;
      gptCfg.modelDir = config.modelDir;
      gptCfg.device = config.device;
      gptCfg.dtype = config.dtype;
      gptCfg.samplerConfig = config.samplerConfig;
      gptCfg.maxNewTokens = config.maxNewTokens;
      gptCfg.maxBatchTokens = config.maxBatchTokens;
      gptCfg.prefillChunkSize = config.prefillChunkSize;
      gptCfg.maxGraphBatch = config.maxGraphBatch;
      gptCfg.tensorParallelSize = config.tensorParallelSize;

      GPTEngine engine(gptCfg);
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
    HttpServer server;
    g_server = &server;

    LOGI("============================================================");
    LOGI("  TinyGPT OpenAI-Compatible API Server");
    LOGI("============================================================");

    if (!server.start(config)) {
      LOGE("Failed to start server");
      rc = 1;
    }
    g_server = nullptr;
  }

  teardownTP();
  return rc;
}
