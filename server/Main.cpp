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

#include "HttpServer.h"

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
  LOGI("  --temperature <f>  Sampling temperature (default: 0.7)");
  LOGI("  --top-p <f>        Top-p sampling (default: 0.9)");
  LOGI("  --min-p <f>        Min-p sampling (default: 0.0)");
  LOGI("  --chat-template <s> Custom chat template (Jinja2 string or file path)");
  LOGI("  --web-dir <path>   Path to web UI directory (auto-detected if omitted)");
  LOGI("  --help             Show this help message");
}

int main(int argc, char** argv) {
  ServerConfig config;
  bool hasModel = false;

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
      config.port = std::atoi(argv[++i]);
    } else if (arg == "--max-tokens" && i + 1 < argc) {
      config.maxNewTokens = std::atoll(argv[++i]);
    } else if (arg == "--temperature" && i + 1 < argc) {
      config.samplerConfig.temperature = std::strtof(argv[++i], nullptr);
    } else if (arg == "--top-p" && i + 1 < argc) {
      config.samplerConfig.topP = std::strtof(argv[++i], nullptr);
    } else if (arg == "--min-p" && i + 1 < argc) {
      config.samplerConfig.minP = std::strtof(argv[++i], nullptr);
    } else if (arg == "--web-dir" && i + 1 < argc) {
      config.webDir = argv[++i];
    } else if (arg == "--chat-template" && i + 1 < argc) {
      std::string val = argv[++i];
      // If the value looks like a file path, read its contents
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

  // register signal handlers
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  HttpServer server;
  g_server = &server;

  LOGI("============================================================");
  LOGI("  TinyGPT OpenAI-Compatible API Server");
  LOGI("============================================================");

  if (!server.start(config)) {
    LOGE("Failed to start server");
    return 1;
  }

  return 0;
}
