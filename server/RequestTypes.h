/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <string>
#include <vector>

#include "engine/GPTEngine.h"

namespace tinygpt::server {

struct ServerConfig {
  std::string modelDir;
  std::string host = "0.0.0.0";
  int port = 8080;
  std::string webDir;

  tinytorch::Device device = tinytorch::DeviceType::CUDA;
  tinytorch::DType dtype = tinytorch::DType::BFloat16;

  SamplerConfig samplerConfig = {0.7f, 0, 0.9f, 0.0f};
  int32_t maxNewTokens = 4096;
  int32_t maxBatchTokens = 8192;

  int32_t prefillChunkSize = 512;
  int32_t maxGraphBatch = 64;

  // 1 = single gpu
  int32_t tensorParallelSize = 1;
  std::string tpInitMethod = "env://";  // env://, tcp://host:port, file://path

  std::string chatTemplate;
};

struct InferenceRequest {
  std::string prompt;
  float temperature;
  float topP;
  float minP = 0.0f;
  int32_t maxTokens;
  bool stream = false;

  std::vector<std::string> stopStrings;
  std::vector<int32_t> stopTokenIds;
  bool includeStopStrInOutput = false;
};

}  // namespace tinygpt::server
