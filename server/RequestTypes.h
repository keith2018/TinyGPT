/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <functional>
#include <future>
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
  int64_t maxNewTokens = 4096;

  std::string chatTemplate;
};

struct InferenceRequest {
  std::string prompt;
  float temperature;
  float topP;
  float minP = 0.0f;
  int64_t maxTokens;
  bool stream = false;

  std::vector<std::string> stopStrings;
  std::vector<int32_t> stopTokenIds;
  bool includeStopStrInOutput = false;
};

struct InferenceTask {
  InferenceRequest request;

  // non-stream
  std::promise<GPTOutput> promise;

  // stream
  std::function<bool(const std::string& chunk)> streamCallback;
  std::function<void(bool success, FinishReason reason)> streamDone;
};

}  // namespace tinygpt::server
