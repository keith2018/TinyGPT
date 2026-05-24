/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <atomic>
#include <memory>

#include "RequestTypes.h"
#include "tokenizer/Tokenizer.h"

namespace tinygpt::server {

class HttpServer {
 public:
  HttpServer();
  ~HttpServer();

  bool start(const ServerConfig& config);
  void stop();

 private:
  void setupRoutes();
  void setupStaticFiles() const;

  void handleListModels(const void* req, void* res) const;
  void handleChatCompletions(const void* req, void* res);
  void handleCompletions(const void* req, void* res);
  void dispatchGenerate(const InferenceRequest& inferReq, bool isChatCompletion, const void* rawReq, void* rawRes);

  std::string generateRequestId() const;
  static std::string buildErrorResponse(int code, const std::string& message, const std::string& type);
  static std::string buildSSEChunk(const std::string& requestId, const std::string& model, const std::string& content,
                                   bool isChatCompletion, const std::string& finishReason = "",
                                   bool isRoleChunk = false);
  static std::string buildSSERoleChunk(const std::string& requestId, const std::string& model, bool isChatCompletion) {
    return buildSSEChunk(requestId, model, "", isChatCompletion, "", true);
  }

  ServerConfig config_;
  std::string modelName_;

  std::unique_ptr<GPTEngine> engine_;
  std::unique_ptr<tokenizer::Tokenizer> tokenizer_;
  std::vector<int32_t> chatTemplateStopIds_;
  bool useChatMLFallback_ = false;
  bool noChatTemplate_ = false;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tinygpt::server

#define CPPHTTPLIB_THREAD_POOL_COUNT 256
#define CPPHTTPLIB_THREAD_POOL_MAX_COUNT 512
#include "cpp-httplib/httplib.h"

struct tinygpt::server::HttpServer::Impl {
  httplib::Server svr;
  std::atomic<int64_t> requestCounter{0};
};
