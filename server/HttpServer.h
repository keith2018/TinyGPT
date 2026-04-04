/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

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

  void workerLoop();

  // OpenAI-compatible API handlers (implemented in ApiHandler.cpp)
  void handleListModels(const void* req, void* res) const;
  void handleChatCompletions(const void* req, void* res);
  void handleCompletions(const void* req, void* res);

  // parse request params, build InferenceRequest, dispatch
  void dispatchGenerate(const InferenceRequest& inferReq, bool isChatCompletion, const void* rawReq, void* rawRes);

  // helpers (implemented in ApiHandler.cpp)
  std::string generateRequestId() const;
  static std::string buildErrorResponse(int code, const std::string& message, const std::string& type);
  static std::string buildSSEChunk(const std::string& requestId, const std::string& model, const std::string& content,
                            bool isChatCompletion, const std::string& finishReason = "");
  static std::string buildSSERoleChunk(const std::string& requestId, const std::string& model, bool isChatCompletion);

  ServerConfig config_;
  std::string modelName_;

  std::unique_ptr<GPTEngine> engine_;
  std::unique_ptr<tokenizer::Tokenizer> tokenizer_;
  std::vector<int32_t> chatTemplateStopIds_;  // extra stop tokens when using ChatML fallback
  bool useChatMLFallback_ = false;            // true when using ChatML fallback template
  bool noChatTemplate_ = false;               // true when model has no chat template at all

  // async task queue
  std::mutex queueMutex_;
  std::condition_variable queueCV_;
  std::queue<std::shared_ptr<InferenceTask>> taskQueue_;
  std::thread workerThread_;
  bool workerRunning_ = false;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tinygpt::server

#include "cpp-httplib/httplib.h"

struct tinygpt::server::HttpServer::Impl {
  httplib::Server svr;
  std::atomic<int64_t> requestCounter{0};
};
