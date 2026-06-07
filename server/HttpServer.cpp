/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "HttpServer.h"

#include "ChatTemplateUtils.h"
#include "util/FileUtils.h"

namespace tinygpt::server {

HttpServer::HttpServer() = default;

HttpServer::~HttpServer() { stop(); }

bool HttpServer::start(const ServerConfig& config) {
  config_ = config;

  // extract model name from directory path
  modelName_ = config_.modelDir;
  auto pos = modelName_.find_last_of("/\\");
  if (pos != std::string::npos) {
    modelName_ = modelName_.substr(pos + 1);
  }

  // load tokenizer (for chat template)
  tokenizer_ = std::make_unique<tokenizer::Tokenizer>();
  std::string tokenizerPath = fileutil::join(config_.modelDir, "tokenizer.json");
  std::string tokenizerCfgPath = fileutil::join(config_.modelDir, "tokenizer_config.json");
  if (!tokenizer_->initWithConfig(tokenizerPath, tokenizerCfgPath)) {
    LOGE("HttpServer: failed to load tokenizer from: %s", config_.modelDir.c_str());
    return false;
  }

  // apply chat template: CLI override > model built-in > ChatML fallback
  if (!config_.chatTemplate.empty()) {
    LOGI("HttpServer: using custom chat template from --chat-template");
    tokenizer_->setChatTemplate(config_.chatTemplate);
  } else if (!tokenizer_->hasChatTemplate()) {
    // check if vocabulary supports ChatML special tokens
    auto imStartEnc = tokenizer_->encode("<|im_start|>");
    auto imEndEnc = tokenizer_->encode("<|im_end|>");
    bool hasChatMLTokens = (imStartEnc.size() == 1 && imEndEnc.size() == 1);

    if (hasChatMLTokens) {
      LOGW("HttpServer: model has no chat template, falling back to default ChatML");
      tokenizer_->setChatTemplate(kDefaultChatMLTemplate);
      chatTemplateStopIds_.push_back(imEndEnc[0]);
      useChatMLFallback_ = true;
      LOGI("HttpServer: added <|im_end|> (id=%d) as extra stop token for ChatML", imEndEnc[0]);
    } else {
      LOGW(
          "HttpServer: model has no chat template and vocabulary lacks ChatML tokens. "
          "/v1/chat/completions will be unavailable. Use --chat-template to specify one, "
          "or use /v1/completions for raw text generation.");
      noChatTemplate_ = true;
    }
  }

  GPTConfig gptConfig;
  gptConfig.modelDir = config_.modelDir;
  gptConfig.device = config_.device;
  gptConfig.dtype = config_.dtype;
  gptConfig.samplerConfig = config_.samplerConfig;
  gptConfig.maxNewTokens = config_.maxNewTokens;
  gptConfig.maxBatchTokens = config_.maxBatchTokens;
  gptConfig.prefillChunkSize = config_.prefillChunkSize;
  gptConfig.maxGraphBatch = config_.maxGraphBatch;
  gptConfig.tensorParallelSize = config_.tensorParallelSize;

  engine_ = std::make_unique<GPTEngine>(gptConfig);
  if (!engine_->prepare()) {
    LOGE("HttpServer: failed to prepare engine");
    return false;
  }

  impl_ = std::make_unique<Impl>();
  setupRoutes();
  setupStaticFiles();

  LOGI("HttpServer: starting on %s:%d (async mode)", config_.host.c_str(), config_.port);
  LOGI("HttpServer: model loaded: %s", modelName_.c_str());

  if (!impl_->svr.listen(config_.host, config_.port)) {
    LOGE("HttpServer: failed to listen on %s:%d", config_.host.c_str(), config_.port);
    return false;
  }
  return true;
}

void HttpServer::stop() {
  if (impl_) {
    impl_->svr.stop();
  }
}

void HttpServer::setupStaticFiles() const {
  auto& svr = impl_->svr;

  std::string webDir = config_.webDir;

  if (webDir.empty()) {
    std::vector<std::string> candidates = {
        "web",            // <build>/web  (copied by CMake)
        "server/web",     // running from project root
        "../server/web",  // running from build/
    };
    for (const auto& candidate : candidates) {
      if (fileutil::exists(fileutil::join(candidate, "index.html"))) {
        webDir = candidate;
        break;
      }
    }
  }

  if (webDir.empty()) {
    LOGW("HttpServer: web directory not found, chat UI disabled. Use --web-dir to specify.");
    return;
  }

  if (!svr.set_mount_point("/", webDir)) {
    LOGW("HttpServer: failed to mount web directory: %s", webDir.c_str());
    return;
  }

  LOGI("HttpServer: serving web UI from: %s", webDir.c_str());
  LOGI("HttpServer: open http://%s:%d/ in your browser", config_.host.c_str(), config_.port);
}

void HttpServer::setupRoutes() {
  auto& svr = impl_->svr;

  svr.Options("/(.*)", [](const httplib::Request&, httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
    res.status = 204;
  });

  svr.Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) { handleListModels(&req, &res); });

  svr.Post("/v1/chat/completions",
           [this](const httplib::Request& req, httplib::Response& res) { handleChatCompletions(&req, &res); });

  svr.Post("/v1/completions",
           [this](const httplib::Request& req, httplib::Response& res) { handleCompletions(&req, &res); });
}

}  // namespace tinygpt::server
