/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "HttpServer.h"

#include "ChatTemplateUtils.h"
#include "util/PathUtils.h"

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
  std::string tokenizerPath = PathUtils::joinPath(config_.modelDir, "tokenizer.json");
  std::string tokenizerCfgPath = PathUtils::joinPath(config_.modelDir, "tokenizer_config.json");
  if (!tokenizer_->initWithConfig(tokenizerPath, tokenizerCfgPath)) {
    LOGE("HttpServer: failed to load tokenizer from: %s", config_.modelDir.c_str());
    return false;
  }

  // apply chat template with priority:
  //   1. CLI override (--chat-template)
  //   2. Model built-in (from tokenizer_config.json)
  //   3. ChatML (if vocab has <|im_start|>/<|im_end|>)
  //   4. Simple fallback using model's own eos_token
  if (!config_.chatTemplate.empty()) {
    LOGI("HttpServer: using custom chat template from --chat-template");
    tokenizer_->setChatTemplate(config_.chatTemplate);
  } else if (!tokenizer_->hasChatTemplate()) {
    // Check if the vocabulary supports ChatML special tokens
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

  // init inference engine
  GPTConfig gptConfig;
  gptConfig.modelDir = config_.modelDir;
  gptConfig.device = config_.device;
  gptConfig.dtype = config_.dtype;
  gptConfig.samplerConfig = config_.samplerConfig;
  gptConfig.maxNewTokens = config_.maxNewTokens;

  engine_ = std::make_unique<GPTEngine>(gptConfig);
  if (!engine_->prepare()) {
    LOGE("HttpServer: failed to prepare engine");
    return false;
  }

  // start inference worker thread
  workerRunning_ = true;
  workerThread_ = std::thread(&HttpServer::workerLoop, this);

  // setup HTTP server
  impl_ = std::make_unique<Impl>();
  setupRoutes();

  // serve static web files
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

  // signal worker thread to exit
  {
    std::lock_guard<std::mutex> lock(queueMutex_);
    workerRunning_ = false;
  }
  queueCV_.notify_one();

  if (workerThread_.joinable()) {
    workerThread_.join();
  }
}

void HttpServer::workerLoop() {
  LOGI("HttpServer: inference worker started");
  while (true) {
    std::shared_ptr<InferenceTask> task;
    {
      std::unique_lock<std::mutex> lock(queueMutex_);
      queueCV_.wait(lock, [this] { return !taskQueue_.empty() || !workerRunning_; });
      if (!workerRunning_ && taskQueue_.empty()) {
        break;
      }
      task = std::move(taskQueue_.front());
      taskQueue_.pop();
    }

    if (!task) continue;

    const auto& req = task->request;

    // reconfigure engine for this request
    SamplerConfig samplerConfig(req.temperature, 0, req.topP, req.minP);

    // merge stop token IDs: chatTemplateStopIds_ + request-level stopTokenIds
    std::vector<int32_t> allStopTokenIds = chatTemplateStopIds_;
    for (auto id : req.stopTokenIds) {
      allStopTokenIds.push_back(id);
    }
    engine_->reconfigure(samplerConfig, req.maxTokens, allStopTokenIds);

    if (req.stream) {
      // streaming mode: use generateAsync with per-token callback
      auto output = engine_->generateAsync(req.prompt, [&task](const std::string& tokenText) -> bool {
        if (task->streamCallback) {
          return task->streamCallback(tokenText);
        }
        return true;
      });
      if (task->streamDone) task->streamDone(true, output.finishReason);
    } else {
      // non-stream mode: use generateSync, deliver result via promise
      std::vector<std::string> prompts = {req.prompt};
      GPTOutput output = engine_->generateSync(prompts);
      task->promise.set_value(std::move(output));
    }
  }
  LOGI("HttpServer: inference worker stopped");
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
      if (PathUtils::fileExists(PathUtils::joinPath(candidate, "index.html"))) {
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

  // CORS preflight
  svr.Options("/(.*)", [](const httplib::Request&, httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
    res.status = 204;
  });

  // GET /v1/models
  svr.Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) { handleListModels(&req, &res); });

  // POST /v1/chat/completions
  svr.Post("/v1/chat/completions",
           [this](const httplib::Request& req, httplib::Response& res) { handleChatCompletions(&req, &res); });

  // POST /v1/completions
  svr.Post("/v1/completions",
           [this](const httplib::Request& req, httplib::Response& res) { handleCompletions(&req, &res); });
}

}  // namespace tinygpt::server
