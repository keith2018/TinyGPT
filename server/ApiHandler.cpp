/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */
#include <atomic>
#include <chrono>

#include "ChatTemplateUtils.h"
#include "HttpServer.h"
#include "ServerUtils.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "tokenizer/ChatTemplate.h"

namespace tinygpt::server {

namespace rj = rapidjson;

void HttpServer::handleListModels(const void* rawReq, void* rawRes) const {
  (void)rawReq;
  auto& res = *static_cast<httplib::Response*>(rawRes);

  rj::Document doc(rj::kObjectType);
  auto& alloc = doc.GetAllocator();

  doc.AddMember("object", "list", alloc);

  rj::Value dataArr(rj::kArrayType);
  {
    rj::Value modelObj(rj::kObjectType);
    modelObj.AddMember("id", rj::Value(modelName_.c_str(), alloc), alloc);
    modelObj.AddMember("object", "model", alloc);
    auto now = std::chrono::system_clock::now();
    auto created = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    modelObj.AddMember("created", created, alloc);
    modelObj.AddMember("owned_by", "tinygpt", alloc);

    // Capabilities: tell the frontend which API modes are available
    rj::Value capabilities(rj::kObjectType);
    capabilities.AddMember("chat", !noChatTemplate_, alloc);
    capabilities.AddMember("completion", true, alloc);
    modelObj.AddMember("capabilities", capabilities, alloc);

    dataArr.PushBack(modelObj, alloc);
  }
  doc.AddMember("data", dataArr, alloc);

  rj::StringBuffer buf;
  rj::Writer<rj::StringBuffer> writer(buf);
  doc.Accept(writer);

  res.set_header("Access-Control-Allow-Origin", "*");
  res.set_content(buf.GetString(), "application/json");
}

void HttpServer::handleChatCompletions(const void* rawReq, void* rawRes) {
  auto& req = *static_cast<const httplib::Request*>(rawReq);
  auto& res = *static_cast<httplib::Response*>(rawRes);

  // Reject if no chat template is available
  if (noChatTemplate_) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.status = 400;
    res.set_content(buildErrorResponse(400,
                                       "This model does not have a chat template. "
                                       "Use --chat-template to specify one at startup, "
                                       "or use /v1/completions for raw text generation.",
                                       "invalid_request_error"),
                    "application/json");
    return;
  }

  // parse request JSON
  rj::Document reqDoc;
  reqDoc.Parse(req.body.c_str());
  if (reqDoc.HasParseError() || !reqDoc.IsObject()) {
    res.status = 400;
    res.set_content(buildErrorResponse(400, "Invalid JSON in request body", "invalid_request_error"),
                    "application/json");
    return;
  }

  // extract messages
  if (!reqDoc.HasMember("messages") || !reqDoc["messages"].IsArray()) {
    res.status = 400;
    res.set_content(buildErrorResponse(400, "Missing or invalid 'messages' field", "invalid_request_error"),
                    "application/json");
    return;
  }

  std::vector<tokenizer::ChatMessage> messages;
  const auto& msgArr = reqDoc["messages"].GetArray();
  for (const auto& m : msgArr) {
    if (!m.IsObject() || !m.HasMember("role")) {
      continue;
    }
    tokenizer::ChatMessage msg;
    msg.role = m["role"].GetString();

    // Support content as string or array format
    if (m.HasMember("content")) {
      const auto& contentVal = m["content"];
      if (contentVal.IsString()) {
        msg.content = contentVal.GetString();
      } else if (contentVal.IsArray()) {
        // Extract text parts from content array format (OpenAI multimodal)
        for (const auto& part : contentVal.GetArray()) {
          if (part.IsObject() && part.HasMember("type") && part.HasMember("text")) {
            if (std::string(part["type"].GetString()) == "text") {
              if (!msg.content.empty()) msg.content += "\n";
              msg.content += part["text"].GetString();
            }
          } else if (part.IsString()) {
            if (!msg.content.empty()) msg.content += "\n";
            msg.content += part.GetString();
          }
        }
      }
    }

    messages.push_back(std::move(msg));
  }

  if (messages.empty()) {
    res.status = 400;
    res.set_content(buildErrorResponse(400, "No valid messages provided", "invalid_request_error"), "application/json");
    return;
  }

  // parse add_generation_prompt (default true, per vLLM)
  bool addGenerationPrompt = true;
  if (reqDoc.HasMember("add_generation_prompt") && reqDoc["add_generation_prompt"].IsBool()) {
    addGenerationPrompt = reqDoc["add_generation_prompt"].GetBool();
  }

  // apply chat template
  std::string prompt = tokenizer_->applyChatTemplate(messages, addGenerationPrompt);
  if (prompt.empty()) {
    res.status = 500;
    res.set_content(buildErrorResponse(500, "Failed to apply chat template. Try specifying --chat-template at startup.",
                                       "server_error"),
                    "application/json");
    return;
  }

  // build inference request with defaults
  InferenceRequest inferReq;
  inferReq.prompt = std::move(prompt);
  inferReq.temperature = config_.samplerConfig.temperature;
  inferReq.topP = config_.samplerConfig.topP;
  inferReq.minP = config_.samplerConfig.minP;
  inferReq.maxTokens = config_.maxNewTokens;
  inferReq.stream = false;

  // parse common params (temperature, top_p, min_p, max_tokens, stop, etc.)
  parseCommonInferenceParams(reqDoc, inferReq);

  // validate sampling params
  auto validationError = validateSamplingParams(inferReq);
  if (!validationError.empty()) {
    res.status = 400;
    res.set_content(buildErrorResponse(400, validationError, "invalid_request_error"), "application/json");
    return;
  }

  dispatchGenerate(inferReq, /*isChatCompletion=*/true, rawReq, rawRes);
}

void HttpServer::handleCompletions(const void* rawReq, void* rawRes) {
  auto& req = *static_cast<const httplib::Request*>(rawReq);
  auto& res = *static_cast<httplib::Response*>(rawRes);

  // parse request JSON
  rj::Document reqDoc;
  reqDoc.Parse(req.body.c_str());
  if (reqDoc.HasParseError() || !reqDoc.IsObject()) {
    res.status = 400;
    res.set_content(buildErrorResponse(400, "Invalid JSON in request body", "invalid_request_error"),
                    "application/json");
    return;
  }

  // extract prompt
  if (!reqDoc.HasMember("prompt") || !reqDoc["prompt"].IsString()) {
    res.status = 400;
    res.set_content(buildErrorResponse(400, "Missing or invalid 'prompt' field", "invalid_request_error"),
                    "application/json");
    return;
  }

  // build inference request with defaults
  InferenceRequest inferReq;
  inferReq.prompt = reqDoc["prompt"].GetString();
  inferReq.temperature = config_.samplerConfig.temperature;
  inferReq.topP = config_.samplerConfig.topP;
  inferReq.minP = config_.samplerConfig.minP;
  inferReq.maxTokens = config_.maxNewTokens;
  inferReq.stream = false;

  // parse common params
  parseCommonInferenceParams(reqDoc, inferReq);

  // validate sampling params
  auto validationError = validateSamplingParams(inferReq);
  if (!validationError.empty()) {
    res.status = 400;
    res.set_content(buildErrorResponse(400, validationError, "invalid_request_error"), "application/json");
    return;
  }

  dispatchGenerate(inferReq, /*isChatCompletion=*/false, rawReq, rawRes);
}

void HttpServer::dispatchGenerate(const InferenceRequest& inferReq, bool isChatCompletion, const void* /*rawReq*/,
                                  void* rawRes) {
  auto& res = *static_cast<httplib::Response*>(rawRes);
  res.set_header("Access-Control-Allow-Origin", "*");

  std::string requestId = generateRequestId();

  if (inferReq.stream) {
    // --- SSE Streaming Mode ---
    // Use httplib's chunked content provider for Server-Sent Events
    auto task = std::make_shared<InferenceTask>();
    task->request = inferReq;

    // Synchronization between worker thread (producer) and HTTP chunked provider (consumer)
    auto streamMutex = std::make_shared<std::mutex>();
    auto streamCV = std::make_shared<std::condition_variable>();
    auto streamChunks = std::make_shared<std::queue<std::string>>();
    auto streamFinished = std::make_shared<bool>(false);

    // Client disconnect flag: set by chunked provider when client disconnects,
    // checked by streamCallback to abort generation immediately.
    auto clientDisconnected = std::make_shared<std::atomic<bool>>(false);

    // Buffer for incomplete UTF-8 sequences across token boundaries
    auto utf8Buffer = std::make_shared<std::string>();

    // Accumulated text for stop string checking during streaming
    auto accumulatedText = std::make_shared<std::string>();
    auto sentLength = std::make_shared<size_t>(0);  // how many chars already sent to client
    auto generationAborted = std::make_shared<bool>(false);

    // Build complete list of stop patterns: user stop strings + ChatML/fallback tag patterns
    auto allStopPatterns = std::make_shared<std::vector<std::string>>(inferReq.stopStrings);
    if (isChatCompletion) {
      if (useChatMLFallback_) {
        allStopPatterns->emplace_back("<|im_start|>");
        allStopPatterns->emplace_back("<|im_end|>");
      }
    }

    // Compute stop buffer size = max stop pattern length - 1
    // This prevents sending partial stop strings to the client
    size_t maxStopLen = 0;
    for (const auto& s : *allStopPatterns) {
      maxStopLen = std::max(maxStopLen, s.size());
    }
    auto stopBufferSize = std::make_shared<size_t>(maxStopLen > 0 ? maxStopLen - 1 : 0);

    // Compute prompt tokens for usage stats (before dispatching)
    auto promptTokens = std::make_shared<size_t>(tokenizer_->encode(inferReq.prompt).size());
    auto completionTokens = std::make_shared<size_t>(0);

    // Capture model name and request-level includeStopStrInOutput for closures
    auto modelName = modelName_;
    auto includeStopStr = inferReq.includeStopStrInOutput;

    // Send initial role chunk (per OpenAI SSE protocol, chat completions only)
    if (isChatCompletion) {
      std::string roleChunk = buildSSERoleChunk(requestId, modelName, isChatCompletion);
      std::lock_guard<std::mutex> lock(*streamMutex);
      streamChunks->push(std::move(roleChunk));
      streamCV->notify_one();
    }

    // Worker callback: push SSE chunks into the queue
    task->streamCallback = [=](const std::string& tokenText) -> bool {
      if (*generationAborted || clientDisconnected->load()) return false;

      // Count completion tokens
      (*completionTokens)++;

      // Append new token bytes to the UTF-8 buffer
      utf8Buffer->append(tokenText);

      // Check if the buffer ends with an incomplete UTF-8 sequence
      size_t bufLen = utf8Buffer->size();
      size_t incomplete = incompleteUtf8Tail(*utf8Buffer);

      // Clamp to buffer size (defensive)
      if (incomplete > bufLen) incomplete = bufLen;

      size_t completeLen = bufLen - incomplete;

      if (completeLen == 0) return true;  // wait for more bytes

      // Extract complete UTF-8 portion
      std::string completeText(utf8Buffer->data(), completeLen);
      if (incomplete > 0) {
        *utf8Buffer = utf8Buffer->substr(completeLen);
      } else {
        utf8Buffer->clear();
      }

      // Accumulate text for stop string checking
      accumulatedText->append(completeText);

      // Check for stop patterns in accumulated text (search only in newly added region)
      for (const auto& stop : *allStopPatterns) {
        if (stop.empty()) continue;
        // Only search where the new text could have completed a match
        size_t searchStart = 0;
        if (accumulatedText->size() > completeText.size() + stop.size()) {
          searchStart = accumulatedText->size() - completeText.size() - stop.size();
        }
        auto pos = accumulatedText->find(stop, searchStart);
        if (pos != std::string::npos) {
          // Stop matched! Determine truncation point
          size_t truncateAt = includeStopStr ? pos + stop.size() : pos;

          // Send any unsent text up to truncation point
          if (truncateAt > *sentLength) {
            std::string toSend = accumulatedText->substr(*sentLength, truncateAt - *sentLength);
            std::string sseData = buildSSEChunk(requestId, modelName, toSend, isChatCompletion);
            {
              std::lock_guard<std::mutex> lock(*streamMutex);
              streamChunks->push(std::move(sseData));
            }
            streamCV->notify_one();
          }
          *sentLength = truncateAt;
          *generationAborted = true;
          return false;  // abort generation
        }
      }

      // Stop buffer: hold back characters to avoid sending partial stop strings
      if (accumulatedText->size() > *sentLength + *stopBufferSize) {
        size_t safeLen = accumulatedText->size() - *sentLength - *stopBufferSize;
        std::string toSend = accumulatedText->substr(*sentLength, safeLen);
        *sentLength += safeLen;

        std::string sseData = buildSSEChunk(requestId, modelName, toSend, isChatCompletion);
        {
          std::lock_guard<std::mutex> lock(*streamMutex);
          streamChunks->push(std::move(sseData));
        }
        streamCV->notify_one();
      }
      return true;
    };

    task->streamDone = [=](bool success, FinishReason reason) {
      // If client already disconnected, skip all finalization — nobody is listening
      if (clientDisconnected->load()) {
        std::lock_guard<std::mutex> lock(*streamMutex);
        *streamFinished = true;
        streamCV->notify_one();
        return;
      }

      // Flush any remaining bytes in the UTF-8 buffer
      if (!utf8Buffer->empty()) {
        accumulatedText->append(*utf8Buffer);
        utf8Buffer->clear();
      }

      // Determine finish_reason
      std::string finishStr;
      if (*generationAborted) {
        finishStr = "stop";
      } else {
        finishStr = (reason == FinishReason::Stop) ? "stop" : "length";
      }

      // Flush remaining accumulated text (stop buffer residue)
      if (!*generationAborted && accumulatedText->size() > *sentLength) {
        std::string remaining = accumulatedText->substr(*sentLength);

        // Final check for stop patterns in the remaining text
        for (const auto& stop : *allStopPatterns) {
          if (stop.empty()) continue;
          auto pos = remaining.find(stop);
          if (pos != std::string::npos) {
            remaining = includeStopStr ? remaining.substr(0, pos + stop.size()) : remaining.substr(0, pos);
            finishStr = "stop";
            break;
          }
        }

        if (!remaining.empty()) {
          std::string sseData = buildSSEChunk(requestId, modelName, remaining, isChatCompletion);
          std::lock_guard<std::mutex> lock(*streamMutex);
          streamChunks->push(std::move(sseData));
        }
      }

      // Send final chunk with finish_reason
      {
        rj::Document doc(rj::kObjectType);
        auto& alloc = doc.GetAllocator();
        doc.AddMember("id", rj::Value(requestId.c_str(), alloc), alloc);
        doc.AddMember(
            "object",
            isChatCompletion ? rj::Value("chat.completion.chunk", alloc) : rj::Value("text_completion", alloc), alloc);
        auto now = std::chrono::system_clock::now();
        auto created = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
        doc.AddMember("created", created, alloc);
        doc.AddMember("model", rj::Value(modelName.c_str(), alloc), alloc);

        rj::Value choices(rj::kArrayType);
        rj::Value choice(rj::kObjectType);
        choice.AddMember("index", 0, alloc);
        if (isChatCompletion) {
          rj::Value delta(rj::kObjectType);
          choice.AddMember("delta", delta, alloc);
        } else {
          choice.AddMember("text", "", alloc);
        }
        choice.AddMember("finish_reason", rj::Value(finishStr.c_str(), alloc), alloc);
        choices.PushBack(choice, alloc);
        doc.AddMember("choices", choices, alloc);

        // Usage statistics in the final chunk
        rj::Value usage(rj::kObjectType);
        usage.AddMember("prompt_tokens", static_cast<int64_t>(*promptTokens), alloc);
        usage.AddMember("completion_tokens", static_cast<int64_t>(*completionTokens), alloc);
        usage.AddMember("total_tokens", static_cast<int64_t>(*promptTokens + *completionTokens), alloc);
        doc.AddMember("usage", usage, alloc);

        rj::StringBuffer buf;
        rj::Writer<rj::StringBuffer> writer(buf);
        doc.Accept(writer);

        {
          std::string finalChunk = "data: " + std::string(buf.GetString()) + "\n\n";
          std::lock_guard<std::mutex> lock(*streamMutex);
          streamChunks->push(std::move(finalChunk));
          streamChunks->emplace("data: [DONE]\n\n");
          *streamFinished = true;
        }
        streamCV->notify_one();
      }
    };

    // Enqueue task
    {
      std::lock_guard<std::mutex> lock(queueMutex_);
      taskQueue_.push(task);
    }
    queueCV_.notify_one();

    // Set up chunked response (SSE)
    res.set_header("Content-Type", "text/event-stream");
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");

    res.set_chunked_content_provider("text/event-stream", [=](size_t /*offset*/, httplib::DataSink& sink) -> bool {
      while (true) {
        std::string chunk;
        bool finished = false;
        {
          std::unique_lock<std::mutex> lock(*streamMutex);
          streamCV->wait(lock, [&] { return !streamChunks->empty() || *streamFinished; });

          if (!streamChunks->empty()) {
            chunk = std::move(streamChunks->front());
            streamChunks->pop();
          }
          finished = *streamFinished && streamChunks->empty();
        }

        if (!chunk.empty()) {
          if (!sink.write(chunk.c_str(), chunk.size())) {
            // Client disconnected — signal worker to abort generation
            clientDisconnected->store(true);
            return false;
          }
        }

        if (finished) {
          sink.done();
          return true;
        }
      }
    });

  } else {
    // --- Non-Stream Async Mode ---
    // Submit task to worker queue, block on future for result
    auto task = std::make_shared<InferenceTask>();
    task->request = inferReq;
    auto future = task->promise.get_future();

    {
      std::lock_guard<std::mutex> lock(queueMutex_);
      taskQueue_.push(task);
    }
    queueCV_.notify_one();

    // Wait for worker to complete inference (HTTP thread yields to other connections via httplib)
    GPTOutput output = future.get();

    if (output.texts.empty()) {
      res.status = 500;
      res.set_content(buildErrorResponse(500, "Generation failed", "server_error"), "application/json");
      return;
    }

    // build response
    auto now = std::chrono::system_clock::now();
    auto created = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();

    // post-process: strip template markers for chat completions
    std::string outputText = output.texts[0];
    if (isChatCompletion) {
      if (useChatMLFallback_) {
        outputText = stripChatMLTags(outputText);
      }
    }

    // Determine finish_reason from engine
    std::string finishStr = (output.finishReason == FinishReason::Stop) ? "stop" : "length";

    // Check stop strings and truncate if matched
    if (!inferReq.stopStrings.empty()) {
      auto [truncated, found] = checkStopStrings(outputText, inferReq.stopStrings, inferReq.includeStopStrInOutput);
      if (found) {
        outputText = truncated;
        finishStr = "stop";
      }
    }

    auto promptTokens = tokenizer_->encode(inferReq.prompt).size();
    auto completionTokens = static_cast<size_t>(output.tokenIds.size());

    rj::Document respDoc(rj::kObjectType);
    auto& alloc = respDoc.GetAllocator();

    respDoc.AddMember("id", rj::Value(requestId.c_str(), alloc), alloc);
    respDoc.AddMember(
        "object", isChatCompletion ? rj::Value("chat.completion", alloc) : rj::Value("text_completion", alloc), alloc);
    respDoc.AddMember("created", created, alloc);
    respDoc.AddMember("model", rj::Value(modelName_.c_str(), alloc), alloc);

    // choices
    rj::Value choices(rj::kArrayType);
    {
      rj::Value choice(rj::kObjectType);
      choice.AddMember("index", 0, alloc);

      if (isChatCompletion) {
        rj::Value message(rj::kObjectType);
        message.AddMember("role", "assistant", alloc);
        message.AddMember("content", rj::Value(outputText.c_str(), alloc), alloc);
        choice.AddMember("message", message, alloc);
      } else {
        choice.AddMember("text", rj::Value(outputText.c_str(), alloc), alloc);
      }

      choice.AddMember("finish_reason", rj::Value(finishStr.c_str(), alloc), alloc);
      choices.PushBack(choice, alloc);
    }
    respDoc.AddMember("choices", choices, alloc);

    // usage
    rj::Value usage(rj::kObjectType);
    usage.AddMember("prompt_tokens", static_cast<int64_t>(promptTokens), alloc);
    usage.AddMember("completion_tokens", static_cast<int64_t>(completionTokens), alloc);
    usage.AddMember("total_tokens", static_cast<int64_t>(promptTokens + completionTokens), alloc);
    respDoc.AddMember("usage", usage, alloc);

    rj::StringBuffer buf;
    rj::Writer<rj::StringBuffer> writer(buf);
    respDoc.Accept(writer);

    res.set_content(buf.GetString(), "application/json");
  }
}

std::string HttpServer::generateRequestId() const {
  auto now = std::chrono::system_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  auto counter = impl_->requestCounter.fetch_add(1);
  return "chatcmpl-" + std::to_string(ms) + "-" + std::to_string(counter);
}

std::string HttpServer::buildErrorResponse(int code, const std::string& message, const std::string& type) {
  rj::Document doc(rj::kObjectType);
  auto& alloc = doc.GetAllocator();

  rj::Value errorObj(rj::kObjectType);
  errorObj.AddMember("message", rj::Value(message.c_str(), alloc), alloc);
  errorObj.AddMember("type", rj::Value(type.c_str(), alloc), alloc);
  errorObj.AddMember("code", code, alloc);
  doc.AddMember("error", errorObj, alloc);

  rj::StringBuffer buf;
  rj::Writer<rj::StringBuffer> writer(buf);
  doc.Accept(writer);
  return buf.GetString();
}

std::string HttpServer::buildSSEChunk(const std::string& requestId, const std::string& model,
                                      const std::string& content, bool isChatCompletion,
                                      const std::string& finishReason) {
  rj::Document doc(rj::kObjectType);
  auto& alloc = doc.GetAllocator();

  doc.AddMember("id", rj::Value(requestId.c_str(), alloc), alloc);
  doc.AddMember("object",
                isChatCompletion ? rj::Value("chat.completion.chunk", alloc) : rj::Value("text_completion", alloc),
                alloc);
  auto now = std::chrono::system_clock::now();
  auto created = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
  doc.AddMember("created", created, alloc);
  doc.AddMember("model", rj::Value(model.c_str(), alloc), alloc);

  rj::Value choices(rj::kArrayType);
  {
    rj::Value choice(rj::kObjectType);
    choice.AddMember("index", 0, alloc);

    if (isChatCompletion) {
      rj::Value delta(rj::kObjectType);
      delta.AddMember("content", rj::Value(content.c_str(), alloc), alloc);
      choice.AddMember("delta", delta, alloc);
    } else {
      choice.AddMember("text", rj::Value(content.c_str(), alloc), alloc);
    }

    if (!finishReason.empty()) {
      choice.AddMember("finish_reason", rj::Value(finishReason.c_str(), alloc), alloc);
    } else {
      choice.AddMember("finish_reason", rj::Value(rj::kNullType), alloc);
    }

    choices.PushBack(choice, alloc);
  }
  doc.AddMember("choices", choices, alloc);

  rj::StringBuffer buf;
  rj::Writer<rj::StringBuffer> writer(buf);
  doc.Accept(writer);

  return "data: " + std::string(buf.GetString()) + "\n\n";
}

std::string HttpServer::buildSSERoleChunk(const std::string& requestId, const std::string& model,
                                          bool isChatCompletion) {
  rj::Document doc(rj::kObjectType);
  auto& alloc = doc.GetAllocator();

  doc.AddMember("id", rj::Value(requestId.c_str(), alloc), alloc);
  doc.AddMember("object",
                isChatCompletion ? rj::Value("chat.completion.chunk", alloc) : rj::Value("text_completion", alloc),
                alloc);
  auto now = std::chrono::system_clock::now();
  auto created = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
  doc.AddMember("created", created, alloc);
  doc.AddMember("model", rj::Value(model.c_str(), alloc), alloc);

  rj::Value choices(rj::kArrayType);
  {
    rj::Value choice(rj::kObjectType);
    choice.AddMember("index", 0, alloc);

    if (isChatCompletion) {
      rj::Value delta(rj::kObjectType);
      delta.AddMember("role", "assistant", alloc);
      delta.AddMember("content", "", alloc);
      choice.AddMember("delta", delta, alloc);
    } else {
      choice.AddMember("text", "", alloc);
    }

    choice.AddMember("finish_reason", rj::Value(rj::kNullType), alloc);
    choices.PushBack(choice, alloc);
  }
  doc.AddMember("choices", choices, alloc);

  rj::StringBuffer buf;
  rj::Writer<rj::StringBuffer> writer(buf);
  doc.Accept(writer);

  return "data: " + std::string(buf.GetString()) + "\n\n";
}

}  // namespace tinygpt::server
