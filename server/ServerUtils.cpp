/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "ServerUtils.h"

#include "rapidjson/document.h"

namespace tinygpt::server {

namespace rj = rapidjson;

size_t incompleteUtf8Tail(const std::string& s) {
  if (s.empty()) return 0;

  // scan backwards: count continuation bytes (10xxxxxx)
  size_t i = s.size();
  size_t continuations = 0;
  while (i > 0) {
    auto c = static_cast<uint8_t>(s[i - 1]);
    if ((c & 0xC0) == 0x80) {
      // continuation byte
      ++continuations;
      --i;
    } else {
      // leading byte (or ASCII)
      break;
    }
  }

  if (i == 0) {
    // all continuation bytes with no leading byte — all incomplete
    return continuations;
  }

  auto lead = static_cast<uint8_t>(s[i - 1]);
  size_t expected = 0;  // expected continuation bytes for this leading byte

  if ((lead & 0x80) == 0x00) {
    expected = 0;  // ASCII
  } else if ((lead & 0xE0) == 0xC0) {
    expected = 1;  // 110xxxxx -> 2-byte sequence
  } else if ((lead & 0xF0) == 0xE0) {
    expected = 2;  // 1110xxxx -> 3-byte sequence
  } else if ((lead & 0xF8) == 0xF0) {
    expected = 3;  // 11110xxx -> 4-byte sequence
  } else {
    return 0;  // invalid leading byte, treat as complete
  }

  if (continuations < expected) {
    // incomplete: return the leading byte + all its continuation bytes so far
    return continuations + 1;
  }

  return 0;  // sequence is complete
}

std::pair<std::string, bool> checkStopStrings(const std::string& text, const std::vector<std::string>& stopStrings,
                                              bool includeStop) {
  if (stopStrings.empty()) return {text, false};

  size_t earliest = std::string::npos;
  size_t matchLen = 0;
  for (const auto& stop : stopStrings) {
    if (stop.empty()) continue;
    auto pos = text.find(stop);
    if (pos != std::string::npos && (earliest == std::string::npos || pos < earliest)) {
      earliest = pos;
      matchLen = stop.size();
    }
  }

  if (earliest == std::string::npos) {
    return {text, false};
  }

  if (includeStop) {
    return {text.substr(0, earliest + matchLen), true};
  } else {
    return {text.substr(0, earliest), true};
  }
}

std::string validateSamplingParams(const InferenceRequest& req) {
  if (req.temperature < 0.0f) return "'temperature' must be >= 0, got " + std::to_string(req.temperature);
  if (req.topP <= 0.0f || req.topP > 1.0f) return "'top_p' must be in (0, 1], got " + std::to_string(req.topP);
  if (req.minP < 0.0f || req.minP > 1.0f) return "'min_p' must be in [0, 1], got " + std::to_string(req.minP);
  if (req.maxTokens < 1) return "'max_tokens' must be >= 1, got " + std::to_string(req.maxTokens);
  return "";
}

void parseCommonInferenceParams(const rj::Document& reqDoc, InferenceRequest& inferReq) {
  if (reqDoc.HasMember("temperature") && reqDoc["temperature"].IsNumber()) {
    inferReq.temperature = reqDoc["temperature"].GetFloat();
  }
  if (reqDoc.HasMember("top_p") && reqDoc["top_p"].IsNumber()) {
    inferReq.topP = reqDoc["top_p"].GetFloat();
  }
  if (reqDoc.HasMember("min_p") && reqDoc["min_p"].IsNumber()) {
    inferReq.minP = reqDoc["min_p"].GetFloat();
  }
  if (reqDoc.HasMember("max_tokens") && reqDoc["max_tokens"].IsInt64()) {
    inferReq.maxTokens = reqDoc["max_tokens"].GetInt64();
  }
  // max_completion_tokens as alias for max_tokens (OpenAI compatibility)
  if (reqDoc.HasMember("max_completion_tokens") && reqDoc["max_completion_tokens"].IsInt64()) {
    inferReq.maxTokens = reqDoc["max_completion_tokens"].GetInt64();
  }
  if (reqDoc.HasMember("stream") && reqDoc["stream"].IsBool()) {
    inferReq.stream = reqDoc["stream"].GetBool();
  }

  // stop strings (string or array of strings)
  if (reqDoc.HasMember("stop")) {
    const auto& stopVal = reqDoc["stop"];
    if (stopVal.IsString()) {
      inferReq.stopStrings.emplace_back(stopVal.GetString());
    } else if (stopVal.IsArray()) {
      for (const auto& s : stopVal.GetArray()) {
        if (s.IsString()) {
          inferReq.stopStrings.emplace_back(s.GetString());
        }
      }
    }
  }

  // stop_token_ids
  if (reqDoc.HasMember("stop_token_ids") && reqDoc["stop_token_ids"].IsArray()) {
    for (const auto& id : reqDoc["stop_token_ids"].GetArray()) {
      if (id.IsInt()) {
        inferReq.stopTokenIds.push_back(static_cast<int32_t>(id.GetInt()));
      }
    }
  }

  // include_stop_str_in_output
  if (reqDoc.HasMember("include_stop_str_in_output") && reqDoc["include_stop_str_in_output"].IsBool()) {
    inferReq.includeStopStrInOutput = reqDoc["include_stop_str_in_output"].GetBool();
  }
}

}  // namespace tinygpt::server
