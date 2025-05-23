/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "ConfigGPT2.h"

#include <fstream>
#include <sstream>
#include <string>

#include "rapidjson/document.h"

namespace tinygpt::tokenizer::gpt2 {

static bool loadEncoder(ConfigGPT2& cfg, const std::string& encoderPath) {
  std::ifstream in(encoderPath);
  if (!in) {
    LOGE("Cannot open file: %s", encoderPath.c_str());
    return false;
  }
  std::stringstream buffer;
  buffer << in.rdbuf();
  std::string content = buffer.str();

  rapidjson::Document doc;
  if (doc.Parse(content.c_str()).HasParseError()) {
    LOGE("Parse encoder error: %s", "JSON parse error");
    return false;
  }

  if (!doc.IsObject() || doc.ObjectEmpty()) {
    LOGE("Encoder file empty: %s", encoderPath.c_str());
    return false;
  }

  for (auto it = doc.MemberBegin(); it != doc.MemberEnd(); ++it) {
    std::string k = it->name.GetString();
    int32_t v = it->value.GetInt();
    cfg.vocab.insert({k, v});
  }
  return true;
}

static bool loadVocab(ConfigGPT2& cfg, const std::string& vocabPath) {
  std::ifstream in(vocabPath);
  if (!in) {
    LOGE("Cannot open file: %s", vocabPath.c_str());
    return false;
  }

  std::string line;
  int32_t idx = 0;
  while (std::getline(in, line)) {
    // skip empty line or comments
    if (line.empty() || line[0] == '#') {
      continue;
    }
    size_t sep = line.find(' ');
    cfg.merges[{line.substr(0, sep), line.substr(sep + 1)}] = idx;
    idx++;
  }

  return true;
}

bool load(ConfigGPT2& cfg, const std::string& encoderPath, const std::string& vocabPath) {
  return loadEncoder(cfg, encoderPath) && loadVocab(cfg, vocabPath);
}

}  // namespace tinygpt::tokenizer::gpt2