/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "Base.h"
#include "Split.h"
#include "TemplateProcessing.h"

namespace tinygpt::tokenizer::huggingface {

struct ConfigAddedToken {
  int32_t id;
  std::string content;
  bool singleWord;
  bool lStrip;
  bool rStrip;
  bool normalized;
  bool special;
};

struct Config {
  ComponentType type;
  virtual ~Config() = default;
};

struct ConfigByteLevel : Config {
  bool addPrefixSpace;
  bool trimOffsets;
  bool useRegex;
};

struct ConfigSplit : Config {
  std::string pattern;
  SplitDelimiterBehavior behavior;
  bool invert;
};

struct ConfigBPE : Config {
  bool ignoreMerges;
  ankerl::unordered_dense::map<std::string, int32_t> vocab;
  ankerl::unordered_dense::map<StringPair, int32_t, StringPairHash> merges;
};

struct ConfigTemplateProcessing : Config {
  std::vector<TemplateElement> single;
  std::vector<TemplateElement> pair;
  ankerl::unordered_dense::map<std::string, std::vector<int32_t>> specialTokens;
};

struct ConfigSequence : Config {
  std::vector<std::unique_ptr<Config>> configs;
};

struct ConfigHuggingface {
  // version
  std::string version;

  // truncation

  // padding

  // added_tokens
  std::vector<ConfigAddedToken> addedTokens;

  // normalizer
  std::unique_ptr<Config> normalizer;

  // preTokenizer
  std::unique_ptr<Config> preTokenizer;

  // model
  std::unique_ptr<Config> model;

  // postProcessor;
  std::unique_ptr<Config> postProcessor;

  // decoder
  std::unique_ptr<Config> decoder;

  // config
  bool addBosToken;
  bool addEosToken;

  ConfigAddedToken bosToken;
  ConfigAddedToken eosToken;
  ConfigAddedToken padToken;

  int32_t modelMaxLength;
  std::string chatTemplate;
};

bool load(ConfigHuggingface& cfg, const std::string& tokenizerPath, const std::string& cfgPath);

std::unique_ptr<Component> createComponent(const std::unique_ptr<Config>& cfg);

}  // namespace tinygpt::tokenizer::huggingface
