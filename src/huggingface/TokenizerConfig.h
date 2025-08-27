/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "tokenizer/Base.h"
#include "tokenizer/Split.h"
#include "tokenizer/TemplateProcessing.h"

namespace tinygpt::huggingface::tokenizer {

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
  virtual ~Config() = default;
  tinygpt::tokenizer::ComponentType type;
};

struct ConfigByteLevel : Config {
  bool addPrefixSpace;
  bool trimOffsets;
  bool useRegex;
};

struct ConfigSplit : Config {
  std::string pattern;
  tinygpt::tokenizer::SplitDelimiterBehavior behavior;
  bool invert;
};

struct ConfigBPE : Config {
  bool ignoreMerges;
  ankerl::unordered_dense::map<std::string, int32_t> vocab;
  ankerl::unordered_dense::map<tinygpt::tokenizer::StringPair, int32_t, tinygpt::tokenizer::StringPairHash> merges;
};

struct ConfigTemplateProcessing : Config {
  std::vector<tinygpt::tokenizer::TemplateElement> single;
  std::vector<tinygpt::tokenizer::TemplateElement> pair;
  ankerl::unordered_dense::map<std::string, std::vector<int32_t>> specialTokens;
};

struct ConfigSequence : Config {
  std::vector<std::unique_ptr<Config>> configs;
};

struct TokenizerConfig {
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

bool load(TokenizerConfig& cfg, const std::string& tokenizerPath, const std::string& cfgPath);

std::unique_ptr<tinygpt::tokenizer::Component> createComponent(const std::unique_ptr<Config>& cfg);

}  // namespace tinygpt::huggingface::tokenizer
