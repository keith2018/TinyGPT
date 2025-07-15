/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Base.h"

namespace tinygpt::tokenizer::gpt2 {

struct ConfigGPT2 {
  ankerl::unordered_dense::map<std::string, int32_t> vocab;
  ankerl::unordered_dense::map<StringPair, int32_t, StringPairHash> merges;
};

bool load(ConfigGPT2& cfg, const std::string& vocabPath, const std::string& mergesPath);

}  // namespace tinygpt::tokenizer::gpt2